// ns-3 transport for the EGAT-MAPPO mesh simulator (protocol: docs/ns3/protocol.md).
// Reads JSON lines on stdin (init, step..., end), runs one step of packet-level 802.11ax per
// step message, writes one JSON result line on stdout. Logs go to stderr only.
//
// Forwarding is done here, not by an IP stack: each packet carries (source id, ttl) in its
// first payload bytes, every hop sends it with WifiNetDevice::Send to the next hop's MAC on the
// common channel, and a gateway counts it as delivered. This keeps routing exactly the
// action's next_hop tree, with no ARP or routing-protocol traffic.
#include "ns3/core-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/propagation-module.h"
#include "ns3/wifi-module.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace ns3;

// ---------- minimal JSON (numbers, strings, null, bool, arrays, objects) ----------
struct J
{
    enum Kind { NUL, NUM, STR, ARR, OBJ } kind{NUL};
    double num{0};
    std::string str;
    std::vector<J> arr;
    std::map<std::string, J> obj;

    const J& operator[](const std::string& k) const
    {
        auto it = obj.find(k);
        if (it == obj.end())
        {
            throw std::runtime_error("missing key: " + k);
        }
        return it->second;
    }
    bool has(const std::string& k) const { return obj.count(k) && obj.at(k).kind != NUL; }
};

struct JParser
{
    const std::string& s;
    size_t i{0};

    void ws()
    {
        while (i < s.size() && isspace(static_cast<unsigned char>(s[i])))
        {
            ++i;
        }
    }
    J parse()
    {
        ws();
        if (i >= s.size())
        {
            throw std::runtime_error("unexpected end of JSON");
        }
        J v;
        char c = s[i];
        if (c == '{')
        {
            v.kind = J::OBJ;
            ++i;
            ws();
            if (s[i] == '}')
            {
                ++i;
                return v;
            }
            while (true)
            {
                ws();
                std::string k = parse().str;
                ws();
                if (s[i++] != ':')
                {
                    throw std::runtime_error("expected ':'");
                }
                v.obj[k] = parse();
                ws();
                if (s[i] == ',')
                {
                    ++i;
                    continue;
                }
                if (s[i++] != '}')
                {
                    throw std::runtime_error("expected '}'");
                }
                return v;
            }
        }
        if (c == '[')
        {
            v.kind = J::ARR;
            ++i;
            ws();
            if (s[i] == ']')
            {
                ++i;
                return v;
            }
            while (true)
            {
                v.arr.push_back(parse());
                ws();
                if (s[i] == ',')
                {
                    ++i;
                    continue;
                }
                if (s[i++] != ']')
                {
                    throw std::runtime_error("expected ']'");
                }
                return v;
            }
        }
        if (c == '"')
        {
            v.kind = J::STR;
            ++i;
            while (s[i] != '"')
            {
                if (s[i] == '\\')
                {
                    ++i;
                }
                v.str += s[i++];
            }
            ++i;
            return v;
        }
        if (s.compare(i, 4, "null") == 0)
        {
            i += 4;
            return v;
        }
        if (s.compare(i, 4, "true") == 0)
        {
            i += 4;
            v.kind = J::NUM;
            v.num = 1;
            return v;
        }
        if (s.compare(i, 5, "false") == 0)
        {
            i += 5;
            v.kind = J::NUM;
            return v;
        }
        size_t end;
        v.kind = J::NUM;
        v.num = std::stod(s.substr(i), &end);
        i += end;
        return v;
    }
};

static J
ParseJson(const std::string& line)
{
    JParser p{line};
    return p.parse();
}

// ---------- bridge state ----------
// Radio model: every node has one WifiNetDevice per channel, each fixed on its channel for the
// whole run. The action's two channels per node say which of them are in use; a channel that
// enters a node's set becomes usable switch_delay_s later. Nothing retunes at runtime: ns-3.47
// runtime channel switching stalls queues (ADDBA timeouts cancelled, Txop access lost) and
// aborts on responses scheduled across the switch.
static const uint16_t PROTO = 0x88B5; // local experimental ethertype
static const uint32_t HDR = 3;        // payload bytes: src (2), ttl (1)

struct Bridge
{
    uint32_t n{0};
    uint32_t k{0}; // number of channels
    std::vector<bool> isGw;
    double stepS{0.1};
    uint32_t pktBytes{1200};
    double switchDelayS{0.1};
    double routeHoldS{0.1};
    bool flushOnSwitch{true}; // setting (a): leaving a channel loses its queue; false = setting (b)

    NodeContainer nodes;
    std::vector<std::vector<Ptr<WifiNetDevice>>> dev;                // [node][channel]
    std::map<Mac48Address, std::pair<uint32_t, uint32_t>> macOwner; // mac -> (node, channel)

    std::vector<std::vector<int>> channels;   // [node] -> the action's two channel indices
    std::vector<std::vector<double>> readyAt; // [node][channel] usable from this time
    std::vector<int> nextHop;
    std::vector<bool> alive;
    std::vector<double> holdUntil;
    std::vector<std::vector<Ptr<Packet>>> pending; // packets waiting for a route hold or a channel
    std::vector<double> flushAt;                   // latest scheduled release of pending, per node
    uint32_t queuePkts{1};
    std::vector<double> carryBytes;

    // per-step counters (reset each step)
    std::vector<double> delivered, dropOverflow, dropMac, dropSwitch, dropNoroute;
    std::map<std::pair<uint32_t, uint32_t>, std::vector<double>> links; // (u,v) -> ok, fail, snrSum, snrCnt, rateSum, rateCnt
    // cumulative per source, for in-flight bytes
    std::vector<double> genTotal, doneTotal;
};

static Bridge B;

static Ptr<Packet>
MakePacket(uint32_t src, uint8_t ttl)
{
    std::vector<uint8_t> buf(B.pktBytes, 0);
    buf[0] = src & 0xff;
    buf[1] = (src >> 8) & 0xff;
    buf[2] = ttl;
    return Create<Packet>(buf.data(), buf.size());
}

static void
ReadTag(Ptr<const Packet> p, uint32_t& src, uint8_t& ttl)
{
    uint8_t h[HDR];
    p->CopyData(h, HDR);
    src = h[0] | (h[1] << 8);
    ttl = h[2];
}

// our payload in one MSDU (LLC/SNAP header + payload), or nullptr
static Ptr<Packet>
MsduPayload(Ptr<const Packet> msdu)
{
    Ptr<Packet> p = msdu->Copy();
    LlcSnapHeader llc;
    if (p->GetSize() < llc.GetSerializedSize() + HDR)
    {
        return nullptr;
    }
    p->RemoveHeader(llc);
    return llc.GetType() == PROTO ? p : nullptr;
}

// our payloads inside a queued MPDU: one, or several when it is an A-MSDU
static std::vector<Ptr<Packet>>
MpduPayloads(Ptr<const WifiMpdu> mpdu)
{
    std::vector<Ptr<Packet>> out;
    if (!mpdu->GetHeader().IsData())
    {
        return out;
    }
    if (mpdu->GetHeader().IsQosAmsdu())
    {
        for (auto it = mpdu->begin(); it != mpdu->end(); ++it)
        {
            if (auto p = MsduPayload(it->first))
            {
                out.push_back(p);
            }
        }
    }
    else if (auto p = MsduPayload(mpdu->GetPacket()))
    {
        out.push_back(p);
    }
    return out;
}

static int
SourceOf(Ptr<const Packet> p)
{
    uint32_t src;
    uint8_t ttl;
    ReadTag(p, src, ttl);
    return src < B.n ? static_cast<int>(src) : -1;
}

static void
Drop(std::vector<double>& counter, uint32_t src, double bytes)
{
    counter[src] += bytes;
    B.doneTotal[src] += bytes;
}

// common channel of u and v: largest index in both action sets, as mesh.sim.graph.common_channel
static int
CommonChannel(uint32_t u, uint32_t v)
{
    int best = -1;
    for (int a : B.channels[u])
    {
        for (int b : B.channels[v])
        {
            if (a == b && a > best)
            {
                best = a;
            }
        }
    }
    return best;
}

static void Forward(uint32_t u, Ptr<Packet> p);

static void
FlushPending(uint32_t u)
{
    auto held = std::move(B.pending[u]);
    B.pending[u].clear();
    for (auto& p : held)
    {
        Forward(u, p); // packets still not ready are held again
    }
}

// Held packets share the node's queue limit (a policy that changes parent every step would
// otherwise grow them without bound), and one flush is scheduled per node and release time.
static void
Hold(uint32_t u, Ptr<Packet> p, double until)
{
    if (B.pending[u].size() >= B.queuePkts)
    {
        uint32_t src;
        uint8_t ttl;
        ReadTag(p, src, ttl);
        Drop(B.dropOverflow, src, p->GetSize());
        return;
    }
    B.pending[u].push_back(p);
    if (B.flushAt[u] < until - 1e-12)
    {
        B.flushAt[u] = until;
        Simulator::Schedule(Seconds(until - Simulator::Now().GetSeconds()), &FlushPending, u);
    }
}

static void
Forward(uint32_t u, Ptr<Packet> p)
{
    uint32_t src;
    uint8_t ttl;
    ReadTag(p, src, ttl);
    const double bytes = p->GetSize();
    const double now = Simulator::Now().GetSeconds() + 1e-12;
    if (!B.alive[u])
    {
        Drop(B.dropSwitch, src, bytes);
        return;
    }
    if (B.isGw[u])
    {
        B.delivered[src] += bytes;
        B.doneTotal[src] += bytes;
        return;
    }
    if (now < B.holdUntil[u])
    {
        Hold(u, p, B.holdUntil[u]);
        return;
    }
    const int v = B.nextHop[u];
    const int c = v >= 0 ? CommonChannel(u, v) : -1;
    if (c < 0 || ttl > B.n)
    {
        Drop(B.dropNoroute, src, bytes);
        return;
    }
    const double ready = std::max(B.readyAt[u][c], B.readyAt[v][c]);
    if (now < ready)
    {
        Hold(u, p, ready);
        return;
    }
    // ns-3.47's FCFS scheduler drops the newest MPDU of a full queue without any trace, so
    // overflow is detected here
    Ptr<WifiMacQueue> queue = B.dev[u][c]->GetMac()->GetTxopQueue(AC_BE);
    if (queue->GetNPackets() >= queue->GetMaxSize().GetValue())
    {
        Drop(B.dropOverflow, src, bytes);
        return;
    }
    Ptr<Packet> q = MakePacket(src, ttl + 1);
    B.dev[u][c]->Send(q, Mac48Address::ConvertFrom(B.dev[v][c]->GetAddress()), PROTO);
}

static bool
OnReceive(uint32_t node, Ptr<NetDevice>, Ptr<const Packet> p, uint16_t proto, const Address&)
{
    if (proto == PROTO)
    {
        Forward(node, p->Copy());
    }
    return true;
}

static std::vector<double>&
LinkRow(uint32_t u, uint32_t v)
{
    auto& row = B.links[{u, v}];
    if (row.empty())
    {
        row.assign(6, 0.0);
    }
    return row;
}

static void
OnDropped(uint32_t u, WifiMacDropReason reason, Ptr<const WifiMpdu> mpdu)
{
    auto payloads = MpduPayloads(mpdu);
    for (auto& p : payloads)
    {
        int src = SourceOf(p);
        if (src < 0)
        {
            continue;
        }
        Drop(reason == WIFI_MAC_DROP_REACHED_RETRY_LIMIT ? B.dropMac : B.dropOverflow, src, B.pktBytes);
    }
    auto it = B.macOwner.find(mpdu->GetHeader().GetAddr1());
    if (reason == WIFI_MAC_DROP_REACHED_RETRY_LIMIT && it != B.macOwner.end())
    {
        LinkRow(u, it->second.first)[1] += payloads.size();
    }
}

static void
OnAcked(uint32_t u, Ptr<const WifiMpdu> mpdu)
{
    auto it = B.macOwner.find(mpdu->GetHeader().GetAddr1());
    if (it != B.macOwner.end())
    {
        LinkRow(u, it->second.first)[0] += MpduPayloads(mpdu).size();
    }
}

static void
OnSniffRx(uint32_t v,
          Ptr<const Packet> packet,
          uint16_t,
          WifiTxVector txVector,
          MpduInfo info,
          SignalNoiseDbm sn,
          uint16_t)
{
    Ptr<Packet> c = packet->Copy();
    if (info.type != NORMAL_MPDU)
    {
        AmpduSubframeHeader sub;
        c->RemoveHeader(sub);
    }
    WifiMacHeader hdr;
    if (c->GetSize() < hdr.GetSerializedSize() || c->PeekHeader(hdr) == 0 || !hdr.IsData())
    {
        return;
    }
    auto it = B.macOwner.find(hdr.GetAddr2());
    if (it == B.macOwner.end() || it->second.first == v)
    {
        return;
    }
    auto& row = LinkRow(it->second.first, v);
    row[2] += sn.signal - sn.noise;
    row[3] += 1;
    row[4] += txVector.GetMode().GetDataRate(txVector) / 1e6;
    row[5] += 1;
}

// a node stops using channel c: setting (a) drops that radio's queue, setting (b) re-forwards it
static void
LeaveChannel(uint32_t u, uint32_t c)
{
    // Only MPDUs never sent are removed. One already given a sequence number is in flight or in
    // a Block Ack window, and pulling it out corrupts ns-3's BA state (bad_optional_access); it
    // finishes or fails on its own.
    Ptr<WifiMacQueue> q = B.dev[u][c]->GetMac()->GetTxopQueue(AC_BE);
    std::vector<Ptr<WifiMpdu>> fresh;
    for (Ptr<WifiMpdu> m = q->PeekFirstAvailable(SINGLE_LINK_OP_ID); m;
         m = q->PeekFirstAvailable(SINGLE_LINK_OP_ID, m))
    {
        if (!m->IsInFlight() && !m->HasSeqNoAssigned())
        {
            fresh.push_back(m);
        }
    }
    std::vector<Ptr<Packet>> moved;
    for (auto& mpdu : fresh)
    {
        q->Remove(mpdu);
        for (auto& p : MpduPayloads(mpdu))
        {
            if (B.flushOnSwitch)
            {
                if (int src = SourceOf(p); src >= 0)
                {
                    Drop(B.dropSwitch, src, B.pktBytes);
                }
            }
            else
            {
                moved.push_back(p);
            }
        }
    }
    for (auto& p : moved)
    {
        Forward(u, p);
    }
}

static void
Generate(uint32_t src)
{
    B.genTotal[src] += B.pktBytes;
    Forward(src, MakePacket(src, 0));
}

static double
QueueBytes(uint32_t u)
{
    double q = 0;
    for (auto& d : B.dev[u])
    {
        q += d->GetMac()->GetTxopQueue(AC_BE)->GetNBytes();
    }
    for (auto& p : B.pending[u])
    {
        q += p->GetSize();
    }
    return q;
}

// ---------- messages ----------
static void
Init(const J& m)
{
    B.n = static_cast<uint32_t>(m["n"].num);
    B.k = static_cast<uint32_t>(m["channels"].arr.size());
    B.stepS = m["step_s"].num;
    B.pktBytes = static_cast<uint32_t>(m["pkt_bytes"].num);
    B.switchDelayS = m["switch_delay_s"].num;
    B.routeHoldS = m["route_hold_s"].num;
    B.flushOnSwitch = !(m.has("queue_preserve") && m["queue_preserve"].num != 0);
    const uint32_t queuePkts =
        std::max<uint32_t>(1, static_cast<uint32_t>(m["queue_bytes"].num / B.pktBytes));
    B.queuePkts = queuePkts;
    if (m.has("ext_interf"))
    {
        std::cerr << "warning: ext_interf is not modelled in mesh_bridge; ignored\n";
    }
    if (m.has("seed"))
    {
        RngSeedManager::SetRun(static_cast<uint64_t>(m["seed"].num) + 1);
    }
    Config::SetDefault("ns3::WifiMacQueue::MaxSize",
                       QueueSizeValue(QueueSize(std::to_string(queuePkts) + "p")));
    // the sim keeps backlog until q_max overflows; do not expire queued packets on age
    Config::SetDefault("ns3::WifiMacQueue::MaxDelay", TimeValue(Seconds(3600)));

    const uint32_t n = B.n;
    B.isGw.assign(n, false);
    for (auto& g : m["gateways"].arr)
    {
        B.isGw[static_cast<uint32_t>(g.num)] = true;
    }
    B.channels.assign(n, {0, 1}); // MeshEnv.reset: every node starts on channels (0, 1)
    B.readyAt.assign(n, std::vector<double>(B.k, 0.0));
    B.nextHop.assign(n, -1);
    B.alive.assign(n, true);
    B.holdUntil.assign(n, 0.0);
    B.pending.assign(n, {});
    B.flushAt.assign(n, -1.0);
    B.carryBytes.assign(n, 0.0);
    B.genTotal.assign(n, 0.0);
    B.doneTotal.assign(n, 0.0);

    B.nodes.Create(n);
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> pos = CreateObject<ListPositionAllocator>();
    for (auto& xy : m["pos"].arr)
    {
        pos->Add(Vector(xy.arr[0].num, xy.arr[1].num, 0.0));
    }
    mobility.SetPositionAllocator(pos);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(B.nodes);

    // same propagation as mesh/sim/radio.py: log-distance, exponent 3, 46.6777 dB at 1 m
    Ptr<YansWifiChannel> channel = CreateObject<YansWifiChannel>();
    Ptr<LogDistancePropagationLossModel> loss = CreateObject<LogDistancePropagationLossModel>();
    loss->SetAttribute("Exponent", DoubleValue(3.0));
    loss->SetAttribute("ReferenceLoss", DoubleValue(46.6777));
    loss->SetAttribute("ReferenceDistance", DoubleValue(1.0));
    channel->SetPropagationLossModel(loss);
    channel->SetPropagationDelayModel(CreateObject<ConstantSpeedPropagationDelayModel>());

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211ax);
    // MAC options (init keys, all optional): "rts": 1 = RTS/CTS before every data frame;
    // "ampdu_bytes": max A-MPDU (ns-3 default 65535, 802.11ax allows up to 6500631);
    // "amsdu_bytes": max A-MSDU (ns-3 default 0 = off; 802.11ax up to 11398)
    const bool rts = m.has("rts") && m["rts"].num != 0;
    wifi.SetRemoteStationManager("ns3::IdealWifiManager", "RtsCtsThreshold", UintegerValue(rts ? 0 : 65535));
    WifiMacHelper mac;
    mac.SetType("ns3::AdhocWifiMac",
                "BE_MaxAmpduSize",
                UintegerValue(m.has("ampdu_bytes") ? static_cast<uint32_t>(m["ampdu_bytes"].num) : 65535),
                "BE_MaxAmsduSize",
                UintegerValue(m.has("amsdu_bytes") ? static_cast<uint32_t>(m["amsdu_bytes"].num) : 0));

    // The sim's channels differ only by index (same propagation and MCS table), so channel
    // index c is the 20 MHz 5 GHz channel 36 + 4c.
    B.dev.assign(n, {});
    for (uint32_t i = 0; i < n; ++i)
    {
        for (uint32_t c = 0; c < B.k; ++c)
        {
            YansWifiPhyHelper phy;
            phy.SetChannel(channel);
            phy.Set("ChannelSettings", StringValue("{" + std::to_string(36 + 4 * c) + ", 20, BAND_5GHZ, 0}"));
            phy.Set("TxPowerStart", DoubleValue(20.0));
            phy.Set("TxPowerEnd", DoubleValue(20.0));
            phy.Set("RxNoiseFigure", DoubleValue(7.0)); // -101 dBm thermal (20 MHz) + 7 = -94 dBm
            NetDeviceContainer d = wifi.Install(phy, mac, B.nodes.Get(i));
            Ptr<WifiNetDevice> wd = DynamicCast<WifiNetDevice>(d.Get(0));
            B.dev[i].push_back(wd);
            B.macOwner[Mac48Address::ConvertFrom(wd->GetAddress())] = {i, c};
            wd->SetReceiveCallback(MakeBoundCallback(&OnReceive, i));
            wd->GetMac()->TraceConnectWithoutContext("DroppedMpdu", MakeBoundCallback(&OnDropped, i));
            wd->GetMac()->TraceConnectWithoutContext("AckedMpdu", MakeBoundCallback(&OnAcked, i));
            wd->GetPhy()->TraceConnectWithoutContext("MonitorSnifferRx", MakeBoundCallback(&OnSniffRx, i));
        }
    }
}

static std::string
Arr(const std::vector<double>& v)
{
    std::ostringstream o;
    o.precision(10);
    o << "[";
    for (size_t i = 0; i < v.size(); ++i)
    {
        o << (i ? "," : "") << v[i];
    }
    o << "]";
    return o.str();
}

static bool
Has(const std::vector<int>& set, int c)
{
    return std::find(set.begin(), set.end(), c) != set.end();
}

static std::string
Step(const J& m)
{
    const uint32_t n = B.n;
    const double now = Simulator::Now().GetSeconds();
    for (auto* c : {&B.delivered, &B.dropOverflow, &B.dropMac, &B.dropSwitch, &B.dropNoroute})
    {
        c->assign(n, 0.0);
    }
    B.links.clear();

    // pass 1: new channel sets, readiness, route holds; queues are moved only after every node's
    // new state is known, so re-forwarded packets see this step's action
    std::vector<std::vector<uint32_t>> left(n);
    for (uint32_t i = 0; i < n; ++i)
    {
        const bool alive = m["alive"].arr[i].num > 0;
        if (B.alive[i] && !alive)
        {
            // ponytail: only the dead node's own in-flight bytes are written off; relayed packets
            // stuck in its queues stay in flight. Exact per-source accounting needs a queue walk.
            for (auto& d : B.dev[i])
            {
                d->GetPhy()->SetOffMode();
            }
            const double lost = B.genTotal[i] - B.doneTotal[i];
            if (lost > 0)
            {
                Drop(B.dropSwitch, i, lost);
            }
            B.pending[i].clear();
        }
        B.alive[i] = alive;
        std::vector<int> next = {static_cast<int>(m["channels"].arr[i].arr[0].num),
                                 static_cast<int>(m["channels"].arr[i].arr[1].num)};
        for (int c : B.channels[i])
        {
            if (!Has(next, c))
            {
                left[i].push_back(c);
            }
        }
        for (int c : next)
        {
            if (!Has(B.channels[i], c))
            {
                B.readyAt[i][c] = now + B.switchDelayS; // retune: unusable for the switch delay
            }
        }
        B.channels[i] = next;
        const int nh = static_cast<int>(m["next_hop"].arr[i].num);
        if (nh != B.nextHop[i] && !B.isGw[i] && alive && B.routeHoldS > 0)
        {
            B.holdUntil[i] = now + B.routeHoldS;
        }
        B.nextHop[i] = nh;
    }
    for (uint32_t i = 0; i < n; ++i)
    {
        for (uint32_t c : left[i])
        {
            LeaveChannel(i, c);
        }
    }

    for (uint32_t s = 0; s < n; ++s)
    {
        const double d = m["demand_mbps"].arr[s].num;
        if (B.isGw[s] || !B.alive[s] || d <= 0)
        {
            continue;
        }
        const double bytes = d * 1e6 * B.stepS / 8.0 + B.carryBytes[s];
        const uint32_t k = static_cast<uint32_t>(bytes / B.pktBytes);
        B.carryBytes[s] = bytes - static_cast<double>(k) * B.pktBytes;
        const double gap = B.stepS / std::max<uint32_t>(k, 1);
        for (uint32_t j = 0; j < k; ++j)
        {
            Simulator::Schedule(Seconds((j + 0.5) * gap), &Generate, s);
        }
    }

    Simulator::Stop(Seconds(B.stepS));
    Simulator::Run();

    std::vector<double> deliveredMbps(n), queue(n), srcQueue(n);
    for (uint32_t i = 0; i < n; ++i)
    {
        deliveredMbps[i] = B.delivered[i] * 8.0 / B.stepS / 1e6;
        queue[i] = QueueBytes(i);
        srcQueue[i] = std::max(0.0, B.genTotal[i] - B.doneTotal[i]);
    }
    std::ostringstream o;
    o.precision(10);
    o << "{\"type\":\"result\",\"t\":" << static_cast<int>(m["t"].num)
      << ",\"delivered_mbps\":" << Arr(deliveredMbps) << ",\"queue_bytes\":" << Arr(queue)
      << ",\"src_queue_bytes\":" << Arr(srcQueue) << ",\"drop_bytes\":{\"overflow\":"
      << Arr(B.dropOverflow) << ",\"mac\":" << Arr(B.dropMac) << ",\"switch\":"
      << Arr(B.dropSwitch) << ",\"noroute\":" << Arr(B.dropNoroute) << "},\"links\":[";
    bool first = true;
    for (auto& [uv, row] : B.links)
    {
        o << (first ? "" : ",") << "[" << uv.first << "," << uv.second << "," << row[0] << ","
          << row[1] << "," << (row[3] > 0 ? row[2] / row[3] : 0.0) << ","
          << (row[5] > 0 ? row[4] / row[5] : 0.0) << "]";
        first = false;
    }
    o << "]}";
    return o.str();
}

int
main(int argc, char* argv[])
{
    CommandLine cmd(__FILE__);
    cmd.Parse(argc, argv);
    std::ios::sync_with_stdio(false);
    std::string line;
    bool ready = false;
    try
    {
        while (std::getline(std::cin, line))
        {
            if (line.empty())
            {
                continue;
            }
            J m = ParseJson(line);
            const std::string type = m["type"].str;
            if (type == "init")
            {
                Init(m);
                ready = true;
                std::cout << "{\"type\":\"ready\"}" << std::endl;
            }
            else if (type == "step")
            {
                if (!ready)
                {
                    throw std::runtime_error("step before init");
                }
                std::cout << Step(m) << std::endl;
            }
            else if (type == "end")
            {
                break;
            }
            else
            {
                throw std::runtime_error("unknown message type: " + type);
            }
        }
    }
    catch (const std::exception& e)
    {
        std::string msg = e.what();
        for (auto& ch : msg)
        {
            if (ch == '"' || ch == '\\')
            {
                ch = '\'';
            }
        }
        std::cout << "{\"type\":\"error\",\"msg\":\"" << msg << "\"}" << std::endl;
        Simulator::Destroy();
        return 1;
    }
    Simulator::Destroy();
    return 0;
}
