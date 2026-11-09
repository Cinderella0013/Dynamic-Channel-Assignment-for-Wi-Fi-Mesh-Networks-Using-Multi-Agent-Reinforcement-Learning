## Table 5: seed-pair composite and 95% t-interval, setting (a), A+B

| reference | 0-19 | 20-39 | 40-59 |
|---|---|---|---|
| Ko | +0.260296 [+0.230390, +0.290203] (df 4.00) | +0.252957 [+0.221978, +0.283936] (df 4.00) | +0.245951 [+0.215532, +0.276371] (df 4.00) |
| Li | +0.202719 [+0.068658, +0.336780] (df 4.46) | +0.220239 [+0.107932, +0.332546] (df 4.63) | +0.178952 [+0.033497, +0.324406] (df 4.37) |
| Yuan | +0.241429 [+0.200836, +0.282023] (df 7.96) | +0.256531 [+0.212274, +0.300787] (df 7.60) | +0.247645 [+0.213764, +0.281526] (df 6.62) |
| M0 | +0.311090 [+0.290933, +0.331247] (df 4.00) | +0.297449 [+0.270170, +0.324728] (df 4.00) | +0.315790 [+0.299750, +0.331830] (df 4.00) |
| Q-neutral | +0.736031 [+0.725599, +0.746463] (df 4.50) | +0.735009 [+0.723244, +0.746774] (df 4.71) | +0.737183 [+0.725508, +0.748858] (df 4.37) |
| Li+H | +0.132943 [+0.099125, +0.166760] (df 7.18) | +0.155407 [+0.115031, +0.195784] (df 7.88) | +0.107321 [+0.074599, +0.140043] (df 7.16) |
| Yuan+H | +0.085097 [+0.042557, +0.127636] (df 8.00) | +0.088794 [+0.041795, +0.135793] (df 7.54) | +0.066841 [+0.031338, +0.102345] (df 7.42) |
| EGAT+GRU+H | +0.069355 [+0.025709, +0.113001] (df 7.87) | +0.039590 [-0.011710, +0.090889] (df 7.48) | +0.049191 [+0.007170, +0.091211] (df 7.94) |

Run/config, topologies 0-19:
- commit 8aaaf1cddab7, overrides {"airtime": "domain", "contention": 0.0, "mac_eff": 0.65}, agent {"switch_margin": 2.0}, eval seeds 0-19: 5 files, e.g. `runs\v10\t0\a\suite_egat_nogru_h.json`
- commit 75d17f896b0a, overrides {"airtime": "domain", "contention": 0.0, "mac_eff": 0.65}, agent {"switch_margin": 2.0}, eval seeds 0-19: 5 files, e.g. `runs\v10\t0\a\suite_egat_h2.json`
- commit c7c16c78ee29, overrides {"airtime": "domain", "contention": 0.0, "mac_eff": 0.65}, agent {}, eval seeds 0-19: 16 files, e.g. `runs\v10\t0\a\suite_li.json`
- commit 75d17f896b0a, overrides {"airtime": "domain", "contention": 0.0, "mac_eff": 0.65}, agent {"switch_margin": 1}, eval seeds 0-19: 10 files, e.g. `runs\v10\t0\a\suite_li_h.json`

Run/config, topologies 20-39:
- commit 8aaaf1cddab7, overrides {"airtime": "domain", "contention": 0.0, "mac_eff": 0.65}, agent {"switch_margin": 2.0}, eval seeds 20-39: 5 files, e.g. `runs\v10\fresh\t0\a\suite_egat_nogru_h.json`
- commit 75d17f896b0a, overrides {"airtime": "domain", "contention": 0.0, "mac_eff": 0.65}, agent {"switch_margin": 2.0}, eval seeds 20-39: 5 files, e.g. `runs\v10\fresh\t0\a\suite_egat_h2.json`
- commit 75d17f896b0a, overrides {"airtime": "domain", "contention": 0.0, "mac_eff": 0.65}, agent {}, eval seeds 20-39: 16 files, e.g. `runs\v10\fresh\t0\a\suite_li.json`
- commit 75d17f896b0a, overrides {"airtime": "domain", "contention": 0.0, "mac_eff": 0.65}, agent {"switch_margin": 1}, eval seeds 20-39: 10 files, e.g. `runs\v10\fresh\t0\a\suite_li_h.json`

Run/config, topologies 40-59:
- commit 8aaaf1cddab7, overrides {"airtime": "domain", "contention": 0.0, "mac_eff": 0.65}, agent {"switch_margin": 2}, eval seeds 40-59: 5 files, e.g. `runs\v10\fresh2\t0\a\suite_egat_nogru_h.json`
- commit 8aaaf1cddab7, overrides {"airtime": "domain", "contention": 0.0, "mac_eff": 0.65}, agent {"switch_margin": 2.0}, eval seeds 40-59: 5 files, e.g. `runs\v10\fresh2\t0\a\suite_egat_h2.json`
- commit 8aaaf1cddab7, overrides {"airtime": "domain", "contention": 0.0, "mac_eff": 0.65}, agent {}, eval seeds 40-59: 16 files, e.g. `runs\v10\fresh2\t0\a\suite_li.json`
- commit 8aaaf1cddab7, overrides {"airtime": "domain", "contention": 0.0, "mac_eff": 0.65}, agent {"switch_margin": 1}, eval seeds 40-59: 10 files, e.g. `runs\v10\fresh2\t0\a\suite_li_h.json`

## Table 6: raw means over 11 scenarios (scenario mean of all runs, then mean over scenarios), setting (a), topologies 0-19

| method | aggregate | jain | p5 | loss_rate | n_channel_switch | n_route_change | stability |
|---|---|---|---|---|---|---|---|
| M0 | 26.865868 | 0.892526 | 0.858405 | 0.515763 | 0.000000 | 0.186364 | 0.186364 |
| Ko | 47.319173 | 0.776517 | 1.410292 | 0.248425 | 456.900000 | 158.613636 | 615.513636 |
| Li | 47.058570 | 0.789929 | 1.219653 | 0.245528 | 165.974545 | 3.140909 | 169.115455 |
| Yuan | 49.378679 | 0.766961 | 1.296057 | 0.220687 | 346.227273 | 5.260909 | 351.488182 |
| Li+H | 47.002135 | 0.802129 | 1.277757 | 0.244080 | 40.206364 | 3.132727 | 43.339091 |
| Yuan+H | 49.703897 | 0.787276 | 1.361463 | 0.213198 | 40.262727 | 3.058182 | 43.320909 |
| EGAT+GRU+H | 51.348088 | 0.787900 | 1.442078 | 0.193213 | 46.836364 | 4.880909 | 51.717273 |
| EGAT+H | 53.196328 | 0.785896 | 1.532758 | 0.172321 | 39.480909 | 2.764545 | 42.245455 |
| Q-neutral | 0.170778 | 0.918764 | 0.000000 | 0.974082 | 6263.291818 | 1787.219091 | 8050.510909 |

## Table 7: per-term contributions of the pooled-mean composite, topologies 0-19 (full precision)

| reference | aggregate | jain | p5 | loss_rate | stability | total (sum of terms) | report 'mean' column |
|---|---|---|---|---|---|---|---|
| Ko | +0.028270 | +0.002489 | +0.017447 | +0.073351 | +0.138740 | +0.260296 | +0.260296 |
| Li | +0.029749 | -0.001053 | +0.049875 | +0.072500 | +0.106561 | +0.257632 | +0.257632 |
| Yuan | +0.017528 | +0.005021 | +0.035373 | +0.060013 | +0.125796 | +0.243731 | +0.243731 |
| Li+H | +0.029962 | -0.004068 | +0.038966 | +0.068259 | +0.003165 | +0.136285 | +0.136285 |
| Yuan+H | +0.015985 | -0.000366 | +0.024292 | +0.049280 | +0.003575 | +0.092766 | +0.092766 |
| M0 | +0.205242 | -0.023621 | +0.151134 | +0.128957 | -0.150000 | +0.311712 | +0.311712 |
| EGAT+GRU+H | +0.007977 | -0.000530 | +0.012024 | +0.028874 | +0.026885 | +0.075230 | +0.075230 |

## Yuan+H: 0.093 vs 0.094

Terms recomputed from the 3-decimal means printed in `runs/v10/final/composite_a_nogru_test.md` (the input `make_figures.decomposition()` used):

aggregate +0.015985 | jain -0.000386 | p5 +0.024280 | loss_rate +0.050287 | stability +0.003575 | total +0.093741

## Zero-reference check (Q-neutral P5)

Q-neutral mean P5 per scenario (topologies 0-19): S1 0.000e+00, S2 0.000e+00, S3 0.000e+00, S4 0.000e+00, S5 0.000e+00, S6 0.000e+00, B1 0.000e+00, B2 0.000e+00, B3 0.000e+00, B4 0.000e+00, B5 0.000e+00

## Table 8: three-seed ablation (training seeds 0-2, margin 2), setting (a), A+B, seed-pair mean

| variant | vs Ko | vs Yuan+H | vs full GRU [95% CI] |
|---|---|---|---|
| Full GRU | +0.225964 | +0.006090 | reference |
| No edge features | +0.248845 | +0.044214 | +0.030328 [-0.055355, +0.116011] |
| No GRU | +0.260438 | +0.086532 | +0.069618 [-0.011469, +0.150704] |
| No Ko prior | +0.173359 | -0.019256 | -0.046331 [-0.140852, +0.048189] |
| No behaviour cloning | +0.219924 | +0.013728 | +0.001128 [-0.082160, +0.084416] |

Run/config:
- commit 75d17f896b0a, overrides {"airtime": "domain", "contention": 0.0, "mac_eff": 0.65}, agent {"switch_margin": 2.0}, eval seeds 0-19: 15 files, e.g. `runs\v10\t0\a\suite_egat_h2.json`
- commit c7c16c78ee29, overrides {"airtime": "domain", "contention": 0.0, "mac_eff": 0.65}, agent {}, eval seeds 0-19: 1 files, e.g. `runs\v10\t0\a\suite_heur.json`

## Table 9: S2/B1/B3, seed-pair mean [95% t-interval]

| reference | simulator (topologies 0-19) | ns-3 RTS/CTS | ns-3 base MAC |
|---|---|---|---|
| Ko | +0.291109 [+0.262517, +0.319701] (df 4.00) | +0.230892 [+0.154543, +0.307241] (df 4.00) | +0.218016 [+0.143806, +0.292227] (df 4.00) |
| Li | +0.245700 [+0.087262, +0.404139] (df 4.36) | +0.436644 [+0.350838, +0.522450] (df 5.67) | +0.460927 [+0.379034, +0.542820] (df 5.56) |
| Yuan | +0.278125 [+0.228298, +0.327953] (df 6.74) | +0.363532 [+0.222666, +0.504397] (df 5.51) | +0.372369 [+0.225577, +0.519161] (df 5.20) |
| M0 | +0.362975 [+0.345553, +0.380397] (df 4.00) | +0.525914 [+0.486485, +0.565342] (df 4.00) | +0.554767 [+0.523346, +0.586188] (df 4.00) |
| Q-neutral | +0.722458 [+0.714180, +0.730737] (df 4.68) | +0.692732 [+0.681501, +0.703963] (df 7.24) | +0.680169 [+0.670187, +0.690152] (df 6.18) |
| Li+H | +0.164836 [+0.122850, +0.206822] (df 8.00) | +0.364546 [+0.321551, +0.407542] (df 5.02) | +0.395226 [+0.356819, +0.433634] (df 4.75) |
| Yuan+H | +0.089693 [+0.045994, +0.133392] (df 7.97) | +0.164388 [+0.003731, +0.325044] (df 5.22) | +0.183524 [+0.021179, +0.345870] (df 5.02) |
| EGAT+GRU+H | +0.080573 [+0.029050, +0.132096] (df 7.50) | +0.153052 [+0.017656, +0.288449] (df 6.24) | +0.176649 [+0.044404, +0.308894] (df 6.19) |

Run/config, ns-3 RTS/CTS (topologies 0-4):
- commit 8aaaf1cddab7, overrides {"airtime": "domain", "contention": 0.0, "mac_eff": 0.65}, agent {"switch_margin": 2}, eval seeds 0-4: 5 files, e.g. `runs\ns3\v10\t0\suite_egat_nogru_h.json`
- commit 75d17f896b0a, overrides {"airtime": "domain", "contention": 0.0, "mac_eff": 0.65}, agent {"switch_margin": 2.0}, eval seeds 0-4: 5 files, e.g. `runs\ns3\v10\t0\suite_egat_h2.json`
- commit c7c16c78ee29, overrides {"airtime": "domain", "contention": 0.0, "mac_eff": 0.65}, agent {}, eval seeds 0-4: 6 files, e.g. `runs\ns3\v10\t0\suite_rl.json`
- commit 75d17f896b0a, overrides {"airtime": "domain", "contention": 0.0, "mac_eff": 0.65}, agent {"switch_margin": 1.0}, eval seeds 0-4: 10 files, e.g. `runs\ns3\v10\t0\suite_li_h.json`

Run/config, ns-3 base MAC (topologies 0-4):
- commit 8aaaf1cddab7, overrides {"airtime": "domain", "contention": 0.0, "mac_eff": 0.65}, agent {"switch_margin": 2}, eval seeds 0-4: 5 files, e.g. `runs\ns3\v10base\t0\suite_egat_nogru_h.json`
- commit 75d17f896b0a, overrides {"airtime": "domain", "contention": 0.0, "mac_eff": 0.65}, agent {"switch_margin": 2.0}, eval seeds 0-4: 5 files, e.g. `runs\ns3\v10base\t0\suite_egat_h2.json`
- commit 65c49b625e9d, overrides {"airtime": "domain", "contention": 0.0, "mac_eff": 0.65}, agent {}, eval seeds 0-4: 6 files, e.g. `runs\ns3\v10base\t0\suite_rl.json`
- commit 75d17f896b0a, overrides {"airtime": "domain", "contention": 0.0, "mac_eff": 0.65}, agent {"switch_margin": 1.0}, eval seeds 0-4: 10 files, e.g. `runs\ns3\v10base\t0\suite_li_h.json`

Raw means, ns-3 S2 (ours vs Ko):

- ns-3 RTS/CTS EGAT+H: aggregate 68.062848, jain 0.693275, p5 1.292406, loss_rate 0.265659, stability 44.680000
- ns-3 RTS/CTS Ko: aggregate 67.100832, jain 0.722368, p5 1.318421, loss_rate 0.276779, stability 790.600000
- ns-3 base MAC EGAT+H: aggregate 63.815558, jain 0.662589, p5 0.954942, loss_rate 0.309967, stability 44.680000
- ns-3 base MAC Ko: aggregate 66.354528, jain 0.698268, p5 1.084361, loss_rate 0.292223, stability 791.400000

