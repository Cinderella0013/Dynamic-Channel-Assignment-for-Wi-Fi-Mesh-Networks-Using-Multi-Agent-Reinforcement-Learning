#!/bin/bash
# slot scheduler: pops first line of queue.txt when < 3 training jobs run
cd /e/MESH; q=runs/v10/queue.txt
while [ -s $q ]; do
  n=$(ps -ef | grep -c "[r]uns/v10/job.sh")
  if [ "$n" -lt 3 ]; then
    line=$(head -1 $q); sed -i 1d $q
    echo "$(date +%F_%T) start $line" >> runs/v10/events.log
    nohup bash runs/v10/job.sh $line >/dev/null 2>&1 &
    sleep 5
  else sleep 60; fi
done
echo "$(date +%F_%T) queue empty" >> runs/v10/events.log
