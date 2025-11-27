#!/bin/bash

INPUT_FILE="3outof4.dat"

while read run; do
    jcache get /mss/hallc/xem2/analysis/OFFLINE/REPLAYS/HMS/PRODUCTION/pass2/hms_replay_production_${run}_-1.root -e ashard@jlab.org -D 60
done < "$INPUT_FILE"
