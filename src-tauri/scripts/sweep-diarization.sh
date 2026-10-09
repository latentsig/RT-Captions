#!/bin/bash
# Tries parameter sets (RT_CAPTIONS_TUNE strings, "" = defaults) on ES2002a and b in parallel, empty DB each.
#   sweep-diarization.sh <data-dir> "gap=8" "stages=1.5/3/6" ...
# Prints one line per set: DER and number of "Voice N" labels per meeting.
D=$1; shift
HERE=$(cd "$(dirname "$0")" && pwd)
i=0
for spec in "$@"; do
  i=$((i+1)); tag=sw$$-$i
  for m in a b; do
    ( RT_CAPTIONS_TUNE="$spec" "$HERE/eval-diarization.sh" "$D" "$tag-$m" $m 2>/dev/null > "$D/$tag-$m.txt" ) &
  done
done
wait
i=0
for spec in "$@"; do
  i=$((i+1)); tag=sw$$-$i
  line="[$spec]"
  for m in a b; do
    der=$(grep -o "^DER [0-9.]*%" "$D/$tag-$m.txt" | head -1)
    v=$(grep -o "hypothesis labels [0-9]* ([0-9]* with" "$D/$tag-$m.txt" | head -1 | sed 's/hypothesis labels //; s/ with//')
    pc=$(grep -o "purity [0-9.]*%  coverage [0-9.]*%" "$D/$tag-$m.txt" | head -1)
    line="$line  $m: $der labels $v $pc"
  done
  echo "$line"
  rm -rf "$D/db-$tag-a" "$D/db-$tag-b"
done
