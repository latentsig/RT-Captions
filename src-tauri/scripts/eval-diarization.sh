#!/bin/bash
# Diarization evaluation on AMI ES2002 (see tasks/diarization-eval.md).
#   eval-diarization.sh <data-dir> <tag> [meetings...]     empty-DB DER per meeting (labels are "Voice N")
#   eval-diarization.sh <data-dir> <tag> --identity        enroll the 4 speakers from ES2002a, then ES2002b and c
# <data-dir> holds ES2002{a,b,c}.16k.wav and ES2002{a,b,c}.rttm. Set RT_CAPTIONS_TUNE to try parameters.
# Everything runs on a scratch people database (RT_CAPTIONS_DATA), never the real one.
set -euo pipefail
D=$1; TAG=$2; shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=${RT_CAPTIONS_BIN:-$HERE/../target/release/rt-captions}
DER="python3 -I $HERE/der.py"
export RT_CAPTIONS_DATA=$D/db-$TAG
rm -rf "$RT_CAPTIONS_DATA"; mkdir -p "$RT_CAPTIONS_DATA"
NAMES=FEE005=Fee,MEE006=Mee6,MEE007=Mee7,MEE008=Mee8
if [ "${1:-}" = "--identity" ]; then
  # up to 60 s of each speaker's lone speech (no one else talking), cut from the reference RTTM
  # (MEE006 and MEE007 say less than 60 s in ES2002a in total: they get all of it)
  python3 -I - "$D/ES2002a.rttm" "$D/ES2002a.16k.wav" "$D/enroll-$TAG" <<'PY'
import sys, subprocess, os
rttm, wav, out = sys.argv[1:]
os.makedirs(out, exist_ok=True)
segs = [(float(p[3]), float(p[3]) + float(p[4]), p[7]) for p in (l.split() for l in open(rttm))]
for spk in sorted({s for _, _, s in segs}):
    mine = [(a, b) for a, b, s in segs if s == spk]
    others = [(a, b) for a, b, s in segs if s != spk]
    lone = []
    for a, b in mine:
        a, b = a + 0.05, b - 0.05
        if b - a < 0.5 or any(x < b + 0.1 and y > a - 0.1 for x, y in others):
            continue
        lone.append((a, b))
    lone.sort(key=lambda x: x[0] - x[1])  # longest stretches first
    pick, tot = [], 0.0
    for a, b in lone:
        if tot >= 60: break
        b = min(b, a + 60 - tot); pick.append((a, b)); tot += b - a
    flt = "".join(f"[0:a]atrim={a}:{b},asetpts=N/SR/TB[s{i}];" for i, (a, b) in enumerate(pick))
    flt += "".join(f"[s{i}]" for i in range(len(pick))) + f"concat=n={len(pick)}:v=0:a=1"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-filter_complex", flt, "-ar", "16000", "-ac", "1", f"{out}/{spk}.wav"], check=True)
    print(f"{spk}: {tot:.0f} s from {len(pick)} stretches", file=sys.stderr)
PY
  for kv in ${NAMES//,/ }; do "$BIN" enroll "${kv#*=}" "$D/enroll-$TAG/${kv%%=*}.wav" >/dev/null; done
  for m in b c; do
    "$BIN" diarize "$D/ES2002$m.16k.wav" --rttm-out "$D/ES2002$m.$TAG.rttm" 2>/dev/null
    echo "== ES2002$m ($TAG, enrolled from ES2002a)"; $DER "$D/ES2002$m.rttm" "$D/ES2002$m.$TAG.rttm" --names "$NAMES"
  done
  exit 0
fi
for m in "${@:-a b}"; do
  "$BIN" diarize "$D/ES2002$m.16k.wav" --rttm-out "$D/ES2002$m.$TAG.rttm" 2>/dev/null
  echo "== ES2002$m ($TAG)"; $DER "$D/ES2002$m.rttm" "$D/ES2002$m.$TAG.rttm"
done
