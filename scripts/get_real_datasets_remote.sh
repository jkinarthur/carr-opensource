#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/carr-opensource
mkdir -p data/raw data/datasets/{ML-1M,Beauty,Toys,Yelp,Steam}

fetch() {
  local out="$1"
  shift
  rm -f "$out"
  for u in "$@"; do
    if curl -fL --retry 3 --connect-timeout 20 "$u" -o "$out"; then
      if [ -s "$out" ]; then
        return 0
      fi
    fi
  done
  return 1
}

cd data/raw

# ML-1M
fetch ml-1m.zip http://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip -oq ml-1m.zip
python3 - << 'PY'
from pathlib import Path
raw = Path('/home/ubuntu/carr-opensource/data/raw/ml-1m/ratings.dat')
out = Path('/home/ubuntu/carr-opensource/data/datasets/ML-1M/interactions.tsv')
out.parent.mkdir(parents=True, exist_ok=True)
with raw.open('r', encoding='latin-1') as f, out.open('w', encoding='utf-8') as w:
    w.write('user_id\titem_id\ttimestamp\n')
    for line in f:
        p = line.rstrip('\n').split('::')
        if len(p) >= 4:
            w.write(f"{p[0]}\t{p[1]}\t{p[3]}\n")
PY

# RecSysDatasets repo (for Beauty/Toys/Yelp/Steam)
if [ ! -d /tmp/RecSysDatasets/.git ]; then
  rm -rf /tmp/RecSysDatasets
  git clone --depth 1 https://github.com/RUCAIBox/RecSysDatasets.git /tmp/RecSysDatasets >/dev/null 2>&1 || true
fi

convert_inter_to_tsv() {
  local infile="$1"
  local outfile="$2"
  python3 - "$infile" "$outfile" << 'PY'
import sys
from pathlib import Path
inp = Path(sys.argv[1])
out = Path(sys.argv[2])
out.parent.mkdir(parents=True, exist_ok=True)
with inp.open('r', encoding='utf-8', errors='ignore') as f, out.open('w', encoding='utf-8') as w:
    header = f.readline().rstrip('\n').split('\t')
    lower = [h.lower() for h in header]
    def pick(prefixes):
        for p in prefixes:
            for i, h in enumerate(lower):
                if h.startswith(p):
                    return i
        return None
    iu = pick(['user_id', 'user'])
    ii = pick(['item_id', 'business_id', 'item'])
    it = pick(['timestamp', 'time'])
    if iu is None or ii is None:
        raise SystemExit(2)
    w.write('user_id\titem_id\ttimestamp\n')
    t = 1
    for line in f:
        p = line.rstrip('\n').split('\t')
        if iu >= len(p) or ii >= len(p):
            continue
        ts = p[it] if it is not None and it < len(p) and p[it] else str(t)
        w.write(f"{p[iu]}\t{p[ii]}\t{ts}\n")
        t += 1
PY
}

pick_and_convert() {
  local ds="$1"
  shift
  local out="/home/ubuntu/carr-opensource/data/datasets/$ds/interactions.tsv"
  local found=""
  for pat in "$@"; do
    found=$(find /tmp/RecSysDatasets -type f -iname "$pat" | head -n 1 || true)
    if [ -n "$found" ]; then
      break
    fi
  done
  if [ -n "$found" ]; then
    convert_inter_to_tsv "$found" "$out" || true
  fi
}

pick_and_convert Beauty '*beauty*.inter'
pick_and_convert Toys '*toys*games*.inter' '*toys*.inter'
pick_and_convert Yelp '*yelp*.inter'
pick_and_convert Steam '*steam*.inter'

printf "%-10s %-70s %-12s %-8s\n" Dataset Path Lines Status
for d in ML-1M Beauty Toys Yelp Steam; do
  f="/home/ubuntu/carr-opensource/data/datasets/$d/interactions.tsv"
  if [ -f "$f" ]; then
    n=$(wc -l "$f" | awk '{print $1}')
    if [ "$n" -gt 1 ]; then
      s=READY
    else
      s=MISSING
    fi
  else
    n=0
    s=MISSING
  fi
  printf "%-10s %-70s %-12s %-8s\n" "$d" "$f" "$n" "$s"
done
