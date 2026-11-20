#!/usr/bin/env bash
# Parallel ranged download for GEO files over a slow link.
# usage: fetch_parallel.sh URL OUTFILE CHUNKS
# Every part is checked against its expected byte count; short parts are
# resumed individually.  The assembled file must match Content-Length.
set -u
url="$1"; out="$2"; n="${3:-6}"
tmp="${out}.parts"
mkdir -p "$tmp"
size=$(curl --ssl-no-revoke -sSI "$url" | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}')
[ -z "$size" ] && { echo "no content-length"; exit 1; }
chunk=$(( (size + n - 1) / n ))

fetch_part() {
  local s=$1 e=$2 part=$3 want=$4 attempt
  for attempt in 1 2 3 4 5; do
    curl --ssl-no-revoke -sS --retry 3 --retry-delay 2 -r ${s}-${e} -o "$part" "$url" || true
    [ "$(stat -c%s "$part" 2>/dev/null || echo 0)" -eq "$want" ] && return 0
    echo "part $(basename "$part") short ($(stat -c%s "$part" 2>/dev/null || echo 0)/$want), retry $attempt"
    sleep 2
  done
  return 1
}

pids=()
for i in $(seq 0 $((n-1))); do
  s=$(( i*chunk )); e=$(( s+chunk-1 )); [ $e -ge $size ] && e=$((size-1))
  want=$(( e - s + 1 ))
  part="$tmp/part.$(printf %02d $i)"
  fetch_part "$s" "$e" "$part" "$want" &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ $fail -ne 0 ] && { echo "chunk failed"; exit 1; }
cat "$tmp"/part.* > "$out"
got=$(stat -c%s "$out")
if [ "$got" -ne "$size" ]; then
  echo "size mismatch: got $got expected $size"
  exit 1
fi
rm -rf "$tmp"
echo "done $out $got bytes"
