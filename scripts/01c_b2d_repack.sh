#!/usr/bin/env bash
# Repack the modalities the phase-2 loader never reads (depth / semantic /
# instance cameras, lidar, radar) into ONE uncompressed extras.tar per clip and
# delete the loose files. Why: the /bulk user quota is on FILE COUNT (7M hard
# limit) and every clip unpacks to ~7k files; these five modalities are ~70% of
# them. Bytes are unchanged and `tar xf extras.tar` restores everything.
#
#     nohup bash scripts/01c_b2d_repack.sh > logs/b2d_repack.log 2>&1 &
#
# Per clip: count the files, tar them, verify the archive lists the same count,
# then rm the directories and write .repacked. Idempotent; a clip whose verify
# fails keeps its loose files and prints FAIL. After the pass it runs
# 01c_b2d_extract.sh for any tarballs still waiting, then repacks those.
set -uo pipefail
ROOT=/bulk/datasets/bench2drive
OUT=$ROOT/extracted
JOBS=${JOBS:-8}

one() {
    d=$1; clip=$(basename "$d")
    [ -f "$d/.done" ] || { echo "SKIP $clip (not extracted)"; return 0; }
    [ -f "$d/.repacked" ] && { echo "SKIP $clip"; return 0; }
    cd "$d" || return 1
    dirs=$(ls -d camera/depth_* camera/semantic_* camera/instance_* lidar radar 2>/dev/null)
    [ -n "$dirs" ] || { touch .repacked; echo "OK   $clip nothing to repack"; return 0; }
    expected=$(find $dirs -type f | wc -l)
    rm -f extras.tar
    if ! tar cf extras.tar $dirs; then echo "FAIL tar $clip"; rm -f extras.tar; return 1; fi
    found=$(tar tf extras.tar | grep -vc '/$')
    if [ "$found" -gt 0 ] && [ "$found" = "$expected" ]; then
        rm -rf $dirs && touch .repacked && echo "OK   $clip $found files -> extras.tar"
    else
        echo "FAIL count $clip expected=$expected found=$found (loose files kept)"; rm -f extras.tar; return 1
    fi
}
export -f one
ls -d "$OUT"/*/ | xargs -P "$JOBS" -I{} bash -c 'one "$@"' _ {}
echo "REPACK PASS 1 DONE $(date -Is)"; quota -s -u vla 2>&1 | tail -1
if ls "$ROOT"/*.tar.gz >/dev/null 2>&1; then
    echo "extracting the remaining $(ls "$ROOT"/*.tar.gz | wc -l) tarballs"
    bash "$(dirname "$0")/01c_b2d_extract.sh"
    ls -d "$OUT"/*/ | xargs -P "$JOBS" -I{} bash -c 'one "$@"' _ {}
    echo "REPACK PASS 2 DONE $(date -Is)"; quota -s -u vla 2>&1 | tail -1
fi
echo "ALL DONE $(date -Is): $(ls "$OUT"/*/.repacked | wc -l) repacked, $(ls "$ROOT"/*.tar.gz 2>/dev/null | wc -l) tarballs left"
