#!/usr/bin/env bash
# Extract the Bench2Drive-Base tarballs in place and DELETE each archive once its
# extraction has been verified (user instruction 2026-09-28: keeping both copies
# would double the footprint). /bulk has ~5 TB free, so every modality is kept.
#
#     nohup bash scripts/01c_b2d_extract.sh > logs/b2d_extract.log 2>&1 &
#
# Per clip: list the archive (file count), extract, count files on disk, and
# only on an exact match write <clip>/.done and rm the tarball. A mismatch or a
# missing top-level dir leaves the tarball alone and prints FAIL. Idempotent:
# clips with .done are skipped, partial dirs are wiped and redone.
set -uo pipefail
ROOT=/bulk/datasets/bench2drive
OUT=$ROOT/extracted
JOBS=${JOBS:-6}
mkdir -p "$OUT"

one() {
    f=$1; clip=$(basename "$f" .tar.gz); dst=$OUT/$clip
    [ -f "$dst/.done" ] && { echo "SKIP $clip"; return 0; }
    rm -rf "$dst"
    expected=$(pigz -dc "$f" | tar t | grep -vc '/$')
    if ! pigz -dc "$f" | tar x -C "$OUT"; then echo "FAIL extract $clip"; return 1; fi
    [ -d "$dst" ] || { echo "FAIL no-dir $clip (archive root differs from name)"; return 1; }
    found=$(find "$dst" -type f | wc -l)
    if [ "$found" -gt 0 ] && [ "$found" = "$expected" ]; then
        touch "$dst/.done" && rm -f "$f" && echo "OK   $clip $found files"
    else
        echo "FAIL count $clip expected=$expected found=$found (tarball kept)"; return 1
    fi
}
export -f one; export OUT
ls "$ROOT"/*.tar.gz 2>/dev/null | xargs -P "$JOBS" -I{} bash -c 'one "$@"' _ {}
echo "DONE $(date -Is): $(ls "$OUT" | wc -l) dirs, $(ls "$ROOT"/*.tar.gz 2>/dev/null | wc -l) tarballs left"
