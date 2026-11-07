#!/bin/bash
# Submit a submission zip to Codabench (competition 13967, MIND News Rec).
#
# WHY THIS SCRIPT LOOKS THE WAY IT DOES
# ------------------------------------
# The previous version of this script was wrong in FOUR independent ways.
# Each was verified against the live Codabench API and the upstream source
# (github.com/codalab/codabench):
#
#  1. AUTH PREFIX. The DRF TokenAuthentication expects
#     `Authorization: Token <key>`. Sending `Bearer <key>` reaches a
#     different code path and returns HTTP 500; `Token <key>` with a bad
#     key correctly returns 403 {"detail":"Invalid token."}.
#
#  2. THERE IS NO `files` FIELD. SubmissionCreationSerializer
#     (src/apps/api/serializers/submissions.py) has fields
#     data / phase / status / status_details / secret / md5 / tasks /
#     selected_groups / fact_sheet_answers / organization / queue /
#     created_when / scores. The file is attached via `data`, a
#     SlugRelatedField pointing at a Data row's UUID `key`.
#
#  3. THE FILE UPLOAD IS A SEPARATE 3-STEP FLOW (see src/static/js/ours/
#     client.js `create_dataset`):
#        a) POST /api/datasets/          -> {"key": ..., "sassy_url": ...}
#        b) PUT  <sassy_url>             -> upload the zip bytes
#        c) PUT  /api/datasets/completed/<key>/
#     The `sassy_url` is a presigned S3/Azure PUT URL; it must be sent
#     with Content-Type: application/zip (the signature covers it).
#
#  4. THE PHASE. The leaderboard we are ranked on (leaderboard 15401)
#     contains our only entry with task=29311, which belongs to phase
#     23196 "Official Test" -- NOT phase 23195 "Development" that the old
#     script used. Phase 23196 has task 29311; phase 23195 has task 29310.
#
# The zip itself must contain exactly one member named `prediction.txt`
# (the official msnews/MIND evaluate.py hardcodes that name). Use
# scripts/build_submission_zip.sh to build it.
#
# Usage:
#   CODEBENCH_KEY=<key> bash scripts/submit_codabench.sh [zip_path]
#   bash scripts/submit_codabench.sh <key> [zip_path]
#
# Env overrides:
#   CODEBENCH_PHASE   (default 23196 = Official Test)
#   CODEBENCH_TASK    (default 29311)
#   CODEBENCH_COMP    (default 13967)
#   CODEBENCH_DRYRUN  (default 0; if 1, validate everything but upload nothing)
set -euo pipefail
cd /import/ml-sc-scratch5/chenw/MiniOneRec

API=https://www.codabench.org/api
COMP=${CODEBENCH_COMP:-13967}
PHASE=${CODEBENCH_PHASE:-23196}
TASK=${CODEBENCH_TASK:-29311}
DRYRUN=${CODEBENCH_DRYRUN:-0}

# Arg parsing: the key may come from $1 or $CODEBENCH_KEY; the zip may come
# from the other positional slot or $CODEBENCH_ZIP. A positional arg that
# names an existing file is the zip; anything else is the key.
ZIP="${CODEBENCH_ZIP:-results/submissions/MINDlarge_test_lg9b_v4.zip}"
KEY="${CODEBENCH_KEY:-}"
for a in "$@"; do
  if [ -f "$a" ]; then
    ZIP="$a"
  elif [ -z "$KEY" ] || [ "$KEY" = "$CODEBENCH_KEY" ]; then
    KEY="$a"
  fi
done
[ -f "$ZIP" ] || { echo "ERROR: missing $ZIP" >&2; exit 1; }

mkdir -p logs
LOG=logs/codabench_submit_last.json

# ---- sanity 1: the zip must contain exactly one member named prediction.txt ----
python3 - "$ZIP" <<'PY'
import sys, zipfile
z = sys.argv[1]
with zipfile.ZipFile(z) as f:
    names = f.namelist()
assert names == ["prediction.txt"], f"{z}: members must be exactly ['prediction.txt'], got {names}"
print(f"zip OK: {z} -> prediction.txt")
PY

# ---- sanity 2: the phase/task pair must match the live competition ----
COMPJSON=$(curl -s -m 120 "${API}/competitions/${COMP}/")
PHASE_TASKS=$(echo "$COMPJSON" | jq -r --argjson p "$PHASE" \
  '.phases[] | select(.id==$p) | [.tasks[].id] | @csv')
PHASE_NAME=$(echo "$COMPJSON" | jq -r --argjson p "$PHASE" \
  '.phases[] | select(.id==$p) | .name')
if [ -z "$PHASE_TASKS" ]; then
  echo "ERROR: phase ${PHASE} not found in competition ${COMP}" >&2
  exit 1
fi
echo "phase ${PHASE} (${PHASE_NAME}) tasks: ${PHASE_TASKS}"
if ! echo ",${PHASE_TASKS}," | grep -q ",${TASK},"; then
  echo "ERROR: task ${TASK} is not part of phase ${PHASE} (has ${PHASE_TASKS})" >&2
  exit 1
fi

SIZE=$(stat -c %s "$ZIP")
NAME="mind_$(basename "$ZIP" .zip)_$(date +%Y%m%d_%H%M%S)"

echo "=== Codabench submit ==="
echo "  zip   : $ZIP ($SIZE bytes)"
echo "  comp  : $COMP   phase: $PHASE ($PHASE_NAME)   task: $TASK"
echo "  name  : $NAME"

if [ "$DRYRUN" = "1" ]; then
  echo "DRYRUN: validation passed; stopping before upload."
  exit 0
fi

if [ -z "$KEY" ]; then
  echo "ERROR: no api_key. Pass it as \$1 or set CODEBENCH_KEY." >&2
  exit 1
fi
AUTH="Authorization: Token ${KEY}"

# ---- 0. pre-flight: may we submit to this phase at all? ----
echo "--- pre-flight: can_make_submission ---"
PRE=$(curl -s -m 120 -H "$AUTH" "${API}/can_make_submission/${PHASE}/")
echo "$PRE"
if ! echo "$PRE" | grep -q '"can": *true'; then
  echo "ERROR: phase ${PHASE} refuses submissions: $PRE" >&2
  exit 1
fi

# ---- 1. create the Data row -> get {key, sassy_url} ----
echo "--- step 1: POST /api/datasets/ ---"
CREATE=$(curl -s -m 300 -X POST "${API}/datasets/" \
  -H "$AUTH" -H "Content-Type: application/json" \
  -d "$(jq -nc \
        --arg name "$NAME" \
        --arg fname "$(basename "$ZIP")" \
        --argjson size "$SIZE" \
        --argjson comp "$COMP" \
        '{type: "submission", competition: $comp, name: $name,
          description: "MIND large test predictions",
          license: "MIT",
          file_size: $size, file_name: $fname,
          request_sassy_file_name: $fname}')")
echo "$CREATE"
KEYDATA=$(echo "$CREATE" | jq -r '.key // empty')
SASSY=$(echo "$CREATE" | jq -r '.sassy_url // empty')
if [ -z "$KEYDATA" ] || [ -z "$SASSY" ]; then
  echo "ERROR: dataset creation failed: $CREATE" >&2
  exit 1
fi
echo "  data key: $KEYDATA"

# ---- 2. PUT the zip bytes to the presigned URL ----
echo "--- step 2: PUT sassy_url (${SIZE} bytes) ---"
case "$SASSY" in
  *blob.core.windows.net*)
    PUT_HDRS=(-H "Content-Type: application/zip"
              -H "x-ms-blob-type: BlockBlob"
              -H "x-ms-version: 2018-03-28") ;;
  *)
    PUT_HDRS=(-H "Content-Type: application/zip") ;;
esac
HTTP=$(curl -s -m 3600 -o /tmp/cb_put_body.txt -w '%{http_code}' \
  -X PUT "${SASSY}" "${PUT_HDRS[@]}" --data-binary "@${ZIP}")
echo "  sassy PUT -> HTTP ${HTTP}"
if [ "$HTTP" != "200" ] && [ "$HTTP" != "201" ] && [ "$HTTP" != "204" ]; then
  echo "ERROR: sassy upload failed (HTTP ${HTTP}):" >&2
  head -c 500 /tmp/cb_put_body.txt >&2; echo >&2
  exit 1
fi

# ---- 3. tell Codabench the upload finished ----
echo "--- step 3: PUT /api/datasets/completed/${KEYDATA}/ ---"
DONE=$(curl -s -m 300 -X PUT "${API}/datasets/completed/${KEYDATA}/" -H "$AUTH")
echo "$DONE"

# ---- 4. create the submission ----
# `selected_groups` is deliberately omitted: the server then uses every
# participant group the user belongs to (group_ids=None), which is the
# default routing. Passing an empty list would instead filter the groups
# down to none and could route the job to the wrong queue.
echo "--- step 4: POST /api/submissions/ ---"
SUB=$(curl -s -m 600 -X POST "${API}/submissions/" \
  -H "$AUTH" -H "Content-Type: application/json" \
  -d "$(jq -nc \
        --arg data "$KEYDATA" \
        --argjson phase "$PHASE" \
        --argjson task "$TASK" \
        '{data: $data, phase: $phase, tasks: [$task]}')")
echo "$SUB"
echo "$SUB" > "$LOG"

SUBID=$(echo "$SUB" | jq -r '.id // .pk // empty')
if [ -z "$SUBID" ]; then
  echo "ERROR: submission creation failed: $SUB" >&2
  exit 1
fi
echo "=== submission id ${SUBID} created ==="
echo "    poll: curl -s -H 'Authorization: Token <key>' ${API}/submissions/${SUBID}/"
