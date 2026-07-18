#!/usr/bin/env bash
# End-to-end smoke test against a RUNNING Sentinel instance (+ llama-server).
#
#   ./scripts/smoke_test.sh
#
# Prereqs: llama-server up (scripts/run_model.sh), app up (python3 -m sentinel),
# demo data seeded (python3 scripts/seed_demo.py). Uses the live model, so the
# two parse steps and the image evaluation can each take a minute-plus on CPU.
#
# Exercises: health, UI page, sensor listing, live rule parsing (numeric +
# image), confirm flow, numeric reading -> alert with NO model call, real
# photo -> vision evaluation, image serving, ack. Prints PASS/FAIL per step.
set -u

BASE="${BASE:-http://127.0.0.1:8000}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CURL="curl -s --max-time 420"
PASS=0; FAIL=0

ok()   { PASS=$((PASS+1)); echo "  PASS  $1"; }
bad()  { FAIL=$((FAIL+1)); echo "  FAIL  $1"; }
note() { echo "  note  $1"; }

json() { python3 -c "import sys,json;d=json.load(sys.stdin);print($1)" 2>/dev/null; }

echo "== 1. Health =="
H=$($CURL "$BASE/api/health")
[ "$(echo "$H" | json 'd["db"]')" = "True" ] && ok "db reachable" || bad "db: $H"
if [ "$(echo "$H" | json 'd["llama_server"]')" = "True" ]; then
  ok "llama-server reachable"
else
  bad "llama-server not reachable — is run_model.sh running?"
  echo "Aborting: the remaining tests need the model."; exit 1
fi

echo "== 2. UI page =="
CODE=$($CURL -o /dev/null -w '%{http_code}' "$BASE/")
[ "$CODE" = "200" ] && ok "GET / -> 200" || bad "GET / -> $CODE"

echo "== 3. Sensors =="
SENSORS=$($CURL "$BASE/api/sensors")
echo "$SENSORS" | json '[s["name"] for s in d]' | grep -q breakroom_cam \
  && ok "breakroom_cam exists" || bad "breakroom_cam missing — run scripts/seed_demo.py"
echo "$SENSORS" | json '[s["name"] for s in d]' | grep -q keg_scale \
  && ok "keg_scale exists" || bad "keg_scale missing — run scripts/seed_demo.py"

echo "== 4. Live parse: numeric rule (model call, ~1 min cold) =="
R1=$($CURL -X POST "$BASE/api/rules/parse" -H 'Content-Type: application/json' \
  -d '{"text":"alert me when the keg drops below 15 pounds"}')
KEG_ID=$(echo "$R1" | json 'd["id"]')
if [ -n "$KEG_ID" ] && [ "$(echo "$R1" | json 'd["status"]')" = "pending_confirm" ]; then
  ok "parsed to pending_confirm (rule $KEG_ID)"
  note "summary: $(echo "$R1" | json 'd["summary"]')"
  [ "$(echo "$R1" | json 'd["parsed"]["condition"]["operator"]')" = "lt" ] \
    && ok "condition is lt threshold" || bad "unexpected condition: $R1"
  $CURL -X POST "$BASE/api/rules/$KEG_ID/confirm" >/dev/null && ok "confirmed"
else
  bad "parse failed: $R1"
fi

echo "== 5. Numeric alert WITHOUT model call =="
if [ -n "${KEG_ID:-}" ]; then
  $CURL -X POST "$BASE/api/readings" -H 'Content-Type: application/json' \
    -d '{"sensor":"keg_scale","value":12}' >/dev/null
  CY=$($CURL -X POST "$BASE/api/cycle")
  note "cycle: $CY"
  EV=$($CURL "$BASE/api/evaluations?limit=20")
  ROW=$(echo "$EV" | json "[e for e in d if e['rule_id']==$KEG_ID][0]")
  RES=$(echo "$EV" | json "[e for e in d if e['rule_id']==$KEG_ID][0]['result']")
  MA=$(echo "$EV" | json "[e for e in d if e['rule_id']==$KEG_ID][0]['model_answer']")
  [ "$RES" = "triggered" ] && ok "evaluation triggered" || bad "evaluation: $ROW"
  [ "$MA" = "None" ] && ok "model_answer is NULL (pure code path)" \
                     || bad "model_answer not NULL: $MA"
  AL=$($CURL "$BASE/api/alerts?unacked=1")
  AID=$(echo "$AL" | json "[a for a in d if a['rule_id']==$KEG_ID][0]['id']")
  if [ -n "$AID" ]; then
    ok "alert created (id $AID) — you should have seen a macOS notification"
    $CURL -X POST "$BASE/api/alerts/$AID/ack" >/dev/null && ok "alert acked"
  else
    bad "no alert for rule $KEG_ID: $AL"
  fi
fi

echo "== 6. Live parse: image rule (model call) =="
R2=$($CURL -X POST "$BASE/api/rules/parse" -H 'Content-Type: application/json' \
  -d '{"text":"alert me if any snack basket in the break room is empty"}')
IMG_ID=$(echo "$R2" | json 'd["id"]')
if [ -n "$IMG_ID" ] && [ "$(echo "$R2" | json 'd["status"]')" = "pending_confirm" ]; then
  ok "parsed to pending_confirm (rule $IMG_ID)"
  note "question: $(echo "$R2" | json 'd["parsed"]["condition"]["question"]')"
  $CURL -X POST "$BASE/api/rules/$IMG_ID/confirm" >/dev/null && ok "confirmed"
else
  bad "parse failed: $R2"
fi

echo "== 7. Real photo -> vision evaluation (slowest step) =="
PHOTO="$ROOT/examples/images/snack_wall_2.jpg"
if [ ! -f "$PHOTO" ]; then
  note "examples/images missing (patch not applied?) — using newest seeded inbox image instead"
  PHOTO=""
fi
if [ -n "${IMG_ID:-}" ]; then
  if [ -n "$PHOTO" ]; then
    # Honor the same env vars the app uses, else the copy lands in the wrong inbox.
    INBOX="${SENTINEL_INBOX:-${SENTINEL_BASE_DIR:-$ROOT}/inbox}"
    mkdir -p "$INBOX/breakroom_cam"
    cp "$PHOTO" "$INBOX/breakroom_cam/smoke_$(date +%s).jpg"
  fi
  CY=$($CURL -X POST "$BASE/api/cycle")
  note "cycle: $CY"
  EV=$($CURL "$BASE/api/evaluations?limit=20")
  ROW=$(echo "$EV" | json "[e for e in d if e['rule_id']==$IMG_ID][0]")
  MA=$(echo "$EV" | json "[e for e in d if e['rule_id']==$IMG_ID][0]['model_answer']")
  LAT=$(echo "$EV" | json "[e for e in d if e['rule_id']==$IMG_ID][0]['latency_ms']")
  if [ -n "$ROW" ] && [ "$MA" != "None" ] && [ -n "$MA" ]; then
    ok "vision evaluation recorded (latency ${LAT} ms)"
    note "model said: $MA"
    echo "$MA" | grep -q '"answer": *"yes"' \
      && ok "model answered YES (empty basket found) — alert should have fired" \
      || note "model answer was not yes — check reasoning above; snack_wall_2 contains an empty basket"
  else
    bad "no vision evaluation for rule $IMG_ID: $ROW"
  fi
fi

echo "== 8. Image serving =="
IMG=$($CURL "$BASE/api/sensors" | json "[s['latest_reading']['image_path'] for s in d if s['name']=='breakroom_cam'][0]")
if [ -n "$IMG" ] && [ "$IMG" != "None" ]; then
  NAME=$(basename "$IMG")
  CODE=$($CURL -o /dev/null -w '%{http_code}' "$BASE/images/$NAME")
  [ "$CODE" = "200" ] && ok "thumbnail served ($NAME)" || bad "/images/$NAME -> $CODE"
else
  note "no image reading yet for breakroom_cam"
fi

echo "== 9. Cleanup (disable smoke-test rules) =="
for ID in ${KEG_ID:-} ${IMG_ID:-}; do
  $CURL -X POST "$BASE/api/rules/$ID/disable" >/dev/null && note "disabled rule $ID"
done

echo
echo "================================"
echo "  $PASS passed, $FAIL failed"
echo "================================"
[ "$FAIL" -eq 0 ]
