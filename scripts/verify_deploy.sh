#!/bin/sh
# Smoke-test a deployed recommender. Usage:
#   RECOMMENDER_URL=https://your-service ./scripts/verify_deploy.sh
#   RECOMMENDER_URL=http://localhost:8000 RECOMMENDER_API_KEY=secret ./scripts/verify_deploy.sh
set -eu

BASE_URL="${RECOMMENDER_URL:-http://localhost:8000}"
BASE_URL="${BASE_URL%/}"
TOKEN="${RECOMMENDER_API_KEY:-}"
TEST_USER_ID="${TEST_USER_ID:-0}"

auth_header() {
  if [ -n "$TOKEN" ]; then
    printf '%s' "-H" "X-Internal-Token: $TOKEN"
  fi
}

echo "==> GET $BASE_URL/health"
curl -fsS "$BASE_URL/health" | python -m json.tool

echo ""
echo "==> GET $BASE_URL/ready"
READY_JSON=$(curl -fsS "$BASE_URL/ready")
echo "$READY_JSON" | python -m json.tool

USERS=$(echo "$READY_JSON" | python -c "import json,sys; print(json.load(sys.stdin).get('users', 0))")
CAN_SERVE=$(echo "$READY_JSON" | python -c "import json,sys; print(json.load(sys.stdin).get('can_serve_recommendations', False))")

if [ "$USERS" = "0" ]; then
  echo ""
  echo "ERROR: users=0 — /recommend will return 404 (user_not_found) for every request."
  echo "Fix: set EXPRESS_INTERNAL_URL, then POST /sync/bootstrap with X-Internal-Token."
  exit 1
fi

if [ "$CAN_SERVE" != "True" ] && [ "$CAN_SERVE" != "true" ]; then
  echo ""
  echo "ERROR: can_serve_recommendations=false — index not ready for recommendations."
  exit 1
fi

echo ""
echo "==> POST $BASE_URL/recommend (user_id=$TEST_USER_ID)"
REC_ARGS="-fsS -X POST $BASE_URL/recommend -H Content-Type:application/json"
if [ -n "$TOKEN" ]; then
  REC_ARGS="$REC_ARGS -H X-Internal-Token:$TOKEN"
fi
# shellcheck disable=SC2086
curl $REC_ARGS -d "{\"user_id\": \"$TEST_USER_ID\", \"top_k\": 3}" | python -m json.tool

echo ""
echo "OK: deploy verification passed."
