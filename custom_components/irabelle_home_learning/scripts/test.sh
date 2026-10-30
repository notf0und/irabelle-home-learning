#!/usr/bin/env bash
# Run the test suite inside the Home Assistant container.
#
# The tests build a real `HomeAssistant` object and call the component's own code, so they
# run where Home Assistant is: in the container, against the same version the house is
# running. No pytest, no test harness, no network.
#
#     scripts/test.sh          # everything
#     scripts/test.sh -v       # (extra arguments are passed to the runner)
set -euo pipefail

CONTAINER="${HA_CONTAINER:-homeassistant}"
TESTS="/config/custom_components/irabelle_home_learning/tests/run.py"

if ! docker exec "$CONTAINER" test -f "$TESTS"; then
  echo "cannot find $TESTS in container '$CONTAINER'" >&2
  echo "set HA_CONTAINER if the container has another name" >&2
  exit 2
fi

docker exec "$CONTAINER" python3 "$TESTS" "$@"
