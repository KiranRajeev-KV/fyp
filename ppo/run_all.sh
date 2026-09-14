#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# The whole compute plan, relaunchable.
#
#   nohup bash ppo/run_all.sh >> ppo/runs/run_all.log 2>&1 &
#
# Runs V5 (ppo/run_v5.sh) and then the walk-forward (ppo/run_walkforward.sh).
# Both are idempotent: finished runs are skipped and partial runs resume from
# their per-generation checkpoint, so after a shutdown or crash simply run the
# same command again and it picks up where it stopped. Two overnight shutdowns
# (2026-09-09 and 2026-09-14) each cost a full run before this existed.
#
# Optional: relaunch automatically after every boot with
#   crontab -e   ->   @reboot sleep 120 && cd /home/nandu/uni/fyp/fyp && bash ppo/run_all.sh >> ppo/runs/run_all.log 2>&1
# ---------------------------------------------------------------------------
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
say() { echo "[$(date '+%F %T')] run_all: $*"; }

if pgrep -f "run_coevolution.py|run_walkforward.py" >/dev/null; then
  say "engine processes already running; not starting a second copy"; exit 0
fi

say "=== V5 ==="
bash "$REPO/ppo/run_v5.sh" >> "$REPO/ppo/runs/v5.log" 2>&1
say "V5 script exited ($?)"

say "=== walk-forward ==="
bash "$REPO/ppo/run_walkforward.sh" >> "$REPO/ppo/runs/walkforward.log" 2>&1
say "walk-forward script exited ($?)"
say "ALL DONE"
