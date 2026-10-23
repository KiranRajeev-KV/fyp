"""SYNTH_V2 (Phase 2, P2): SYNTH_V1 with a weaker regime signal.

Uses the same mixes, seeds, and splits as synth_market.py, with a noisier
regime_signal and drifts scaled by DRIFT_SCALE. The historical dev-shift probe
reported AUC 0.55 +/- 0.02 (V1: 0.60). Signal noise alone was insufficient:
V1's 19 price features gave AUC 0.590 in the original local Phase 2 ledger.
Generated metadata records the SYNTH_V2 name and parameters, without source
hashes. Parameter changes should be treated as a new benchmark.

Examples from ppo/src:
    python synth_market_v2.py --out ../data/synth_v2 --check   # build dev + cheap check
    python synth_market_v2.py --out ../data/synth_v2 --sealed  # once, at the end
"""

import argparse
import copy
import json
import os

import synth_market as v1

PARAMS = copy.deepcopy(v1.PARAMS)
DRIFT_SCALE = 0.6               # Historical SYNTH_V2 parameter choice.
PARAMS["signal_noise"] = 12.0   # V1: 2.0
PARAMS["drift"] = {r: d * DRIFT_SCALE for r, d in v1.PARAMS["drift"].items()}
AUC_BAND = (0.53, 0.57)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--sealed", action="store_true")
    args = ap.parse_args()
    if not args.check_only:
        v1.build(args.out, sealed=args.sealed, p=PARAMS)
        if not args.sealed:   # identify the V2 parameter set in generated metadata
            with open(os.path.join(args.out, "params.json"), "w") as metadata:
                json.dump({"benchmark": "SYNTH_V2", "params": PARAMS}, metadata, indent=2)
    if args.check or args.check_only:
        res = v1.check(args.out)
        s = res["dev_shift"]
        res["passed"] = s["timer_sharpe"] > s["bh_sharpe"] and AUC_BAND[0] <= s["auc"] <= AUC_BAND[1]
        print("SYNTH_V2 passed:", res["passed"])
        with open(os.path.join(args.out, "cheap_check.json"), "w") as check:
            json.dump(res, check, indent=2)


if __name__ == "__main__":
    main()
