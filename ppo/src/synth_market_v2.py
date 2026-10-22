"""SYNTH_V2 (Phase 2, P2): SYNTH_V1 with a weaker regime signal.

Identical to synth_market.py (same mixes, seeds, splits; that file and its hash
8070ed6ffe695667 stay untouched) except a noisier `regime_signal` and regime
drifts scaled by DRIFT_SCALE, so the logistic probe's AUC on dev_shift is
0.55 +/- 0.02 (V1: 0.60). Weakening the signal alone cannot do it: V1's 19 price
features alone already give AUC 0.590 (ledger, Phase 2). The V2 hash covers
both files.

    python synth_market_v2.py --out ../data/synth_v2 --check   # build dev + cheap check
    python synth_market_v2.py --out ../data/synth_v2 --sealed  # once, at the end
"""

import argparse
import copy
import hashlib
import json
import os

import synth_market as v1

PARAMS = copy.deepcopy(v1.PARAMS)
DRIFT_SCALE = 0.6               # Frozen once the cheap check passes (R10).
PARAMS["signal_noise"] = 12.0   # V1: 2.0
PARAMS["drift"] = {r: d * DRIFT_SCALE for r, d in v1.PARAMS["drift"].items()}
AUC_BAND = (0.53, 0.57)


def code_hash():
    h = hashlib.sha256()
    for f in (v1.__file__, __file__):
        h.update(open(f, "rb").read())
    return h.hexdigest()[:16]


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
        if not args.sealed:   # v1.build stamps V1's hash; restamp with V2's
            json.dump({"benchmark": "SYNTH_V2", "params": PARAMS, "code_sha256_16": code_hash()},
                      open(os.path.join(args.out, "params.json"), "w"), indent=2)
    if args.check or args.check_only:
        res = v1.check(args.out)
        s = res["dev_shift"]
        res["passed"] = s["timer_sharpe"] > s["bh_sharpe"] and AUC_BAND[0] <= s["auc"] <= AUC_BAND[1]
        print("SYNTH_V2 passed:", res["passed"], "hash", code_hash())
        json.dump(res, open(os.path.join(args.out, "cheap_check.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
