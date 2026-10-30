import argparse
import glob
import json
import os
import pickle
import pprint
from datetime import datetime


MODEL_BANK_DIR = "model_bank/"
CHAMPION_FILE = MODEL_BANK_DIR + "champion.json"
PROPOSAL_FILE = MODEL_BANK_DIR + "champion_proposal.json"


def load_candidates(train_date_str):
    # every artefact trained on this date is a candidate, whatever the algorithm
    suffix = train_date_str.replace('-', '_') + '.pkl'
    rows = []
    for path in sorted(glob.glob(MODEL_BANK_DIR + "credit_model_*_" + suffix)):
        with open(path, 'rb') as f:
            artefact = pickle.load(f)
        r = artefact['results']
        rows.append({
            "model_name": os.path.basename(path),
            "auc_train": round(float(r['auc_train']), 4),
            "auc_test": round(float(r['auc_test']), 4),
            "auc_oot": round(float(r['auc_oot']), 4),
            "gini_oot": round(float(r['gini_oot']), 4),
            "overfit_gap": round(float(r['auc_train'] - r['auc_test']), 4),
            "hp_params": {k: (v if isinstance(v, (int, float, str)) else str(v))
                          for k, v in artefact['hp_params'].items()},
        })
    return rows


def main(snapshotdate, promote=False):
    print('\n\n---starting model selection job---\n\n')

    candidates = load_candidates(snapshotdate)
    if not candidates:
        # nothing was trained this month, so there is nothing to select;
        # the champion chosen earlier stays in place
        print("skip: no models trained on " + snapshotdate)
        return

    # pick by out-of-time AUC: it is the window closest to how the model will be
    # used in production. test AUC breaks ties, and a model that overfits badly
    # (train - test gap above 0.1) is not allowed to win on a lucky OOT month
    eligible = [c for c in candidates if c["overfit_gap"] <= 0.10] or candidates
    champion = max(eligible, key=lambda c: (c["auc_oot"], c["auc_test"]))

    record = {
        "model_name": champion["model_name"],
        "selected_on": snapshotdate,
        "selected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rule": "highest OOT AUC among candidates with train-test AUC gap <= 0.10; "
                "ties broken by test AUC",
        "champion": champion,
        "candidates": candidates,
    }
    pprint.pprint({k: v for k, v in record.items() if k != "candidates"})

    # the first deployment is automatic; after that a new winner is only a
    # proposal until a person re-runs this with --promote (SOP step 4:
    # models never promote themselves)
    if os.path.exists(CHAMPION_FILE) and not promote:
        with open(CHAMPION_FILE) as f:
            current = json.load(f)["model_name"]
        with open(PROPOSAL_FILE, 'w') as f:
            json.dump(record, f, indent=1)
        print(f"champion stays {current}; proposal written to {PROPOSAL_FILE} "
              f"-- run with --promote to accept it")
        return
    with open(CHAMPION_FILE, 'w') as f:
        json.dump(record, f, indent=1)
    if os.path.exists(PROPOSAL_FILE):
        os.remove(PROPOSAL_FILE)
    print("champion written to", CHAMPION_FILE)
    print('\n\n---model selection job completed---\n\n')


def resolve_model_name(modelname):
    """'champion' -> the file named in champion.json; anything else unchanged.
    Returns None when no champion has been chosen yet."""
    if modelname != "champion":
        return modelname
    if not os.path.exists(CHAMPION_FILE):
        return None
    with open(CHAMPION_FILE) as f:
        return json.load(f)["model_name"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="choose the champion among the models trained on a date")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--promote", action="store_true",
                        help="replace the current champion with this date's winner (human step)")
    args = parser.parse_args()
    main(args.snapshotdate, args.promote)
