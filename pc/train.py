"""Train and evaluate the gesture decision tree, then export it for the Pico.

Usage:  python train.py ../data/*.csv [--depth 5]

  - Compares tree depths 3, 5 and 8. With recordings from 2+ people it uses
    leave-one-person-out testing (train on everyone else, test on the person
    left out), otherwise 5-fold cross-validation.
  - Writes results/depth_results.csv, results/confusion_depth<D>.png and
    results/model_d<D>.py (one per depth, for the on-device benchmark).
  - Writes results/per_person.csv (accuracy for each person left out) and
    results/baselines.csv (two simple baselines the tree has to beat).
  - Trains the final tree on all data and writes ../pico/model.py.
Needs: pip install scikit-learn matplotlib
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
PICO = os.path.join(HERE, "..", "pico")
sys.path.insert(0, PICO)
import features  # noqa: E402  (shared with the Pico)
from export_tree import tree_to_micropython  # noqa: E402

DEPTHS = (3, 5, 8)


def load(paths):
    X, y, groups = [], [], []
    n = features.WINDOW
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.reader(f):
                if not row:
                    continue
                person, label, values = row[0], row[1], [float(v) for v in row[2:]]
                if len(values) != 6 * n:
                    print("Skipping bad row in", path)
                    continue
                window = [values[k * n:(k + 1) * n] for k in range(6)]
                X.append(features.extract(window))
                y.append(label)
                groups.append(person)
    return np.array(X), np.array(y), np.array(groups)


def make_tree(depth):
    return DecisionTreeClassifier(max_depth=depth, min_samples_leaf=3, random_state=0)


def splitter(y, groups):
    if len(set(groups)) >= 2:
        return LeaveOneGroupOut(), "leave-one-person-out"
    folds = max(2, min(5, int(min(np.unique(y, return_counts=True)[1]))))
    return StratifiedKFold(folds, shuffle=True, random_state=0), "%d-fold cross-validation" % folds


def baselines(X, y, groups, cv):
    """Controls: always the most common class, and thresholds on the rotation energy alone."""
    energy = [features.FEATURE_NAMES.index("gyro_mag_mean")]
    rows = []
    for name, model, cols in (("majority class", DummyClassifier(strategy="most_frequent"), slice(None)),
                              ("rotation-energy thresholds", make_tree(3), energy)):
        pred = cross_val_predict(model, X[:, cols], y, groups=groups, cv=cv)
        rows.append({"model": name, "accuracy": round(accuracy_score(y, pred), 4),
                     "macro_f1": round(f1_score(y, pred, average="macro", zero_division=0), 4)})
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def save_confusion(cm, labels, path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping", path)
        return
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(paths, depth, results_dir, model_out):
    X, y, groups = load(paths)
    if len(y) == 0:
        raise SystemExit("No recordings found")
    os.makedirs(results_dir, exist_ok=True)
    labels = sorted(set(y.tolist()))
    print("%d recordings, %d people, classes: %s" % (len(y), len(set(groups)), labels))
    cv, cv_name = splitter(y, groups)
    print("Testing with", cv_name)

    rows, per_person = [], []
    for d in DEPTHS:
        pred = cross_val_predict(make_tree(d), X, y, groups=groups, cv=cv)
        acc = accuracy_score(y, pred)
        f1 = f1_score(y, pred, average="macro", zero_division=0)
        for p in sorted(set(groups.tolist())):
            m = groups == p
            per_person.append({"depth": d, "person": p, "recordings": int(m.sum()),
                               "accuracy": round(accuracy_score(y[m], pred[m]), 4)})
        full = make_tree(d).fit(X, y)
        code = tree_to_micropython(full, full.classes_, features.FEATURE_NAMES, "Evaluation model.")
        with open(os.path.join(results_dir, "model_d%d.py" % d), "w") as f:
            f.write(code)
        rows.append({"depth": d, "accuracy": round(acc, 4), "macro_f1": round(f1, 4),
                     "leaves": full.get_n_leaves(), "code_bytes": len(code.encode())})
        cm = confusion_matrix(y, pred, labels=labels)
        save_confusion(cm, labels, os.path.join(results_dir, "confusion_depth%d.png" % d),
                       "Depth %d, accuracy %.1f%% (%s)" % (d, 100 * acc, cv_name))
        print("depth %d: accuracy %.3f, macro-F1 %.3f, %d leaves, %d bytes of code" %
              (d, acc, f1, full.get_n_leaves(), len(code.encode())))
        if d == depth:
            print(classification_report(y, pred, zero_division=0))

    write_csv(os.path.join(results_dir, "depth_results.csv"), rows)
    write_csv(os.path.join(results_dir, "per_person.csv"), per_person)
    base = baselines(X, y, groups, cv)
    write_csv(os.path.join(results_dir, "baselines.csv"), base)
    for b in base:
        print("baseline %s: accuracy %.3f, macro-F1 %.3f" % (b["model"], b["accuracy"], b["macro_f1"]))

    final = make_tree(depth).fit(X, y)
    code = tree_to_micropython(final, final.classes_, features.FEATURE_NAMES,
                               "Trained on %d recordings from %d people." % (len(y), len(set(groups))))
    with open(model_out, "w") as f:
        f.write(code)
    print("Wrote", model_out)
    return final, X, y, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--depth", type=int, default=5, help="depth of the model exported to the Pico")
    ap.add_argument("--results", default=os.path.join(HERE, "..", "results"))
    ap.add_argument("--model-out", default=os.path.join(PICO, "model.py"))
    args = ap.parse_args()
    paths = [p for pattern in args.csv for p in glob.glob(pattern)]
    run(paths, args.depth, args.results, args.model_out)


if __name__ == "__main__":
    main()
