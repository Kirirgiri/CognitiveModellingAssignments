"""Helper functions for the Cognitive Modeling course (Homework 1).

You do not need to change anything in this file.
"""
from __future__ import annotations

import pandas as pd
from pyprims import Model, run_trials

__all__ = ["transfer_experiment", "learning_curve", "operator_prims", "shared_prims"]


# ---------------------------------------------------------------- experiments
def transfer_experiment(test, train=None, n_train=100, n_test=30, runs=20, label=None, **params):
    """Run a transfer experiment and return the data as a DataFrame.

    ``test`` and ``train`` are models: a .prims file, PRIMs text, or a
    notebook Session (its current task is used, e.g. a model you wrote in a
    ``%%prims`` cell).

    For each of ``runs`` model runs a fresh model is made.  If ``train`` is
    given (a .prims file), the model first does ``n_train`` trials of it; then
    it does ``n_test`` trials of ``test``.  Only the test trials are returned.

    ``label`` names the condition in the data (by default "after <train>" or
    "no training").  The run number is also the random seed, so the results
    are reproducible.
    """
    if label is None:
        label = "no training" if train is None else f"after {n_train} trials of {_stem(train)}"
    rows = []
    for r in range(runs):
        m = Model(seed=r, trace=False, batch_mode=True)
        if train is not None:
            run_trials(m, n_train, _load_task(train))
        rows += run_trials(m, n_test, _load_task(test), run=r, condition=label)
    data = pd.DataFrame(rows)
    failed = (data.outcome != "success").sum()
    if failed:
        print(f"Warning: {failed} of {len(data)} test trials did not end with success "
              f"(outcomes: {data.outcome.value_counts().to_dict()})")
    return data


def learning_curve(data, block=5, ax=None, title=None):
    """Plot mean trial time per block of trials, one line per condition (±1 SE over runs).

    Returns a table with the means, one row per condition and one column per block.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    if not isinstance(data, pd.DataFrame):
        data = pd.concat(data, ignore_index=True)
    d = data.assign(block=(data.trial - 1) // block + 1)
    per_run = d.groupby(["condition", "run", "block"], sort=False).result.mean().reset_index()
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 3.8))
    table = {}
    for cond, grp in per_run.groupby("condition", sort=False):
        s = grp.groupby("block").result.agg(["mean", "std", "count"])
        se = s["std"].fillna(0) / np.sqrt(s["count"])
        line, = ax.plot(s.index, s["mean"], marker="o", markersize=4, label=cond)
        ax.fill_between(s.index, s["mean"] - se, s["mean"] + se, alpha=0.2, color=line.get_color())
        table[cond] = s["mean"].round(2)
    ax.set_xlabel(f"Block of {block} trials")
    ax.set_ylabel("Mean trial time (s)")
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    plt.show()
    return pd.DataFrame(table).T.rename_axis(columns="block")


# ---------------------------------------------------------------- shared PRIMs
def operator_prims(source):
    """The PRIMs of each operator in a model, as a DataFrame.

    ``source`` is a .prims file, PRIMs text, or a pyprims Session/Model (then
    all operators of all loaded tasks are listed).  Constants are shown as C1,
    C2 ... in the order they appear in the operator, which is how PRIMs
    represents them: ``count-fact -> RT1`` and ``answer -> RT1`` are the same
    PRIM if the constant is the first one in both operators.
    """
    m = _as_model(source, whole_session=True)
    rows = []
    for ch in m.dm.chunks.values():
        if ch.type != "operator":
            continue
        cond = ch.slotvals.get("condition")
        act = ch.slotvals.get("action")
        rows.append(dict(task=", ".join(m.tasks[t].name for t in ch.defined_in if 0 <= t < len(m.tasks)),
                         operator=ch.name,
                         conditions=" ; ".join(cond.split(";")) if isinstance(cond, str) else "",
                         actions=" ; ".join(act.split(";")) if isinstance(act, str) else ""))
    return pd.DataFrame(rows)


def shared_prims(task_a, task_b, minimum=2):
    """Which PRIM sequences can be shared by operators of two models?

    PRIMs learns productions that carry out a *sequence* of PRIMs.  A learned
    production is used for an operator when what remains to be done of that
    operator matches it exactly.  So two operators can share a learned
    production when they **end** in the same sequence of PRIMs: the same
    actions at the end, or, if all actions are the same, the same actions and
    the same last conditions.

    For each pair of operators (one from each model) this function gives the
    longest sequence they share at the end, if it is at least ``minimum``
    PRIMs long.  ``task_a`` and ``task_b`` are .prims files (or PRIMs text).
    """
    ops_a = _ops(task_a)
    ops_b = _ops(task_b)
    rows = []
    for na, (ca, aa) in ops_a.items():
        for nb, (cb, ab) in ops_b.items():
            act = _common_suffix(aa, ab)
            seq = act
            if len(act) == len(aa) == len(ab):           # all actions shared: also look at the conditions
                seq = _common_suffix(ca, cb) + act
            if len(seq) >= minimum:
                rows.append({f"operator in {_stem(task_a)}": na, f"operator in {_stem(task_b)}": nb,
                             "shared PRIMs": len(seq),
                             "shared sequence": " ; ".join(seq),
                             "whole operator": len(seq) == len(ca) + len(aa) == len(cb) + len(ab)})
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"No shared sequences of {minimum} or more PRIMs.")
        return None
    return df.sort_values("shared PRIMs", ascending=False).reset_index(drop=True)


def _ops(source):
    m = _as_model(source)
    out = {}
    for ch in m.dm.chunks.values():
        if ch.type == "operator":
            c = ch.slotvals.get("condition")
            a = ch.slotvals.get("action")
            out[ch.name] = (c.split(";") if isinstance(c, str) else [],
                            a.split(";") if isinstance(a, str) else [])
    return out


def _common_suffix(x, y):
    n = 0
    while n < min(len(x), len(y)) and x[-1 - n] == y[-1 - n]:
        n += 1
    return x[len(x) - n:] if n else []


def _is_session(x):
    return hasattr(x, "model") and hasattr(x, "entries")


def _as_model(source, whole_session=False):
    if _is_session(source) and whole_session:
        return source.model
    if isinstance(source, Model):
        return source
    m = Model(seed=0, trace=False, batch_mode=True)
    m.load(_load_task(source))
    return m


def _load_task(source):
    from pyprims import load_task, parse_prims
    if _is_session(source):
        source = source.source()
    s = str(source)
    if "\n" in s or "define " in s:
        return parse_prims(s)
    return load_task(s)


def _stem(source):
    if _is_session(source):
        return source.current_task
    s = str(source)
    if "\n" in s:
        import re
        mt = re.search(r"define\s+task\s+([\w-]+)", s)
        return mt.group(1) if mt else "model"
    import os
    return os.path.splitext(os.path.basename(s))[0]
