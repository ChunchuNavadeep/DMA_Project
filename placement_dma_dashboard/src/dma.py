"""Data-mining utilities for the Placement DMA dashboard.

Association mining uses mlxtend when it is installed and working. If mlxtend is
missing, or raises for any reason (API changes between releases are common), the
module transparently falls back to the self-contained Apriori / FP-Growth
implementations below, so the dashboard keeps working. Association rules are
always derived from the frequent itemsets by this module, which removes any
dependence on the changing ``association_rules`` signature.
"""
from __future__ import annotations

import math
import time
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, auc, classification_report, confusion_matrix,
    mean_absolute_error, precision_score, r2_score, recall_score, roc_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:  # optional accelerator
    from mlxtend.frequent_patterns import apriori as _mlx_apriori
    from mlxtend.frequent_patterns import fpgrowth as _mlx_fpgrowth
except Exception:  # pragma: no cover - depends on the environment
    _mlx_apriori = _mlx_fpgrowth = None

PLACED_ITEM = "Placement_Placed"
NOT_PLACED_ITEM = "Placement_NotPlaced"
RULE_COLUMNS = ["antecedents", "consequents", "support", "confidence", "lift", "leverage"]
MAX_ITEMSETS = 150_000  # safety valve against combinatorial explosions


# --------------------------------------------------------------------------- #
# Cleaning helpers
# --------------------------------------------------------------------------- #
def normalize_binary(series: pd.Series) -> pd.Series:
    """Normalize common yes/no and placement labels to 1/0; unknowns become NA."""
    text = series.astype(str).str.strip().str.lower()
    mapped = text.map({
        "yes": 1, "y": 1, "true": 1, "1": 1, "1.0": 1, "placed": 1,
        "no": 0, "n": 0, "false": 0, "0": 0, "0.0": 0,
        "not placed": 0, "not_placed": 0, "unplaced": 0,
    })
    numeric = pd.to_numeric(series, errors="coerce")
    result = mapped.where(mapped.notna(), numeric)
    return result.where(result.isin([0, 1])).astype("float64")


def clean_students(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names and types of a student-level table."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    if "placed" in df.columns:
        df["placed"] = normalize_binary(df["placed"])
    if "internship" in df.columns:
        df["internship"] = normalize_binary(df["internship"])
    for col in ["cgpa", "aptitude_score", "communication_score", "projects",
                "certifications", "backlogs", "salary_lpa"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "skills" in df.columns:
        df["skills"] = df["skills"].fillna("").astype(str)
    return df


# --------------------------------------------------------------------------- #
# Transactions
# --------------------------------------------------------------------------- #
def _encode(transactions) -> pd.DataFrame:
    """One-hot encode a list of item lists into a boolean DataFrame."""
    columns = sorted({item for t in transactions for item in t})
    index = {item: i for i, item in enumerate(columns)}
    matrix = np.zeros((len(transactions), len(columns)), dtype=bool)
    for row, items in enumerate(transactions):
        for item in items:
            matrix[row, index[item]] = True
    return pd.DataFrame(matrix, columns=columns)


def _skill_item(skill: str) -> str:
    skill = skill.strip().replace(" ", "_")
    if skill.islower():
        skill = skill.upper() if len(skill) <= 3 else skill.title()
    return f"Skill_{skill}"


def _level(value, low, high, names):
    if pd.isna(value):
        return None
    return names[0] if value < low else names[1] if value < high else names[2]


def prepare_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Build one-hot transactions from a student-level table.

    Item names and cut-offs match the bundled transaction files, so results from
    uploaded data are directly comparable (CGPA <7 / 7-<8 / >=8, aptitude <55 /
    55-<75 / >=75, communication >=7, projects >=3, certifications >=2, backlogs >0).
    """
    transactions = []
    for _, row in df.iterrows():
        items = set()
        for skill in str(row.get("skills", "")).replace(",", ";").split(";"):
            skill = skill.strip()
            if skill and skill.lower() not in {"nan", "none", "null"}:
                items.add(_skill_item(skill))
        internship = normalize_binary(pd.Series([row.get("internship")])).iloc[0]
        if pd.notna(internship):
            items.add("Internship_Yes" if internship == 1 else "Internship_No")
        level = _level(pd.to_numeric(row.get("cgpa"), errors="coerce"), 7, 8, ("Low", "Medium", "High"))
        if level:
            items.add(f"CGPA_{level}")
        level = _level(pd.to_numeric(row.get("aptitude_score"), errors="coerce"), 55, 75, ("Low", "Medium", "High"))
        if level:
            items.add(f"Aptitude_{level}")
        comm = pd.to_numeric(row.get("communication_score"), errors="coerce")
        if pd.notna(comm):
            items.add("Communication_High" if comm >= 7 else "Communication_Developing")
        projects = pd.to_numeric(row.get("projects"), errors="coerce")
        if pd.notna(projects):
            items.add("Projects_3plus" if projects >= 3 else "Projects_0to2")
        certs = pd.to_numeric(row.get("certifications"), errors="coerce")
        if pd.notna(certs):
            items.add("Certification_2plus" if certs >= 2 else "Certification_0to1")
        backlogs = pd.to_numeric(row.get("backlogs"), errors="coerce")
        if pd.notna(backlogs):
            items.add("Backlogs_Yes" if backlogs > 0 else "Backlogs_No")
        placed = normalize_binary(pd.Series([row.get("placed")])).iloc[0]
        if pd.notna(placed):
            items.add(PLACED_ITEM if placed == 1 else NOT_PLACED_ITEM)
        transactions.append(sorted(items))
    if not transactions or not any(transactions):
        raise ValueError("No usable transaction items were found in the dataset.")
    return _encode(transactions)


def prepare_precomputed_transactions(path: str) -> pd.DataFrame:
    """Load a transaction_id/items CSV supplied with the dataset."""
    raw = pd.read_csv(path)
    if not {"transaction_id", "items"}.issubset(raw.columns):
        raise ValueError("Transaction CSV must contain transaction_id and items columns.")
    transactions = [
        sorted({item.strip() for item in str(value).split(",") if item.strip() and item.strip().lower() != "nan"})
        for value in raw["items"].fillna("")
    ]
    if not any(transactions):
        raise ValueError("The transaction file contains no items.")
    return _encode(transactions)


# --------------------------------------------------------------------------- #
# Frequent itemset mining (built-in implementations)
# --------------------------------------------------------------------------- #
def _min_count(min_support: float, n: int) -> int:
    return max(1, math.ceil(min_support * n - 1e-9))


def _apriori_builtin(tx: pd.DataFrame, min_support: float, max_len: int) -> dict:
    """Level-wise Apriori with candidate pruning; supports counted with bit-masks."""
    n = len(tx)
    min_count = _min_count(min_support, n)
    masks = {}
    for item in tx.columns:
        mask = 0
        for pos in np.flatnonzero(tx[item].to_numpy(dtype=bool)):
            mask |= 1 << int(pos)
        if mask.bit_count() >= min_count:
            masks[frozenset([item])] = mask
    result = {s: m.bit_count() for s, m in masks.items()}
    current = masks
    size = 2
    while current and size <= max_len:
        ordered = sorted(tuple(sorted(s)) for s in current)
        known = set(current)
        nxt = {}
        for i, a in enumerate(ordered):
            for b in ordered[i + 1:]:
                if a[:-1] != b[:-1]:
                    break  # sorted order: no further shared prefix for `a`
                candidate = frozenset(a) | {b[-1]}
                if candidate in nxt:
                    continue
                if all(frozenset(sub) in known for sub in combinations(sorted(candidate), size - 1)):
                    mask = current[frozenset(a)] & current[frozenset(b)]
                    if mask.bit_count() >= min_count:
                        nxt[candidate] = mask
        result.update({s: m.bit_count() for s, m in nxt.items()})
        if len(result) > MAX_ITEMSETS:
            raise ValueError("Too many frequent itemsets. Increase minimum support or reduce maximum length.")
        current = nxt
        size += 1
    return {s: c / n for s, c in result.items()}


class _Node:
    __slots__ = ("item", "count", "parent", "children")

    def __init__(self, item, parent):
        self.item, self.count, self.parent, self.children = item, 0, parent, {}


def _fp_mine(paths, suffix, min_count, max_len, result):
    """Mine one (conditional) database given as [(ordered items, count)]."""
    root = _Node(None, None)
    header = {}
    for items, count in paths:
        node = root
        for item in items:
            child = node.children.get(item)
            if child is None:
                child = node.children[item] = _Node(item, node)
                header.setdefault(item, []).append(child)
            child.count += count
            node = child
    for item, nodes in header.items():
        support = sum(n.count for n in nodes)
        if support < min_count:
            continue
        pattern = suffix | {item}
        result[pattern] = support
        if len(result) > MAX_ITEMSETS:
            raise ValueError("Too many frequent itemsets. Increase minimum support or reduce maximum length.")
        if len(pattern) >= max_len:
            continue
        conditional = []
        for node in nodes:
            path, parent = [], node.parent
            while parent.item is not None:
                path.append(parent.item)
                parent = parent.parent
            if path:
                conditional.append((path[::-1], node.count))
        if conditional:
            totals = Counter()
            for path, count in conditional:
                for it in path:
                    totals[it] += count
            conditional = [([it for it in path if totals[it] >= min_count], count) for path, count in conditional]
            conditional = [entry for entry in conditional if entry[0]]
            if conditional:
                _fp_mine(conditional, pattern, min_count, max_len, result)


def _fpgrowth_builtin(tx: pd.DataFrame, min_support: float, max_len: int) -> dict:
    """FP-tree construction followed by recursive conditional-pattern mining."""
    n = len(tx)
    min_count = _min_count(min_support, n)
    columns = list(tx.columns)
    matrix = tx.to_numpy(dtype=bool)
    totals = matrix.sum(axis=0)
    frequent = {columns[i]: int(totals[i]) for i in range(len(columns)) if totals[i] >= min_count}
    rank = {item: r for r, (item, _) in enumerate(sorted(frequent.items(), key=lambda kv: (-kv[1], kv[0])))}
    paths = []
    for row in matrix:
        items = sorted((columns[i] for i in np.flatnonzero(row) if columns[i] in rank), key=rank.__getitem__)
        if items:
            paths.append((items, 1))
    result: dict = {}
    _fp_mine(paths, frozenset(), min_count, max_len, result)
    return {s: c / n for s, c in result.items()}


def mine_itemsets(tx: pd.DataFrame, min_support: float, max_len: int, algorithm: str = "apriori"):
    """Return ({frozenset: support}, engine_name). ``algorithm`` is 'apriori' or 'fpgrowth'."""
    algorithm = algorithm.lower().replace("-", "").replace("_", "").replace(" ", "")
    if algorithm not in {"apriori", "fpgrowth"}:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    tx = tx.astype(bool)
    if tx.empty or tx.shape[1] == 0:
        return {}, "none"
    library = _mlx_apriori if algorithm == "apriori" else _mlx_fpgrowth
    if library is not None:
        try:
            frequent = library(tx, min_support=min_support, use_colnames=True, max_len=max_len)
            supports = {frozenset(items): float(s) for items, s in zip(frequent["itemsets"], frequent["support"])}
            return supports, "mlxtend"
        except Exception:
            pass  # fall through to the built-in implementation
    builtin = _apriori_builtin if algorithm == "apriori" else _fpgrowth_builtin
    return builtin(tx, min_support, max_len), "built-in"


# --------------------------------------------------------------------------- #
# Association rules
# --------------------------------------------------------------------------- #
def itemset_table(supports: dict, n_transactions: int) -> pd.DataFrame:
    rows = [
        {"itemset": ", ".join(sorted(s)), "length": len(s), "support": round(v, 3), "count": int(round(v * n_transactions))}
        for s, v in supports.items()
    ]
    if not rows:
        return pd.DataFrame(columns=["itemset", "length", "support", "count"])
    return pd.DataFrame(rows).sort_values(["support", "length"], ascending=[False, True]).reset_index(drop=True)


def build_rules(supports: dict, min_confidence: float) -> pd.DataFrame:
    """Generate rules A -> C from frequent itemsets (support, confidence, lift, leverage)."""
    rows = []
    for itemset, support in supports.items():
        if len(itemset) < 2:
            continue
        items = sorted(itemset)
        for r in range(1, len(items)):
            for antecedent in combinations(items, r):
                a = frozenset(antecedent)
                c = itemset - a
                sa, sc = supports.get(a), supports.get(c)
                if not sa or not sc:
                    continue
                confidence = support / sa
                if confidence + 1e-12 < min_confidence:
                    continue
                rows.append((", ".join(sorted(a)), ", ".join(sorted(c)), support, confidence,
                             confidence / sc, support - sa * sc))
    if not rows:
        return pd.DataFrame(columns=RULE_COLUMNS)
    rules = pd.DataFrame(rows, columns=RULE_COLUMNS)
    rules = rules.sort_values(["lift", "confidence", "support"], ascending=False)
    return rules.round(3).reset_index(drop=True)


def _rules(tx, min_support, min_confidence, max_len, algorithm):
    supports, _ = mine_itemsets(tx, min_support, max_len, algorithm)
    return build_rules(supports, min_confidence)


def apriori_rules(tx, min_support, min_confidence, max_len):
    return _rules(tx, min_support, min_confidence, max_len, "apriori")


def fp_growth_rules(tx, min_support, min_confidence, max_len):
    return _rules(tx, min_support, min_confidence, max_len, "fpgrowth")


def compare_algorithms(tx, min_support, min_confidence, max_len) -> pd.DataFrame:
    """Run Apriori and FP-Growth on the same transactions and compare outputs."""
    rows, results = [], {}
    for label, key in (("Apriori", "apriori"), ("FP-Growth", "fpgrowth")):
        start = time.perf_counter()
        supports, engine = mine_itemsets(tx, min_support, max_len, key)
        rules = build_rules(supports, min_confidence)
        elapsed = time.perf_counter() - start
        results[label] = supports
        rows.append({"algorithm": label, "engine": engine, "frequent itemsets": len(supports),
                     "rules": len(rules), "runtime (s)": round(elapsed, 3)})
    a, f = results["Apriori"], results["FP-Growth"]
    same = set(a) == set(f) and all(abs(a[k] - f[k]) < 1e-9 for k in a)
    table = pd.DataFrame(rows)
    table["same itemsets & supports"] = "Yes" if same else "No"
    return table


# --------------------------------------------------------------------------- #
# Model features, clustering, classification, regression
# --------------------------------------------------------------------------- #
def _features(df):
    """Use available non-outcome predictors; avoid IDs, salary and target leakage."""
    features = {}
    for column in ["cgpa", "aptitude_score", "communication_score", "projects", "certifications", "backlogs"]:
        if column in df.columns:
            features[column] = pd.to_numeric(df[column], errors="coerce")
    if "internship" in df.columns:
        features["internship"] = normalize_binary(df["internship"])
    if not features:
        raise ValueError("No supported predictor columns found. Include CGPA, aptitude, projects, etc.")
    x = pd.DataFrame(features, index=df.index).astype("float64")
    x = x.dropna(axis=1, how="all")  # entirely missing columns cannot help
    x = x.loc[:, x.nunique(dropna=True) > 1]  # constant columns cannot help
    if x.empty:
        raise ValueError("The available predictor columns contain no usable variation.")
    return x.fillna(x.median(numeric_only=True)).fillna(0)


def model_features(df):
    """Public alias of the predictor table used by every model."""
    return _features(df)


def clustering(df, k):
    x = _features(df)
    if len(x) < k:
        raise ValueError(f"K-Means needs at least {k} student records; found {len(x)}.")
    scaled = StandardScaler().fit_transform(x)
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = pd.Series(model.fit_predict(scaled), index=df.index, name="cluster")
    unique_labels = np.unique(labels)
    silhouette = silhouette_score(scaled, labels) if 1 < len(unique_labels) < len(df) else None
    profile = x.copy()
    profile["cluster"] = labels
    summary = profile.groupby("cluster").mean().round(2)
    summary["students"] = profile.groupby("cluster").size()
    return labels, model.cluster_centers_, summary.reset_index(), silhouette


def cluster_diagnostics(df, k_max=8) -> pd.DataFrame:
    """Inertia (elbow) and silhouette for K = 2..k_max."""
    x = _features(df)
    scaled = StandardScaler().fit_transform(x)
    rows = []
    for k in range(2, min(k_max, len(x) - 1) + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10).fit(scaled)
        sil = silhouette_score(scaled, model.labels_) if len(np.unique(model.labels_)) > 1 else np.nan
        rows.append({"k": k, "inertia": model.inertia_, "silhouette": sil})
    return pd.DataFrame(rows, columns=["k", "inertia", "silhouette"])


def cluster_projection(df, labels):
    """2-D PCA projection of the standardised features for plotting clusters."""
    x = _features(df)
    scaled = StandardScaler().fit_transform(x)
    components = min(2, scaled.shape[1], len(scaled))
    pca = PCA(n_components=components, random_state=42)
    coords = pca.fit_transform(scaled)
    if components == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(len(coords))])
    ratio = list(pca.explained_variance_ratio_) + [0.0] * (2 - components)
    frame = pd.DataFrame({"pc1": coords[:, 0], "pc2": coords[:, 1], "cluster": np.asarray(labels)}, index=df.index)
    return frame, ratio


def train_placement_model(df, max_depth=4):
    if "placed" not in df.columns:
        raise ValueError("Missing required target column: placed")
    x = _features(df)
    y = normalize_binary(df["placed"])
    valid = y.notna()
    x, y = x.loc[valid], y.loc[valid].astype(int)
    if y.nunique() < 2:
        raise ValueError("The placed column must contain both placed and not-placed examples.")
    if len(x) < 10 or y.value_counts().min() < 2:
        raise ValueError("At least 10 valid records and at least 2 examples per class are required.")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, stratify=y)
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=2, class_weight="balanced", random_state=42)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    proba = model.predict_proba(x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    metrics = {
        "accuracy": accuracy_score(y_test, prediction),
        "precision": precision_score(y_test, prediction, zero_division=0),
        "recall": recall_score(y_test, prediction, zero_division=0),
        "report": classification_report(y_test, prediction, zero_division=0),
        "confusion": confusion_matrix(y_test, prediction, labels=[0, 1]),
        "baseline": float(max(y_test.mean(), 1 - y_test.mean())),
        "roc": (fpr, tpr, float(auc(fpr, tpr))),
        "n_train": len(x_train),
        "n_test": len(x_test),
    }
    importance = pd.DataFrame({"feature": x.columns, "importance": model.feature_importances_})
    importance = importance.sort_values("importance", ascending=False).reset_index(drop=True)
    return metrics, importance, model


def salary_regression(df):
    if not {"salary_lpa", "placed"}.issubset(df.columns):
        raise ValueError("Salary analysis requires placed and salary_lpa columns.")
    placed_flag = normalize_binary(df["placed"])
    salary = pd.to_numeric(df["salary_lpa"], errors="coerce")
    valid = placed_flag.eq(1) & salary.notna()
    placed = df.loc[valid].copy()
    if len(placed) < 8:
        raise ValueError(f"At least 8 placed students with valid salary values are recommended; found {len(placed)}.")
    x, y = _features(placed), salary.loc[valid]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42, min_samples_leaf=2)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    metrics = {
        "r2": r2_score(y_test, prediction),
        "mae": mean_absolute_error(y_test, prediction),
        "baseline_mae": mean_absolute_error(y_test, np.full(len(y_test), y_train.mean())),
        "actual": y_test.to_numpy(),
        "predicted": prediction,
        "n": len(placed),
    }
    importance = pd.DataFrame({"feature": x.columns, "importance": model.feature_importances_})
    importance = importance.sort_values("importance", ascending=False).reset_index(drop=True)
    return metrics, importance
