from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import charts
from src.dma import (
    PLACED_ITEM, build_rules, clean_students, cluster_diagnostics, cluster_projection, clustering,
    compare_algorithms, itemset_table, mine_itemsets, model_features, prepare_precomputed_transactions,
    prepare_transactions, salary_regression, train_placement_model,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

st.set_page_config(page_title="Placement DMA Intelligence", page_icon="🎓", layout="wide")
st.title("🎓 Placement Analytics — DMA Intelligence")
st.caption("Apriori • FP-Growth • K-Means • Decision Tree • Random Forest Regression")


def show(fig, caption=None):
    """Render a matplotlib figure and release it (pyplot keeps figures alive otherwise)."""
    if fig is None:
        return False
    st.pyplot(fig)
    plt.close(fig)
    if caption:
        st.caption(caption)
    return True


@st.cache_data
def load_default_students():
    return pd.read_csv(DATA_DIR / "placement_students_500.csv")


@st.cache_data
def load_bundled_transactions(filename):
    return prepare_precomputed_transactions(str(DATA_DIR / filename))


@st.cache_data(show_spinner="Mining frequent itemsets…")
def mine(tx, algorithm, min_support, max_len):
    return mine_itemsets(tx, min_support, max_len, algorithm)


@st.cache_data(show_spinner="Comparing algorithms…")
def compare(tx, min_support, min_conf, max_len):
    return compare_algorithms(tx, min_support, min_conf, max_len)


# ----------------------------------------------------------------------------- data
uploaded = st.sidebar.file_uploader("Upload student-level placement CSV", type=["csv"])
if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded)
        dataset_label = f"Uploaded: {uploaded.name}"
    except Exception as exc:
        st.error(f"Could not read uploaded CSV: {exc}")
        st.stop()
else:
    raw = load_default_students()
    dataset_label = "Bundled synthetic dataset (500 students)"

df = clean_students(raw)  # lower-cases column names and normalises types once, up front

required = {"cgpa", "aptitude_score", "internship", "projects", "skills", "placed", "salary_lpa"}
missing = required - set(df.columns)
if missing:
    st.error("Missing required columns: " + ", ".join(sorted(missing)))
    st.info("Use the bundled data/placement_students_500.csv or download data/placement_template.csv.")
    st.stop()
if len(df) < 3:
    st.error("The dataset needs at least 3 student records.")
    st.stop()

st.sidebar.subheader("Dataset")
st.sidebar.caption(dataset_label)
st.sidebar.write(f"{len(df):,} student records")
st.sidebar.write(f"{len(df.columns)} columns")
st.sidebar.download_button("Download CSV template", (DATA_DIR / "placement_template.csv").read_bytes(),
                           "placement_template.csv", "text/csv")
st.sidebar.download_button("Download 500-student dataset", (DATA_DIR / "placement_students_500.csv").read_bytes(),
                           "placement_students_500.csv", "text/csv")

valid_outcomes = df["placed"].dropna()
placed_count = int(valid_outcomes.sum())
placement_rate = valid_outcomes.mean() * 100 if len(valid_outcomes) else 0
c1, c2, c3, c4 = st.columns(4)
c1.metric("Students", f"{len(df):,}")
c2.metric("Placed", f"{placed_count:,}")
c3.metric("Placement rate", f"{placement_rate:.1f}%")
c4.metric("Average CGPA", f"{df['cgpa'].mean():.2f}" if df["cgpa"].notna().any() else "N/A")

st.caption("The bundled student and transaction datasets are synthetic academic data, not actual institutional placement statistics.")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "🧩 Apriori & FP-Growth", "👥 K-Means", "🌳 Placement classification", "💰 Salary analysis"
])

# ----------------------------------------------------------------------------- overview
with tab1:
    try:
        left, right = st.columns(2)
        with left:
            show(charts.outcome_bar(df))
        with right:
            show(charts.cgpa_distribution(df))
        left, right = st.columns(2)
        with left:
            show(charts.cgpa_aptitude_scatter(df))
        with right:
            show(charts.salary_histogram(df))
        show(charts.placement_rate_panel(df))
        left, right = st.columns(2)
        with left:
            show(charts.skills_chart(df))
        with right:
            show(charts.correlation_heatmap(df))
    except Exception as exc:
        st.error(f"Some overview charts could not be drawn: {exc}")
    st.subheader("Student records preview")
    st.dataframe(df.head(30), hide_index=True)
    st.download_button("Download cleaned dataset", df.to_csv(index=False).encode("utf-8"),
                       "cleaned_placement_students.csv", "text/csv")

# ----------------------------------------------------------------------------- association rules
with tab2:
    st.subheader("Frequent itemsets and association rules")
    st.write("Apriori and FP-Growth run on the same transactions (one per student) and must return identical "
             "itemsets. Rules show co-occurrence, not causation. Lift > 1 means the items occur together more "
             "often than expected by chance.")
    k1, k2, k3, k4 = st.columns(4)
    min_support = k1.slider("Minimum support", 0.02, 0.80, 0.05, 0.01)
    min_conf = k2.slider("Minimum confidence", 0.10, 1.00, 0.30, 0.05)
    min_lift = k3.slider("Minimum lift", 0.0, 3.0, 1.0, 0.1)
    max_len = k4.slider("Maximum itemset length", 2, 5, 4)
    o1, o2 = st.columns(2)
    algo = o1.radio("Algorithm", ["Apriori", "FP-Growth"], horizontal=True)
    focus = o2.radio("Show", ["Rules that predict placement", "All rules"], horizontal=True)

    try:
        if uploaded is None:
            tx = load_bundled_transactions("apriori_transactions_500.csv" if algo == "Apriori"
                                           else "fp_growth_transactions_500.csv")
        else:
            tx = prepare_transactions(df)
        supports, engine = mine(tx, algo, min_support, max_len)
        rules_all = build_rules(supports, min_conf)
        rules_all = rules_all[rules_all["lift"] >= min_lift].reset_index(drop=True)
        rules = rules_all[rules_all["consequents"] == PLACED_ITEM].reset_index(drop=True) \
            if focus.startswith("Rules that") else rules_all
        itemsets = itemset_table(supports, len(tx))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Transactions", f"{len(tx):,}")
        m2.metric("Frequent itemsets", f"{len(itemsets):,}")
        m3.metric("Rules shown", f"{len(rules):,}")
        m4.metric("Engine", engine)

        if itemsets.empty:
            st.warning("No frequent itemsets. Lower the minimum support.")
        elif rules.empty:
            if focus.startswith("Rules that"):
                st.warning("No rules predicting placement at these thresholds. Lower the minimum support "
                           "(placed students are a minority) or confidence, or switch to “All rules”.")
            else:
                st.warning("No rules found. Lower the minimum support, confidence or lift.")

        r_tab, c_tab, i_tab, a_tab = st.tabs(["Rules", "Charts", "Frequent itemsets", "Apriori vs FP-Growth"])
        with r_tab:
            if not rules.empty:
                st.dataframe(rules.head(500), hide_index=True)
                if len(rules) > 500:
                    st.caption(f"Showing the top 500 of {len(rules):,} rules (sorted by lift). Download for all.")
                st.download_button("Download rules CSV", rules.to_csv(index=False).encode("utf-8"),
                                   f"{algo.lower().replace('-', '_')}_rules.csv", "text/csv")
        with c_tab:
            left, right = st.columns(2)
            with left:
                show(charts.top_itemsets(itemsets))
            with right:
                show(charts.rule_scatter(rules))
            show(charts.top_rules(rules))
            left, right = st.columns(2)
            with left:
                show(charts.rule_heatmap(rules_all))
            with right:
                show(charts.metric_distribution(rules))
            if rules.empty:
                st.info("Charts need at least one rule. Adjust the thresholds above.")
        with i_tab:
            if not itemsets.empty:
                st.dataframe(itemsets.head(500), hide_index=True)
                st.download_button("Download itemsets CSV", itemsets.to_csv(index=False).encode("utf-8"),
                                   "frequent_itemsets.csv", "text/csv")
        with a_tab:
            st.write("Both algorithms are run on identical transactions with the current thresholds.")
            st.dataframe(compare(tx, min_support, min_conf, max_len), hide_index=True)
            st.caption("Runtimes are indicative only; on a dataset this small the difference is negligible.")
    except Exception as exc:
        st.error(f"Association mining failed: {exc}")

# ----------------------------------------------------------------------------- K-Means
with tab3:
    st.subheader("K-Means student segmentation")
    max_k = min(8, len(df) - 1)
    if max_k < 2:
        st.warning("At least 3 students are needed for clustering.")
    else:
        k = st.slider("Number of clusters (K)", 2, max_k, min(3, max_k))
        try:
            labels, centers, summary, silhouette = clustering(df, k)
            projection, ratio = cluster_projection(df, labels)
            a, b = st.columns(2)
            a.metric("Silhouette score", f"{silhouette:.3f}" if silhouette is not None else "Unavailable")
            b.metric("Features used", ", ".join(model_features(df).columns))
            left, right = st.columns(2)
            with left:
                show(charts.cluster_scatter(projection, ratio))
            with right:
                show(charts.elbow_chart(cluster_diagnostics(df, max_k), k))
            show(charts.cluster_profile_heatmap(summary))
            show(charts.cluster_size_rate(labels, df["placed"]))
            st.subheader("Cluster profiles")
            st.dataframe(summary, hide_index=True)
            clustered = df.copy()
            clustered["cluster"] = labels
            st.download_button("Download clustered students", clustered.to_csv(index=False).encode("utf-8"),
                               "student_clusters.csv", "text/csv")
            st.caption("Cluster numbers are arbitrary. Interpret the feature profiles, not the numeric labels. "
                       "A low silhouette (< 0.25) means the groups overlap heavily.")
        except Exception as exc:
            st.error(f"Clustering could not run: {exc}")

# ----------------------------------------------------------------------------- classification
with tab4:
    st.subheader("Decision Tree placement classification")
    st.warning("Metrics use a stratified held-out test split. They are estimates, not proof of generalization "
               "or suitable for individual hiring decisions.")
    depth = st.slider("Maximum tree depth", 1, 8, 4)
    try:
        metrics, importance, model = train_placement_model(df, max_depth=depth)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        m2.metric("Precision", f"{metrics['precision']:.3f}")
        m3.metric("Recall", f"{metrics['recall']:.3f}")
        m4.metric("ROC AUC", f"{metrics['roc'][2]:.3f}")
        m5.metric("Majority baseline", f"{metrics['baseline']:.3f}")
        st.caption(f"Trained on {metrics['n_train']} and tested on {metrics['n_test']} students. The model uses "
                   "class-balanced weights, so accuracy can sit below the majority-class baseline while recall "
                   "for placed students is higher.")
        left, right = st.columns(2)
        with left:
            show(charts.confusion_chart(metrics["confusion"]))
        with right:
            show(charts.roc_chart(metrics["roc"]))
        left, right = st.columns(2)
        with left:
            show(charts.importance_chart(importance, "Decision tree feature importance"))
        with right:
            st.dataframe(importance, hide_index=True)
        with st.expander("Show fitted tree"):
            show(charts.tree_chart(model, list(model.feature_names_in_)))
        st.text(metrics["report"])
    except Exception as exc:
        st.error(f"Classification could not run: {exc}")

# ----------------------------------------------------------------------------- salary
with tab5:
    st.subheader("Salary relationship analysis")
    try:
        metrics, importance = salary_regression(df)
        a, b, c = st.columns(3)
        a.metric("R²", f"{metrics['r2']:.3f}")
        b.metric("MAE (LPA)", f"{metrics['mae']:.2f}")
        c.metric("Baseline MAE (mean salary)", f"{metrics['baseline_mae']:.2f}")
        left, right = st.columns(2)
        with left:
            show(charts.actual_vs_predicted(metrics["actual"], metrics["predicted"]))
        with right:
            show(charts.importance_chart(importance, "Salary model feature importance"))
        left, right = st.columns(2)
        with left:
            show(charts.salary_vs_cgpa(df))
        with right:
            show(charts.salary_histogram(df))
        st.dataframe(importance, hide_index=True)
        st.caption(f"Only the {metrics['n']} placed students with known salary are used. Results are exploratory "
                   "and depend on the synthetic data and split.")
    except Exception as exc:
        st.error(f"Salary analysis could not run: {exc}")

st.divider()
st.caption("Academic project • Replace synthetic data with verified, authorized institutional data before making real-world conclusions.")
