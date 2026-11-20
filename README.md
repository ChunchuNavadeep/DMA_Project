# Placement Analytics Dashboard — DMA Edition

A Streamlit academic dashboard combining the placement analytics application with the supplied 500-record Data Mining dataset.

## Included analyses
- **Overview**: outcome, CGPA distribution, CGPA-vs-aptitude scatter, placement rate by internship / projects / backlogs / certifications, placement rate by skill, correlation heat map, salary histogram.
- **Apriori** and **FP-Growth**: frequent itemsets and association rules (support, confidence, lift, leverage) with filters (support, confidence, lift, "rules that predict placement"), plus charts (top itemsets, support-vs-confidence scatter, top rules by lift, lift heat map, metric histograms) and an Apriori-vs-FP-Growth comparison.
- **K-Means**: student segmentation, cluster profiles, silhouette score, elbow / silhouette curves, PCA scatter, profile heat map and placement rate per cluster.
- **Decision Tree**: placement classification with held-out accuracy, precision, recall, ROC AUC, majority baseline, confusion matrix, ROC curve, feature importance and the fitted tree.
- **Random Forest Regression**: exploratory salary prediction for placed students with known salaries (R², MAE, baseline MAE, actual-vs-predicted, feature importance, salary-vs-CGPA).

If `mlxtend` is missing or fails, the app automatically uses built-in Apriori / FP-Growth implementations (the "Engine" metric shows which one ran), so the dashboard keeps working.
## 📸 Dashboard Preview

### 1. Main Dashboard
![Main Dashboard](Visulzations/Screenshot%202026-11-21%20010923.png)

### 2. Data Analysis and Visualizations
![Data Analysis](Visulzations/Screenshot%202026-11-21%20010954.png)

### 3. Data Mining Results
![Data Mining Results](Visulzations/Screenshot%202026-11-21%20011011.png)

### 4. Dashboard Analytics
![Dashboard Analytics](Visulzations/Screenshot%202026-11-21%20011031.png)

## Merged datasets
All supplied datasets are in `data/`:
- `placement_students_500.csv` — 500 synthetic student records; primary default dashboard dataset.
- `apriori_transactions_500.csv` — 500 precomputed transactions for Apriori.
- `fp_growth_transactions_500.csv` — matching transactions for FP-Growth.
- `data_dictionary.csv` — field descriptions.
- `README.txt` — original dataset notes.
- `sample_placements.csv` and `placement_template.csv` — original sample and upload template.

The Apriori and FP-Growth transaction files contain the same records so algorithm results can be compared on equivalent input. The dashboard uses these bundled transactions by default and builds transactions from an uploaded student CSV when one is supplied (same item names and cut-offs: CGPA <7 / 7–<8 / ≥8, aptitude <55 / 55–<75 / ≥75, communication ≥7, projects ≥3, certifications ≥2, backlogs >0).

## Run locally
Python 3.10 or newer.

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`).

## Upload format
The uploaded student-level CSV must include `cgpa`, `aptitude_score`, `internship`, `projects`, `skills`, `placed`, and `salary_lpa`. Column names are case-insensitive. Optional numeric predictors include `communication_score`, `certifications`, and `backlogs`. The blank template is available in `data/placement_template.csv`.

## Important interpretation notes
- All supplied student and transaction records are **synthetic**. They are not actual college placement statistics.
- The bundled data has 500 students; 98 are marked placed and 402 not placed. Salary analysis uses the 98 placed students with known salary.
- Association rules show co-occurrence, not causation. Only 19.6% of bundled students are placed, so rules leading to placement need a low minimum support; the default is 0.05.
- On the bundled data the decision tree is weak (accuracy 0.57 against a 0.81 majority baseline, ROC AUC 0.61).
- K-Means cluster IDs are arbitrary; interpret cluster profiles rather than the numeric labels.
- Model metrics are a single held-out split and can vary with the dataset. They are not proof of generalization or suitable for individual hiring decisions.
- Salary regression is exploratory and uses only placed students with valid salary values.
- Do not upload personally identifiable student information without authorization.

## Project structure
```text
placement_dma_dashboard/
├── app.py
├── src/
│   ├── __init__.py
│   ├── dma.py
│   └── charts.py
├── data/
│   ├── placement_students_500.csv
│   ├── apriori_transactions_500.csv
│   ├── fp_growth_transactions_500.csv
│   ├── data_dictionary.csv
│   ├── README.txt
│   ├── sample_placements.csv
│   └── placement_template.csv
├── requirements.txt
└── README.md
```
