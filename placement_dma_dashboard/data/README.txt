PLACEMENT DMA DATASET — README

All records are SYNTHETIC and generated for academic demonstration.
They are not real student records or actual institutional placement statistics.

FILES
1. placement_students_500.csv
   500 student-level records for K-Means, Decision Tree and salary regression.

2. apriori_transactions_500.csv
   One transaction per student. Split the 'items' field on ', ' to create
   transactions for Apriori.

3. fp_growth_transactions_500.csv
   Same transaction data, ready for FP-Growth.

4. data_dictionary.csv
   Field descriptions and data types.

SUGGESTED USE
- Decision Tree: target = placed. Exclude student_id and salary_lpa from inputs.
- K-Means: use numeric features such as cgpa, aptitude_score,
  communication_score, projects, certifications and backlogs; standardize them.
- Random Forest Regression: filter placed == 1 and predict salary_lpa.
- Apriori / FP-Growth: convert each row's items field into a list of items.

IMPORTANT
The simulated placement and salary outcomes were generated from artificial
relationships. Do not report results as real placement statistics or use
these data for actual hiring decisions.
