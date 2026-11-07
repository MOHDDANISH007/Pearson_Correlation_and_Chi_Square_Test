# Pearson Correlation and Chi-Square Test

These are two statistical methods used in Exploratory Data Analysis (EDA).

- **Pearson Correlation:** Measures the strength and direction of a linear relationship between two numerical variables.
- **Chi-Square Test:** Checks whether two categorical variables are associated with each other.

---

# 1. Pearson Correlation

## 1.1 What is Pearson Correlation?

Pearson correlation measures how strongly two numerical variables move together in a linear way.

For example, in an insurance dataset, we may want to understand the relationship between `age` and `charges`.

- If age increases and charges also tend to increase, the relationship is positive.
- If age increases and charges tend to decrease, the relationship is negative.
- If there is no clear linear pattern, correlation may be close to zero.

## 1.2 Pearson Correlation Coefficient (r)

The Pearson correlation coefficient is represented by **r**.

Its value always lies between -1 and +1.

| Correlation | Meaning |
|---|---|
| +1 | Perfect positive linear relationship |
| Close to +1 | Strong positive relationship |
| Around 0 | Little or no linear relationship |
| Close to -1 | Strong negative relationship |
| -1 | Perfect negative linear relationship |

**Important:** Correlation close to zero does not necessarily mean there is no relationship. The relationship might be non-linear.

## 1.3 Example

Suppose we have this data:

| Age | Charges |
|---:|---:|
| 20 | 2000 |
| 30 | 3000 |
| 40 | 4000 |
| 50 | 5000 |

As age increases, charges also increase.

This represents a positive correlation.

## 1.4 Pearson Correlation in Python

```python
from scipy.stats import pearsonr

r, p_value = pearsonr(df["age"], df["charges"])

print("Correlation:", r)
print("P-value:", p_value)
```

- `r`: Correlation coefficient.
- `p_value`: Helps determine whether there is statistical evidence of a non-zero population correlation.

Both variables should contain paired numerical observations. Handle missing values before calculating the test.

## 1.5 Correlation for Multiple Features

To calculate the correlation between individual features and the target:

```python
from scipy.stats import pearsonr

correlation = {
    feature: pearsonr(df[feature], df["charges"])[0]
    for feature in numerical_columns
}

print(correlation)
```

To calculate a complete correlation matrix:

```python
correlation_matrix = df.corr(numeric_only=True)
```

**Difference:**

- `pearsonr()` calculates the correlation between two selected variables.
- `df.corr()` calculates pairwise correlations among numerical columns.

## 1.6 What is the P-value in Pearson Correlation?

Pearson correlation uses two hypotheses:

- **H0 (Null Hypothesis):** There is no linear correlation in the population (`ρ = 0`).
- **H1 (Alternative Hypothesis):** There is a non-zero linear correlation (`ρ ≠ 0`).

A commonly used significance level is 0.05.

| P-value | Interpretation |
|---|---|
| Less than 0.05 | Reject H0; evidence of a non-zero linear correlation |
| Greater than or equal to 0.05 | Do not reject H0; insufficient evidence of a non-zero linear correlation |

A small p-value does not mean the relationship is strong. Check the value of `r` to understand its strength and direction.

## 1.7 When Should We Use Pearson Correlation?

Use Pearson correlation when:

- Both variables are numerical.
- You want to measure a linear relationship.
- You want to examine relationships between numerical features and a numerical target.

Examples:

- Age vs. Charges
- BMI vs. Charges
- Age vs. BMI

## 1.8 Limitations of Pearson Correlation

1. It measures linear relationships, not every possible relationship.
2. Outliers can strongly affect the result.
3. Correlation does not prove causation.
4. A low correlation does not automatically mean a feature is useless for machine learning.

---

# 2. Chi-Square Test

## 2.1 What is the Chi-Square Test?

The **Chi-Square Test of Independence** checks whether two categorical variables are associated with each other.

In simple words, it helps answer:

"Are these two categorical variables related, or could the observed difference in counts be due to random variation?"

For example, in an insurance dataset, we may want to check whether `sex` and `smoker` are associated.

Both are categorical variables:

- `sex`: Male or Female
- `smoker`: Yes or No

The test works with category counts, not the category names as numerical measurements.

## 2.2 Contingency Table

A contingency table displays the number of observations belonging to each combination of categories.

Example:

| Sex | Smoker: Yes | Smoker: No |
|---|---:|---:|
| Female | 100 | 500 |
| Male | 150 | 450 |

These are example counts only.

The Chi-Square test compares:

- **Observed frequencies:** Actual counts in the table.
- **Expected frequencies:** Counts expected if the two variables were independent.

If the observed counts differ sufficiently from the expected counts, the test may provide evidence of an association.

## 2.3 Hypotheses of the Chi-Square Test

- **H0 (Null Hypothesis):** The two categorical variables are independent. There is no association.
- **H1 (Alternative Hypothesis):** The two categorical variables are associated.

For example:

- H0: Sex and smoking status are independent.
- H1: Sex and smoking status are associated.

## 2.4 Chi-Square Test in Python

First, create a contingency table using `pd.crosstab()`.

Then perform the test using `chi2_contingency()`.

```python
import pandas as pd
from scipy.stats import chi2_contingency

# Create contingency table
table = pd.crosstab(df["sex"], df["smoker"])

# Perform Chi-Square test
chi2, p_value, dof, expected = chi2_contingency(table)

print("Contingency Table:")
print(table)

print("Chi-Square Statistic:", chi2)
print("P-value:", p_value)
print("Degrees of Freedom:", dof)
print("Expected Frequencies:")
print(expected)
```

## 2.5 Understand the Output

The function returns four values:

| Output | Meaning |
|---|---|
| `chi2` | Test statistic measuring the discrepancy between observed and expected counts |
| `p_value` | Helps determine whether there is evidence of an association |
| `dof` | Degrees of freedom |
| `expected` | Expected frequencies under the independence assumption |

### Degrees of Freedom Formula

For a contingency table:

\[
df = (r-1)(c-1)
\]

Where:

- `r` = Number of rows
- `c` = Number of columns

For a 2 × 2 table:

\[
df = (2-1)(2-1) = 1
\]

## 2.6 How to Interpret the P-value?

A common significance level is 0.05.

| P-value | Interpretation |
|---|---|
| Less than 0.05 | Reject H0; evidence of an association |
| Greater than or equal to 0.05 | Do not reject H0; insufficient evidence of an association |

**Important:** The Chi-Square test does not tell us that one category causes another. It also does not directly tell us the direction or strength of the association.

## 2.7 Assumptions of the Chi-Square Test

1. Both variables should be categorical.
2. Each observation should belong to one category in each variable.
3. Observations should be independent.
4. Expected cell counts should be sufficiently large for the usual Chi-Square approximation to be reliable.

A common guideline is that expected frequencies should generally be at least 5. If expected counts are too small, consider an exact test, such as Fisher's exact test for a 2 × 2 table.

## 2.8 When Should We Use the Chi-Square Test?

Use it when you have two categorical variables and want to check whether they are associated.

Examples from the insurance dataset:

- Sex vs. Smoker
- Region vs. Smoker
- BMI Category vs. Smoker

Do not apply this test directly to a continuous target such as `charges`. Use a suitable statistical method for a continuous outcome instead.

---

# 3. Pearson Correlation vs. Chi-Square Test

| Point | Pearson Correlation | Chi-Square Test |
|---|---|---|
| Purpose | Measures linear relationship | Tests association |
| Data type | Two numerical variables | Two categorical variables |
| Uses | Numerical values | Category counts |
| Main output | Correlation coefficient `r` and p-value | Chi-square statistic, p-value and degrees of freedom |
| Direction | Positive or negative | No positive or negative direction |
| Example | Age vs. Charges | Sex vs. Smoker |

## 3.1 Which Test Should I Choose?

| Question | Suitable Method |
|---|---|
| Is age linearly related to charges? | Pearson Correlation |
| Is BMI linearly related to charges? | Pearson Correlation |
| Are sex and smoker associated? | Chi-Square Test |
| Are region and smoker associated? | Chi-Square Test |

---

# 4. Chi-Square Test vs. Chi-Square Feature Selection

In Python, there are two commonly encountered uses of Chi-Square.

## A. `scipy.stats.chi2_contingency`

This is a statistical test of independence between two categorical variables.

```python
from scipy.stats import chi2_contingency

table = pd.crosstab(df["sex"], df["smoker"])

chi2, p_value, dof, expected = chi2_contingency(table)
```

## B. `sklearn.feature_selection.chi2`

This is used for feature selection in classification problems.

```python
from sklearn.feature_selection import chi2

scores, p_values = chi2(X, y)
```

Important points:

- Input features must be non-negative.
- The target should contain class labels.
- It is not directly suitable for a continuous regression target such as insurance `charges`.

Do not confuse these two methods.

---

# 5. Quick Revision

| Concept | Remember |
|---|---|
| Pearson Correlation | Relationship between two numerical variables |
| Correlation coefficient | Value between -1 and +1 |
| Positive correlation | Variables tend to increase together |
| Negative correlation | One tends to increase as the other decreases |
| Zero correlation | No linear relationship detected |
| Pearson p-value | Tests evidence against zero population correlation |
| Chi-Square Test | Association between two categorical variables |
| Contingency Table | Table of observed category counts |
| Expected Frequencies | Counts expected under independence |
| Chi-Square p-value | Tests evidence against independence |
| Significance level | Commonly 0.05 |
| Causation | Neither test proves causation |

**Final takeaway:**

- Use **Pearson correlation** for two numerical variables when you want to measure a linear relationship.
- Use the **Chi-Square test** for two categorical variables when you want to check whether they are associated.
- Neither test alone determines whether a feature should be included in a machine-learning model.
