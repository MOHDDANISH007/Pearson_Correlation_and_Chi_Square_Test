# Data Distribution, EDA, Data Cleaning & Data Processing

These notes cover how to understand the distribution of numerical data,
identify potential outliers, clean a dataset, and prepare it for machine
learning. Examples use the medical insurance dataset.

------------------------------------------------------------------------

# Part 1: Understanding Data Distribution in EDA

## 1. What is Data Distribution?

Data distribution tells us how values in a dataset are spread.

During Exploratory Data Analysis (EDA), we want to understand:

-   Where most data values are concentrated.
-   Where the middle value is located.
-   How widely the values are spread.
-   Whether unusually high or low values exist.
-   Whether the data is symmetrical or skewed.

Two useful ways to visualize data distribution are:

1.  Box plot.
2.  Bell curve (normal distribution).

## 2. Box Plot

A box plot is a visualization that helps us understand how numerical
data is distributed using quartiles, the median, and whiskers.

It helps us identify:

-   The median (middle value).
-   The spread of the middle 50% of data.
-   The spread of non-outlier observations.
-   Potential outliers.

### Five important parts

1.  **Lower whisker:** The smallest observation that is not beyond the
    lower outlier fence.
2.  **Q1 (first quartile):** The 25th percentile.
3.  **Median (Q2):** The 50th percentile, or middle value.
4.  **Q3 (third quartile):** The 75th percentile.
5.  **Upper whisker:** The largest observation that is not beyond the
    upper outlier fence.

### Box plot diagram

``` text
                  BOX PLOT

 Lower whisker     Q1       Median       Q3      Upper whisker
       |------------[==========|==========]------------|
                    <------ BOX ------->
                    Middle 50% of data
```

The box extends from Q1 to Q3. The line inside the box represents the
median.

**Important:** Whiskers do not always represent the actual minimum and
maximum. In the common 1.5 × IQR rule, they extend to the most extreme
observed values that remain within the outlier fences. Individual points
beyond the whiskers are plotted as potential outliers.

## 3. Quartiles

Quartiles divide sorted data into four parts.

-   **Q1:** 25% of observations are at or below this value.
-   **Q2:** 50% of observations are at or below this value. Q2 is the
    median.
-   **Q3:** 75% of observations are at or below this value.

Example dataset:

`10, 20, 30, 40, 50, 60, 70`

Using the median-of-halves method:

-   Q1 = 20
-   Q2 = 40
-   Q3 = 60

The median tells us where the middle is. Quartiles tell us where the
25th and 75th percentiles are.

**Note:** Different software packages and quartile calculation methods
can produce slightly different quartile values for some datasets. Use
the method specified by your tool or assignment.

## 4. IQR (Interquartile Range)

IQR measures the spread of the middle 50% of the data.

### Formula

\[ IQR = Q3 - Q1 \]

### Example 1

Dataset:

`10, 20, 30, 40, 50, 60, 70`

-   Q1 = 20
-   Median = 40
-   Q3 = 60

Therefore:

\[ IQR = 60 - 20 = 40 \]

### What does 40 mean?

It means the distance between Q1 and Q3 is 40 units. The middle 50% of
observations lie between 20 and 60.

It **does not** mean that 50% of people have a value of 40.

### Example 2: Insurance charges

Suppose insurance charges, in thousands, are:

`10, 12, 15, 18, 20, 22, 25, 30, 80`

Using the median-of-halves method:

Lower half: `10, 12, 15, 18`

Upper half: `22, 25, 30, 80`

-   Q1 = (12 + 15) / 2 = 13.5
-   Median = 20
-   Q3 = (25 + 30) / 2 = 27.5

\[ IQR = 27.5 - 13.5 = 14 \]

### Interpretation

The middle 50% of insurance charges lie between 13.5 thousand and 27.5
thousand. The gap between these two values is 14 thousand.

Remember:

-   **Median:** Tells us where the middle value is.
-   **IQR:** Tells us how spread out the middle 50% of observations are.

A smaller IQR means the middle 50% is more closely packed. A larger IQR
means it is more spread out. Whether an IQR is small or large should be
judged in context or by comparing it with another feature or dataset.

## 5. Identifying Potential Outliers Using IQR

The IQR method uses lower and upper fences to flag potential outliers.

### Formulas

\[ Lower Fence = Q1 - 1.5 `\times `{=tex}IQR \]

\[ Upper Fence = Q3 + 1.5 `\times `{=tex}IQR \]

Using the insurance example:

-   Q1 = 13.5
-   Q3 = 27.5
-   IQR = 14

Lower fence:

\[ 13.5 - (1.5 `\times 14`{=tex}) = -7.5 \]

Upper fence:

\[ 27.5 + (1.5 `\times 14`{=tex}) = 48.5 \]

Therefore:

-   Values below -7.5 are potential lower outliers.
-   Values above 48.5 are potential upper outliers.

The value 80 is a potential outlier because it is greater than 48.5.

**Important:** An outlier is not automatically an error. It may be a
genuine observation that needs investigation.

## 6. Bell Curve (Normal Distribution)

A bell curve represents a normal distribution. It shows how frequently
values occur around the mean (average).

In a normal distribution:

-   Most values are concentrated around the mean.
-   Fewer values occur farther away from the mean.
-   The left and right sides are symmetrical.

### Bell curve diagram

``` text
                    Frequency
                        |
                        |          .-''''-.
                        |        .'        '.
                        |      .'            '.
                        |    .'                '.
________________________|___'__________________'________
                       Mean
```

The highest point is around the mean because values near the mean occur
most frequently in a normal distribution.

### Empirical rule (68--95--99.7 rule)

For a normal distribution:

-   Approximately 68% of values lie within 1 standard deviation of the
    mean.
-   Approximately 95% lie within 2 standard deviations.
-   Approximately 99.7% lie within 3 standard deviations.

This rule helps us understand how values are distributed around the
mean. It applies to a normal distribution, not automatically to every
dataset.

## 7. Standard Deviation (SD)

Standard deviation measures how spread out values are around the mean.

-   **Small standard deviation:** Values tend to be closer to the mean.
-   **Large standard deviation:** Values tend to be more spread out from
    the mean.

Standard deviation is used when calculating Z-scores.

## 8. Z-Score

A Z-score tells us how far an individual data value is from the mean,
measured in standard deviation units.

### Formula

\[ Z = `\frac{X - Mean}{Standard\ Deviation}`{=tex} \]

Where:

-   **X:** Individual data value.
-   **Mean:** Average of the dataset.
-   **Standard deviation:** Measure of spread around the mean.

### Example

Suppose:

-   Mean = 50
-   Standard deviation = 10
-   Individual value = 90

\[ Z = `\frac{90 - 50}{10}`{=tex} = 4 \]

### Interpretation

A Z-score of +4 means the value 90 is four standard deviations above the
mean.

Similarly:

-   Z = 0: The value is exactly at the mean.
-   Z = +2: The value is two standard deviations above the mean.
-   Z = -2: The value is two standard deviations below the mean.

### Common Z-score guidelines

  Z-score                Interpretation
  ---------------------- --------------------------
  Between -2 and +2      Often not very unusual
  Below -2 or above +2   Potentially unusual
  Below -3 or above +3   Strong potential outlier
  Below -4 or above +4   Very unusual

These are common guidelines, not universal rules. A Z-score above +3 or
below -3 does not guarantee that a value is an outlier. It is a signal
to investigate further.

The usual ±3 rule is most meaningful when the data is approximately
normally distributed. Z-scores can be affected by extreme values because
both the mean and standard deviation are sensitive to them.

## 9. Box Plot vs. Bell Curve

  -----------------------------------------------------------------------
  Box plot                            Bell curve
  ----------------------------------- -----------------------------------
  Summarizes data using quartiles and Shows a normal distribution around
  median                              the mean

  Uses Q1, Q2, and Q3                 Uses mean and standard deviation

  IQR measures the spread of the      Standard deviation measures spread
  middle 50%                          around the mean

  Can help flag potential outliers    Helps understand a normal
                                      distribution

  Useful for skewed data too          Best suited to approximately normal
                                      data
  -----------------------------------------------------------------------

### IQR vs. Z-score

  IQR method                         Z-score method
  ---------------------------------- -------------------------------------------
  Uses Q1 and Q3                     Uses mean and standard deviation
  Uses 1.5 × IQR fences              Commonly uses ±3 as a flagging threshold
  More resistant to extreme values   Sensitive to extreme values
  Useful for skewed data             Most useful for approximately normal data

## 10. Final Summary for EDA

When exploring numerical data:

1.  Use a box plot to understand the median, quartiles, spread, and
    potential outliers.
2.  Use IQR to measure the spread of the middle 50% of observations.
3.  Use a bell curve to understand a normal distribution around the
    mean.
4.  Use standard deviation to measure spread around the mean.
5.  Use Z-scores to measure how far individual values are from the mean
    in standard deviation units.

### Most important concepts

-   **Median:** Middle value.
-   **IQR:** Distance between Q1 and Q3; it represents the spread of the
    middle 50%.
-   **Mean:** Average value.
-   **Standard deviation:** Measures spread around the mean.
-   **Z-score:** Distance of an individual value from the mean in
    standard deviation units.

**Final reminder:** Never remove an outlier automatically. First
investigate whether it is a genuine observation or a data-quality issue.

------------------------------------------------------------------------

# Part 2: Data Cleaning

Data cleaning means identifying and correcting data-quality problems
before using the dataset for machine learning.

## Step 1. Handle Missing Values

Missing values mean some observations are absent from the dataset.

Example:

    age    bmi   charges
  ----- ------ ---------
     25   27.5     12000
     30    NaN     15000
    NaN   31.2     18000

### Identify missing values

``` python
df.isnull().sum()
```

Check the percentage of missing values:

``` python
(df.isnull().mean() * 100).round(2)
```

### Possible ways to handle missing values

**For numerical columns:**

-   **Mean:** May be suitable when data is reasonably symmetric and has
    no strong outliers.
-   **Median:** Often useful when data is skewed or has outliers.

Example:

``` python
df['bmi'] = df['bmi'].fillna(df['bmi'].median())
```

**For categorical columns:**

-   **Mode:** Fill with the most frequently occurring category.
-   **Unknown category:** Use a separate category such as `"Unknown"`
    when missingness itself may be meaningful.

Example:

``` python
df['smoker'] = df['smoker'].fillna(df['smoker'].mode()[0])
```

**Important:** Do not fill missing values blindly. First understand why
they are missing and choose a suitable method. For machine learning,
calculate imputation values from the training data only, then apply them
to the test data.

## Step 2. Handle Duplicate Records

Duplicate records mean the same observation appears more than once.

Check the number of duplicate rows:

``` python
df.duplicated().sum()
```

View duplicate rows:

``` python
df[df.duplicated()]
```

Remove exact duplicate rows when they are genuinely accidental:

``` python
df.drop_duplicates(inplace=True)
```

**Important:** Two people can have identical feature values and still be
two different genuine observations. Do not remove records just because
their values look similar.

## Step 3. Handle Incorrect or Invalid Values

Data can contain values that are impossible or inconsistent.

  Feature    Example of an invalid value
  ---------- -----------------------------
  age        -5
  bmi        -10
  children   -2
  smoker     "maybe"

Check categorical values:

``` python
df['age'].unique()
df['sex'].unique()
df['smoker'].unique()
df['region'].unique()
```

Check some numerical conditions:

``` python
df[
    (df['age'] < 0) |
    (df['bmi'] < 0) |
    (df['children'] < 0)
]
```

Also check whether values fall outside the valid range for each feature.

Correct invalid values using reliable source information where possible.
Do not replace them with arbitrary values.

## Step 4. Investigate Outliers

Outliers are observations that are unusually far from the rest of the
data.

Two methods discussed in these notes are the IQR method and Z-score
method.

### A. IQR method

``` python
Q1 = df['charges'].quantile(0.25)
Q3 = df['charges'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[
    (df['charges'] < lower) |
    (df['charges'] > upper)
]
```

### B. Z-score method

``` python
from scipy.stats import zscore

z_scores = zscore(df['charges'])

outliers = df[abs(z_scores) > 3]
```

### What should you do after identifying potential outliers?

-   Keep genuine observations.
-   Correct data-entry errors.
-   Remove records only when they are confirmed to be erroneous or
    unsuitable.
-   Consider transformations or robust models if extreme values affect
    modelling.

**Important for the insurance dataset:** A person with very high
insurance charges may be a genuine customer, particularly if they are a
smoker. Do not remove that observation automatically.

------------------------------------------------------------------------

# Part 3: Data Processing

Data processing converts cleaned data into a suitable format for a
machine learning algorithm.

## Step 5. Separate Features and Target

For supervised machine learning, separate the input features (X) from
the target (y).

In the insurance dataset:

**Features (X):**

-   age
-   sex
-   bmi
-   children
-   smoker
-   region

**Target (y):**

-   charges

``` python
X = df.drop('charges', axis=1)

y = df['charges']
```

### Why do we do this?

We want the model to learn the relationship between the input features
and insurance charges. The target is the value the model is trying to
predict.

## Step 6. Train-Test Split

Divide the dataset into two parts:

-   **Training data:** Used to teach the model.
-   **Testing data:** Used to evaluate how well the model performs on
    unseen data.

``` python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Here:

-   80% of the data is used for training.
-   20% of the data is used for testing.
-   `random_state=42` makes the split reproducible.

**Very important:** Split the data before learning preprocessing values.
Fit preprocessing steps such as imputation, encoding, and scaling using
training data only. Then apply the learned transformations to the test
data. This prevents data leakage.

## Step 7. Handle Categorical Variables (Encoding)

Machine learning algorithms generally require numerical inputs. The
insurance dataset contains categorical values such as `sex`, `smoker`,
and `region`.

Encoding converts categories into numerical representations.

### A. One-Hot Encoding

One-hot encoding is used for nominal categories, where categories have
no natural order.

Original data:

  region
  -----------
  southwest
  southeast
  northwest

After one-hot encoding:

    southwest   southeast   northwest
  ----------- ----------- -----------
            1           0           0
            0           1           0
            0           0           1

Python example:

``` python
import pandas as pd

encoded_data = pd.get_dummies(
    df,
    columns=['region'],
    dtype=int
)
```

For a train-test workflow, prefer fitting an encoder on training data
and applying it to test data, for example with scikit-learn's
`OneHotEncoder(handle_unknown='ignore')` inside a pipeline.

### B. Binary Mapping

For a binary feature such as `smoker`, map the two categories to 0 and
1.

``` python
df['smoker'] = df['smoker'].map({
    'yes': 1,
    'no': 0
})
```

Similarly:

``` python
df['sex'] = df['sex'].map({
    'male': 1,
    'female': 0
})
```

These numbers are category labels, not measurements of quantity.

## Step 8. Feature Scaling

Feature scaling brings numerical features onto comparable scales.

Example:

    age    bmi
  ----- ------
     25   27.5
     40   32.0
     55   35.0

The numerical ranges of age and BMI are different.

Some algorithms are sensitive to feature scales, such as:

-   KNN
-   K-Means
-   SVM
-   Linear models with regularization
-   Neural networks

Two common scaling methods are Standardization and Min-Max Scaling.

### A. Standardization (StandardScaler)

Standardization transforms features so that they have approximately mean
0 and standard deviation 1 on the training data.

Formula:

\[ Z = `\frac{X - Mean}{Standard\ Deviation}`{=tex} \]

Python:

``` python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

`fit_transform()` learns the scaling values from the training data and
transforms it. `transform()` uses those same learned values on the test
data.

### B. Min-Max Scaling (MinMaxScaler)

Min-Max Scaling usually scales values into a range between 0 and 1.

Formula:

\[ X\_{scaled} = `\frac{X - X_{min}}{X_{max} - X_{min}}`{=tex} \]

Python:

``` python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

**Important:** Apply scaling to numerical input features, not
automatically to every column. The choice of scaling method depends on
the algorithm and data. Tree-based models generally do not require
feature scaling.

## Step 9. Feature Transformation

Sometimes the distribution of a feature is highly skewed.

For example, insurance charges may have a long right tail. A log
transformation can sometimes make a highly right-skewed feature more
manageable.

``` python
import numpy as np

df['charges_log'] = np.log1p(df['charges'])
```

`log1p(x)` calculates `log(1 + x)`, which also handles zero values.

Whether to transform the target or input features depends on the
modelling objective and algorithm. If you transform the target, remember
to convert predictions back to the original scale when interpreting
insurance charges. Learn any transformation parameters from training
data only.

## Step 10. Feature Engineering

Feature engineering means creating useful new features from existing
features.

For example, in the insurance dataset:

``` python
df['age_group'] = pd.cut(
    df['age'],
    bins=[17, 30, 45, 65],
    labels=['Young', 'Middle-aged', 'Older']
)
```

This creates a new feature called `age_group`.

Other possible ideas:

-   Create BMI categories.
-   Combine useful features.
-   Extract meaningful information from dates, if available.

Only create features when there is a reasonable explanation for why they
may help the model. Feature engineering is optional; a new feature is
not automatically better.

------------------------------------------------------------------------

# Part 4: Final Preparation

## Step 11. Prepare a Reproducible Processing Pipeline

Instead of performing every transformation manually, use a pipeline.

A pipeline helps ensure that the same preprocessing operations are
applied consistently during training and prediction.

A typical workflow can include:

1.  Missing-value imputation.
2.  Categorical encoding.
3.  Numerical scaling, when needed.
4.  Model training.

Use scikit-learn's `ColumnTransformer` and `Pipeline` to combine these
operations. This also helps prevent data leakage by keeping
preprocessing inside the training workflow.

## Step 12. Final Data Validation

Before model training, check the shapes:

``` python
print(X_train.shape)
print(X_test.shape)

print(y_train.shape)
print(y_test.shape)
```

Also verify:

-   No unexpected missing values remain.
-   Invalid values have been investigated and handled.
-   Training and testing data have consistent columns.
-   The target variable is not accidentally included in X.
-   Preprocessing has not introduced data leakage.
-   The test set has not been used to fit preprocessing steps.

Then the dataset is ready for model training.

------------------------------------------------------------------------

# Part 5: Final Summary

    Order Task                  Purpose
  ------- --------------------- -------------------------------------------------
        1 Missing values        Handle absent data
        2 Duplicates            Remove accidental repeated records
        3 Invalid values        Correct data-quality issues
        4 Outliers              Investigate unusual observations
        5 Separate X and y      Identify inputs and target
        6 Train-test split      Separate learning and evaluation data
        7 Encoding              Convert categories into numerical form
        8 Scaling               Bring numerical features onto comparable scales
        9 Transformation        Adjust distributions when useful
       10 Feature engineering   Create useful new features
       11 Pipeline              Make preprocessing consistent
       12 Validation            Confirm data is ready for modelling

## Final Reminder

You do not need to apply every technique to every dataset.

For every step, ask:

1.  What am I checking?
2.  Why am I checking it?
3.  What did I discover?
4.  How will this discovery affect my machine learning model?

EDA helps you understand the data. Data cleaning fixes data-quality
problems. Data processing prepares the data for modelling.
