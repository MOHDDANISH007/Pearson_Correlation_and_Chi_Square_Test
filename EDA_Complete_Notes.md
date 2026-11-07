# Exploratory Data Analysis (EDA) -- Complete Notes

## 1. What is EDA?

Exploratory Data Analysis (EDA) is the process of understanding,
investigating, and analyzing a dataset before building a Machine
Learning model.

The main purpose of EDA is to understand:

-   What information is available in the dataset?
-   Is the data clean and complete?
-   How are the values distributed?
-   Are there unusual observations or outliers?
-   What relationships exist between features?
-   Do the features contain useful information for predicting the
    target?

**Why do we perform EDA?**

We perform EDA to understand the quality, structure, distribution, and
relationships within our data so that we can make appropriate decisions
before training a Machine Learning model.

EDA does not directly select the best model. It helps us understand the
data and decide how to prepare it.

------------------------------------------------------------------------

## 2. Eight Important Questions to Ask During EDA

### Step 1: Do I Understand My Data and Problem?

Before analyzing the dataset, understand the problem you are solving.

Questions to ask:

1.  What problem am I trying to solve?
2.  What is my target variable?
3.  Which columns will be my input features?
4.  What does each column represent?
5.  Is this a classification or regression problem?

**Example: House Price Prediction**

-   Input features: Area, location, bedrooms, house age.
-   Target variable: House price.

**Purpose:** Understand what information is available and what the model
is expected to predict.

Useful functions:

``` python
df.head()
df.tail()
df.shape
df.columns
df.info()
```

### Step 2: Is My Dataset Complete and Correct?

Investigate the quality of the dataset.

Questions to ask:

1.  Are there missing values?
2.  Are there duplicate records?
3.  Are there incorrect or invalid values?
4.  Are numerical columns stored in the correct data types?
5.  Are categorical columns stored correctly?

**Example:**

    Age   Salary Location
  ----- -------- ----------
     25    30000 Lucknow
    NaN    45000 Delhi
     32      NaN Mumbai

Here, Age and Salary contain missing values.

Useful functions:

``` python
df.info()
df.isnull().sum()
df.duplicated().sum()
df.dtypes
```

**Purpose:** Identify data quality problems before model training.

### Step 3: What Does the Distribution of My Data Look Like?

Understand how numerical values are distributed.

Questions to ask:

1.  What is the average value?
2.  What is the median value?
3.  How spread out are the values?
4.  Are values concentrated in a particular range?
5.  Is the distribution symmetric or skewed?
6.  Are there unusual values?

**Important statistical measures:**

-   **Mean:** Average value of the data.
-   **Median:** Middle value after sorting the data.
-   **Standard Deviation:** Measures how spread out values are around
    the mean.
-   **25th percentile (Q1):** Approximately 25% of observations are at
    or below this value.
-   **50th percentile (Q2):** Median; approximately 50% of observations
    are at or below this value.
-   **75th percentile (Q3):** Approximately 75% of observations are at
    or below this value.
-   **IQR:** Measures the spread of the middle 50% of observations.

Formula:

``` text
IQR = Q3 - Q1
```

Useful functions:

``` python
df.describe()
df.hist()
```

Useful visualizations:

-   Histogram
-   KDE plot
-   Boxplot

**Purpose:** Understand the distribution, variation, and possible
skewness of the data.

Possible decisions:

-   Investigate skewed distributions.
-   Consider feature scaling when required.
-   Consider transformations when appropriate.

### Step 4: Are There Any Outliers or Unusual Observations?

Outliers are observations that are unusually far from the majority of
the data.

Questions to ask:

1.  Are there extremely high or low values?
2.  Are these values genuine or incorrect?
3.  Are they affecting the distribution?
4.  Should I retain, correct, transform, or remove them?

Important techniques:

-   IQR Method
-   Z-score
-   Boxplot

**IQR formulas:**

``` text
IQR = Q3 - Q1

Lower Boundary = Q1 - 1.5 × IQR

Upper Boundary = Q3 + 1.5 × IQR
```

Values outside these boundaries are flagged as potential outliers.

**Purpose:** Identify unusual observations and investigate whether they
require special treatment.

**Important:** Every outlier is not an error. Do not remove outliers
without understanding the data.

### Step 5: How Are My Features Related to Each Other?

Investigate relationships between input features.

Questions to ask:

1.  Are two numerical features correlated?
2.  Are some features providing similar information?
3.  Is there a strong positive or negative relationship?
4.  Are there potentially redundant features?

Important techniques:

-   Pearson Correlation
-   Correlation Matrix
-   Heatmap
-   Pairplot
-   Scatterplot

Example:

``` python
corr = df.corr(numeric_only=True)

sns.heatmap(corr, annot=True)
```

**Purpose:** Understand relationships between features and investigate
potential redundancy.

**Important:** Correlation does not prove causation. Highly correlated
features do not always need to be removed.

### Step 6: How Are My Features Related to the Target Variable?

In supervised Machine Learning, this is particularly important.

Questions to ask:

1.  Does a feature have a relationship with the target?
2.  Which features might contain useful predictive information?
3.  Are relationships linear or nonlinear?
4.  Are categorical features associated with different target
    distributions?

**Example: House Price Prediction**

``` python
sns.scatterplot(
    data=df,
    x='area',
    y='price'
)
```

For categorical features:

``` python
sns.boxplot(
    data=df,
    x='location',
    y='price'
)
```

**Purpose:** Understand which features may help predict the target
variable.

**Important:** A feature with weak individual correlation may still be
useful when combined with other features.

### Step 7: Is My Target Variable Balanced?

This step is particularly important for classification problems.

**Example: Spam Classification**

  Class             Records
  --------------- ---------
  Normal Emails        9500
  Spam Emails           500

Here, the target variable is imbalanced because one class has
significantly more observations than the other.

Questions to ask:

1.  How many observations belong to each class?
2.  Is one class dominating the dataset?
3.  Are some classes represented by very few observations?

Useful function:

``` python
df['target'].value_counts()
```

Visualization:

``` python
sns.countplot(
    data=df,
    x='target'
)
```

**Purpose:** Identify class imbalance because a model can achieve high
accuracy while performing poorly on minority classes.

Possible approaches:

-   Class weights
-   Oversampling
-   Undersampling
-   Appropriate evaluation metrics

### Step 8: Is My Data Suitable for Training and Testing?

Before training the model, investigate whether the dataset can provide a
reliable evaluation.

Questions to ask:

1.  Do I have enough observations?
2.  Does my dataset represent the real-world problem?
3.  Is there any data leakage?
4.  Are all features available at prediction time?
5.  Should I use a random split or a time-based split?

**Purpose:** Ensure that the model is evaluated fairly and can
generalize to unseen data.

------------------------------------------------------------------------

## 3. Complete EDA Checklist

  ------------------------------------------------------------------------
                          Step Investigation         Main Purpose
  ---------------------------- --------------------- ---------------------
                             1 Understand the        Identify features and
                               problem and dataset   target

                             2 Check data quality    Find missing values,
                                                     duplicates and
                                                     incorrect types

                             3 Analyze distributions Understand spread,
                                                     skewness and
                                                     variation

                             4 Detect potential      Investigate unusual
                               outliers              observations

                             5 Analyze feature       Understand
                               relationships         correlation and
                                                     redundancy

                             6 Analyze               Investigate
                               feature-target        predictive patterns
                               relationships         

                             7 Check target balance  Identify class
                                                     imbalance

                             8 Check training        Investigate leakage
                               suitability           and
                                                     representativeness
  ------------------------------------------------------------------------

------------------------------------------------------------------------

## 4. Important Points to Remember

1.  EDA is not about executing functions without understanding their
    purpose.
2.  Every EDA technique should answer a question about the dataset.
3.  EDA helps us make informed preprocessing and modelling decisions.
4.  Outliers should not be removed blindly.
5.  Correlation does not establish causation.
6.  EDA requirements depend on the type of Machine Learning problem.
7.  Not every dataset requires every EDA technique.

------------------------------------------------------------------------

## 5. Final Machine Learning Workflow

``` text
Understand the Problem
        |
        v
Understand the Dataset
        |
        v
Exploratory Data Analysis
        |
        v
Data Cleaning
        |
        v
Data Preprocessing
        |
        v
Feature Engineering
        |
        v
Train-Test Split
        |
        v
Model Selection
        |
        v
Model Training
        |
        v
Model Evaluation
        |
        v
Model Improvement
```

## Final Takeaway

EDA helps us understand what information our dataset contains, what
problems exist within it, what patterns are present, and what decisions
we should consider before training a Machine Learning model.
