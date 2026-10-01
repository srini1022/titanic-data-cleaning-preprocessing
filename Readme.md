# Titanic Data Cleaning & Preprocessing

This project prepares the Titanic passenger dataset through data exploration, missing-value handling, IQR-based outlier analysis, categorical encoding, and numerical feature scaling. It was completed as Task 1 of an AI & ML internship.

## Objective

Explore and clean the dataset, then prepare selected features for machine-learning workflows. The notebook documents the choices made at each step and visualizes the outlier rule before and after row removal.

## Dataset

`Titanic-Dataset.csv` contains passenger information including passenger class, survival, age, sex, family counts, ticket, fare, cabin, and port of embarkation. The initial dataset has 891 rows and 12 columns, with missing values in `Age`, `Cabin`, and `Embarked`.

## Tools

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Jupyter Notebook

## Preprocessing Workflow

The notebook's section labels are `1`, `2`, `5`, `3`, and `4` in execution order. Outlier analysis is placed before categorical encoding and feature scaling so it uses the original feature units.

### 1. Data Exploration

The notebook checks the dataset dimensions, previews records, lists column names, inspects data types and non-missing counts, summarizes numerical and categorical columns, counts missing values, and checks for duplicate rows. These checks describe the data before it is modified.

### 2. Missing Value Handling

- `Age`: missing values are filled with the median.
- `Embarked`: missing values are filled with the most frequent category (the mode).
- `Cabin`: the first character is extracted into a `Deck` feature; missing cabin values become `Unknown`. The original `Cabin` column is then dropped because it is highly incomplete and has many distinct labels.

### 5. Outlier Detection and Removal

The notebook inspects `Age`, `SibSp`, `Parch`, and `Fare` with boxplots and the interquartile range (IQR) fences:

```text
IQR = Q3 - Q1
Lower fence = Q1 - 1.5 * IQR
Upper fence = Q3 + 1.5 * IQR
```

It reports IQR flags for all four features and shows before-and-after boxplots. Rows are removed only when `Age` or `Fare` is outside its fence. `SibSp` and `Parch` are discrete family counts; their IQR flags are reported but do not independently remove rows, since valid family counts can be flagged as outliers (in this dataset, `Parch` has an IQR of zero).

### 3. Categorical Encoding

- `Sex` is mapped to 0 for male and 1 for female.
- `Embarked` and `Deck` are one-hot encoded with Scikit-learn's `OneHotEncoder` and `handle_unknown="ignore"`.
- `Name` and `Ticket` are not encoded because they have many distinct values. They remain text columns and need further handling before use with a numeric-only model.

### 4. Feature Scaling

`StandardScaler` is applied to `PassengerId`, `Pclass`, `Age`, `SibSp`, `Parch`, and `Fare`. `Survived` is not scaled because it is the target variable. Standardization centers each selected feature around zero and scales it by its standard deviation; it does not remove outliers or make a distribution normal.

**Modeling note:** The notebook fits the scaler on the full dataframe. For a model evaluation, split the data first, fit preprocessing only on the training data, then use the fitted transformations on validation and test data to avoid data leakage.

## Project Files

```text
Task1/
├── Readme.md
├── titanic-data-cleaning-preprocessing.ipynb
├── Titanic-Dataset.csv
└── task 1.pdf
```

## How to Run

1. Open `titanic-data-cleaning-preprocessing.ipynb` in Jupyter Notebook or VS Code.
2. Ensure `Titanic-Dataset.csv` is in the same directory as the notebook.
3. Run the notebook cells sequentially from the top in a fresh kernel.

## Key Learning Outcomes

- Data exploration and profiling
- Missing-value analysis and imputation
- Feature extraction from `Cabin`
- IQR-based outlier detection and boxplot interpretation
- Binary and one-hot categorical encoding
- Feature standardization and preprocessing considerations