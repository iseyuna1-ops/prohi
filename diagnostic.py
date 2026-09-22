
import streamlit as st
import pandas as pd


from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

st.title("Diagnostic Analysis")

st.write(
    "Which vital signs most strongly predict maternal risk?"
)

# Load data
data_path = Path(__file__).parent / "Maternal Health Risk Data Set.csv"
df = pd.read_csv(data_path)

# Select physiological features
FEATURES = [
    "SystolicBP",
    "DiastolicBP",
    "BS",
    "BodyTemp",
    "HeartRate"
]

X = df[FEATURES]
y = df["RiskLevel"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train decision tree
model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

# Feature importance
importance = pd.Series(
    model.feature_importances_,
    index=FEATURES
)

st.header("Feature Importance")

st.bar_chart(
    importance.sort_values()
)

# Age-group analysis
st.header("Feature Importance by Age Group")

age_groups = {
    "Under 20": df[df["Age"] < 20],
    "20-34": df[(df["Age"] >= 20) & (df["Age"] < 35)],
    "35 and over": df[df["Age"] >= 35]
}

for group_name, group_data in age_groups.items():

    st.subheader(group_name)

    st.write("Number of records:", len(group_data))

    X_group = group_data[FEATURES]
    y_group = group_data["RiskLevel"]

    group_model = DecisionTreeClassifier(
        max_depth=4,
        random_state=42
    )


    # Cross-validation accuracy
    cv_scores = cross_val_score(
        group_model,
        X_group,
        y_group,
        cv=5,
        scoring="accuracy"
    )

    st.write(
        "Cross-validation accuracy:",
        f"{cv_scores.mean():.2%}"
    )

    group_model.fit(X_group, y_group)

    group_importance = pd.Series(
        group_model.feature_importances_,
        index=FEATURES
    )

    st.bar_chart(
        group_importance.sort_values()
    )
    
# Risk distribution by age group
st.header("Risk Distribution by Age Group")

risk_table = pd.crosstab(
    pd.cut(
        df["Age"],
        bins=[0, 19, 34, float("inf")],
        labels=["Under 20", "20-34", "35 and over"]
    ),
    df["RiskLevel"]
)

st.write("Number of records")
st.dataframe(risk_table)

st.write("Risk distribution (%)")

risk_percent = risk_table.div(
    risk_table.sum(axis=1),
    axis=0
) * 100

st.dataframe(risk_percent.round(1))

st.bar_chart(risk_percent)


# Sensitivity analysis: duplicate rows
st.header("Sensitivity Analysis: Duplicate Records")

# Use the same training data for both models
train_df = df.loc[X_train.index].copy()

# Remove exact duplicates from the training data
train_unique = train_df.drop_duplicates()

st.write("Original training records:", len(train_df))

st.write(
    "Training records after removing exact duplicates:",
    len(train_unique)
)

# Train a new model without duplicate records
unique_model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

X_unique = train_unique[FEATURES]
y_unique = train_unique["RiskLevel"]

unique_model.fit(X_unique, y_unique)

# Feature importance without duplicates
importance_unique = pd.Series(
    unique_model.feature_importances_,
    index=FEATURES
)

# Compare with the original training model
comparison = pd.DataFrame({
    "Original": importance,
    "Without duplicates": importance_unique
})

st.write("Feature importance comparison")

st.dataframe(comparison.round(3))

st.bar_chart(comparison)


# Age-group sensitivity analysis
st.header("Age-Group Sensitivity Analysis")

st.write(
    "Comparing feature importance before and after "
    "removing exact duplicate records within each age group."
)

for group_name, group_data in age_groups.items():

    st.subheader(group_name)

    # Remove exact duplicate records
    group_unique = group_data.drop_duplicates()

    st.write(
        "Original records:",
        len(group_data)
    )

    st.write(
        "Records without duplicates:",
        len(group_unique)
    )

    # Model trained on original group data
    original_model = DecisionTreeClassifier(
        max_depth=4,
        random_state=42
    )

    original_model.fit(
        group_data[FEATURES],
        group_data["RiskLevel"]
    )

    # Model trained on data without duplicates
    unique_model = DecisionTreeClassifier(
        max_depth=4,
        random_state=42
    )

    unique_model.fit(
        group_unique[FEATURES],
        group_unique["RiskLevel"]
    )

    # Compare feature importance
    original_importance = pd.Series(
        original_model.feature_importances_,
        index=FEATURES
    )

    unique_importance = pd.Series(
        unique_model.feature_importances_,
        index=FEATURES
    )

    comparison_group = pd.DataFrame({
        "Original": original_importance,
        "Without duplicates": unique_importance
    })

    st.write("Feature importance comparison")

    st.dataframe(
        comparison_group.round(3)
    )


# Group-aware cross-validation
st.header("Group-Aware Cross-Validation")

st.write(
    "Evaluating model accuracy while keeping identical "
    "feature combinations in the same validation group."
)

for group_name, group_data in age_groups.items():

    st.subheader(group_name)

    X_group = group_data[FEATURES]
    y_group = group_data["RiskLevel"]

    # Group records with identical physiological measurements
    grouping_columns = ["Age"] + FEATURES

    groups = pd.util.hash_pandas_object(
        group_data[grouping_columns],
        index=False
    )

    # Define a new decision tree
    validation_model = DecisionTreeClassifier(
        max_depth=4,
        random_state=42
    )

    # Keep identical feature combinations in the same fold
    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        validation_model,
        X_group,
        y_group,
        groups=groups,
        cv=cv,
        scoring="accuracy"
    )

    st.write(
        "Group-aware CV accuracy:",
        f"{scores.mean():.2%}"
    )

    st.write(
        "Individual fold accuracies:",
        [round(score, 3) for score in scores]
    )


# Baseline model comparison
st.header("Decision Tree vs Baseline Model")

st.write(
    "Comparing the Decision Tree with a simple model "
    "that always predicts the most frequent risk class."
)

for group_name, group_data in age_groups.items():

    st.subheader(group_name)

    X_group = group_data[FEATURES]
    y_group = group_data["RiskLevel"]

    # Group identical measurements
    grouping_columns = ["Age"] + FEATURES

    groups = pd.util.hash_pandas_object(
        group_data[grouping_columns],
        index=False
    )

    # Group-aware cross-validation
    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    tree_scores = []
    baseline_scores = []

    for train_idx, test_idx in cv.split(
        X_group, y_group, groups
    ):

        X_train_fold = X_group.iloc[train_idx]
        X_test_fold = X_group.iloc[test_idx]

        y_train_fold = y_group.iloc[train_idx]
        y_test_fold = y_group.iloc[test_idx]

        # Decision Tree
        tree = DecisionTreeClassifier(
            max_depth=4,
            random_state=42
        )

        tree.fit(X_train_fold, y_train_fold)

        tree_prediction = tree.predict(X_test_fold)

        tree_scores.append(
            accuracy_score(
                y_test_fold,
                tree_prediction
            )
        )

        # Baseline: most frequent class in training data
        most_common = y_train_fold.mode()[0]

        baseline_prediction = [
            most_common
        ] * len(y_test_fold)

        baseline_scores.append(
            accuracy_score(
                y_test_fold,
                baseline_prediction
            )
        )

    # Display results
    st.write(
        "Decision Tree accuracy:",
        f"{sum(tree_scores) / len(tree_scores):.2%}"
    )

    st.write(
        "Baseline accuracy:",
        f"{sum(baseline_scores) / len(baseline_scores):.2%}"
    )