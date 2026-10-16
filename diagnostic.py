import streamlit as st
import pandas as pd

from pathlib import Path
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests


st.title("Diagnostic Analysis")

st.write(
    "Do physiological measurements differ across low-, "
    "mid-, and high-risk groups, and which measurements "
    "show the largest group differences?"
)

# Load data
data_path = (
    Path(__file__).parent / "Maternal Health Risk Data Set.csv"
)
df = pd.read_csv(data_path)

FEATURES = [
    "SystolicBP",
    "DiastolicBP",
    "BS",
    "BodyTemp",
    "HeartRate"
]

RISK_ORDER = [
    "low risk",
    "mid risk",
    "high risk"
]


def analyse_risk_groups(data):
    results = []
    summaries = []

    for variable in FEATURES:
        samples = [
            data.loc[
                data["RiskLevel"] == risk, variable
            ].dropna()
            for risk in RISK_ORDER
        ]

        if any(len(sample) < 5 for sample in samples):
            raise ValueError(
                f"Insufficient observations for {variable}."
            )

        H, p = kruskal(*samples)

        n = sum(len(sample) for sample in samples)
        k = len(samples)

        # Kruskal–Wallis effect-size estimate:
        # eta-squared based on H = (H - k + 1) / (n - k)
        eta_squared_H = max(
            0.0, (H - k + 1) / (n - k)
        )

        results.append({
            "Variable": variable,
            "H statistic": H,
            "p-value": p,
            "Effect size (eta²_H)": eta_squared_H
        })

        for risk, sample in zip(RISK_ORDER, samples):
            summaries.append({
                "Variable": variable,
                "Risk group": risk,
                "n": len(sample),
                "Median": sample.median(),
                "Q1": sample.quantile(0.25),
                "Q3": sample.quantile(0.75)
            })

    results = pd.DataFrame(results)

    # Holm correction across the five KW tests
    results["Holm-adjusted p"] = multipletests(
        results["p-value"],
        method="holm"
    )[1]

    results["Significant after correction"] = (
        results["Holm-adjusted p"] < 0.05
    )

    results = results.sort_values(
        "Effect size (eta²_H)",
        ascending=False
    )

    return results, pd.DataFrame(summaries)


st.header("Differences Across Maternal Risk Groups")

original_tab, unique_tab = st.tabs([
    "Original records",
    "Without exact duplicates"
])

# Exact duplicates are assessed using all original columns,
# consistent with the shared data preparation.
datasets = [
    (original_tab, df),
    (unique_tab, df.drop_duplicates())
]

for tab, analysis_data in datasets:
    with tab:
        st.write(
            "Number of records:",
            len(analysis_data)
        )

        results, summaries = analyse_risk_groups(
            analysis_data
        )

        st.subheader("Kruskal–Wallis results")

        display_results = results.copy()

        display_results["H statistic"] = (
            display_results["H statistic"].round(2)
        )

        display_results["Effect size (eta²_H)"] = (
            display_results["Effect size (eta²_H)"].round(3)
        )

        for column in ["p-value", "Holm-adjusted p"]:
            display_results[column] = (
                display_results[column].map(
                    lambda value: f"{value:.3e}"
                )
            )

        st.dataframe(
            display_results,
            hide_index=True,
            use_container_width=True
        )

        st.subheader("Comparison of Effect Sizes")

        st.write(
            "Larger effect-size estimates indicate greater "
            "separation between risk groups in the ranked "
            "measurements. They do not establish clinical "
            "importance or causation."
        )

        st.bar_chart(
            results.set_index("Variable")[
                "Effect size (eta²_H)"
            ]
        )

        st.subheader("Group Medians and Quartiles")

        st.dataframe(
            summaries.round(2),
            hide_index=True,
            use_container_width=True
        )

st.caption(
    "Significant Kruskal–Wallis results indicate differences "
    "among risk groups but do not identify which pairs differ. "
    "Patient independence cannot be verified because patient "
    "IDs are unavailable. Removing exact duplicates does not "
    "guarantee independent observations."
)

# Compare original and deduplicated results
st.header("Effect Sizes Before and After Duplicate Removal")

original_results, _ = analyse_risk_groups(df)
unique_results, _ = analyse_risk_groups(df.drop_duplicates())

effect_column = "Effect size (eta²_H)"

variable_labels = {
    "BS": "Blood glucose",
    "SystolicBP": "Systolic blood pressure",
    "DiastolicBP": "Diastolic blood pressure",
    "HeartRate": "Heart rate",
    "BodyTemp": "Body temperature"
}

variable_order = [
    "BS",
    "SystolicBP",
    "DiastolicBP",
    "HeartRate",
    "BodyTemp"
]

original_label = f"Original dataset (n = {len(df):,})"
unique_label = (
    f"Without duplicates (n = {len(df.drop_duplicates()):,})"
)

comparison = pd.DataFrame({
    original_label: original_results.set_index("Variable")[
        effect_column
    ],
    unique_label: unique_results.set_index("Variable")[
        effect_column
    ]
})

comparison = comparison.loc[variable_order].rename(
    index=variable_labels
)

comparison.index.name = "Variable name"

st.subheader("Effect-Size Comparison Table")

st.dataframe(
    comparison.style.format("{:.3f}"),
    use_container_width=True
)

st.subheader("Effect-Size Comparison Chart")

# Convert to long format for grouped bars
chart_data = comparison.reset_index().melt(
    id_vars="Variable name",
    var_name="Dataset",
    value_name="Effect size"
)

import altair as alt

chart = (
    alt.Chart(chart_data)
    .mark_bar()
    .encode(
        x=alt.X(
            "Variable name:N",
            sort=list(variable_labels.values()),
            title=None,
            axis=alt.Axis(labelAngle=-25, labelLimit=0)
        ),
        xOffset=alt.XOffset(
            "Dataset:N",
            sort=[original_label, unique_label]
        ),
        y=alt.Y(
            "Effect size:Q",
            title="Effect size (η²_H)",
            scale=alt.Scale(domain=[0, 0.35])
        ),
        color=alt.Color(
            "Dataset:N",
            sort=[original_label, unique_label],
            scale=alt.Scale(range=["#4472C4", "#ED7D31"]),
            legend=alt.Legend(title=None, orient="bottom", columns=1, labelLimit=0)
        ),
        tooltip=[
            "Physiological measurement:N",
            "Dataset:N",
            alt.Tooltip("Effect size:Q", format=".3f")
        ]
    )
    .properties(height=400)
)

st.altair_chart(chart, use_container_width=True)

st.caption(
    "Larger η²_H estimates indicate greater separation "
    "between risk groups in the ranked measurements. "
    "The estimates do not establish causation or "
    "clinical importance."
)