# Sample Datasets

This document provides an overview of the sample datasets included in this repository, primarily located in the `sample_data/` directory. These datasets are derived from or inspired by the UCI Adult dataset and are provided to help users test and understand the functionalities of this bias and fairness analysis toolkit.

## 1. `sample_data/sample_data_adult_binary.csv`

*   **Description:** This dataset is a simplified and reduced version inspired by the UCI Adult dataset, tailored for binary income classification (e.g., <=50K vs >50K). It contains a mix of continuous and categorical features.
*   **Purpose:** Intended for demonstrating and testing binary classification fairness metrics and bias checks.
*   **Target Variable:** `income-label` (0 or 1)
*   **Common Protected Attributes for Examples:** `sex`, `race`

*   **Columns:**
    *   `age`: (Integer) Age of the individual.
    *   `workclass`: (String) Type of employment (e.g., `Private`, `State-gov`, `Self-emp-not-inc`).
    *   `education`: (String) Highest level of education achieved (e.g., `Bachelors`, `HS-grad`, `Masters`).
    *   `education-num`: (Integer) Numerical representation of education level (e.g., 9 for HS-grad, 13 for Bachelors).
    *   `marital-status`: (String) Marital status (e.g., `Never-married`, `Married-civ-spouse`, `Divorced`).
    *   `occupation`: (String) Occupation type (e.g., `Adm-clerical`, `Exec-managerial`, `Prof-specialty`).
    *   `relationship`: (String) Relationship status in household (e.g., `Husband`, `Wife`, `Not-in-family`, `Own-child`).
    *   `race`: (String) Race of the individual (e.g., `White`, `Black`, `Asian-Pac-Islander`, `Other`).
    *   `sex`: (String) Sex of the individual (`Male`, `Female`).
    *   `capital-gain`: (Integer) Capital gains recorded.
    *   `capital-loss`: (Integer) Capital losses recorded.
    *   `hours-per-week`: (Integer) Hours worked per week.
    *   `native-country`: (String) Country of origin (e.g., `United-States`, `Other`).
    *   `income-label`: (Integer) Target variable indicating income level. `0` typically represents "<=50K", `1` represents ">50K".

## 2. `sample_data/sample_data_adult_multiclass.csv`

*   **Description:** This dataset uses features similar to `sample_data_adult_binary.csv` but is adapted for a multi-class classification task based on education level.
*   **Purpose:** Intended for demonstrating and exploring fairness analysis in multi-class scenarios, typically using a One-vs-Rest approach with the existing tools.
*   **Target Variable:** `education-level-multiclass` (String: `HighSchool`, `Bachelors`, `Graduate`)
*   **Common Protected Attributes for Examples:** `sex`, `race`

*   **Columns:**
    *   `age`: (Integer)
    *   `workclass`: (String)
    *   `education`: (String)
    *   `education-num`: (Integer) - Used to derive the target variable.
    *   `marital-status`: (String)
    *   `occupation`: (String)
    *   `relationship`: (String)
    *   `race`: (String)
    *   `sex`: (String)
    *   `capital-gain`: (Integer)
    *   `capital-loss`: (Integer)
    *   `hours-per-week`: (Integer)
    *   `native-country`: (String)
    *   `education-level-multiclass`: (String) Target variable with three categories:
        *   `HighSchool`: Corresponds to `education-num` < 13.
        *   `Bachelors`: Corresponds to `education-num` == 13.
        *   `Graduate`: Corresponds to `education-num` > 13.

These datasets provide a starting point for users to run the analysis scripts and understand their outputs. Users are encouraged to replace these with their own datasets for meaningful analysis.

## 3. `sample_data/sample_hallbayes_prompts.csv`

*   **Description:** A small CSV listing example prompts paired with group labels. The prompts touch on gender and disability representation to encourage fairness-focused evaluation and are intended for experimenting with the `hallucination_fairness_from_csv` utility in `hallbayes_fairness.py`.
*   **Purpose:** Demonstrates how hallucination metrics from HallBayes can be gathered for different groups and later fed into bias and fairness checks.
*   **Columns:**
    *   `group`: Identifier for the group or attribute associated with the prompt.
    *   `prompt`: The text prompt to be evaluated.
