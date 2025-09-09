# Bias and Fairness Analysis Toolkit

This repository provides Python scripts to analyze datasets and AI model outputs for potential bias and fairness issues. It leverages the AIF360 toolkit.

## Overview

The toolkit currently includes two main scripts:
-   `bias_check.py`: Focuses on metrics related to dataset bias.
-   `fairness.py`: Focuses on fairness metrics typically evaluated on model predictions (or actual labels).

These tools help in identifying disparities across different demographic groups defined by protected attributes.

## Sample Datasets

This toolkit includes sample datasets to help you get started and test its features. These are located in the `sample_data/` directory.
For detailed information about these datasets, including column descriptions and their intended use for binary vs. multi-class classification examples, please refer to [DATASETS.md](./DATASETS.md).

## Prerequisites

-   Python 3.x
-   pandas
-   aif360

## Installation

1.  **Clone this repository:**
    ```bash
    git clone <repository_url>
    # Replace <repository_url> with the actual URL of this repository
    cd <repository_directory>
    ```

2.  **Install dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    pip install pandas aif360
    ```
    (If a `requirements.txt` file is provided with specific versions, you can use `pip install -r requirements.txt` instead).

## Usage

Both scripts require input data in CSV format. You need to specify the label (outcome) column, protected attribute(s), and how privileged/unprivileged groups and favorable/unfavorable outcomes are defined.

### `bias_check.py`

This script computes dataset bias metrics.

```python
from bias_check import bias_check

input_file = 'your_data.csv'       # Path to your input CSV file
output_file = 'bias_metrics.csv'   # Path to save the results
label_name = 'outcome'             # Column name for the target variable (e.g., loan_approved)
protected_attribute_names = ['sex', 'race'] # List of protected attribute column names
favorable_label_value = 1.0        # Value in 'label_name' considered favorable (e.g., 1 for approved)
unfavorable_label_value = 0.0      # Value in 'label_name' considered unfavorable (e.g., 0 for denied)

# Define privileged and unprivileged groups based on your data's encoding.
# Example: For 'sex', 1 might be privileged (e.g., male) and 0 unprivileged (e.g., female).
# For 'race', 'White' might be privileged and 'Black' unprivileged.
# These must be lists of dictionaries.
privileged_groups = [{'sex': 1}, {'race': 'White'}]
unprivileged_groups = [{'sex': 0}, {'race': 'Black'}]

bias_check(
    input_file=input_file,
    output_file=output_file,
    label_name=label_name,
    protected_attribute_names=protected_attribute_names,
    privileged_groups=privileged_groups,
    unprivileged_groups=unprivileged_groups,
    favorable_label_value=favorable_label_value,
    unfavorable_label_value=unfavorable_label_value
)

print(f"Bias metrics saved to {output_file}")
```
This will analyze `your_data.csv` and save computed bias metrics to `bias_metrics.csv`.

### `fairness.py` (using `fairness_check`)

This script computes fairness metrics, often applied to model predictions but can also analyze original labels.

```python
from fairness import fairness_check # Ensure this matches the function name in fairness.py

input_file = 'your_data_with_predictions.csv' # Your data, potentially with model predictions
output_file = 'fairness_metrics.csv'
label_name = 'actual_outcome'          # Column name for the true outcome
# If evaluating model predictions, this would be the true label.
# If evaluating dataset labels, it's the label itself.

protected_attribute_names = ['age_group'] # Example: ['age_group']
favorable_label_value = 1.0               # Favorable outcome (e.g., loan_granted = 1)
unfavorable_label_value = 0.0             # Unfavorable outcome (e.g., loan_granted = 0)

# Example: 'age_group': 'adult' is privileged, 'young' is unprivileged
privileged_groups = [{'age_group': 'adult'}]
unprivileged_groups = [{'age_group': 'young'}]


# Note: For fairness_check, the 'data' (first argument to ClassificationMetric) and
# 'classified_dataset' (second argument) are both derived from input_file.
# This means we are evaluating the fairness of the labels present in input_file.
# If you have a separate file or columns for model predictions, AIF360 allows
# comparing the original dataset with a dataset containing predictions.
# For simplicity, this example evaluates the input data directly.

fairness_check(
    input_file=input_file,
    output_file=output_file,
    label_name=label_name,
    protected_attribute_names=protected_attribute_names,
    privileged_groups=privileged_groups,
    unprivileged_groups=unprivileged_groups,
    favorable_label_value=favorable_label_value,
    unfavorable_label_value=unfavorable_label_value
)

print(f"Fairness metrics saved to {output_file}")
```
This will analyze the specified dataset and save fairness metrics to `fairness_metrics.csv`.

### `hallbayes_fairness.py`

The repository also integrates the [HallBayes](https://github.com/leochlon/hallbayes)
hallucination-risk toolkit to help assess disparities in large language model
responses. After installing HallBayes and setting your OpenAI API key, you can
run hallucination analysis across prompts associated with different groups and
then apply existing fairness checks to the resulting decisions.

```bash
pip install --upgrade openai
pip install git+https://github.com/leochlon/hallbayes.git
export OPENAI_API_KEY=sk-...
```

```python
from hallbayes_fairness import hallucination_fairness_analysis
from fairness import fairness_check

prompts = ["Who won the 2019 Nobel Prize in Physics?", "Who won the 2019 Nobel Prize in Physics?"]
groups = ["group_a", "group_b"]

metrics_df = hallucination_fairness_analysis(prompts, groups, output_file='llm_metrics.csv')

fairness_check(
    input_file='llm_metrics.csv',
    output_file='llm_fairness.csv',
    label_name='decision_answer',
    protected_attribute_names=['group'],
    privileged_groups=[{'group': 'group_a'}],
    unprivileged_groups=[{'group': 'group_b'}],
    favorable_label_value=1,
    unfavorable_label_value=0,
)
```

The resulting `llm_fairness.csv` file contains standard fairness metrics based
on the HallBayes decision outputs, enabling systematic evaluation of LLM
behavior across groups.

## Available Metrics

The following metrics are calculated by the scripts:

### From `bias_check.py`

*   **Disparate Impact**
    *   **Description:** Measures the ratio of the rate of favorable outcomes for the unprivileged group to that of the privileged group.
        `DI = P(Y=favorable | G=unprivileged) / P(Y=favorable | G=privileged)`
    *   **Interpretation:** Values close to 1.0 are preferred. Values significantly less than 1 (e.g., < 0.8) may indicate bias against the unprivileged group, while values significantly greater than 1 (e.g., > 1.25) may indicate bias against the privileged group. The "four-fifths rule" (DI < 0.8) is a common, though not definitive, threshold.

*   **Statistical Parity Difference**
    *   **Description:** Difference in the rate of favorable outcomes between unprivileged and privileged groups.
        `SPD = P(Y=favorable | G=unprivileged) - P(Y=favorable | G=privileged)`
    *   **Interpretation:** Values closer to 0 indicate better parity. Positive values indicate higher rates for the unprivileged group and negative values indicate higher rates for the privileged group.

*   **Mean Difference**
    *   **Description:** Difference in mean outcomes between unprivileged and privileged groups.
    *   **Interpretation:** Values closer to 0 suggest less disparity in average outcomes.

### From `fairness.py` (via `fairness_check`)

*   **Accuracy**
    *   **Description:** Standard classification accuracy: `(TP + TN) / (TP + TN + FP + FN)`.
    *   **Interpretation:** Overall correctness of the labels/predictions. Does not by itself indicate fairness.

*   **Balanced Accuracy**
    *   **Description:** The average of True Positive Rate (Recall/Sensitivity) and True Negative Rate (Specificity). Useful for imbalanced datasets.
        `Balanced Accuracy = (TPR + TNR) / 2`
    *   **Interpretation:** Provides a more balanced view of accuracy, especially when classes are imbalanced.

*   **Demographic Parity Difference (Statistical Parity Difference)**
    *   **Description:** The difference in the rate of favorable outcomes received by the unprivileged group and the privileged group.
        `DPD = P(Y=favorable | G=unprivileged) - P(Y=favorable | G=privileged)`
    *   **Interpretation:** Values closer to 0 indicate better demographic parity. Positive values mean the unprivileged group has a higher rate of favorable outcomes; negative values mean the privileged group does.

*   **Equal Opportunity Difference**
    *   **Description:** The difference in true positive rates (TPR) between unprivileged and privileged groups.
        `EOD = TPR_unprivileged - TPR_privileged`
    *   **Interpretation:** Measures whether individuals who should receive a favorable outcome have an equal chance of doing so, regardless of group membership. Values closer to 0 are preferred. Negative values indicate the unprivileged group has a lower TPR.

*   **Equalized Odds Difference**
    *   **Description:** Measures the average of absolute differences in False Positive Rates (FPR) and True Positive Rates (TPR) between the unprivileged and privileged groups. A value of 0 indicates perfect equality in FPR and TPR.
        `Equalized Odds Difference = 0.5 * [abs(FPR_unprivileged - FPR_privileged) + abs(TPR_unprivileged - TPR_privileged)]`
    *   **Interpretation:** Values closer to 0 are preferred. Higher values indicate disparities in error rates (FPR) and correct classification rates for positive instances (TPR) across groups.

*   **False Positive Rate Difference**
    *   **Description:** Calculates the difference in False Positive Rates (FPR) between unprivileged and privileged groups (`FPR_unprivileged - FPR_privileged`). FPR is the proportion of actual negatives incorrectly classified as positive.
    *   **Interpretation:** Values closer to 0 indicate similar FPRs across groups. Positive values mean the unprivileged group has a higher FPR; negative values mean the privileged group has a higher FPR.

*   **False Negative Rate Difference**
    *   **Description:** Calculates the difference in False Negative Rates (FNR) between unprivileged and privileged groups (`FNR_unprivileged - FNR_privileged`). FNR is the proportion of actual positives incorrectly classified as negative.
    *   **Interpretation:** Values closer to 0 indicate similar FNRs across groups. Positive values mean the unprivileged group has a higher FNR; negative values mean the privileged group has a higher FNR.

## Current Limitations

*   **Binary Classification Focus:** The tools are primarily designed for binary classification tasks where there is a clear favorable and unfavorable outcome. This is due to the use of `aif360.datasets.BinaryLabelDataset` and associated metrics.
*   **CSV Input Only:** Input data must be provided in CSV format.
*   **Group Definition:** Users must correctly define `privileged_groups` and `unprivileged_groups`. These are provided as lists of dictionaries, where each dictionary specifies a protected attribute and its value for that group (e.g., `[{'sex': 1, 'race': 'White'}]`). The values must match those in the input CSV.
*   **Favorable/Unfavorable Outcome Definition:** The meaning of "favorable" (e.g., loan approved, hired) and "unfavorable" outcomes is critical and must be explicitly defined by the user via `favorable_label_value` and `unfavorable_label_value`.
*   **Single Protected Attribute for some Metrics:** While `protected_attribute_names` can be a list, some AIF360 metrics and visualizations are often most straightforward when analyzing one protected attribute at a time or carefully constructed combined groups. The examples primarily show single attribute group definitions.

## Bias Mitigation

Understanding and addressing fairness in machine learning often involves not just measuring bias but also applying techniques to mitigate it. For a conceptual overview of different categories of bias mitigation techniques (pre-processing, in-processing, and post-processing) and examples of common algorithms, please see the [Bias Mitigation Overview](./bias_mitigation.md) document.

### Available Mitigation Functions

#### Applying Reweighing

This repository includes an implementation of the Reweighing pre-processing technique, which assigns weights to instances in the dataset to improve group fairness. The `apply_reweighing` function in `mitigation_techniques.py` can be used for this purpose.

**Function Signature:**
```python
apply_reweighing(
    input_file: str,
    output_file: str,
    label_name: str,
    protected_attribute_names: list[str],
    privileged_groups: list[dict],
    unprivileged_groups: list[dict],
    favorable_label_value: float = 1.0,
    unfavorable_label_value: float = 0.0
)
```

**Parameters:**
*   `input_file (str)`: Path to the input CSV file.
*   `output_file (str)`: Path where the reweighed data (original data + instance weights) will be saved.
*   `label_name (str)`: The name of the target variable column.
*   `protected_attribute_names (list[str])`: List of names of the protected attribute columns.
*   `privileged_groups (list[dict])`: Definitions for privileged groups (e.g., `[{'sex': 1}]`).
*   `unprivileged_groups (list[dict])`: Definitions for unprivileged groups (e.g., `[{'sex': 0}]`).
*   `favorable_label_value (float, optional)`: Value in the label column considered favorable. Defaults to `1.0`.
*   `unfavorable_label_value (float, optional)`: Value in the label column considered unfavorable. Defaults to `0.0`.

**Usage Example:**

```python
from mitigation_techniques import apply_reweighing

apply_reweighing(
    input_file='path/to/your/input.csv',
    output_file='path/to/your/reweighed_data.csv',
    label_name='your_label_column',
    protected_attribute_names=['your_protected_attribute'],
    privileged_groups=[{'your_protected_attribute': 'privileged_value'}],
    unprivileged_groups=[{'your_protected_attribute': 'unprivileged_value'}],
    favorable_label_value=1.0,
    unfavorable_label_value=0.0
)
```
This will create a new CSV file at `path/to/your/reweighed_data.csv`. This output file will contain all the original data from `input.csv` plus an additional column named `instance_weights`. These weights can then be used in training fairness-aware machine learning models or in other fairness-sensitive parts of a data processing pipeline.

#### Applying Disparate Impact Remover

This repository also includes a utility for applying the Disparate Impact Remover pre-processing technique. This algorithm modifies feature values in the dataset to reduce disparate impact related to a specified sensitive attribute. The goal is to transform features such that they become less correlated with the sensitive attribute, while trying to preserve utility for downstream tasks.

**Function Signature:**
```python
apply_disparate_impact_remover(
    input_file: str,
    output_file: str,
    protected_attribute_names: list[str],
    sensitive_attribute_name: str,
    label_name_for_dataset_init: str,
    favorable_label_for_dataset_init: float = 1.0,
    unfavorable_label_for_dataset_init: float = 0.0,
    repair_level: float = 1.0
)
```

**Parameters:**
*   `input_file (str)`: Path to the input CSV file.
*   `output_file (str)`: Path where the repaired data will be saved.
*   `protected_attribute_names (list[str])`: List of all protected attribute column names. This is used to initialize the AIF360 `BinaryLabelDataset`.
*   `sensitive_attribute_name (str)`: The specific protected attribute name that the Disparate Impact Remover should focus on for repair. This must be one of the names included in `protected_attribute_names`.
*   `label_name_for_dataset_init (str)`: The name of the label column. While Disparate Impact Remover is an unsupervised technique (it doesn't use labels for its repair logic), AIF360's `BinaryLabelDataset` structure requires a label.
*   `favorable_label_for_dataset_init (float, optional)`: Favorable outcome value for dataset initialization. Defaults to `1.0`.
*   `unfavorable_label_for_dataset_init (float, optional)`: Unfavorable outcome value for dataset initialization. Defaults to `0.0`.
*   `repair_level (float, optional)`: The level of repair to apply, ranging from 0.0 (no repair) to 1.0 (full repair). Defaults to `1.0`.

**Usage Example:**

```python
from mitigation_techniques import apply_disparate_impact_remover

apply_disparate_impact_remover(
    input_file='sample_data/sample_data_adult_binary.csv',
    output_file='path/to/your/repaired_data.csv',
    protected_attribute_names=['sex', 'race'],
    sensitive_attribute_name='sex',
    label_name_for_dataset_init='income-label',
    favorable_label_for_dataset_init=1.0,
    unfavorable_label_for_dataset_init=0.0,
    repair_level=0.8
)
```
This will create a new CSV file at `path/to/your/repaired_data.csv`. The feature values in this file (especially those correlated with the `sensitive_attribute_name`) may be altered compared to the input file, aiming to reduce disparate impact. The original label and protected attribute columns are preserved.

## Reporting Features

### HTML Analysis Report

The toolkit is designed to generate a consolidated HTML report summarizing the bias and fairness metrics for all analyzed protected attributes. This provides a convenient way to view all results in a single document.

**Configuration:**

You can control HTML report generation via the `config_template.yaml` file, within the `visualization_params` section:

```yaml
visualization_params:
  # ... other visualization params ...
  generate_charts: true # Example, may be configured separately
  charts_to_generate:
    - "Disparate Impact"
    # ... other charts ...
  chart_format: "png"
  generate_html_report: true  # Set to true to enable HTML report generation
  html_report_filename: "fairness_analysis_report.html" # Specify the desired filename
```

**Intended Content:**
The HTML report is intended to include:
*   A summary of the analysis configuration (input file, label name).
*   Timestamp of the report generation.
*   For each protected attribute analyzed:
    *   Tables for bias metrics (if `bias_check` was run).
    *   Tables for fairness metrics (if `fairness_check` was run).
    *   Embedded charts (if chart generation was enabled for specific metrics).

**Note on Current Status:** While the configuration options for HTML reporting (`generate_html_report` and `html_report_filename`) are available in `config_template.yaml` and parsed by `run_analysis.py`, the actual generation of the HTML report file is **not yet implemented** in the current version due to development tool limitations. This feature is planned for completion in a future update. When implemented, `run_analysis.py` will produce this HTML file in the specified `output_directory`.

## Unit Tests

The repository includes `test_bias_fairness.py` for unit testing the core functions. To run these tests:

```bash
python -m unittest test_bias_fairness.py
```
Ensure your environment is set up with the necessary libraries and that `sample_test_data_sex.csv` is present in the root directory.

## Contributing

Contributions are welcome! Please feel free to fork the repository, make your changes, and submit a pull request. Ensure that your changes are well-documented and include unit tests where appropriate.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details. (Assuming a LICENSE file exists or will be added).
```
*Note: The "Authors" section was removed as it's often maintained via Git history or a dedicated AUTHORS file. If "Emre Tasar" is the sole author and wishes to be credited directly in the README, that part can be re-added.*
```


