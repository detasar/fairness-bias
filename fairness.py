# fairness_check.py
import pandas as pd
from aif360.metrics import ClassificationMetric
from aif360.datasets import BinaryLabelDataset

def fairness_check(input_file: str, output_file: str, label_name: str, protected_attribute_names: list[str], privileged_groups: list[dict], unprivileged_groups: list[dict], favorable_label_value: float = 1.0, unfavorable_label_value: float = 0.0)-> None:
    """
    Checks for multiple types of fairness in an input dataset and outputs a scoring table.

    Parameters:
    input_file (str): Path to the input CSV file.
    output_file (str): Path to the output CSV file where the scoring table will be saved.
    label_name (str): The name of the label column in the input dataset.
    protected_attribute_names (list[str]): A list of names of the protected attribute columns.
    privileged_groups (list[dict]): A list of dictionaries representing privileged groups.
                                     Example: [{'sex': 1}]
    unprivileged_groups (list[dict]): A list of dictionaries representing unprivileged groups.
                                       Example: [{'sex': 0}]
    favorable_label_value (float, optional): Value representing the favorable outcome in the label column.
                                            Defaults to 1.0.
    unfavorable_label_value (float, optional): Value representing the unfavorable outcome in the label column.
                                              Defaults to 0.0.

    Returns:
    None
    """
    # Read in the input dataset
    input_df = pd.read_csv(input_file)

    # --- Start Validation (Similar to bias_check.py) ---
    if label_name not in input_df.columns:
        raise ValueError(f"Label name '{label_name}' not found in input CSV columns: {input_df.columns.tolist()}")

    for attr_name in protected_attribute_names:
        if attr_name not in input_df.columns:
            raise ValueError(f"Protected attribute name '{attr_name}' not found in input CSV columns: {input_df.columns.tolist()}")

    if len(protected_attribute_names) != len(set(protected_attribute_names)):
        raise ValueError(f"Protected attribute names must be unique. Found: {protected_attribute_names}")

    label_values = input_df[label_name].unique()
    if favorable_label_value not in label_values:
        raise ValueError(f"Favorable label value '{favorable_label_value}' not found in label column '{label_name}'. Present values: {label_values}")
    if unfavorable_label_value not in label_values:
        raise ValueError(f"Unfavorable label value '{unfavorable_label_value}' not found in label column '{label_name}'. Present values: {label_values}")

    if not isinstance(privileged_groups, list) or not all(isinstance(g, dict) for g in privileged_groups):
        raise ValueError("privileged_groups must be a list of dictionaries.")
    if not privileged_groups: # Ensure not empty
        raise ValueError("privileged_groups cannot be empty.")

    if not isinstance(unprivileged_groups, list) or not all(isinstance(g, dict) for g in unprivileged_groups):
        raise ValueError("unprivileged_groups must be a list of dictionaries.")
    if not unprivileged_groups: # Ensure not empty
        raise ValueError("unprivileged_groups cannot be empty.")

    for group_list_name, group_list in [("privileged_groups", privileged_groups), ("unprivileged_groups", unprivileged_groups)]:
        for group_dict in group_list:
            if not group_dict: # Ensure dict itself is not empty
                raise ValueError(f"Empty dictionary found in {group_list_name}.")
            for key in group_dict.keys():
                if key not in protected_attribute_names:
                    raise ValueError(f"Key '{key}' in {group_list_name} definition {group_dict} is not among protected_attribute_names: {protected_attribute_names}")
    # --- End Validation ---

    try:
        data = BinaryLabelDataset(df=input_df, label_names=[label_name],
                                    protected_attribute_names=protected_attribute_names,
                                    favorable_label=favorable_label_value,
                                    unfavorable_label=unfavorable_label_value)

        # When evaluating the dataset itself (not a model's predictions on it),
        # dataset_true and dataset_pred are the same.
        classified_dataset = data
        metric = ClassificationMetric(data, classified_dataset, # dataset_true, dataset_pred
                                        unprivileged_groups=unprivileged_groups,
                                        privileged_groups=privileged_groups)

        accuracy = metric.accuracy()
        tpr = metric.true_positive_rate()
        tnr = metric.true_negative_rate()
        balanced_accuracy = (tpr + tnr) / 2
        demographic_parity_difference = metric.statistical_parity_difference()
        equal_opportunity_difference = metric.equal_opportunity_difference()

        # Add new metrics
        equalized_odds_diff = metric.equalized_odds_difference()
        fpr_diff = metric.false_positive_rate_difference()
        fnr_diff = metric.false_negative_rate_difference()

        scoring_table = {
            'Metric': [
                'Accuracy', 'Balanced Accuracy', 'Demographic Parity Difference',
                'Equal Opportunity Difference', 'Equalized Odds Difference',
                'False Positive Rate Difference', 'False Negative Rate Difference'
            ],
            'Score': [
                accuracy, balanced_accuracy, demographic_parity_difference,
                equal_opportunity_difference, equalized_odds_diff,
                fpr_diff, fnr_diff
            ]
        }
        pd.DataFrame(scoring_table).to_csv(output_file, index=False)

    except Exception as e:
        raise RuntimeError(f"AIF360 error during fairness check: {e}") from e
