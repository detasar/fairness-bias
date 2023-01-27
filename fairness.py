# fairness_check.py
import pandas as pd
from aif360.metrics import ClassificationMetric
from aif360.datasets import BinaryLabelDataset

def fairness_check(input_file:str, output_file:str)-> None:
    """
    This function checks for multiple types of fairness in an input dataset, and outputs a scoring table for an AI model.

    Parameters:
    input_file (str): path to the input csv file
    output_file (str): path to the output csv file where the scoring table will be saved

    Returns:
    None

    """
    # Read in the input dataset
    input_df = pd.read_csv(input_file)

    # Create a BinaryLabelDataset object
    data = BinaryLabelDataset(df=input_df, label_names=['label'], protected_attribute_names=['sex'])

    # Create a metric object for the dataset
    metric = ClassificationMetric(data, data.favorable_label, data.unfavorable_label,
                                  unprivileged_groups=data
    privileged_groups=data.protected_attribute_groups[1])

    # Compute the fairness metrics
    accuracy = metric.accuracy()
    balanced_accuracy = metric.balanced_accuracy()
    demographic_parity_difference = metric.demographic_parity_difference()
    equal_opportunity_difference = metric.equal_opportunity_difference()

    # Output the fairness metrics to a scoring table
    scoring_table = {'Metric': ['Accuracy', 'Balanced Accuracy', 'Demographic Parity Difference', 'Equal Opportunity Difference'],
                     'Score': [accuracy, balanced_accuracy, demographic_parity_difference, equal_opportunity_difference]}
    pd.DataFrame(scoring_table).to_csv(output_file, index=False)
