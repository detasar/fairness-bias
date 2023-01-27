# bias_check.py
import pandas as pd
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.datasets import BinaryLabelDataset

def bias_check(input_file:str, output_file:str)-> None:
    """
    This function checks for multiple types of biases in an input dataset, and outputs a scoring table for an AI model.

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
    metric = BinaryLabelDatasetMetric(data, unprivileged_groups=data.protected_attribute_groups[0], privileged_groups=data.protected_attribute_groups[1])

    # Compute the bias metrics
    disparate_impact = metric.disparate_impact()
    average_odds_difference = metric.average_odds_difference()
    theil_index = metric.theil_index()

    # Output the bias metrics to a scoring table
    scoring_table = {'Metric': ['Disparate Impact', 'Average Odds Difference', 'Theil Index'],
                     'Score': [disparate_impact, average_odds_difference, theil_index]}
    pd.DataFrame(scoring_table).to_csv(output_file, index=False)
