# mitigation_techniques.py
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing

def apply_reweighing(input_file: str, output_file: str,
                       label_name: str, protected_attribute_names: list[str],
                       privileged_groups: list[dict], unprivileged_groups: list[dict],
                       favorable_label_value: float = 1.0, unfavorable_label_value: float = 0.0) -> None:
    input_df = pd.read_csv(input_file)
    if label_name not in input_df.columns:
        raise ValueError(f"Label name '{label_name}' not found.")
    for attr in protected_attribute_names:
        if attr not in input_df.columns:
            raise ValueError(f"Protected attribute '{attr}' not found.")

    dataset = BinaryLabelDataset(df=input_df,
                                   label_names=[label_name],
                                   protected_attribute_names=protected_attribute_names,
                                   favorable_label=favorable_label_value,
                                   unfavorable_label=unfavorable_label_value)
    RW = Reweighing(unprivileged_groups=unprivileged_groups,
                      privileged_groups=privileged_groups)
    dataset_transformed = RW.fit_transform(dataset)
    output_df = input_df.copy()
    output_df['instance_weights'] = dataset_transformed.instance_weights
    output_df.to_csv(output_file, index=False)
    print(f"Reweighing applied. Output saved to {output_file}")


from aif360.algorithms.preprocessing import DisparateImpactRemover

def apply_disparate_impact_remover(input_file: str, output_file: str,
                                     protected_attribute_names: list[str], # For BinaryLabelDataset
                                     sensitive_attribute_name: str,    # For DisparateImpactRemover
                                     label_name_for_dataset_init: str,
                                     favorable_label_for_dataset_init: float = 1.0,
                                     unfavorable_label_for_dataset_init: float = 0.0,
                                     repair_level: float = 1.0) -> None:

    if sensitive_attribute_name not in protected_attribute_names:
        raise ValueError(f"Sensitive attribute '{sensitive_attribute_name}' must be in protected_attribute_names: {protected_attribute_names}")

    input_df = pd.read_csv(input_file)

    # DisparateImpactRemover needs a BinaryLabelDataset
    # The label for dataset init is used just for the AIF360 dataset structure,
    # DisparateImpactRemover itself is unsupervised w.r.t labels.
    dataset_orig = BinaryLabelDataset(
        df=input_df.copy(),
        label_names=[label_name_for_dataset_init],
        protected_attribute_names=protected_attribute_names, # All PAs for dataset structure
        favorable_label=favorable_label_for_dataset_init,
        unfavorable_label=unfavorable_label_for_dataset_init
    )

    # Initialize DisparateImpactRemover
    DIR = DisparateImpactRemover(repair_level=repair_level,
                                   sensitive_attribute=sensitive_attribute_name) # Specific PA for repair

    dataset_repaired = DIR.fit_transform(dataset_orig)

    # Convert repaired dataset back to DataFrame
    # This creates a DataFrame from the features of the repaired dataset,
    # including any modifications made by the DisparateImpactRemover.
    # The original label and protected attributes are preserved from dataset_repaired.
    df_repaired = dataset_repaired.convert_to_dataframe()[0]
    # convert_to_dataframe returns a tuple (df, label_maps, protected_attribute_maps)

    df_repaired.to_csv(output_file, index=False)
    print(f"Disparate Impact Remover applied. Output saved to {output_file}")
