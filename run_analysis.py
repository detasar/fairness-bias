# run_analysis.py
import yaml
import argparse
import os
from bias_check import bias_check
from fairness import fairness_check
import pandas as pd # Will be needed soon

def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration file: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Run bias and fairness analysis based on a config file.")
    parser.add_argument(
        '--config',
        type=str,
        default='config_template.yaml',
        help='Path to the YAML configuration file (default: config_template.yaml)'
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if not config:
        return

    print("Configuration loaded successfully.")
    # print(yaml.dump(config, indent=2)) # Optional: keep for debugging if desired

    # General parameters
    input_file = config.get('input_file')
    output_dir = config.get('output_directory', 'analysis_results')
    analysis_params = config.get('analysis_params', {})
    label_name = analysis_params.get('label_name')
    favorable_label_value = analysis_params.get('favorable_label_value', 1.0)
    unfavorable_label_value = analysis_params.get('unfavorable_label_value', 0.0)

    # Output filenames (optional from config)
    output_filenames = config.get('output_filenames', {})
    default_bias_report_name_template = "bias_metrics_{attribute_name}.csv"
    default_fairness_report_name_template = "fairness_metrics_{attribute_name}.csv"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    if not input_file or not label_name:
        print("Error: 'input_file' and 'analysis_params.label_name' must be defined in the config.")
        return

    protected_attributes_definitions = analysis_params.get('protected_attributes_definitions', [])
    if not protected_attributes_definitions:
        print("Warning: No 'protected_attributes_definitions' found in config. Nothing to analyze.")
        return

    analyses_to_run = config.get('analyses_to_run', {})
    run_bias_check = analyses_to_run.get('bias_check', False)
    run_fairness_check = analyses_to_run.get('fairness_check', False)

    for attr_def in protected_attributes_definitions:
        attr_name = attr_def.get('name')
        privileged_groups = attr_def.get('privileged_groups')
        unprivileged_groups = attr_def.get('unprivileged_groups')

        if not attr_name or not privileged_groups or not unprivileged_groups:
            print(f"Warning: Skipping attribute definition due to missing 'name', 'privileged_groups', or 'unprivileged_groups': {attr_def}")
            continue

        print(f"\nProcessing protected attribute: {attr_name}")

        if run_bias_check:
            bias_output_filename = output_filenames.get('bias_report', default_bias_report_name_template).format(attribute_name=attr_name)
            bias_output_path = os.path.join(output_dir, bias_output_filename)
            print(f"  Running bias check... Output will be saved to {bias_output_path}")
            try:
                bias_check(
                    input_file=input_file,
                    output_file=bias_output_path,
                    label_name=label_name,
                    protected_attribute_names=[attr_name], # bias_check expects a list
                    privileged_groups=privileged_groups,
                    unprivileged_groups=unprivileged_groups,
                    favorable_label_value=favorable_label_value,
                    unfavorable_label_value=unfavorable_label_value
                )
                print(f"  Bias check for {attr_name} completed.")
            except Exception as e:
                print(f"  Error during bias check for {attr_name}: {e}")

        if run_fairness_check:
            fairness_output_filename = output_filenames.get('fairness_report', default_fairness_report_name_template).format(attribute_name=attr_name)
            fairness_output_path = os.path.join(output_dir, fairness_output_filename)
            print(f"  Running fairness check... Output will be saved to {fairness_output_path}")
            try:
                fairness_check(
                    input_file=input_file,
                    output_file=fairness_output_path,
                    label_name=label_name,
                    protected_attribute_names=[attr_name], # fairness_check expects a list
                    privileged_groups=privileged_groups,
                    unprivileged_groups=unprivileged_groups,
                    favorable_label_value=favorable_label_value,
                    unfavorable_label_value=unfavorable_label_value
                )
                print(f"  Fairness check for {attr_name} completed.")
            except Exception as e:
                print(f"  Error during fairness check for {attr_name}: {e}")

    print("\nAnalysis run complete.")

if __name__ == "__main__":
    main()
