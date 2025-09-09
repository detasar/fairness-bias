import unittest
import pandas as pd
from bias_check import bias_check
from fairness import fairness_check # Corrected import
import os # Added for potential cleanup, though not strictly used in this diff

class TestBiasCheck(unittest.TestCase):
    def test_bias_check(self):
        input_file = 'sample_test_data_sex.csv' # Use static test data
        output_file = 'test_output.csv'
        
        # Run the bias check function
        bias_check(
            input_file=input_file,
            output_file=output_file,
            label_name='outcome',
            protected_attribute_names=['sex'],
            privileged_groups=[{'sex': 1}],
            unprivileged_groups=[{'sex': 0}],
            favorable_label_value=1.0,
            unfavorable_label_value=0.0
        )
        
        # Read the output file
        output_data = pd.read_csv(output_file)
        
        # Assert that the output file has the correct columns
        self.assertEqual(list(output_data.columns), ['Metric', 'Score'])
        
        # Assert that the output file has the correct number of rows (bias_check produces 3 metrics)
        self.assertEqual(len(output_data), 3)

        # Assert specific metric: Disparate Impact
        # Expected Disparate Impact = P(Y=1|G=unprivileged) / P(Y=1|G=privileged)
        # P(Y=1|sex=0) = 3/4 = 0.75
        # P(Y=1|sex=1) = 1/4 = 0.25
        # DI = 0.75 / 0.25 = 3.0
        disparate_impact_row = output_data[output_data['Metric'] == 'Disparate Impact']
        self.assertFalse(disparate_impact_row.empty, "Disparate Impact metric not found in output.")
        self.assertAlmostEqual(disparate_impact_row['Score'].iloc[0], 3.0, places=5, msg="Disparate Impact score incorrect.")
        
class TestFairnessCheck(unittest.TestCase):
    def test_fairness_check(self):
        input_file = 'sample_test_data_sex.csv' # Use static test data
        output_file = 'test_output.csv'
        
        # Run the fairness check function
        fairness_check( # Corrected function call
            input_file=input_file,
            output_file=output_file,
            label_name='outcome',
            protected_attribute_names=['sex'],
            privileged_groups=[{'sex': 1}],
            unprivileged_groups=[{'sex': 0}],
            favorable_label_value=1.0,
            unfavorable_label_value=0.0
        )
        
        # Read the output file
        output_data = pd.read_csv(output_file)
        
        # Assert that the output file has the correct columns
        self.assertEqual(list(output_data.columns), ['Metric', 'Score'])
        
        # Assert that the output file has the correct number of rows (fairness_check produces 4 + 3 = 7 metrics)
        self.assertEqual(len(output_data), 7)

        # Assert specific metric: Demographic Parity Difference
        # Expected Demographic Parity Difference = P(Y=1|G=unprivileged) - P(Y=1|G=privileged)
        # DPD = 0.75 - 0.25 = 0.50
        demographic_parity_row = output_data[output_data['Metric'] == 'Demographic Parity Difference']
        self.assertFalse(demographic_parity_row.empty, "Demographic Parity Difference metric not found.")
        self.assertAlmostEqual(demographic_parity_row['Score'].iloc[0], 0.50, places=5, msg="Demographic Parity Difference score incorrect.")

        # Assertions for new metrics (expected to be 0.0 as actuals are used as predictions)
        eq_odds_row = output_data[output_data['Metric'] == 'Equalized Odds Difference']
        self.assertFalse(eq_odds_row.empty, "Equalized Odds Difference metric not found.")
        self.assertAlmostEqual(eq_odds_row['Score'].iloc[0], 0.0, places=5, msg="Equalized Odds Difference score incorrect.")

        fpr_diff_row = output_data[output_data['Metric'] == 'False Positive Rate Difference']
        self.assertFalse(fpr_diff_row.empty, "False Positive Rate Difference metric not found.")
        self.assertAlmostEqual(fpr_diff_row['Score'].iloc[0], 0.0, places=5, msg="False Positive Rate Difference score incorrect.")

        fnr_diff_row = output_data[output_data['Metric'] == 'False Negative Rate Difference']
        self.assertFalse(fnr_diff_row.empty, "False Negative Rate Difference metric not found.")
        self.assertAlmostEqual(fnr_diff_row['Score'].iloc[0], 0.0, places=5, msg="False Negative Rate Difference score incorrect.")

if __name__ == '__main__':
    unittest.main()

class TestBiasCheckValidation(unittest.TestCase):
    def setUp(self):
        self.sample_csv = 'sample_test_data_sex.csv'
        self.dummy_df_data = {'feature': [1,0], 'label': [1,0], 'sex': [0,1], 'race': ['W','B']}
        self.dummy_input_file = 'dummy_test_input_validation.csv'
        pd.DataFrame(self.dummy_df_data).to_csv(self.dummy_input_file, index=False)
        self.output_file = 'test_validation_output.csv'

        self.valid_params_dummy = {
            'input_file': self.dummy_input_file,
            'output_file': self.output_file,
            'label_name': 'label',
            'protected_attribute_names': ['sex'],
            'privileged_groups': [{'sex': 1}],
            'unprivileged_groups': [{'sex': 0}],
            'favorable_label_value': 1.0,
            'unfavorable_label_value': 0.0
        }
        self.valid_params_sample = {
            'input_file': self.sample_csv, # Uses 'outcome' as label, 'sex' as protected
            'output_file': self.output_file,
            'label_name': 'outcome',
            'protected_attribute_names': ['sex'],
            'privileged_groups': [{'sex': 1}],
            'unprivileged_groups': [{'sex': 0}],
            'favorable_label_value': 1.0,
            'unfavorable_label_value': 0.0
        }

    def tearDown(self):
        if os.path.exists(self.output_file):
            os.remove(self.output_file)
        if os.path.exists(self.dummy_input_file):
            os.remove(self.dummy_input_file)

    def test_invalid_label_name(self):
        params = self.valid_params_sample.copy()
        params['label_name'] = 'non_existent_label'
        with self.assertRaisesRegex(ValueError, "Label name 'non_existent_label' not found"):
            bias_check(**params)

    def test_invalid_protected_attr_name(self):
        params = self.valid_params_sample.copy()
        params['protected_attribute_names'] = ['non_existent_attr']
        with self.assertRaisesRegex(ValueError, "Protected attribute name 'non_existent_attr' not found"):
            bias_check(**params)

    def test_duplicate_protected_attr_names(self):
        params = self.valid_params_dummy.copy()
        params['protected_attribute_names'] = ['sex', 'sex']
        with self.assertRaisesRegex(ValueError, "Protected attribute names must be unique"):
            bias_check(**params)

    def test_favorable_label_not_in_data(self):
        params = self.valid_params_sample.copy()
        params['favorable_label_value'] = 3.0
        with self.assertRaisesRegex(ValueError, "Favorable label value '3.0' not found"):
            bias_check(**params)

    def test_unfavorable_label_not_in_data(self):
        params = self.valid_params_sample.copy()
        params['unfavorable_label_value'] = 4.0
        with self.assertRaisesRegex(ValueError, "Unfavorable label value '4.0' not found"):
            bias_check(**params)

    def test_invalid_privileged_groups_type_not_list(self):
        params = self.valid_params_sample.copy()
        params['privileged_groups'] = {'sex': 1}
        with self.assertRaisesRegex(ValueError, "privileged_groups must be a list of dictionaries"):
            bias_check(**params)

    def test_empty_privileged_groups_list(self):
        params = self.valid_params_sample.copy()
        params['privileged_groups'] = []
        with self.assertRaisesRegex(ValueError, "privileged_groups cannot be empty"):
            bias_check(**params)

    def test_invalid_privileged_groups_element_type_not_dict(self):
        params = self.valid_params_sample.copy()
        params['privileged_groups'] = [1]
        with self.assertRaisesRegex(ValueError, "privileged_groups must be a list of dictionaries"):
            bias_check(**params)

    def test_empty_dict_in_privileged_groups(self):
        params = self.valid_params_sample.copy()
        params['privileged_groups'] = [{}]
        with self.assertRaisesRegex(ValueError, "Empty dictionary found in privileged_groups"):
            bias_check(**params)

    def test_invalid_group_key_in_privileged_groups(self):
        params = self.valid_params_sample.copy()
        params['privileged_groups'] = [{'wrong_key': 1}]
        # 'protected_attribute_names' is ['sex'] for valid_params_sample
        with self.assertRaisesRegex(ValueError, "Key 'wrong_key' in privileged_groups definition.*not among protected_attribute_names"):
            bias_check(**params)

    def test_invalid_unprivileged_groups_type_not_list(self):
        params = self.valid_params_sample.copy()
        params['unprivileged_groups'] = {'sex': 0}
        with self.assertRaisesRegex(ValueError, "unprivileged_groups must be a list of dictionaries"):
            bias_check(**params)

    def test_empty_unprivileged_groups_list(self):
        params = self.valid_params_sample.copy()
        params['unprivileged_groups'] = []
        with self.assertRaisesRegex(ValueError, "unprivileged_groups cannot be empty"):
            bias_check(**params)

    def test_invalid_unprivileged_groups_element_type_not_dict(self):
        params = self.valid_params_sample.copy()
        params['unprivileged_groups'] = ["group"]
        with self.assertRaisesRegex(ValueError, "unprivileged_groups must be a list of dictionaries"):
            bias_check(**params)

    def test_empty_dict_in_unprivileged_groups(self):
        params = self.valid_params_sample.copy()
        params['unprivileged_groups'] = [{}]
        with self.assertRaisesRegex(ValueError, "Empty dictionary found in unprivileged_groups"):
            bias_check(**params)

    def test_invalid_group_key_in_unprivileged_groups(self):
        params = self.valid_params_sample.copy()
        params['unprivileged_groups'] = [{'wrong_key': 0}]
        with self.assertRaisesRegex(ValueError, "Key 'wrong_key' in unprivileged_groups definition.*not among protected_attribute_names"):
            bias_check(**params)

class TestFairnessCheckValidation(unittest.TestCase):
    def setUp(self):
        self.sample_csv = 'sample_test_data_sex.csv'
        self.dummy_df_data = {'feature': [1,0], 'label': [1,0], 'sex': [0,1], 'race': ['W','B']}
        self.dummy_input_file = 'dummy_test_input_validation_fc.csv' # Use a different name
        pd.DataFrame(self.dummy_df_data).to_csv(self.dummy_input_file, index=False)
        self.output_file = 'test_validation_output_fc.csv' # Use a different name

        self.valid_params_dummy = {
            'input_file': self.dummy_input_file,
            'output_file': self.output_file,
            'label_name': 'label',
            'protected_attribute_names': ['sex'],
            'privileged_groups': [{'sex': 1}],
            'unprivileged_groups': [{'sex': 0}],
            'favorable_label_value': 1.0,
            'unfavorable_label_value': 0.0
        }
        self.valid_params_sample = {
            'input_file': self.sample_csv,
            'output_file': self.output_file,
            'label_name': 'outcome',
            'protected_attribute_names': ['sex'],
            'privileged_groups': [{'sex': 1}],
            'unprivileged_groups': [{'sex': 0}],
            'favorable_label_value': 1.0,
            'unfavorable_label_value': 0.0
        }

    def tearDown(self):
        if os.path.exists(self.output_file):
            os.remove(self.output_file)
        if os.path.exists(self.dummy_input_file):
            os.remove(self.dummy_input_file)

    def test_invalid_label_name(self):
        params = self.valid_params_sample.copy()
        params['label_name'] = 'non_existent_label'
        with self.assertRaisesRegex(ValueError, "Label name 'non_existent_label' not found"):
            fairness_check(**params)

    def test_invalid_protected_attr_name(self):
        params = self.valid_params_sample.copy()
        params['protected_attribute_names'] = ['non_existent_attr']
        with self.assertRaisesRegex(ValueError, "Protected attribute name 'non_existent_attr' not found"):
            fairness_check(**params)

    def test_duplicate_protected_attr_names(self):
        params = self.valid_params_dummy.copy()
        params['protected_attribute_names'] = ['sex', 'sex']
        with self.assertRaisesRegex(ValueError, "Protected attribute names must be unique"):
            fairness_check(**params)

    def test_favorable_label_not_in_data(self):
        params = self.valid_params_sample.copy()
        params['favorable_label_value'] = 3.0
        with self.assertRaisesRegex(ValueError, "Favorable label value '3.0' not found"):
            fairness_check(**params)

    def test_unfavorable_label_not_in_data(self):
        params = self.valid_params_sample.copy()
        params['unfavorable_label_value'] = 4.0
        with self.assertRaisesRegex(ValueError, "Unfavorable label value '4.0' not found"):
            fairness_check(**params)

    def test_invalid_privileged_groups_type_not_list(self):
        params = self.valid_params_sample.copy()
        params['privileged_groups'] = {'sex': 1}
        with self.assertRaisesRegex(ValueError, "privileged_groups must be a list of dictionaries"):
            fairness_check(**params)

    def test_empty_privileged_groups_list(self):
        params = self.valid_params_sample.copy()
        params['privileged_groups'] = []
        with self.assertRaisesRegex(ValueError, "privileged_groups cannot be empty"):
            fairness_check(**params)

    def test_invalid_privileged_groups_element_type_not_dict(self):
        params = self.valid_params_sample.copy()
        params['privileged_groups'] = [1]
        with self.assertRaisesRegex(ValueError, "privileged_groups must be a list of dictionaries"):
            fairness_check(**params)

    def test_empty_dict_in_privileged_groups(self):
        params = self.valid_params_sample.copy()
        params['privileged_groups'] = [{}]
        with self.assertRaisesRegex(ValueError, "Empty dictionary found in privileged_groups"):
            fairness_check(**params)

    def test_invalid_group_key_in_privileged_groups(self):
        params = self.valid_params_sample.copy()
        params['privileged_groups'] = [{'wrong_key': 1}]
        with self.assertRaisesRegex(ValueError, "Key 'wrong_key' in privileged_groups definition.*not among protected_attribute_names"):
            fairness_check(**params)

    def test_invalid_unprivileged_groups_type_not_list(self):
        params = self.valid_params_sample.copy()
        params['unprivileged_groups'] = {'sex': 0}
        with self.assertRaisesRegex(ValueError, "unprivileged_groups must be a list of dictionaries"):
            fairness_check(**params)

    def test_empty_unprivileged_groups_list(self):
        params = self.valid_params_sample.copy()
        params['unprivileged_groups'] = []
        with self.assertRaisesRegex(ValueError, "unprivileged_groups cannot be empty"):
            fairness_check(**params)

    def test_invalid_unprivileged_groups_element_type_not_dict(self):
        params = self.valid_params_sample.copy()
        params['unprivileged_groups'] = ["group"]
        with self.assertRaisesRegex(ValueError, "unprivileged_groups must be a list of dictionaries"):
            fairness_check(**params)

    def test_empty_dict_in_unprivileged_groups(self):
        params = self.valid_params_sample.copy()
        params['unprivileged_groups'] = [{}]
        with self.assertRaisesRegex(ValueError, "Empty dictionary found in unprivileged_groups"):
            fairness_check(**params)

    def test_invalid_group_key_in_unprivileged_groups(self):
        params = self.valid_params_sample.copy()
        params['unprivileged_groups'] = [{'wrong_key': 0}]
        with self.assertRaisesRegex(ValueError, "Key 'wrong_key' in unprivileged_groups definition.*not among protected_attribute_names"):
            fairness_check(**params)

class TestMitigationTechniques(unittest.TestCase):
    def setUp(self):
        self.input_file = 'sample_test_data_sex.csv'
        self.output_file = 'test_reweighed_output.csv'
        self.label_name = 'outcome'
        self.protected_attribute_names = ['sex']
        self.unprivileged_groups = [{'sex': 0}]
        self.privileged_groups = [{'sex': 1}]
        self.favorable_label_value = 1.0
        self.unfavorable_label_value = 0.0
        # Clean up any previous output file
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def tearDown(self):
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def test_apply_reweighing_minimal(self):
        from mitigation_techniques import apply_reweighing # Import here

        apply_reweighing(
            input_file=self.input_file,
            output_file=self.output_file,
            label_name=self.label_name,
            protected_attribute_names=self.protected_attribute_names,
            privileged_groups=self.privileged_groups,
            unprivileged_groups=self.unprivileged_groups,
            favorable_label_value=self.favorable_label_value,
            unfavorable_label_value=self.unfavorable_label_value
        )
        self.assertTrue(os.path.exists(self.output_file))
        output_df = pd.read_csv(self.output_file)
        self.assertIn('instance_weights', output_df.columns)

        input_df_len = len(pd.read_csv(self.input_file))
        self.assertEqual(len(output_df), input_df_len)

        # Basic check: not all weights are 1.0
        if not output_df.empty: # Ensure dataframe is not empty before accessing all()
            self.assertFalse((output_df['instance_weights'] == 1.0).all(),
                             "All instance weights are 1.0, reweighing may not have had desired effect.")

    def test_apply_disparate_impact_remover_minimal(self):
        # This test focuses on ensuring the function runs and creates an output file.
        # It uses the new sample data.
        from mitigation_techniques import apply_disparate_impact_remover # Import here

        input_csv = 'sample_test_data_sex.csv'
        if not os.path.exists(input_csv):
            self.skipTest(f"{input_csv} not found, skipping test.")

        output_file_test = 'test_dir_repaired_minimal.csv'
        if os.path.exists(output_file_test):
            os.remove(output_file_test) # Clean up before run

        common_params = {
            'input_file': input_csv,
            'protected_attribute_names': ['sex'],
            'sensitive_attribute_name': 'sex',
            'label_name_for_dataset_init': 'outcome',
            'favorable_label_for_dataset_init': 1.0,
            'unfavorable_label_for_dataset_init': 0.0,
            'repair_level': 1.0
        }

        try:
            apply_disparate_impact_remover(**common_params, output_file=output_file_test)
            self.assertTrue(os.path.exists(output_file_test), "Output file was not created.")

            # Basic check: output file is not empty and is a valid CSV
            output_df = pd.read_csv(output_file_test)
            self.assertFalse(output_df.empty, "Output CSV file is empty.")

            input_df = pd.read_csv(input_csv)
            self.assertEqual(len(output_df), len(input_df), "Output CSV has different number of rows than input.")

        finally: # Ensure cleanup
            if os.path.exists(output_file_test):
                os.remove(output_file_test)
