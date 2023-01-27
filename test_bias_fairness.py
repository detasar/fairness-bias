import unittest
import pandas as pd
from bias_check import bias_check
from fairness import fairness

class TestBiasCheck(unittest.TestCase):
    def test_bias_check(self):
        input_file = 'test_input.csv'
        output_file = 'test_output.csv'
        
        # Create a test input file
        test_data = {'feature1': [1, 2, 3, 4], 'feature2': [5, 6, 7, 8], 'label': [1, 0, 1, 0]}
        pd.DataFrame(test_data).to_csv(input_file, index=False)
        
        # Run the bias check function
        bias_check(input_file, output_file)
        
        # Read the output file
        output_data = pd.read_csv(output_file)
        
        # Assert that the output file has the correct columns
        self.assertEqual(list(output_data.columns), ['Metric', 'Score'])
        
        # Assert that the output file has the correct number of rows
        self.assertEqual(len(output_data), 4)
        
class TestFairnessCheck(unittest.TestCase):
    def test_fairness_check(self):
        input_file = 'test_input.csv'
        output_file = 'test_output.csv'
        
        # Create a test input file
        test_data = {'feature1': [1, 2, 3, 4], 'feature2': [5, 6, 7, 8], 'label': [1, 0, 1, 0], 'protected_attribute': ['A', 'B', 'A', 'B']}
        pd.DataFrame(test_data).to_csv(input_file, index=False)
        
        # Run the fairness check function
        fairness(input_file, output_file)
        
        # Read the output file
        output_data = pd.read_csv(output_file)
        
        # Assert that the output file has the correct columns
        self.assertEqual(list(output_data.columns), ['Metric', 'Score'])
        
        # Assert that the output file has the correct number of rows
        self.assertEqual(len(output_data), 4)

if __name__ == '__main__':
    unittest.main()
