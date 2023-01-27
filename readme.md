# README.md

# Bias and Fairness Check for AI Models

This repository contains two Python scripts, bias_check.py and fairness.py, which can be used to check for multiple types of biases and fairness issues in an input dataset and output scoring table of an AI model.

## Prerequisites
- pandas
- aif360
- Python 3.x

## Installing

### Clone this repository

```bash
git clone https://github.com/your-username/Bias-and-Fairness-Check-for-AI-Models.git

```

### Install the required libraries by running:

```python
pip install -r requirements.txt

```
##Usage
###Bias Check
```python

from bias_check import bias_check

input_file = 'input.csv'
output_file = 'output.csv'

bias_check(input_file, output_file)
```
This will check for multiple types of biases in the input dataset and will write the scores in the output dataset.

##Fairness Check
```python
Copy code
from fairness_check import fairness_check

input_file = 'input.csv'
output_file = 'output.csv'

fairness_check(input_file, output_file)
```
This will check for fairness issues in the input dataset and will write the scores in the output dataset.

###Unit Test
The repository also contains a test_bias_fairness.py file which contains unit tests for bias_check and fairness_check functions. To run these tests, use the following command:


```python -m unittest test_bias_fairness.py
```
##Additional Information
This repository is only an example and the actual implementation may vary depending on the specific use case and dataset. The code also does not cover all possible types of biases and fairness issues, and should be used as a starting point for further testing and analysis.

###Contributing
Please feel free to contribute to this repository by creating a pull request with new features and improvements.

##Authors
Emre Tasar
License
This project is licensed under the MIT License - see the LICENSE.md file for details.


