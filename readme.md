# README.md

# Bias and Fairness Checking

This repository contains two Python scripts for checking for biases and fairness in AI models.

## Dependencies
- pandas
- aif360

## Usage

### Bias Checking

```python
from bias_check import bias_check

bias_check('input.csv', 'output.csv')

from fairness import fairness
fairness('input.csv', 'output.csv')
