# Fairness Analysis for Multi-Class Classification

## Introduction

Assessing and ensuring fairness in multi-class classification tasks presents a greater layer of complexity compared to binary classification. In binary classification, fairness is often framed around a "favorable" and an "unfavorable" outcome, and metrics are designed to measure disparities in these outcomes between different demographic groups. When more than two classes are involved, these concepts become more nuanced.

While the AIF360 toolkit primarily focuses on binary classification fairness metrics and datasets (`BinaryLabelDataset`), its foundational `StandardDataset` can handle multi-class labels. This document outlines the challenges, current approaches using AIF360, and potential future directions for analyzing fairness in multi-class scenarios within this toolkit.

## Challenges with Multi-Class Fairness

1.  **Defining Fairness:**
    *   Many established binary fairness metrics (e.g., those based on true positive/negative rates for a single favorable outcome) do not directly translate to the multi-class setting.
    *   The notion of a single "favorable" or "unfavorable" outcome becomes ambiguous. Is one class always favorable? Or does favorability depend on the context or the specific class being predicted?

2.  **Metric Ambiguity and Granularity:**
    *   Binary classification metrics like True Positive Rate (TPR) or False Positive Rate (FPR) are well-defined. In a multi-class setting, these need to be calculated on a per-class basis (e.g., TPR for Class A, TPR for Class B, etc.).
    *   Group fairness can then be assessed across all classes (e.g., comparing the average TPR across groups) or for each class treated as a separate binary problem (one-vs-rest).

## Using AIF360 for Multi-Class Problems (Current Understanding)

AIF360 can be adapted for multi-class fairness analysis, primarily through the following means:

1.  **`aif360.datasets.StandardDataset`:**
    *   Unlike `BinaryLabelDataset`, the `StandardDataset` class in AIF360 can represent datasets with multi-class labels. This is the starting point for handling multi-class data.

2.  **One-vs-Rest (OvR) Approach:**
    *   This is the most common and practical strategy for leveraging AIF360's existing binary fairness metrics for multi-class problems.
    *   **Explanation:** For each unique class in the target label, the problem is temporarily transformed into a binary classification problem: "this class" versus "all other classes."
    *   **Application:**
        *   Iterate through each unique class label.
        *   For each class, create a temporary binary version of the label column (e.g., if current class is 'A', then 'A' becomes 1 and all other classes 'B', 'C' become 0).
        *   This binarized data can then be loaded into a `BinaryLabelDataset`.
        *   The existing `fairness_check` function (which uses `aif360.metrics.ClassificationMetric`) can be applied to this `BinaryLabelDataset` to calculate fairness metrics for that specific class relative to all others.
    *   This process would be repeated for every class in the dataset.

3.  **Limitations of the OvR Approach with AIF360:**
    *   **Cumbersome Process:** Manually iterating and binarizing for each class can be tedious if not automated by a wrapper function.
    *   **Explosion of Metrics:** This approach results in a full set of fairness metrics for *each class* when treated as the positive class in an OvR scheme. Interpreting and acting upon this large number of metrics can be challenging.
    *   **No Holistic Multi-Class Metrics:** AIF360's `ClassificationMetric` is fundamentally designed for `BinaryLabelDataset`. It does not directly compute holistic multi-class fairness metrics (e.g., a single "multi-class equalized odds" or "multi-class demographic parity" across all classes simultaneously) from a `StandardDataset` with multi-class labels. The metrics obtained are for the binarized OvR problems.
    *   **Loss of Nuance:** The OvR approach simplifies the multi-class problem and may not capture all fairness nuances that arise from the interdependencies between multiple classes.

## Alternative Libraries/Approaches (Brief Mention)

It's worth noting that other fairness toolkits might offer more direct or integrated support for multi-class fairness scenarios. For example:
*   **Fairlearn:** This library has some capabilities for handling multi-class classification and provides tools for assessing and mitigating unfairness that can be more readily applied to multi-class problems.

## Scope for Future Implementation in This Toolkit

To better support multi-class fairness analysis within this toolkit, future enhancements could include:

1.  **One-vs-Rest (OvR) Wrapper Function:**
    *   Develop a wrapper function in `run_analysis.py` or a new module that automates the OvR strategy.
    *   This function would:
        *   Accept a dataset with multi-class labels.
        *   Identify all unique classes in the label column.
        *   Iterate through each unique class, temporarily binarizing the dataset (current class as favorable, all others as unfavorable).
        *   Call the existing `fairness_check` function for each binarized dataset.
        *   Collect and organize the results from each OvR analysis.

2.  **Aggregation of OvR Results:**
    *   Provide options for aggregating the metrics generated from the OvR approach (e.g., reporting the average, minimum, or maximum metric value across the per-class analyses for a given protected attribute).
    *   This could help in summarizing the potentially large number of output metrics.

3.  **Investigation of Custom Multi-Class Metrics:**
    *   Research and potentially implement specific multi-class fairness metrics from academic literature if they can be computed using AIF360's `StandardDataset` structure or its underlying data representations. This would be a more advanced undertaking.

## Conclusion

Fairness in multi-class classification is an evolving and complex field. While AIF360's core strengths lie in binary classification, the One-vs-Rest strategy provides a viable, albeit somewhat indirect, method for applying its rich set of fairness metrics to multi-class problems. Careful problem formulation, clear definitions of what fairness means in the specific multi-class context, and thoughtful interpretation of results are paramount. Future enhancements to this toolkit aim to simplify the application of such strategies.
