"""Integration with HallBayes for LLM fairness analysis.

This module provides a utility function that runs the
HallBayes hallucination-risk calculator on a set of prompts
and returns a DataFrame containing the resulting metrics
along with group labels.  The output can be fed into
existing bias/fairness checks within this repository.

The HallBayes project:
https://github.com/leochlon/hallbayes
"""
from __future__ import annotations

from typing import Sequence, Optional, Dict, Any
import pandas as pd


def hallucination_fairness_analysis(
    prompts: Sequence[str],
    groups: Sequence[str],
    output_file: Optional[str] = None,
    model: str = "gpt-4o-mini",
    planner_params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Run HallBayes hallucination metrics on prompts grouped by attribute.

    Parameters
    ----------
    prompts: Sequence[str]
        The list of prompts to evaluate.
    groups: Sequence[str]
        Group label for each prompt (e.g., demographic group).
    output_file: Optional[str]
        If provided, the metrics DataFrame is written to this CSV file.
    model: str
        OpenAI model name to use with HallBayes backend.
    planner_params: Optional[Dict[str, Any]]
        Additional parameters forwarded to ``OpenAIPlanner.run`` and
        ``OpenAIItem``. Common options include ``n_samples``, ``m``,
        ``h_star`` and others described in the HallBayes documentation.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per prompt containing decision and
        hallucination risk metrics such as ``roh_bound`` and ``isr``.
    """
    if len(prompts) != len(groups):
        raise ValueError("prompts and groups must have the same length")

    try:
        from hallbayes.scripts.hallucination_toolkit import (
            OpenAIBackend,
            OpenAIItem,
            OpenAIPlanner,
        )
    except Exception as exc:  # pragma: no cover - dependency issue
        raise ImportError(
            "hallbayes is required. Install it from"
            " https://github.com/leochlon/hallbayes"  # pragma: no cover
        ) from exc

    planner_params = planner_params or {}

    backend = OpenAIBackend(model=model)
    planner = OpenAIPlanner(
        backend,
        temperature=planner_params.get("temperature", 0.3),
    )

    items = [
        OpenAIItem(
            prompt=prompt,
            n_samples=planner_params.get("n_samples", 5),
            m=planner_params.get("m", 6),
            skeleton_policy=planner_params.get("skeleton_policy", "auto"),
        )
        for prompt in prompts
    ]

    metrics = planner.run(
        items,
        h_star=planner_params.get("h_star", 0.05),
        isr_threshold=planner_params.get("isr_threshold", 1.0),
        margin_extra_bits=planner_params.get("margin_extra_bits", 0.2),
        B_clip=planner_params.get("B_clip", 12.0),
        clip_mode=planner_params.get("clip_mode", "one-sided"),
    )

    rows = []
    for grp, prompt, m in zip(groups, prompts, metrics):
        rows.append(
            {
                "group": grp,
                "prompt": prompt,
                "decision_answer": int(getattr(m, "decision_answer", False)),
                "roh_bound": getattr(m, "roh_bound", None),
                "delta_bar": getattr(m, "delta_bar", None),
                "b2t": getattr(m, "b2t", None),
                "isr": getattr(m, "isr", None),
                "q_lo": getattr(m, "q_lo", None),
                "q_bar": getattr(m, "q_bar", None),
            }
        )

    df = pd.DataFrame(rows)
    if output_file:
        df.to_csv(output_file, index=False)
    return df


def hallucination_fairness_from_csv(
    input_file: str,
    group_col: str = "group",
    prompt_col: str = "prompt",
    output_file: Optional[str] = None,
    model: str = "gpt-4o-mini",
    planner_params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Load prompts and groups from a CSV and run hallucination analysis.

    Parameters
    ----------
    input_file: str
        Path to a CSV file containing prompt text and associated group labels.
    group_col: str
        Name of the column holding group identifiers.
    prompt_col: str
        Name of the column holding prompts.
    output_file: Optional[str]
        If provided, the resulting metrics are written to this CSV file.
    model: str
        OpenAI model name to use with HallBayes backend.
    planner_params: Optional[Dict[str, Any]]
        Additional parameters forwarded to ``hallucination_fairness_analysis``.

    Returns
    -------
    pd.DataFrame
        DataFrame with hallucination metrics for each prompt in ``input_file``.
    """
    df = pd.read_csv(input_file)
    if group_col not in df.columns:
        raise ValueError(f"Column '{group_col}' not found in {input_file}")
    if prompt_col not in df.columns:
        raise ValueError(f"Column '{prompt_col}' not found in {input_file}")

    prompts = df[prompt_col].tolist()
    groups = df[group_col].tolist()
    return hallucination_fairness_analysis(
        prompts,
        groups,
        output_file=output_file,
        model=model,
        planner_params=planner_params,
    )
