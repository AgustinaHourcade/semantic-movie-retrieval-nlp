"""
04 — Inter-Annotator Agreement: LLM vs. Human Relevance Judgments
=================================================================

This script implements the manual-review protocol recommended for validating
LLM-generated Qrels (relevance judgments on a 0-5 scale).

Workflow
--------
1. **Sample**: Draw a stratified random sample of 50 query–document pairs from
   ``qrels_to_grade.csv``, ensuring every relevance level is represented.
   Export to ``human_review_sample.csv`` with an empty ``human_relevance``
   column for the human annotator to fill in.

2. **Agree**: Once the human column is filled, compute:
   - **Exact Agreement (%)** — proportion of rows where LLM and human
     assigned the *identical* score.
   - **Cohen's Kappa (κ)** — chance-corrected agreement for ordinal labels.

Usage
-----
    # Step 1 — generate the sample (run once)
    python 04_inter_annotator_agreement.py --sample

    # Step 2 — after filling human_relevance, compute agreement
    python 04_inter_annotator_agreement.py --evaluate
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ── Paths ────────────────────────────────────────────────────────────────────
QRELS_PATH = Path(__file__).parent / "qrels_to_grade.csv"
SAMPLE_PATH = Path(__file__).parent / "human_review_sample.csv"

SAMPLE_SIZE = 50
SEED = 42
RELEVANCE_COL = "relevance"          # LLM-assigned score (0-5)
HUMAN_COL = "human_relevance"        # Human-assigned score (0-5)
LABELS = list(range(6))              # [0, 1, 2, 3, 4, 5]


# ═══════════════════════════════════════════════════════════════════════════
# Step 1 — Extract a stratified random sample
# ═══════════════════════════════════════════════════════════════════════════

def extract_sample(
    qrels_path: Path = QRELS_PATH,
    sample_path: Path = SAMPLE_PATH,
    n: int = SAMPLE_SIZE,
    seed: int = SEED,
) -> pd.DataFrame:
    """Draw a stratified random sample of *n* rows from the qrels file.

    Stratification ensures that every relevance level present in the data
    appears in the sample (at least 1 row per level, remainder allocated
    proportionally).  The result is saved as a CSV with an empty
    ``human_relevance`` column.
    """
    df = pd.read_csv(qrels_path)
    print(f"Loaded {len(df):,} qrels from {qrels_path.name}")
    print(f"Relevance distribution:\n{df[RELEVANCE_COL].value_counts().sort_index()}\n")

    # --- Stratified sampling ------------------------------------------------
    # Guarantee ≥1 row per existing level, then fill remaining quota
    # proportionally to level frequency.
    rng = np.random.RandomState(seed)
    groups = df.groupby(RELEVANCE_COL)
    n_levels = len(groups)

    if n < n_levels:
        raise ValueError(
            f"Sample size ({n}) is smaller than the number of relevance "
            f"levels ({n_levels}). Increase SAMPLE_SIZE."
        )

    # 1 mandatory row per level
    mandatory = groups.apply(lambda g: g.sample(1, random_state=rng))
    mandatory = mandatory.droplevel(0)  # flatten multi-index

    remaining_pool = df.drop(mandatory.index)
    remaining_n = n - len(mandatory)

    # Proportional fill for the rest
    if remaining_n > 0 and len(remaining_pool) > 0:
        extra = remaining_pool.sample(
            n=min(remaining_n, len(remaining_pool)),
            random_state=rng,
        )
        sample = pd.concat([mandatory, extra]).sample(
            frac=1, random_state=rng,  # shuffle
        )
    else:
        sample = mandatory

    # --- Add empty human column and save ------------------------------------
    sample = sample.reset_index(drop=True)
    sample[HUMAN_COL] = np.nan  # annotator fills this

    sample.to_csv(sample_path, index=False)
    print(f"[OK] Saved {len(sample)} rows to {sample_path.name}")
    print(f"  Stratified distribution:\n{sample[RELEVANCE_COL].value_counts().sort_index()}\n")
    print(
        f"-> Next step: open {sample_path.name}, fill the '{HUMAN_COL}' "
        f"column (0-5), then run:\n"
        f"  python {Path(__file__).name} --evaluate"
    )
    return sample


# ═══════════════════════════════════════════════════════════════════════════
# Step 2 — Agreement metrics
# ═══════════════════════════════════════════════════════════════════════════

def exact_agreement(llm: np.ndarray, human: np.ndarray) -> float:
    """Proportion of judgments where LLM and human scores are identical.

    .. math::

        \\text{Exact Agreement} = \\frac{1}{N} \\sum_{i=1}^{N}
        \\mathbb{1}[y_i^{\\text{LLM}} = y_i^{\\text{human}}]

    Returns a value in [0, 1].
    """
    return float(np.mean(llm == human))


def cohens_kappa(llm: np.ndarray, human: np.ndarray, labels: list[int] = LABELS) -> float:
    """Cohen's Kappa (κ) for two raters on a shared set of items.

    Measures agreement beyond what would be expected by chance alone.

    .. math::

        \\kappa = \\frac{p_o - p_e}{1 - p_e}

    where:

    - :math:`p_o` (observed agreement) is the proportion of items on which
      both raters assign the *same* label — identical to Exact Agreement.

    - :math:`p_e` (expected agreement by chance) is computed from the
      marginal distributions of each rater:

      .. math::

          p_e = \\sum_{k \\in \\text{labels}}
          \\frac{n_{k}^{\\text{LLM}}}{N} \\cdot \\frac{n_{k}^{\\text{human}}}{N}

      where :math:`n_k^{\\text{LLM}}` is the count of label *k* assigned
      by the LLM, and analogously for the human.

    Interpretation (Landis & Koch, 1977):

        | κ range     | Interpretation         |
        |-------------|------------------------|
        | < 0.00      | Poor (less than chance) |
        | 0.00 – 0.20 | Slight                 |
        | 0.21 – 0.40 | Fair                   |
        | 0.41 – 0.60 | Moderate               |
        | 0.61 – 0.80 | Substantial            |
        | 0.81 – 1.00 | Almost perfect         |

    Returns a value in [-1, 1], where 1 = perfect agreement.
    """
    n = len(llm)
    if n == 0:
        raise ValueError("Cannot compute κ on an empty array.")

    # Observed agreement (p_o)
    p_o = np.mean(llm == human)

    # Expected agreement by chance (p_e)
    p_e = 0.0
    for k in labels:
        p_llm_k = np.sum(llm == k) / n
        p_human_k = np.sum(human == k) / n
        p_e += p_llm_k * p_human_k

    # Edge case: if p_e == 1 both raters always pick the same single label
    if p_e == 1.0:
        return 1.0

    kappa = (p_o - p_e) / (1.0 - p_e)
    return float(kappa)


def interpret_kappa(kappa: float) -> str:
    """Return Landis & Koch (1977) qualitative label for a κ value."""
    if kappa < 0.00:
        return "Poor (less than chance)"
    elif kappa <= 0.20:
        return "Slight"
    elif kappa <= 0.40:
        return "Fair"
    elif kappa <= 0.60:
        return "Moderate"
    elif kappa <= 0.80:
        return "Substantial"
    else:
        return "Almost perfect"


def evaluate(sample_path: Path = SAMPLE_PATH) -> None:
    """Load the annotated sample and report agreement metrics."""
    df = pd.read_csv(sample_path)

    if HUMAN_COL not in df.columns:
        print(f"ERROR: Column '{HUMAN_COL}' not found in {sample_path.name}.")
        sys.exit(1)

    missing = df[HUMAN_COL].isna().sum()
    if missing > 0:
        print(f"WARNING: {missing} rows still have empty '{HUMAN_COL}'. Dropping them.\n")
        df = df.dropna(subset=[HUMAN_COL])

    if len(df) == 0:
        print("ERROR: No rows with human annotations found. Nothing to evaluate.")
        sys.exit(1)

    llm = df[RELEVANCE_COL].values.astype(int)
    human = df[HUMAN_COL].values.astype(int)

    # ── Compute metrics ──────────────────────────────────────────────────
    ea = exact_agreement(llm, human)
    kappa = cohens_kappa(llm, human)

    # ── Report ───────────────────────────────────────────────────────────
    print("=" * 60)
    print("  INTER-ANNOTATOR AGREEMENT REPORT")
    print("  LLM relevance  vs.  Human relevance")
    print("=" * 60)
    print(f"  Annotated pairs :  {len(df)}")
    print(f"  Exact Agreement :  {ea:.2%}")
    print(f"  Cohen's Kappa   :  {kappa:.4f}  ({interpret_kappa(kappa)})")
    print("=" * 60)

    # ── Confusion matrix ─────────────────────────────────────────────────
    print("\n  Confusion Matrix (rows = LLM, cols = Human):\n")
    confusion = pd.crosstab(
        pd.Categorical(llm, categories=LABELS),
        pd.Categorical(human, categories=LABELS),
        rownames=["LLM"],
        colnames=["Human"],
        dropna=False,
    )
    print(confusion.to_string())

    # ── Disagreements ────────────────────────────────────────────────────
    disagreements = df[llm != human].copy()
    disagreements["diff"] = (disagreements[HUMAN_COL] - disagreements[RELEVANCE_COL]).astype(int)
    if len(disagreements) > 0:
        print(f"\n  Disagreements: {len(disagreements)} / {len(df)}")
        print(f"  Mean absolute diff: {disagreements['diff'].abs().mean():.2f}")
        print(f"  LLM over-rates:     {(disagreements['diff'] < 0).sum()}")
        print(f"  LLM under-rates:    {(disagreements['diff'] > 0).sum()}")
    else:
        print("\n  Perfect agreement — no disagreements found.")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Inter-Annotator Agreement: LLM vs Human Qrels"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--sample",
        action="store_true",
        help="Extract a stratified sample of 50 rows for human review.",
    )
    group.add_argument(
        "--evaluate",
        action="store_true",
        help="Compute agreement metrics on the annotated sample.",
    )
    args = parser.parse_args()

    if args.sample:
        extract_sample()
    elif args.evaluate:
        evaluate()


if __name__ == "__main__":
    main()
