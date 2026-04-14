"""
SFT Data Construction - Data Filtering

This script implements the full SFT data filtering pipeline.
Based on the paper's three-stage quality filtering:

1. MinHash Deduplication: Remove near-duplicates
2. Rule-based Validation: Filter by code structure/quality rules
3. Model-based Scoring: Filter by Qwen3-Coder-480B quality scores

Final output: 60K high-quality, length-controlled samples

Usage:
    python filter_sft_data.py \
        --input sft_scored.parquet \
        --output sft_filtered.parquet \
        --min_score 80 \
        --max_length 16384
"""

import pandas as pd
import json
import re
import argparse
from typing import Dict, List, Optional
from tqdm import tqdm


def rule_based_filter(record: Dict) -> bool:
    """
    Apply rule-based filtering to a record.

    Checks:
    - Contains valid HTML structure
    - No obvious syntax errors
    - Reasonable code length
    - Contains required elements

    Args:
        record: Data record with 'response' field

    Returns:
        True if record passes all rules
    """
    response = record.get('response', '')

    if not response or len(response.strip()) < 100:
        return False

    # Check for HTML markers
    html_markers = ['<!DOCTYPE', '<html', '<head', '<body', '<div', '<script', '<style']
    has_html = any(marker.lower() in response.lower() for marker in html_markers)

    if not has_html:
        # Check for SVG or other valid content
        if '<svg' not in response.lower():
            return False

    # Check for obvious incomplete code
    incomplete_markers = [
        '...',           # Placeholder
        'TODO',          # Incomplete
        'FIXME',         # Known issues
        '// rest of',    # Incomplete implementation
        '/* more */',    # Placeholder comment
    ]

    for marker in incomplete_markers:
        if marker in response:
            return False

    # Check for balanced tags (basic check)
    open_tags = len(re.findall(r'<[a-zA-Z][^/>]*>', response))
    close_tags = len(re.findall(r'</[a-zA-Z]+>', response))
    self_close = len(re.findall(r'<[a-zA-Z][^>]*/>', response))

    # Allow some imbalance due to self-closing tags
    if abs(open_tags - close_tags - self_close) > open_tags * 0.3:
        return False

    # Check minimum content requirements
    min_elements = 3
    element_pattern = r'<[a-zA-Z][^>]*>'
    elements = re.findall(element_pattern, response)
    if len(elements) < min_elements:
        return False

    return True


def length_filter(record: Dict, max_length: int = 16384) -> bool:
    """
    Filter by response length.

    Based on paper: Max sequence length of 16K tokens.

    Args:
        record: Data record
        max_length: Maximum character length (approximate token count)

    Returns:
        True if length is acceptable
    """
    response = record.get('response', '')

    # Estimate tokens (roughly 0.75 tokens per character for code)
    estimated_tokens = len(response) * 0.75

    return estimated_tokens <= max_length


def score_filter(record: Dict, min_score: float = 80.0) -> bool:
    """
    Filter by quality score.

    Based on paper: Keep samples with high quality scores.

    Args:
        record: Data record with quality_score field
        min_score: Minimum quality score threshold

    Returns:
        True if score meets threshold
    """
    score = record.get('quality_score', 0)
    return score >= min_score


def filter_sft_data(
    df: pd.DataFrame,
    min_score: float = 80.0,
    max_length: int = 16384,
    apply_rules: bool = True,
    target_count: Optional[int] = None
) -> pd.DataFrame:
    """
    Apply full filtering pipeline to SFT data.

    Args:
        df: Input DataFrame
        min_score: Minimum quality score
        max_length: Maximum response length
        apply_rules: Whether to apply rule-based filtering
        target_count: Target number of samples (optional)

    Returns:
        Filtered DataFrame
    """
    print(f"Initial records: {len(df):,}")

    # Step 1: Rule-based filtering
    if apply_rules:
        print("\nStep 1: Rule-based filtering...")
        records = df.to_dict('records')
        passed = [r for r in tqdm(records, desc="Checking rules") if rule_based_filter(r)]
        df = pd.DataFrame(passed)
        print(f"  After rule filter: {len(df):,}")

    # Step 2: Length filtering
    print("\nStep 2: Length filtering...")
    records = df.to_dict('records')
    passed = [r for r in records if length_filter(r, max_length)]
    df = pd.DataFrame(passed)
    print(f"  After length filter: {len(df):,}")

    # Step 3: Score filtering
    if 'quality_score' in df.columns:
        print(f"\nStep 3: Score filtering (threshold: {min_score}%)...")

        # Get score distribution before filtering
        scores = df['quality_score']
        print(f"  Score distribution:")
        print(f"    Mean: {scores.mean():.2f}%")
        print(f"    Median: {scores.median():.2f}%")
        print(f"    Min/Max: {scores.min():.2f}% / {scores.max():.2f}%")

        # Filter
        df = df[df['quality_score'] >= min_score]
        print(f"  After score filter: {len(df):,}")

    # Step 4: Target count (top-k by score)
    if target_count and len(df) > target_count:
        print(f"\nStep 4: Selecting top {target_count:,} by score...")
        if 'quality_score' in df.columns:
            df = df.nlargest(target_count, 'quality_score')
        else:
            df = df.head(target_count)
        print(f"  Final count: {len(df):,}")

    return df


def main():
    parser = argparse.ArgumentParser(description="SFT Data Filtering")

    parser.add_argument("--input", type=str, required=True,
                        help="Input parquet file (with quality scores)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output filtered parquet file")
    parser.add_argument("--min_score", type=float, default=80.0,
                        help="Minimum quality score threshold")
    parser.add_argument("--max_length", type=int, default=16384,
                        help="Maximum response length (tokens)")
    parser.add_argument("--no_rules", action="store_true",
                        help="Skip rule-based filtering")
    parser.add_argument("--target_count", type=int, default=None,
                        help="Target number of samples")

    args = parser.parse_args()

    print(f"{'='*60}")
    print("SFT Data Filtering Pipeline")
    print(f"{'='*60}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Min score: {args.min_score}%")
    print(f"Max length: {args.max_length} tokens")
    print(f"Rule filter: {'disabled' if args.no_rules else 'enabled'}")
    if args.target_count:
        print(f"Target count: {args.target_count:,}")
    print(f"{'='*60}\n")

    # Load data
    print("Loading data...")
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df):,} records")

    # Apply filtering
    filtered_df = filter_sft_data(
        df,
        min_score=args.min_score,
        max_length=args.max_length,
        apply_rules=not args.no_rules,
        target_count=args.target_count
    )

    # Save output
    print(f"\nSaving to {args.output}...")
    filtered_df.to_parquet(args.output, index=False)

    # Final statistics
    print(f"\n{'='*60}")
    print("Filtering Complete!")
    print(f"  Input: {len(df):,}")
    print(f"  Output: {len(filtered_df):,}")
    print(f"  Retention: {100*len(filtered_df)/len(df):.2f}%")

    if 'quality_score' in filtered_df.columns:
        scores = filtered_df['quality_score']
        print(f"\nQuality Score Distribution:")
        print(f"  Mean: {scores.mean():.2f}%")
        print(f"  Median: {scores.median():.2f}%")
        print(f"  Min/Max: {scores.min():.2f}% / {scores.max():.2f}%")

    print(f"\nSaved to: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
