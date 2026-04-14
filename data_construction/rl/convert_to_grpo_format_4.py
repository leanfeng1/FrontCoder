"""
RL Data Construction - Convert to GRPO Format

This script converts RL data to the format required by GRPO training.
Based on the paper's data format requirements.

Output format:
{
    "prompt": "User instruction/question",
    "data_source": "frontcoder_rl",
    "extra_info": {
        "question": "Original question",
        "checklist": [20 checklist items],
        "reference": "Reference HTML (optional)"
    }
}

Usage:
    python convert_to_grpo_format.py \
        --input rl_prompts_with_checklist.jsonl \
        --output rl_grpo_format.parquet
"""

import json
import pandas as pd
import argparse
from typing import List, Dict, Any
from tqdm import tqdm


def convert_record(record: Dict) -> Dict:
    """
    Convert a single record to GRPO format.

    Args:
        record: Input record with prompt and checklist

    Returns:
        GRPO-formatted record
    """
    # Get prompt text
    prompt = record.get('prompt', record.get('question', ''))

    # Get or create extra_info
    extra_info = record.get('extra_info', {})

    # Ensure checklist is present
    if 'checklist' not in extra_info:
        extra_info['checklist'] = record.get('checklist', [])

    # Add question to extra_info
    extra_info['question'] = prompt

    # Add reference if available
    if 'reference' in record:
        extra_info['reference'] = record['reference']
    if 'reference_html' in record:
        extra_info['reference'] = record['reference_html']

    return {
        'prompt': prompt,
        'data_source': 'frontcoder_rl',
        'extra_info': extra_info
    }


def main():
    parser = argparse.ArgumentParser(description="Convert to GRPO format")

    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True,
                        help="Output parquet file")
    parser.add_argument("--train_ratio", type=float, default=0.9,
                        help="Ratio for train split")

    args = parser.parse_args()

    print(f"{'='*60}")
    print("Convert to GRPO Format")
    print(f"{'='*60}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Train ratio: {args.train_ratio}")
    print(f"{'='*60}\n")

    # Load data
    print("Loading data...")
    records = []
    with open(args.input, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"Loaded {len(records):,} records")

    # Convert to GRPO format
    print("Converting to GRPO format...")
    converted = []
    for record in tqdm(records, desc="Converting"):
        converted.append(convert_record(record))
    print(f"Converted {len(converted):,} records")

    # Split train/val
    import random
    random.seed(42)
    random.shuffle(converted)

    split_idx = int(len(converted) * args.train_ratio)
    train_data = converted[:split_idx]
    val_data = converted[split_idx:]

    print(f"\nSplit: {len(train_data):,} train, {len(val_data):,} val")

    # Save as parquet
    train_output = args.output.replace('.parquet', '_train.parquet')
    val_output = args.output.replace('.parquet', '_val.parquet')

    print(f"\nSaving train data to {train_output}...")
    train_df = pd.DataFrame(train_data)
    train_df.to_parquet(train_output, index=False)

    print(f"Saving val data to {val_output}...")
    val_df = pd.DataFrame(val_data)
    val_df.to_parquet(val_output, index=False)

    print(f"\n{'='*60}")
    print("Conversion Complete!")
    print(f"  Train: {len(train_data):,} records -> {train_output}")
    print(f"  Val: {len(val_data):,} records -> {val_output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
