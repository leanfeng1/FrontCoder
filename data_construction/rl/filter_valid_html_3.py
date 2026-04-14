#!/usr/bin/env python3
"""
RL Data Construction - Filter Valid HTML Responses

This script filters Gemini responses to keep only those containing valid ```html``` code blocks.

Pipeline:
1. Load responses from generate_html_with_gemini.py output
2. Extract ```html``` code blocks from responses
3. Validate HTML structure (basic checks)
4. Output filtered records with extracted HTML

Usage:
    python filter_valid_html.py \
        --input rl_with_html.jsonl \
        --output rl_filtered.jsonl
"""

import json
import re
import argparse
from typing import List, Dict, Tuple, Optional
from collections import Counter


def extract_html_block(response: str) -> Tuple[Optional[str], str]:
    """
    Extract HTML code block from response.

    Returns:
        Tuple of (extracted_html, extraction_status)
    """
    if not response:
        return None, "empty_response"

    # Pattern 1: ```html ... ``` (most common)
    pattern1 = r'```html\s*([\s\S]*?)```'
    matches = re.findall(pattern1, response, re.IGNORECASE)

    if matches:
        # Take the longest match (likely the main code)
        html = max(matches, key=len).strip()
        if html:
            return html, "html_block"

    # Pattern 2: ```HTML ... ``` (uppercase variant)
    pattern2 = r'```HTML\s*([\s\S]*?)```'
    matches = re.findall(pattern2, response)

    if matches:
        html = max(matches, key=len).strip()
        if html:
            return html, "HTML_block"

    # Pattern 3: ``` ... ``` with DOCTYPE (unmarked HTML block)
    pattern3 = r'```\s*(<!DOCTYPE[\s\S]*?)```'
    matches = re.findall(pattern3, response, re.IGNORECASE)

    if matches:
        html = max(matches, key=len).strip()
        if html:
            return html, "doctype_block"

    # Pattern 4: ``` ... ``` with <html> tag
    pattern4 = r'```\s*(<html[\s\S]*?</html>)[\s\S]*?```'
    matches = re.findall(pattern4, response, re.IGNORECASE)

    if matches:
        html = max(matches, key=len).strip()
        if html:
            return html, "html_tag_block"

    return None, "no_html_block"


def validate_html(html: str) -> Tuple[bool, str]:
    """
    Basic HTML validation.

    Returns:
        Tuple of (is_valid, validation_status)
    """
    if not html or len(html) < 50:
        return False, "too_short"

    html_lower = html.lower()

    # Check for essential HTML structure
    has_doctype = "<!doctype html>" in html_lower or "<!doctype" in html_lower
    has_html_tag = "<html" in html_lower
    has_head = "<head" in html_lower
    has_body = "<body" in html_lower

    # Must have at least <html> and <body>
    if not has_html_tag:
        return False, "no_html_tag"

    if not has_body:
        return False, "no_body_tag"

    # Check for closing tags
    if html_lower.count("<html") != html_lower.count("</html"):
        return False, "unbalanced_html"

    if html_lower.count("<body") != html_lower.count("</body"):
        return False, "unbalanced_body"

    # Check for placeholders/incomplete code
    placeholder_patterns = [
        r'\.\.\.',
        r'// TODO',
        r'\/\* TODO',
        r'PLACEHOLDER',
        r'\[INSERT',
        r'\[YOUR',
        r'<!-- ADD',
    ]

    for pattern in placeholder_patterns:
        if re.search(pattern, html, re.IGNORECASE):
            return False, "has_placeholder"

    # Check minimum content (should have some actual content)
    if len(html) < 200:
        return False, "too_minimal"

    return True, "valid"


def process_records(input_file: str) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Process records and filter valid HTML.

    Returns:
        Tuple of (valid_records, statistics)
    """
    valid_records = []
    stats = Counter()

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["json_error"] += 1
                continue

            stats["total"] += 1

            # Check generation success
            if not record.get("generation_success"):
                stats["generation_failed"] += 1
                continue

            response = record.get("response", "")

            # Extract HTML block
            html, extract_status = extract_html_block(response)
            stats[f"extract_{extract_status}"] += 1

            if not html:
                continue

            # Validate HTML
            is_valid, valid_status = validate_html(html)
            stats[f"valid_{valid_status}"] += 1

            if not is_valid:
                continue

            # Create filtered record
            filtered_record = {
                "index": record.get("index"),
                "question": record.get("question"),
                "response": html,  # Store extracted HTML only
                "checklist": record.get("checklist"),
                "source": record.get("source"),
                "base_task_id": record.get("base_task_id"),
                "base_task_title": record.get("base_task_title"),
                "tech_stack": record.get("tech_stack"),
                "design_style": record.get("design_style"),
                "feature_modifier": record.get("feature_modifier"),
                "extraction_method": extract_status,
                "html_length": len(html)
            }

            valid_records.append(filtered_record)
            stats["valid_total"] += 1

    return valid_records, dict(stats)


def main():
    parser = argparse.ArgumentParser(description="Filter valid HTML responses")

    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL file from generate_html_with_gemini.py")
    parser.add_argument("--output", type=str, default="rl_filtered.jsonl",
                        help="Output JSONL file with filtered records")
    parser.add_argument("--stats_file", type=str, default=None,
                        help="Optional file to save statistics")

    args = parser.parse_args()

    print("=" * 60)
    print("Filter Valid HTML Responses")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print("=" * 60)

    # Process records
    print("\nProcessing records...")
    valid_records, stats = process_records(args.input)

    # Save filtered records
    print(f"\nSaving {len(valid_records)} valid records...")
    with open(args.output, 'w', encoding='utf-8') as f:
        for record in valid_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    # Print statistics
    print("\n" + "=" * 60)
    print("Filtering Complete!")
    print("=" * 60)
    print(f"\nStatistics:")
    print(f"  Total records: {stats.get('total', 0):,}")
    print(f"  Generation failed: {stats.get('generation_failed', 0):,}")
    print(f"\n  Extraction results:")
    for key, value in sorted(stats.items()):
        if key.startswith("extract_"):
            print(f"    {key}: {value:,}")
    print(f"\n  Validation results:")
    for key, value in sorted(stats.items()):
        if key.startswith("valid_"):
            print(f"    {key}: {value:,}")
    print(f"\n  Final valid records: {stats.get('valid_total', 0):,}")
    print(f"  Success rate: {stats.get('valid_total', 0) / max(stats.get('total', 1), 1) * 100:.1f}%")
    print(f"\n  Output: {args.output}")
    print("=" * 60)

    # Save stats if requested
    if args.stats_file:
        with open(args.stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        print(f"\nStatistics saved to: {args.stats_file}")

    # Suggest next step
    print("\nNext step: Convert to GRPO format")
    print(f"  python convert_to_grpo_format.py --input {args.output} --output rl_data.parquet")


if __name__ == "__main__":
    main()
