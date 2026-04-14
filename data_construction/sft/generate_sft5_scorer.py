"""
SFT Data Construction - Quality Scoring System

This script implements the 25-dimension quality scoring system for SFT data filtering.
Based on the paper: Uses Qwen3-Coder-480B for comprehensive quality assessment.

Scoring Dimensions (25 items):
- Code Quality: executability, completeness, standards compliance, engineering
- Functionality: boundary handling, data validation, interaction completeness
- User Experience: design, responsiveness, feedback mechanisms
- Response Quality: requirement understanding, solution rationality, documentation
- Technical Depth: tech selection, performance, modern features
- Innovation: novelty, UX enhancements
- Robustness: redundancy, exception handling
- Compatibility: cross-platform support
- Accessibility: WCAG compliance
- Maintainability: readability, extensibility

Each dimension uses 5-level scoring: 0, 2.5, 5, 7.5, 10 points.

Usage:
    python quality_scorer.py \
        --input sft_raw.parquet \
        --output sft_scored.parquet \
        --workers 2000
"""

import pandas as pd
import json
import os
import re
import argparse
import asyncio
from typing import Dict, List, Any, Optional
from collections import defaultdict
from tqdm import tqdm
from openai import AsyncOpenAI

# 25 Scoring Criteria based on the paper
SCORING_CRITERIA = [
    {"id": 1, "category": "Code Quality", "dimension": "Code Executability",
     "description": "Evaluate if the code can execute directly without syntax or logic errors. 10=perfect, 7.5=minor fixable issues, 5=obvious problems but runs, 2.5=severe errors, 0=non-executable.",
     "maxScore": 10, "weight": 1.2},

    {"id": 2, "category": "Code Quality", "dimension": "Core Function Completeness",
     "description": "Check if all core functions from requirements are implemented. 10=all complete, 7.5=most complete, 5=partial, 2.5=few, 0=missing.",
     "maxScore": 10, "weight": 1.5},

    {"id": 3, "category": "Code Quality", "dimension": "Code Standards",
     "description": "Evaluate naming conventions, indentation, code structure. 10=best practices, 7.5=mostly compliant, 5=partial, 2.5=poor, 0=chaotic.",
     "maxScore": 10, "weight": 1.0},

    {"id": 4, "category": "Code Quality", "dimension": "Engineering Quality",
     "description": "Assess modular design, design patterns, code reuse. 10=excellent, 7.5=good modularity, 5=average, 2.5=poor, 0=global pollution.",
     "maxScore": 10, "weight": 1.0},

    {"id": 5, "category": "Functionality", "dimension": "Boundary Handling",
     "description": "Evaluate handling of edge cases, exceptions, errors. 10=comprehensive, 7.5=main cases, 5=partial, 2.5=minimal, 0=none.",
     "maxScore": 10, "weight": 1.1},

    {"id": 6, "category": "Functionality", "dimension": "Data Validation",
     "description": "Check input validation, data verification, XSS/CSRF protection. 10=complete security, 7.5=thorough, 5=basic, 2.5=minimal, 0=none.",
     "maxScore": 10, "weight": 1.1},

    {"id": 7, "category": "Functionality", "dimension": "Interaction Completeness",
     "description": "Evaluate user interaction implementation (clicks, drags, inputs). 10=complete and smooth, 7.5=main features, 5=basic usable, 2.5=partial, 0=missing.",
     "maxScore": 10, "weight": 1.2},

    {"id": 8, "category": "UX", "dimension": "Design Professionalism",
     "description": "Assess UI/UX design quality (colors, layout, typography). 10=professional, 7.5=good, 5=basic, 2.5=crude, 0=chaotic.",
     "maxScore": 10, "weight": 1.0},

    {"id": 9, "category": "UX", "dimension": "Interaction Smoothness",
     "description": "Check animations, transitions, response speed. 10=smooth, 7.5=mostly smooth, 5=basic, 2.5=slight lag, 0=severe lag.",
     "maxScore": 10, "weight": 1.0},

    {"id": 10, "category": "UX", "dimension": "User Feedback",
     "description": "Evaluate feedback mechanisms (visual, loading states, errors). 10=comprehensive, 7.5=thorough, 5=partial, 2.5=minimal, 0=none.",
     "maxScore": 10, "weight": 1.0},

    {"id": 11, "category": "Response", "dimension": "Requirement Understanding",
     "description": "Assess if response accurately understands the question. 10=complete, 7.5=basic, 5=partial, 2.5=deviation, 0=off-target.",
     "maxScore": 10, "weight": 1.3},

    {"id": 12, "category": "Response", "dimension": "Solution Rationality",
     "description": "Check if solution is reasonable and uses best practices. 10=optimal, 7.5=good, 5=viable, 2.5=suboptimal, 0=unreasonable.",
     "maxScore": 10, "weight": 1.2},

    {"id": 13, "category": "Response", "dimension": "Comment Quality",
     "description": "Evaluate code comments (structure, logic explanation). 10=excellent, 7.5=good, 5=basic, 2.5=minimal, 0=none.",
     "maxScore": 10, "weight": 0.8},

    {"id": 14, "category": "Response", "dimension": "Documentation Completeness",
     "description": "Check for implementation notes, usage guides. 10=complete, 7.5=thorough, 5=simple, 2.5=brief, 0=none.",
     "maxScore": 10, "weight": 0.7},

    {"id": 15, "category": "Technical", "dimension": "Tech Selection",
     "description": "Evaluate appropriateness of tech stack (frameworks, libraries). 10=optimal, 7.5=good, 5=viable, 2.5=suboptimal, 0=unreasonable.",
     "maxScore": 10, "weight": 1.0},

    {"id": 16, "category": "Technical", "dimension": "Performance Optimization",
     "description": "Check algorithm complexity, render optimization, memory management. 10=comprehensive, 7.5=good, 5=partial, 2.5=minimal, 0=none.",
     "maxScore": 10, "weight": 1.1},

    {"id": 17, "category": "Technical", "dimension": "Modern Features",
     "description": "Evaluate use of modern language features (ES6+, CSS3, HTML5). 10=full utilization, 7.5=good, 5=partial, 2.5=minimal, 0=outdated.",
     "maxScore": 10, "weight": 0.9},

    {"id": 18, "category": "Innovation", "dimension": "Innovative Features",
     "description": "Check for innovative functionality or implementation. 10=multiple innovations, 7.5=notable, 5=some, 2.5=minor, 0=none.",
     "maxScore": 10, "weight": 0.8},

    {"id": 19, "category": "Innovation", "dimension": "UX Enhancement",
     "description": "Evaluate extras beyond basic requirements (animations, personalization). 10=excellent, 7.5=good, 5=average, 2.5=minor, 0=none.",
     "maxScore": 10, "weight": 0.7},

    {"id": 20, "category": "Robustness", "dimension": "Code Redundancy",
     "description": "Check for redundant code, duplicates, unrelated functions. 10=none, 7.5=minimal, 5=some, 2.5=significant, 0=severe.",
     "maxScore": 10, "weight": 0.8},

    {"id": 21, "category": "Robustness", "dimension": "Exception Handling",
     "description": "Evaluate handling of network errors, data anomalies, user mistakes. 10=comprehensive, 7.5=thorough, 5=basic, 2.5=minimal, 0=none.",
     "maxScore": 10, "weight": 1.1},

    {"id": 22, "category": "Compatibility", "dimension": "Cross-Platform",
     "description": "Check browser, device, responsive design compatibility. 10=full, 7.5=good, 5=partial, 2.5=poor, 0=none.",
     "maxScore": 10, "weight": 1.0},

    {"id": 23, "category": "Accessibility", "dimension": "A11y Support",
     "description": "Evaluate semantic tags, ARIA, keyboard nav, screen reader support. 10=comprehensive, 7.5=good, 5=basic, 2.5=minimal, 0=none.",
     "maxScore": 10, "weight": 0.7},

    {"id": 24, "category": "Maintainability", "dimension": "Code Readability",
     "description": "Assess code structure clarity, ease of understanding and maintenance. 10=excellent, 7.5=good, 5=average, 2.5=poor, 0=incomprehensible.",
     "maxScore": 10, "weight": 1.0},

    {"id": 25, "category": "Overall", "dimension": "Visual Fidelity & Completeness",
     "description": "Comprehensive evaluation of visual accuracy and professional level. 10=perfect, 7.5=excellent, 5=adequate, 2.5=partial, 0=inadequate.",
     "maxScore": 10, "weight": 1.2}
]


def generate_scoring_prompt(question: str, response: str) -> str:
    """Generate the scoring prompt for LLM evaluation."""
    criteria_text = "\n\n".join([
        f"**Criterion {c['id']}: {c['dimension']}** (Category: {c['category']}, Max: {c['maxScore']}, Weight: {c['weight']})\n{c['description']}"
        for c in SCORING_CRITERIA
    ])

    total_max_score = sum(c['maxScore'] for c in SCORING_CRITERIA)

    return f"""You are an expert code reviewer. Score the following question-response pair using 25 criteria.

# Scoring Criteria (25 items, total {total_max_score} points)

{criteria_text}

# Content to Score

## Question:
{question}

## Response:
{response}

# Requirements

1. **Strict scoring**: Don't give perfect scores easily
2. **Comprehensive**: Evaluate all dimensions
3. **Find issues**: Actively look for defects and improvement areas
4. **Compare to standards**: Compare against industry best practices
5. **Objective**: Base scores on facts, not code length

# Output Format

Return ONLY valid JSON (no other text):

{{
    "scores": [
        {{"criterion_id": 1, "score": 7.5, "reason": "Brief reason"}},
        ... (25 items, scores must be 10, 7.5, 5, 2.5, or 0)
    ],
    "total_score": 187.5,
    "weighted_score": 195.3,
    "max_score": {total_max_score},
    "percentage": 78.1,
    "overall_comment": "Overall evaluation summary",
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"],
    "suggestions": ["suggestion1", "suggestion2"]
}}"""


def parse_json_response(response_text: str) -> Optional[Dict]:
    """Parse JSON response from LLM."""
    if not response_text:
        return None

    try:
        # Clean markdown formatting
        cleaned = response_text.strip()
        if cleaned.startswith('```'):
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)

        # Find JSON object
        json_match = re.search(r'\{[\s\S]*\}', cleaned)
        if json_match:
            result = json.loads(json_match.group())
            if 'scores' in result:
                return result
    except json.JSONDecodeError:
        pass

    return None


class QualityScorer:
    """Quality scorer using LLM-based evaluation."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_concurrent: int = 2000
    ):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=300)
        self.model = model
        self.max_concurrent = max_concurrent
        self.semaphore = None

    async def score_single(
        self,
        idx: int,
        question: str,
        response: str
    ) -> Optional[Dict]:
        """Score a single question-response pair."""
        if self.semaphore is None:
            self.semaphore = asyncio.Semaphore(self.max_concurrent)

        async with self.semaphore:
            try:
                prompt = generate_scoring_prompt(question, response[:8000])

                result = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4096,
                    temperature=0.3
                )

                response_text = result.choices[0].message.content
                parsed = parse_json_response(response_text)

                if parsed:
                    return {
                        "idx": idx,
                        "success": True,
                        "score": parsed.get('percentage', 0),
                        "total_score": parsed.get('total_score', 0),
                        "weighted_score": parsed.get('weighted_score', 0),
                        "details": parsed
                    }
                return None

            except Exception as e:
                print(f"Scoring error for idx {idx}: {e}")
                return None

    async def score_batch(
        self,
        records: List[tuple]
    ) -> List[Dict]:
        """Score a batch of records."""
        tasks = [
            self.score_single(idx, q, r)
            for idx, q, r in records
        ]

        results = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Scoring"):
            result = await coro
            if result:
                results.append(result)

        return results


async def main_async(args):
    """Main async function."""
    print("Loading input data...")
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df):,} records")

    # Load existing progress
    processed_indices = set()
    if args.resume and os.path.exists(args.progress):
        with open(args.progress, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get('success'):
                        processed_indices.add(data['idx'])
                except:
                    continue
        print(f"Resuming: {len(processed_indices):,} already processed")

    # Prepare records to score
    records = [
        (idx, row['question'], row['response'])
        for idx, row in df.iterrows()
        if idx not in processed_indices
    ]

    if not records:
        print("All records already processed!")
        return

    print(f"Scoring {len(records):,} records...")

    # Initialize scorer
    scorer = QualityScorer(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        max_concurrent=args.workers
    )

    # Score in batches
    batch_size = args.batch_size
    all_results = []

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        print(f"\nBatch {i//batch_size + 1}: {len(batch)} records")

        results = await scorer.score_batch(batch)
        all_results.extend(results)

        # Save progress
        with open(args.progress, 'a') as f:
            for r in results:
                f.write(json.dumps(r) + '\n')

    # Merge results into DataFrame
    print("\nMerging results...")
    results_dict = {r['idx']: r for r in all_results}

    df['quality_score'] = df.index.map(lambda x: results_dict.get(x, {}).get('score', 0))
    df['score_details'] = df.index.map(
        lambda x: json.dumps(results_dict.get(x, {}).get('details', {}), ensure_ascii=False)
    )

    # Save output
    df.to_parquet(args.output, index=False)
    print(f"\nSaved to: {args.output}")

    # Statistics
    scored = df[df['quality_score'] > 0]
    print(f"\nStatistics:")
    print(f"  Scored: {len(scored):,} / {len(df):,}")
    print(f"  Mean score: {scored['quality_score'].mean():.2f}%")
    print(f"  Median: {scored['quality_score'].median():.2f}%")
    print(f"  Min/Max: {scored['quality_score'].min():.2f}% / {scored['quality_score'].max():.2f}%")


def main():
    parser = argparse.ArgumentParser(description="SFT Quality Scoring System")

    parser.add_argument("--input", type=str, required=True,
                        help="Input parquet file")
    parser.add_argument("--output", type=str, required=True,
                        help="Output parquet file with scores")
    parser.add_argument("--progress", type=str, default=None,
                        help="Progress file for resume (default: output.jsonl)")
    parser.add_argument("--workers", type=int, default=2000,
                        help="Number of concurrent workers")
    parser.add_argument("--batch_size", type=int, default=10000,
                        help="Batch size")
    parser.add_argument("--base_url", type=str,
                        default="http://localhost:8000/v1",
                        help="LLM API base URL")
    parser.add_argument("--api_key", type=str, default="EMPTY",
                        help="API key")
    parser.add_argument("--model", type=str, default="Qwen3-Coder-480b",
                        help="Model name")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from progress file")

    args = parser.parse_args()

    if args.progress is None:
        args.progress = args.output.replace('.parquet', '_progress.jsonl')

    print(f"{'='*60}")
    print("SFT Quality Scoring System")
    print(f"{'='*60}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Workers: {args.workers}")
    print(f"Model: {args.model}")
    print(f"{'='*60}\n")

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
