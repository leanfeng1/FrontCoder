#!/usr/bin/env python3
"""
RL Data Construction - Generate HTML with Gemini API

This script takes the 2000 RL prompts and uses Gemini-3-Pro-Preview to generate HTML code.

Pipeline:
1. Load prompts from generate_from_trending_demos.py output
2. Call Gemini API for each prompt
3. Save results with generated HTML responses

Usage:
    python generate_html_with_gemini.py \
        --input rl_prompts_2000.jsonl \
        --output rl_with_html.jsonl \
        --api_key YOUR_GEMINI_API_KEY \
        --workers 50
"""

import json
import os
import sys
import asyncio
import argparse
import random
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any, Optional

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
    print("Warning: google-generativeai not installed. Install with: pip install google-generativeai")


# Gemini generation config
GENERATION_CONFIG = {
    "temperature": 0.9,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 16384,
}

# Safety settings - relaxed for code generation
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
} if HAS_GENAI else {}


async def call_gemini_async(
    model: Any,
    prompt: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3
) -> tuple:
    """Call Gemini API with retry mechanism."""
    async with semaphore:
        for retry_count in range(max_retries):
            try:
                # Use generate_content (sync wrapped in async)
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: model.generate_content(prompt)
                )

                if response and response.text:
                    return True, response.text, None

                if retry_count < max_retries - 1:
                    await asyncio.sleep(random.uniform(1, 3) * (retry_count + 1))
                    continue
                return False, None, "Empty response"

            except Exception as e:
                error_msg = str(e)
                if "quota" in error_msg.lower() or "rate" in error_msg.lower():
                    # Rate limit - wait longer
                    wait_time = random.uniform(5, 15) * (retry_count + 1)
                    await asyncio.sleep(wait_time)
                elif retry_count < max_retries - 1:
                    await asyncio.sleep(random.uniform(2, 5) * (retry_count + 1))
                    continue
                else:
                    return False, None, error_msg

        return False, None, f"Failed after {max_retries} retries"


async def process_batch(
    records: List[Dict],
    model: Any,
    output_file: str,
    max_concurrent: int
) -> List[Dict]:
    """Process batch of records with Gemini API."""
    semaphore = asyncio.Semaphore(max_concurrent)
    file_lock = asyncio.Lock()
    pbar = tqdm(total=len(records), desc="Generating HTML")

    results = []

    async def process_single(record: Dict, idx: int):
        question = record.get("question", "")

        success, response, error = await call_gemini_async(
            model=model,
            prompt=question,
            semaphore=semaphore
        )

        result = {
            **record,
            "response": response if success else None,
            "generation_success": success,
            "generation_error": error
        }

        # Thread-safe write
        async with file_lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()

        pbar.update(1)
        return result

    tasks = [process_single(record, i) for i, record in enumerate(records)]
    results = await asyncio.gather(*tasks)
    pbar.close()

    return results


def load_prompts(input_file: str, processed_indices: set = None) -> List[Dict]:
    """Load prompts from JSONL file."""
    records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                if processed_indices is None or record.get("index") not in processed_indices:
                    records.append(record)
    return records


def load_processed_indices(output_file: str) -> set:
    """Load indices that have already been processed."""
    if not os.path.exists(output_file):
        return set()

    indices = set()
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    indices.add(record.get("index", -1))
    except Exception as e:
        print(f"Warning: Could not load processed indices: {e}")
        return set()

    return indices


def main():
    parser = argparse.ArgumentParser(description="Generate HTML with Gemini API")

    # Input/Output
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL file from generate_from_trending_demos.py")
    parser.add_argument("--output", type=str, default="rl_with_html.jsonl",
                        help="Output JSONL file with generated HTML")

    # API options
    parser.add_argument("--api_key", type=str, default=None,
                        help="Gemini API key (or set GOOGLE_API_KEY env var)")
    parser.add_argument("--model", type=str, default="gemini-2.5-pro-preview-06-05",
                        help="Gemini model name")

    # Processing options
    parser.add_argument("--workers", type=int, default=10,
                        help="Concurrent workers (be careful with rate limits)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to process")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous run")

    args = parser.parse_args()

    if not HAS_GENAI:
        print("Error: google-generativeai not installed!")
        print("Install with: pip install google-generativeai")
        sys.exit(1)

    # Get API key
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: API key required. Use --api_key or set GOOGLE_API_KEY env var")
        sys.exit(1)

    print("=" * 60)
    print("Generate HTML with Gemini API")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Model: {args.model}")
    print(f"Workers: {args.workers}")
    print("=" * 60)

    # Initialize Gemini
    print("\nInitializing Gemini...")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=args.model,
        generation_config=GENERATION_CONFIG,
        safety_settings=SAFETY_SETTINGS
    )

    # Load processed indices if resuming
    processed_indices = None
    if args.resume and os.path.exists(args.output):
        processed_indices = load_processed_indices(args.output)
        print(f"Resuming: {len(processed_indices)} already processed")
    else:
        # Clear output file
        open(args.output, 'w').close()

    # Load prompts
    print(f"\nLoading prompts from {args.input}...")
    records = load_prompts(args.input, processed_indices)
    print(f"Loaded {len(records)} prompts to process")

    if not records:
        print("No prompts to process!")
        return

    # Limit samples if specified
    if args.max_samples:
        records = records[:args.max_samples]
        print(f"Limited to {len(records)} samples")

    # Process
    print(f"\nGenerating HTML with Gemini...")
    results = asyncio.run(process_batch(
        records, model, args.output, args.workers
    ))

    # Statistics
    success_count = sum(1 for r in results if r.get("generation_success"))
    failed_count = len(results) - success_count

    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    print(f"  Success: {success_count:,}")
    print(f"  Failed: {failed_count:,}")
    print(f"  Total: {len(results):,}")
    print(f"  Output: {args.output}")
    print("=" * 60)
    print("\nNext step: Filter valid HTML responses")
    print(f"  python filter_valid_html.py --input {args.output} --output rl_filtered.jsonl")


if __name__ == "__main__":
    main()
