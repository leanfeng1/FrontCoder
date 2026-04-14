"""
CPT Data Construction - WebSight Rewriting

This script expands HTML code from WebSight dataset using Qwen3-Coder.
Based on the paper: Uses LLM to analyze and expand existing HTML code.

Pipeline:
1. Load WebSight parquet files:
   - llm_generated_idea: category/theme description
   - text: original HTML code
2. For each HTML, use Qwen3-Coder to expand and improve the code
3. Output: Expanded HTML with design explanations

Usage:
    python generate_websight_expansion.py \
        --data_dir /path/to/websight/parquet \
        --output_file websight_expanded.jsonl \
        --workers 50
"""

import json
import os
import sys
import asyncio
import argparse
import glob
import random
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from openai import AsyncOpenAI


# HTML expansion prompt template
# {theme} will be filled with llm_generated_idea (category/theme description)
# {original_html} will be filled with text field (original HTML code)
EXPANSION_PROMPT = """You are a senior front-end development expert. Given a webpage category/theme description and original HTML code, please professionally expand and improve this HTML code.

**Webpage Category/Theme**: {theme}

**Original HTML Code**:
```html
{original_html}
```

**Task Requirements**:

1. **Maintain Category/Theme Consistency**: The expanded content must be highly relevant to the given category/theme description. Do not deviate from the theme.

2. **Code Expansion** (Key Focus):
   - Expand upon the original code, adding more practical features
   - Enrich page content and interactive effects
   - Add more components and page elements
   - Optimize CSS styling for better aesthetics
   - Enhance JavaScript interactive functionality
   - Use modern front-end libraries and best practices
   - Goal: The expanded HTML should be more complete and professional than the original

3. **Technical Requirements**:
   - Maintain clear code structure with detailed comments
   - Use modern HTML5/CSS3/JavaScript features
   - Ensure code runs directly
   - Focus on responsive design and user experience

4. **Output Format Requirements**:
   - First, write 1-2 paragraphs (200-400 words total) introducing the design philosophy, main features, and technical highlights
   - Then output the complete expanded HTML code wrapped in ```html``` code blocks
   - Ensure the HTML code is complete and runnable

Please begin your expansion work:"""


async def async_chat_with_retry(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    max_retries: int = 3
) -> tuple:
    """Async API call with retry mechanism."""
    for retry_count in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )

            if response and response.choices:
                response_text = response.choices[0].message.content
                if response_text and response_text.strip():
                    return True, response_text, None

            if retry_count < max_retries - 1:
                await asyncio.sleep(random.uniform(1, 3) * (retry_count + 1))
                continue
            return False, None, "Empty response"

        except Exception as e:
            if retry_count < max_retries - 1:
                await asyncio.sleep(random.uniform(2, 5) * (retry_count + 1))
                continue
            return False, None, str(e)

    return False, None, f"Failed after {max_retries} retries"


async def process_single_sample(
    sample_data: dict,
    client: AsyncOpenAI,
    model: str,
    max_tokens: int,
    temperature: float,
    output_file: str,
    file_lock: asyncio.Lock,
    idx: int
) -> dict:
    """Process single sample."""
    try:
        # llm_generated_idea: category/theme description
        # original_html: HTML code from 'text' field
        llm_generated_idea = sample_data['llm_generated_idea']
        original_html = sample_data['original_html']

        # Build prompt (use llm_generated_idea as theme in prompt)
        prompt = EXPANSION_PROMPT.format(
            theme=llm_generated_idea,
            original_html=original_html
        )

        # Call API
        success, response, error = await async_chat_with_retry(
            client=client,
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )

        if success:
            output_data = {
                "text": response,
                "llm_generated_idea": llm_generated_idea,
                "original_html": original_html
            }

            # Thread-safe write
            async with file_lock:
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                    f.flush()

            return {"success": True, "idx": idx, "llm_generated_idea": llm_generated_idea}
        else:
            return {"success": False, "idx": idx, "llm_generated_idea": llm_generated_idea, "error": error}

    except Exception as e:
        return {"success": False, "idx": idx, "error": str(e)}


async def process_batch(
    samples: list,
    client: AsyncOpenAI,
    model: str,
    max_tokens: int,
    temperature: float,
    output_file: str,
    max_concurrent: int
):
    """Process batch with semaphore for concurrency control."""
    semaphore = asyncio.Semaphore(max_concurrent)
    file_lock = asyncio.Lock()
    pbar = tqdm(total=len(samples), desc="Processing")

    async def limited_process(sample, idx):
        async with semaphore:
            result = await process_single_sample(
                sample, client, model, max_tokens, temperature,
                output_file, file_lock, idx
            )
            pbar.update(1)
            return result

    tasks = [limited_process(sample, i) for i, sample in enumerate(samples)]
    results = await asyncio.gather(*tasks)
    pbar.close()
    return results


def load_parquet_files(data_dir: str, max_files: int = None) -> list:
    """Load all parquet files from WebSight dataset.

    Parquet fields:
    - llm_generated_idea: category/theme description for the webpage
    - text: original HTML code
    """
    print(f"\nScanning directory: {data_dir}")
    parquet_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    print(f"Found {len(parquet_files)} parquet files")

    if max_files:
        parquet_files = parquet_files[:max_files]
        print(f"Using first {max_files} files")

    all_samples = []
    for file_path in tqdm(parquet_files, desc="Loading parquet files"):
        try:
            df = pd.read_parquet(file_path)
            for idx, row in df.iterrows():
                # llm_generated_idea: category/theme description
                # text: original HTML code
                llm_generated_idea = row['llm_generated_idea'] if 'llm_generated_idea' in row else ''
                original_html = row['text'] if 'text' in row else ''

                if llm_generated_idea and original_html:
                    sample = {
                        'llm_generated_idea': llm_generated_idea,
                        'original_html': original_html
                    }
                    all_samples.append(sample)
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
            continue

    print(f"Loaded {len(all_samples):,} samples")
    return all_samples


def load_cache(cache_file: str) -> set:
    """Load processed indices from cache."""
    if not os.path.exists(cache_file):
        return set()

    processed = set()
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    processed.add(obj.get('idx', -1))
        print(f"Loaded {len(processed)} cached records")
    except Exception as e:
        print(f"Failed to load cache: {e}")
        return set()
    return processed


def main():
    parser = argparse.ArgumentParser(description="WebSight HTML Expansion Generator")

    # Input/Output
    parser.add_argument("--data_dir", type=str, required=True,
                        help="WebSight parquet directory")
    parser.add_argument("--output_file", type=str, default="websight_expanded.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--cache_file", type=str, default="websight_cache.jsonl",
                        help="Cache file for resume")

    # Data options
    parser.add_argument("--max_files", type=int, default=None,
                        help="Max parquet files to load")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to process")

    # API options
    parser.add_argument("--api_key", type=str, default="EMPTY",
                        help="API key")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1",
                        help="API base URL")
    parser.add_argument("--model", type=str, default="Qwen3-Coder-480b",
                        help="Model name")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Generation temperature")
    parser.add_argument("--max_tokens", type=int, default=16384,
                        help="Max tokens")

    # Concurrency
    parser.add_argument("--workers", type=int, default=50,
                        help="Concurrent workers")

    args = parser.parse_args()

    print("=" * 60)
    print("WebSight HTML Expansion Generator")
    print("=" * 60)
    print(f"Data Dir: {args.data_dir}")
    print(f"Output: {args.output_file}")
    print(f"Model: {args.model}")
    print(f"Workers: {args.workers}")
    print("=" * 60)

    # Load samples
    print("\nLoading parquet files...")
    all_samples = load_parquet_files(args.data_dir, args.max_files)

    if not all_samples:
        print("No samples found!")
        return

    # Limit samples
    if args.max_samples:
        all_samples = all_samples[:args.max_samples]
        print(f"Limited to {len(all_samples)} samples")

    # Add index
    for i, sample in enumerate(all_samples):
        sample['idx'] = i

    # Load cache
    print("\nLoading cache...")
    processed = load_cache(args.cache_file)

    # Filter processed
    remaining = [s for s in all_samples if s['idx'] not in processed]
    print(f"Already processed: {len(processed)}")
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("\nAll samples processed!")
        return

    # Initialize output file
    if not processed:
        open(args.output_file, 'w').close()

    # Process
    print(f"\nStarting expansion generation...")
    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=300.0
    )

    results = asyncio.run(process_batch(
        remaining, client, args.model, args.max_tokens,
        args.temperature, args.output_file, args.workers
    ))

    # Save cache
    with open(args.cache_file, 'a', encoding='utf-8') as f:
        for r in results:
            if r.get('success'):
                f.write(json.dumps({'idx': r['idx']}, ensure_ascii=False) + '\n')

    # Statistics
    success = sum(1 for r in results if r.get('success'))
    failed = len(results) - success

    print("\n" + "=" * 60)
    print("Expansion Complete!")
    print("=" * 60)
    print(f"  Success: {success:,}")
    print(f"  Failed: {failed:,}")
    print(f"  Total processed: {len(processed) + success:,}")
    print(f"  Output: {args.output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
