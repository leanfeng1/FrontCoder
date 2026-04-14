"""
CPT Data Construction - Step 3: Generate HTML Code with MiniMax

This script generates HTML code for prompts using MiniMax API.
Based on the paper: Uses MiniMax-M2 for large-scale HTML generation.

Features:
- Multi-process parallel generation for 25 category nodes
- Dynamic task queue for load balancing
- Real-time progress tracking and caching
- Automatic retry on failures

Usage:
    python generate_html_minimax.py \
        --input prompts_output/ \
        --output html_output/ \
        --api_key YOUR_API_KEY \
        --workers 200
"""

import json
import os
import sys
import argparse
import random
import threading
import time
import fcntl
from pathlib import Path
from openai import OpenAI
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


# Global client (connection reuse)
_global_client = None
_client_lock = threading.Lock()


def get_client(base_url: str, api_key: str) -> OpenAI:
    """Get global reusable OpenAI client."""
    global _global_client
    with _client_lock:
        if _global_client is None:
            _global_client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=300.0,
                max_retries=0
            )
    return _global_client


def minimax_chat(
    client: OpenAI,
    model: str,
    messages: list,
    max_tokens: int,
    temperature: float,
    max_retries: int = 3
) -> tuple:
    """Call MiniMax API with retries."""
    for retry_count in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )

            if completion is None or not completion.choices:
                if retry_count < max_retries - 1:
                    time.sleep(random.uniform(1, 3) * (retry_count + 1))
                    continue
                return False, None, "Empty response"

            response_text = completion.choices[0].message.content

            if response_text and response_text.strip():
                return True, response_text, None
            else:
                if retry_count < max_retries - 1:
                    time.sleep(random.uniform(1, 3))
                    continue
                return False, None, "Empty content"

        except Exception as e:
            if retry_count < max_retries - 1:
                time.sleep(random.uniform(2, 5) * (retry_count + 1))
                continue
            return False, None, str(e)

    return False, None, f"Failed after {max_retries} retries"


def process_single_record(
    record: dict,
    client: OpenAI,
    model: str,
    max_tokens: int,
    temperature: float,
    output_file: str,
    file_lock: threading.Lock
) -> dict:
    """Process single prompt to generate HTML."""
    try:
        prompt = record.get('prompt', '')
        if not prompt:
            return {"success": False, "error": "No prompt"}

        messages = [{"role": "user", "content": prompt}]

        success, response, error = minimax_chat(
            client=client,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )

        if success:
            output_record = record.copy()
            output_record['response'] = response
            output_record['generated_at'] = datetime.now().isoformat()

            # Thread-safe write
            with file_lock:
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                    f.flush()

            return {"success": True, "response_length": len(response)}
        else:
            return {"success": False, "error": error}

    except Exception as e:
        return {"success": False, "error": str(e)}


def process_input_file(
    input_file: str,
    output_file: str,
    base_url: str,
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
    workers_per_file: int
):
    """Process a single input file."""
    print(f"\nProcessing: {input_file}")

    # Load input records
    records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"  Loaded {len(records):,} records")

    # Load existing (for resume)
    existing_count = 0
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            existing_count = sum(1 for _ in f)
        records = records[existing_count:]
        print(f"  Resuming from {existing_count:,}, remaining: {len(records):,}")

    if not records:
        print(f"  Already complete!")
        return {"success": existing_count, "failed": 0}

    # Process
    client = get_client(base_url, api_key)
    file_lock = threading.Lock()

    success_count = 0
    failed_count = 0

    with ThreadPoolExecutor(max_workers=workers_per_file) as executor:
        futures = {
            executor.submit(
                process_single_record,
                record, client, model, max_tokens, temperature,
                output_file, file_lock
            ): i for i, record in enumerate(records)
        }

        for future in as_completed(futures):
            result = future.result()
            if result.get("success"):
                success_count += 1
                if success_count % 1000 == 0:
                    print(f"  Progress: {success_count:,}/{len(records):,}")
            else:
                failed_count += 1

    print(f"  Complete: {success_count:,} success, {failed_count:,} failed")
    return {"success": success_count + existing_count, "failed": failed_count}


def main():
    parser = argparse.ArgumentParser(description="Generate HTML with MiniMax")

    parser.add_argument("--input", type=str, required=True,
                        help="Input directory or file with prompts")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory or file")
    parser.add_argument("--api_key", type=str, required=True,
                        help="MiniMax API key")
    parser.add_argument("--base_url", type=str,
                        default="https://api.minimax.chat/v1",
                        help="API base URL")
    parser.add_argument("--model", type=str, default="MiniMax-M2",
                        help="Model name")
    parser.add_argument("--max_tokens", type=int, default=32768,
                        help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Generation temperature")
    parser.add_argument("--workers", type=int, default=200,
                        help="Workers per file")

    args = parser.parse_args()

    print("="*60)
    print("HTML Generation with MiniMax")
    print("="*60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Model: {args.model}")
    print(f"Workers: {args.workers}")
    print("="*60)

    # Determine input files
    if os.path.isdir(args.input):
        input_files = sorted(Path(args.input).glob("*.jsonl"))
        os.makedirs(args.output, exist_ok=True)
    else:
        input_files = [Path(args.input)]

    print(f"\nFound {len(input_files)} input files")

    # Process each file
    total_success = 0
    total_failed = 0

    for input_file in input_files:
        if os.path.isdir(args.output):
            output_file = os.path.join(
                args.output,
                input_file.stem + "_output.jsonl"
            )
        else:
            output_file = args.output

        stats = process_input_file(
            str(input_file),
            output_file,
            args.base_url,
            args.api_key,
            args.model,
            args.max_tokens,
            args.temperature,
            args.workers
        )

        total_success += stats["success"]
        total_failed += stats["failed"]

    print("\n" + "="*60)
    print("Generation Complete!")
    print(f"  Total success: {total_success:,}")
    print(f"  Total failed: {total_failed:,}")
    print("="*60)


if __name__ == "__main__":
    main()
