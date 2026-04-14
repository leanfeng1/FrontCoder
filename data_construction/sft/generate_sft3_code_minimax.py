"""
SFT Data Construction - Step 3: Generate HTML Code with Minimax-M2

Generates HTML/CSS/JS code for 240K variants using Minimax-M2
"""

import json
import os
import sys
import asyncio
import argparse
import random
from pathlib import Path
import tqdm
from openai import AsyncOpenAI

async def async_minimax_chat(base_url, api_key, model, messages, max_tokens, temperature, max_retries=3):
    """Call Minimax API using OpenAI-compatible interface"""
    for retry_count in range(max_retries):
        try:
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=300.0,
                max_retries=0
            )

            completion = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )

            response_text = completion.choices[0].message.content

            if response_text is not None and response_text.strip():
                return True, response_text, None
            else:
                if retry_count < max_retries - 1:
                    wait_time = random.uniform(1, 3) * (retry_count + 1)
                    await asyncio.sleep(wait_time)
                    continue
                return False, None, "API returned empty response"

        except Exception as e:
            error_msg = str(e)
            if retry_count < max_retries - 1:
                wait_time = random.uniform(2, 5) * (retry_count + 1)
                await asyncio.sleep(wait_time)
                continue
            return False, None, error_msg

    return False, None, f"Failed after {max_retries} retries"


def build_code_generation_prompt(variant_record):
    """Build code generation prompt"""
    return f"""You are a code expert. Please generate a complete, production-ready web application based on the following requirements.

**Category**: {variant_record['class']}
**Sub-Category**: {variant_record['sub_category']}
**Task**: {variant_record['variant_task']}
**Variant Type**: {variant_record['variant_type']}

**Requirements**:
1. Generate a complete HTML file with embedded CSS and JavaScript
2. Include all necessary styling and functionality
3. Make it visually appealing and user-friendly
4. Ensure the code is clean, well-commented, and follows best practices
5. The application should be fully functional and ready to use

**Output Format**:
Please provide ONLY the complete HTML code (including CSS and JavaScript), without any markdown code fences or explanations.

Generate the code:"""


async def process_single_variant(record, args, idx, total, output_file, cache_file):
    """Process a single variant and generate code"""
    try:
        variant_id = record['variant_id']
        
        # Build prompt
        prompt = build_code_generation_prompt(record)

        messages = [
            {"role": "user", "content": prompt}
        ]

        # Call Minimax API
        success, response, error = await async_minimax_chat(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )

        if success:
            # Build output record
            output_record = record.copy()
            output_record['code'] = response

            # Write in real-time
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                f.flush()

            # Save to cache
            cache_obj = {
                'variant_id': variant_id,
                'success': True
            }
            with open(cache_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(cache_obj, ensure_ascii=False) + '\n')
                f.flush()

            return {
                "success": True,
                "idx": idx,
                "variant_id": variant_id,
                "code_length": len(response),
                "error": None
            }
        else:
            return {
                "success": False,
                "idx": idx,
                "variant_id": variant_id,
                "code_length": 0,
                "error": error
            }
    except Exception as e:
        return {
            "success": False,
            "idx": idx,
            "variant_id": record.get('variant_id', -1),
            "code_length": 0,
            "error": str(e)
        }


async def process_batch_with_semaphore(records, args):
    """Batch processing"""
    semaphore = asyncio.Semaphore(args.workers)
    pbar = tqdm.tqdm(total=len(records), desc="Generating code", unit="records")

    async def limited_process(record, idx, total):
        async with semaphore:
            result = await process_single_variant(
                record, args, idx, total,
                args.output_file, args.cache_file
            )
            pbar.update(1)

            if result["success"]:
                pbar.write(f"✅ [{idx+1}/{total}] Variant {result['variant_id']} - Generated ({result['code_length']} chars)")
            else:
                error_msg = str(result['error'])[:50]
                pbar.write(f"❌ [{idx+1}/{total}] Variant {result['variant_id']} - {error_msg}")

            return result

    total = len(records)
    tasks = [limited_process(record, i, total) for i, record in enumerate(records)]

    print(f"🚀 Starting async processing {total} records (Concurrency: {args.workers})...")
    results = await asyncio.gather(*tasks)
    pbar.close()

    return results


def load_records_from_jsonl(input_file):
    """Load records"""
    print(f"\n📁 Reading file: {input_file}")
    records = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        obj = json.loads(line)
                        records.append(obj)
                    except json.JSONDecodeError as e:
                        print(f"⚠️ Line {line_num} parsing failed: {e}")
                        continue
    except Exception as e:
        print(f"❌ Failed to read file: {e}")
        return []

    print(f"✅ Successfully loaded {len(records)} records")
    return records


def load_cache(cache_file):
    """Load cache"""
    if not os.path.exists(cache_file):
        return set()

    processed_ids = set()
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    if obj.get('success'):
                        processed_ids.add(obj['variant_id'])
        print(f"📦 Loaded {len(processed_ids)} cached records")
    except Exception as e:
        print(f"⚠️ Load cachefailed: {e}")
        return set()

    return processed_ids


def parse_args():
    parser = argparse.ArgumentParser(description="🎯 SFT Step 3: Generate HTML Code with Minimax-M2")

    parser.add_argument("--input_file", type=str, default="sft_variants_240k.jsonl",
                       help="Input: 240K variants")
    parser.add_argument("--output_file", type=str, default="sft_final_240k.jsonl",
                       help="Output: 240K code samples")
    parser.add_argument("--cache_file", type=str, default="sft_final_240k_cache.jsonl",
                       help="Cache file")

    parser.add_argument("--max_records", type=int, default=None,
                       help="Maximum records to process (for testing)")

    parser.add_argument("--api_key", type=str, default="EMPTY",
                       help="API key")
    parser.add_argument("--base_url", type=str,
                       default="http://localhost:8000/v1",
                       help="Minimax API base URL")
    parser.add_argument("--model", type=str, default="Minimax-Text-01",
                       help="Modelname")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Generation temperature")
    parser.add_argument("--max_tokens", type=int, default=16384,
                       help="Max tokens")

    parser.add_argument("--workers", type=int, default=100,
                       help="Concurrency")

    parser.add_argument("--test", action="store_true",
                       help="Test mode (only process first2records)")

    return parser.parse_args()


def main():
    args = parse_args()

    print("="*80)
    print("🎯 SFT Step 3: Generate HTML Code with Minimax-M2")
    print("="*80)
    print(f"📁 Input file: {args.input_file}")
    print(f"📝 Output file: {args.output_file}")
    print(f"💾 Cache file: {args.cache_file}")
    print(f"🤖 Model: {args.model}")
    print(f"🎨 Temperature: {args.temperature}")
    print(f"⚡ Concurrency: {args.workers}")
    print("="*80)

    if args.test:
        print("⚠️ Test mode: processing only first2records")
        args.max_records = 2

    print("\n📋 Loading records...")
    all_records = load_records_from_jsonl(args.input_file)

    if not all_records:
        print("❌ No records found!")
        return

    if args.max_records:
        all_records = all_records[:args.max_records]
        print(f"⚠️ Limiting records to: {len(all_records)}")

    print("\n💾 Loading cache...")
    processed_ids = load_cache(args.cache_file)

    remaining_records = [r for r in all_records if r.get('variant_id', -1) not in processed_ids]

    if not remaining_records:
        print("✅ All records processed!")
        return

    print(f"\n📊 To process: {len(remaining_records)} records (total {len(all_records)} records)")

    # Async processing
    results = asyncio.run(process_batch_with_semaphore(remaining_records, args))

    # Statistics
    success_count = sum(1 for r in results if r['success'])
    failed_count = len(results) - success_count
    total_code_chars = sum(r['code_length'] for r in results if r['success'])
    avg_code_length = total_code_chars / success_count if success_count > 0 else 0

    print("\n" + "="*80)
    print("🎉 Step 3 Complete!")
    print("="*80)
    print(f"  📊 Total records: {len(all_records)}")
    print(f"  ✅ Succeeded in this run: {success_count} records")
    print(f"  ❌ Failed in this run: {failed_count} records")
    print(f"  📝 Generating codetotalcharacterscount: {total_code_chars:,}")
    print(f"  📏 Average code length: {avg_code_length:.0f} characters")
    print(f"  📁 Output file: {args.output_file}")
    print("="*80)


if __name__ == "__main__":
    main()
