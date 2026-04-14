"""
SFT Data Construction - Step 1: Expand to 20K Tasks

Expands from 2K subcategories to 20K specific tasks (generates 10 tasks per subcategory).

This script uses an LLM to generate diverse and specific task descriptions
for each subcategory in the SFT data construction pipeline.
"""

import json
import os
import asyncio
import argparse
import random
import re
from pathlib import Path
import tqdm
import utils

# Task generation prompt (no existing task needed)
TASK_GENERATION_PROMPT = """You are a senior software development and requirement analysis expert. I will give you a category and sub-category. Please generate 10 DIFFERENT and DIVERSE specific tasks under this sub-category.

**Category (Major Classification)**: {category}

**Sub-Category**: {sub_category}

**Requirements**:

1. Generate exactly 10 specific tasks that are:
   - Under the category "{category}" and sub-category "{sub_category}"
   - DIFFERENT from each other (no duplicates)
   - Clear, actionable, and specific (5-15 words each in English)
   - Practical and implementable as web applications or tools
   
2. The 10 tasks should cover diverse aspects:
   - Different features or functionalities
   - Different complexity levels (simple to advanced)
   - Different use cases or scenarios
   - Different technical approaches or implementations

**Output Format**:
Please strictly output in the following JSON format, with no additional content:

```json
{{
    "tasks": [
        "First specific task description",
        "Second specific task description",
        "Third specific task description",
        "Fourth specific task description",
        "Fifth specific task description",
        "Sixth specific task description",
        "Seventh specific task description",
        "Eighth specific task description",
        "Ninth specific task description",
        "Tenth specific task description"
    ]
}}
```

Please provide your 10 specific tasks in English:"""


async def async_chat_with_retry(base_url, api_key, model, prompt, max_tokens, temperature, max_retries=3):
    """Async API call with retry mechanism."""
    for retry_count in range(max_retries):
        try:
            openai_args = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature
            }

            response_text = await utils.async_chat(
                base_url=base_url,
                api_key=api_key,
                **openai_args
            )

            if response_text is not None and response_text.strip():
                return True, response_text, None
            else:
                if retry_count < max_retries - 1:
                    wait_time = random.uniform(1, 3) * (retry_count + 1)
                    await asyncio.sleep(wait_time)
                    continue
                return False, None, "API returned empty response"

        except Exception as e:
            if retry_count < max_retries - 1:
                wait_time = random.uniform(2, 5) * (retry_count + 1)
                await asyncio.sleep(wait_time)
                continue
            return False, None, str(e)

    return False, None, f"Failed after {max_retries} retries"


def extract_tasks_from_response(response_text):
    """Extract task list from API response."""
    try:
        result = json.loads(response_text)
        if 'tasks' in result and isinstance(result['tasks'], list):
            return result['tasks']
    except:
        pass

    # Try extracting from code block
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group(1))
            if 'tasks' in result and isinstance(result['tasks'], list):
                return result['tasks']
        except:
            pass

    # Try extracting any JSON object containing tasks field
    json_match = re.search(r'\{[^{}]*"tasks"[^{}]*\[[^\]]*\][^{}]*\}', response_text, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group(0))
            if 'tasks' in result and isinstance(result['tasks'], list):
                return result['tasks']
        except:
            pass

    return None


async def process_single_subcategory(record, args, idx, total, output_file, cache_file):
    """Process a single subcategory and generate 10 tasks"""
    try:
        subcat_id = record['subcat_id']
        category = record['class']
        sub_category = record['sub_category']

        # Build prompt
        prompt = TASK_GENERATION_PROMPT.format(
            category=category,
            sub_category=sub_category
        )

        # Call API to generate 10 tasks
        success, response, error = await async_chat_with_retry(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )

        if success:
            tasks = extract_tasks_from_response(response)

            if tasks and isinstance(tasks, list) and len(tasks) >= 10:
                tasks = tasks[:10]
                new_records = []

                for i, task in enumerate(tasks):
                    task_id = subcat_id * 10 + i
                    new_record = {
                        "task_id": task_id,
                        "subcat_id": subcat_id,
                        "class": category,
                        "sub_category": sub_category,
                        "specific_task": task
                    }

                    new_records.append(new_record)

                    # Write in real-time
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(new_record, ensure_ascii=False) + '\n')
                        f.flush()

                # Save to cache
                cache_obj = {
                    'subcat_id': subcat_id,
                    'success': True,
                    'num_tasks': len(tasks)
                }
                with open(cache_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(cache_obj, ensure_ascii=False) + '\n')
                    f.flush()

                return {
                    "success": True,
                    "idx": idx,
                    "subcat_id": subcat_id,
                    "num_tasks": len(tasks),
                    "error": None
                }
            else:
                error_msg = f"Failed to extract 10 tasks. Got: {len(tasks) if tasks else 0}"
                return {
                    "success": False,
                    "idx": idx,
                    "subcat_id": subcat_id,
                    "num_tasks": 0,
                    "error": error_msg
                }
        else:
            return {
                "success": False,
                "idx": idx,
                "subcat_id": subcat_id,
                "num_tasks": 0,
                "error": error
            }
    except Exception as e:
        return {
            "success": False,
            "idx": idx,
            "subcat_id": record.get('subcat_id', -1),
            "num_tasks": 0,
            "error": str(e)
        }


async def process_batch_with_semaphore(records, args):
    """Batch processing"""
    semaphore = asyncio.Semaphore(args.workers)
    pbar = tqdm.tqdm(total=len(records), desc="Processing progress", unit="records")

    async def limited_process(record, idx, total):
        async with semaphore:
            result = await process_single_subcategory(
                record, args, idx, total,
                args.output_file, args.cache_file
            )
            pbar.update(1)

            if result["success"]:
                pbar.write(f"✅ [{idx+1}/{total}] Subcat {result['subcat_id']} - Generated {result['num_tasks']} tasks")
            else:
                error_msg = str(result['error'])[:50]
                pbar.write(f"❌ [{idx+1}/{total}] Subcat {result['subcat_id']} - {error_msg}")

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
                        processed_ids.add(obj['subcat_id'])
        print(f"📦 Loaded {len(processed_ids)} cached records")
    except Exception as e:
        print(f"⚠️ Load cachefailed: {e}")
        return set()

    return processed_ids


def parse_args():
    parser = argparse.ArgumentParser(description="🎯 SFT Step 1: Expand to 20K Tasks")

    parser.add_argument("--input_file", type=str, default="sft_subcategories_2k.jsonl",
                       help="Input: 2K subcategories")
    parser.add_argument("--output_file", type=str, default="sft_tasks_20k.jsonl",
                       help="Output: 20K tasks")
    parser.add_argument("--cache_file", type=str, default="sft_tasks_20k_cache.jsonl",
                       help="Cache file")

    parser.add_argument("--max_records", type=int, default=None,
                       help="Maximum records to process (for testing)")

    parser.add_argument("--api_key", type=str, default="sk-abc123",
                       help="API key")
    parser.add_argument("--base_url", type=str,
                       default="https://console.siflow.cn/siflow/auriga/skyinfer/fjing/qwen3-480b-0/v1",
                       help="API base URL")
    parser.add_argument("--model", type=str, default="Qwen3-Coder-480b",
                       help="Modelname")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Generation temperature")
    parser.add_argument("--max_tokens", type=int, default=2048,
                       help="Max tokens")

    parser.add_argument("--workers", type=int, default=50,
                       help="Concurrency")

    parser.add_argument("--test", action="store_true",
                       help="Test mode (only process first3records)")

    return parser.parse_args()


def main():
    args = parse_args()

    print("="*80)
    print("🎯 SFT Step 1: Expand to 20K Tasks")
    print("="*80)
    print(f"📁 Input file: {args.input_file}")
    print(f"📝 Output file: {args.output_file}")
    print(f"💾 Cache file: {args.cache_file}")
    print(f"🤖 Model: {args.model}")
    print(f"🎨 Temperature: {args.temperature}")
    print(f"⚡ Concurrency: {args.workers}")
    print("="*80)

    if args.test:
        print("⚠️ Test mode: processing only first3records")
        args.max_records = 3

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

    remaining_records = [r for r in all_records if r.get('subcat_id', -1) not in processed_ids]

    if not remaining_records:
        print("✅ All records processed!")
        return

    print(f"\n📊 To process: {len(remaining_records)} records (total {len(all_records)} records)")
    print(f"Expected to generate: {len(remaining_records) * 10} tasks")

    # Async processing
    results = asyncio.run(process_batch_with_semaphore(remaining_records, args))

    # Statistics
    success_count = sum(1 for r in results if r['success'])
    failed_count = len(results) - success_count
    total_tasks = sum(r['num_tasks'] for r in results if r['success'])

    print("\n" + "="*80)
    print("🎉 Step 1 Complete!")
    print("="*80)
    print(f"  📊 Total records: {len(all_records)}")
    print(f"  ✅ Succeeded in this run: {success_count} records")
    print(f"  ❌ Failed in this run: {failed_count} records")
    print(f"  📝 Tasks generated: {total_tasks} ")
    print(f"  📁 Output file: {args.output_file}")
    print("="*80)


if __name__ == "__main__":
    main()
