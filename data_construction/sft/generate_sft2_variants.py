"""
SFT Data Construction - Step 2: Expand to 240K Variants

Expands from 20K tasks to 240K variants (generates 12 variants per task).

Variant dimensions include:
- Visual style (colors, layout, design style)
- Feature enhancements (interaction methods, additional features)
- User interface (responsive, accessibility, i18n)
- Technical implementation (frameworks, libraries)
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

# 12 predefined variant types
VARIANT_TYPES = [
    {"id": 1, "name": "Color Scheme", "description": "Change to dark/light mode, different color palette"},
    {"id": 2, "name": "Layout Style", "description": "Grid/List/Card layout variations"},
    {"id": 3, "name": "Interaction Mode", "description": "Click/Hover/Drag interaction changes"},
    {"id": 4, "name": "Responsive Design", "description": "Mobile-first or Desktop-first variations"},
    {"id": 5, "name": "Animation Effects", "description": "Add smooth transitions and animations"},
    {"id": 6, "name": "Accessibility Features", "description": "WCAG compliant, keyboard navigation"},
    {"id": 7, "name": "Advanced Features", "description": "Add premium features like filters, search, export"},
    {"id": 8, "name": "Minimalist Design", "description": "Simplified UI with essential features only"},
    {"id": 9, "name": "Data Visualization", "description": "Add charts, graphs, or visual representations"},
    {"id": 10, "name": "Real-time Updates", "description": "Add live data updates and notifications"},
    {"id": 11, "name": "Gamification", "description": "Add points, badges, leaderboards"},
    {"id": 12, "name": "Internationalization", "description": "Multi-language support and locale-specific features"}
]

VARIANT_GENERATION_PROMPT = """You are a senior software development expert. I will give you an original task and a variant specification. Please generate a NEW task description that incorporates the variant requirements.

**Original Task**:
- **Category**: {category}
- **Sub-Category**: {sub_category}
- **Specific Task**: {specific_task}

**Variant Specification**:
- **Variant #{variant_id}**: {variant_name}
- **Description**: {variant_description}

**Requirements**:

1. Generate a NEW specific task (5-20 words) that:
   - Builds upon the original task: "{specific_task}"
   - Incorporates the variant requirements: {variant_name}
   - Remains under the category "{category}" and sub-category "{sub_category}"
   - Is practical and implementable as a web application

2. Make it specific and actionable:
   - Include technical details (e.g., "dark mode with #1a1a1a background")
   - Specify implementation methods (e.g., "using CSS Grid", "with React hooks")
   - Describe user interactions (e.g., "drag-and-drop interface", "hover effects")

3. Ensure distinctness:
   - The task should feel different from the original
   - It should still solve the same problem but with the variant's approach
   - Keep it concise but specific

**Output Format**:
Please output ONLY the new task description (5-20 words), without any JSON wrapping, explanations, or extra text.

Please generate the variant task:"""


async def async_chat_with_retry(base_url, api_key, model, prompt, max_tokens, temperature, max_retries=3):
    """Async API call with retry mechanism"""
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


async def process_single_task(record, args, idx, total, output_file, cache_file):
    """Process a single task and generate 12 variants"""
    try:
        task_id = record['task_id']
        category = record['class']
        sub_category = record['sub_category']
        specific_task = record['specific_task']

        variants_generated = []

        for variant in VARIANT_TYPES:
            variant_id = variant['id']
            
            # Build prompt
            prompt = VARIANT_GENERATION_PROMPT.format(
                category=category,
                sub_category=sub_category,
                specific_task=specific_task,
                variant_id=variant_id,
                variant_name=variant['name'],
                variant_description=variant['description']
            )

            # Call API
            success, response, error = await async_chat_with_retry(
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature
            )

            if success:
                new_task = response.strip()
                
                # Clean markdown
                if new_task.startswith('```') and new_task.endswith('```'):
                    lines = new_task.split('\n')
                    new_task = '\n'.join(lines[1:-1]).strip()

                # Generate variant_id
                global_variant_id = task_id * 12 + (variant_id - 1)

                variant_record = {
                    "variant_id": global_variant_id,
                    "task_id": task_id,
                    "variant_type_id": variant_id,
                    "variant_type": variant['name'],
                    "class": category,
                    "sub_category": sub_category,
                    "original_task": specific_task,
                    "variant_task": new_task
                }

                variants_generated.append(variant_record)

                # Write in real-time
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(variant_record, ensure_ascii=False) + '\n')
                    f.flush()

        # Save to cache
        cache_obj = {
            'task_id': task_id,
            'success': True,
            'num_variants': len(variants_generated)
        }
        with open(cache_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(cache_obj, ensure_ascii=False) + '\n')
            f.flush()

        return {
            "success": True,
            "idx": idx,
            "task_id": task_id,
            "num_variants": len(variants_generated),
            "error": None
        }

    except Exception as e:
        return {
            "success": False,
            "idx": idx,
            "task_id": record.get('task_id', -1),
            "num_variants": 0,
            "error": str(e)
        }


async def process_batch_with_semaphore(records, args):
    """Batch processing"""
    semaphore = asyncio.Semaphore(args.workers)
    pbar = tqdm.tqdm(total=len(records), desc="Processing progress", unit="records")

    async def limited_process(record, idx, total):
        async with semaphore:
            result = await process_single_task(
                record, args, idx, total,
                args.output_file, args.cache_file
            )
            pbar.update(1)

            if result["success"]:
                pbar.write(f"✅ [{idx+1}/{total}] Task {result['task_id']} - Generated {result['num_variants']} variants")
            else:
                error_msg = str(result['error'])[:50]
                pbar.write(f"❌ [{idx+1}/{total}] Task {result['task_id']} - {error_msg}")

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
                        processed_ids.add(obj['task_id'])
        print(f"📦 Loaded {len(processed_ids)} cached records")
    except Exception as e:
        print(f"⚠️ Load cachefailed: {e}")
        return set()

    return processed_ids


def parse_args():
    parser = argparse.ArgumentParser(description="🎯 SFT Step 2: Expand to 240K Variants")

    parser.add_argument("--input_file", type=str, default="sft_tasks_20k.jsonl",
                       help="Input: 20K tasks")
    parser.add_argument("--output_file", type=str, default="sft_variants_240k.jsonl",
                       help="Output: 240K variants")
    parser.add_argument("--cache_file", type=str, default="sft_variants_240k_cache.jsonl",
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
    parser.add_argument("--temperature", type=float, default=0.9,
                       help="Generation temperature")
    parser.add_argument("--max_tokens", type=int, default=512,
                       help="Max tokens")

    parser.add_argument("--workers", type=int, default=30,
                       help="Concurrency")

    parser.add_argument("--test", action="store_true",
                       help="Test mode (only process first2records)")

    return parser.parse_args()


def main():
    args = parse_args()

    print("="*80)
    print("🎯 SFT Step 2: Expand to 240K Variants")
    print("="*80)
    print(f"📁 Input file: {args.input_file}")
    print(f"📝 Output file: {args.output_file}")
    print(f"💾 Cache file: {args.cache_file}")
    print(f"🤖 Model: {args.model}")
    print(f"🎨 Temperature: {args.temperature}")
    print(f"⚡ Concurrency: {args.workers}")
    print(f"🔢 Generate per task: 12 variants")
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

    remaining_records = [r for r in all_records if r.get('task_id', -1) not in processed_ids]

    if not remaining_records:
        print("✅ All records processed!")
        return

    print(f"\n📊 To process: {len(remaining_records)} records (total {len(all_records)} records)")
    print(f"Expected to generate: {len(remaining_records) * 12} variants")

    # Async processing
    results = asyncio.run(process_batch_with_semaphore(remaining_records, args))

    # Statistics
    success_count = sum(1 for r in results if r['success'])
    failed_count = len(results) - success_count
    total_variants = sum(r['num_variants'] for r in results if r['success'])

    print("\n" + "="*80)
    print("🎉 Step 2 Complete!")
    print("="*80)
    print(f"  📊 Total records: {len(all_records)}")
    print(f"  ✅ Succeeded in this run: {success_count} records")
    print(f"  ❌ Failed in this run: {failed_count} records")
    print(f"  📝 Variants generated: {total_variants} ")
    print(f"  📁 Output file: {args.output_file}")
    print("="*80)


if __name__ == "__main__":
    main()
