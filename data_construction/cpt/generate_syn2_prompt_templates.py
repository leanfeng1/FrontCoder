"""
CPT Data Construction - Step 1: Generate Prompt Templates

This script generates HTML prompt templates using Qwen3-Coder.
Based on the paper: Uses LLM to create diverse prompt templates.

Each template contains:
- {web_category}: Hierarchical web category path
- {design_style}: Design style variable
- {features}: Feature specifications
- {color_scheme}: Color scheme
- {layout_type}: Layout type

Usage:
    python generate_prompt_templates.py \
        --output prompt_templates.jsonl \
        --num_samples 2000 \
        --workers 500
"""

import json
import os
import asyncio
import argparse
import random
import re
from openai import AsyncOpenAI
from tqdm import tqdm


# System prompt for template generation
SYSTEM_PROMPT = """You are an expert front-end designer and prompt engineer. Please generate exactly ONE English prompt template for generating beautiful, practical, and fully functional HTML front-end code.

Requirements:
1. The template MUST include {web_category}, which represents a hierarchical web category ending with a specific page function or purpose
   (e.g., "Gaming > Adventure Games > Action-Adventure Games > Tutorial/Help Page" or "Shopping & E-Commerce > Sports Nutrition > Product Listing Page").

2. In addition to {web_category}, the template MUST include **2 or 3 or 4** of the following variables (use curly braces {} to wrap them):
   - {design_style} - Design style (e.g., modern, minimalist, retro)
   - {features} - Feature specifications (e.g., responsive, interactive)
   - {color_scheme} - Color scheme (e.g., dark mode, pastel colors)
   - {layout_type} - Layout type (e.g., grid, flexbox, single column)
   Do NOT include all variables. Choose only the most relevant ones for a realistic and coherent web page scenario.

3. The template should be detailed and specific, describing the type of web page, its purpose, and how it should look or behave.

4. The output of this request must be **the prompt template itself** (no explanations, no numbering, no examples).

5. The template should include **positive and detailed quality requirements** about the HTML code, such as:
  - visually appealing design
  - responsive and accessible layout
  - clean and semantic structure
  - modern components or good UX practices
  - readable indentation and organized structure

6. The template must instruct the model to automatically determine appropriate interaction modes, component libraries and choose suitable front-end technologies or libraries that best fit the specific webpage context.

7. The template must clearly instruct that the model should **output only the HTML code** and nothing else.

8. The template should be in natural, clear, and professional English.

Now generate ONE unique prompt template following these requirements:"""


def extract_variables(template: str) -> list:
    """Extract variables from template."""
    pattern = r'\{([^}]+)\}'
    matches = re.findall(pattern, template)
    return sorted(list(set(matches)))


def validate_template(template: str, variables: list) -> bool:
    """Validate template meets requirements."""
    if len(variables) < 2:
        return False
    if 'web_category' not in variables:
        return False
    if not template or len(template.strip()) < 20:
        return False
    return True


async def generate_single_template(
    client: AsyncOpenAI,
    model: str,
    temperature: float,
    semaphore: asyncio.Semaphore
) -> dict:
    """Generate a single prompt template."""
    async with semaphore:
        max_retries = 3
        for retry in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": SYSTEM_PROMPT}],
                    max_tokens=1024,
                    temperature=temperature
                )

                template = response.choices[0].message.content.strip()

                # Clean markdown
                template = re.sub(r'^```.*?\n', '', template)
                template = re.sub(r'\n```$', '', template)
                template = template.strip()

                variables = extract_variables(template)

                if validate_template(template, variables):
                    return {
                        "prompt_template": template,
                        "variables": variables
                    }

            except Exception as e:
                if retry == max_retries - 1:
                    print(f"Generation error: {e}")
                await asyncio.sleep(random.uniform(0.5, 1.5))

        return None


async def generate_all_templates(
    num_samples: int,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_concurrent: int,
    output_file: str
):
    """Generate all prompt templates."""
    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=120)
    semaphore = asyncio.Semaphore(max_concurrent)

    # Clear output file
    open(output_file, 'w').close()

    async def generate_and_save(idx: int):
        result = await generate_single_template(client, model, temperature, semaphore)
        if result:
            with open(output_file, 'a') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
            return True
        return False

    tasks = [generate_and_save(i) for i in range(num_samples)]

    success_count = 0
    for coro in tqdm(asyncio.as_completed(tasks), total=num_samples,
                     desc="Generating templates"):
        if await coro:
            success_count += 1

    return success_count


def main():
    parser = argparse.ArgumentParser(description="Generate HTML prompt templates")

    parser.add_argument("--output", type=str, default="prompt_templates.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--num_samples", type=int, default=2000,
                        help="Number of templates to generate")
    parser.add_argument("--workers", type=int, default=500,
                        help="Number of concurrent workers")
    parser.add_argument("--base_url", type=str,
                        default="http://localhost:8000/v1",
                        help="LLM API base URL")
    parser.add_argument("--api_key", type=str, default="EMPTY",
                        help="API key")
    parser.add_argument("--model", type=str, default="Qwen3-Coder-480b",
                        help="Model name")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Generation temperature")

    args = parser.parse_args()

    print(f"{'='*60}")
    print("HTML Prompt Template Generator")
    print(f"{'='*60}")
    print(f"Output: {args.output}")
    print(f"Templates: {args.num_samples}")
    print(f"Workers: {args.workers}")
    print(f"Model: {args.model}")
    print(f"{'='*60}\n")

    success = asyncio.run(generate_all_templates(
        args.num_samples,
        args.base_url,
        args.api_key,
        args.model,
        args.temperature,
        args.workers,
        args.output
    ))

    print(f"\n{'='*60}")
    print(f"Complete! Generated {success}/{args.num_samples} templates")
    print(f"Saved to: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
