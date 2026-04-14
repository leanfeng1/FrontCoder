"""
CPT Data Construction - Generate Category Tree

This script generates a fine-grained website category tree using LLM.
Based on the paper: 25 top-level categories → ~80K leaf categories (depth 5)

Pipeline:
1. Start with 25 root categories (defined in class.txt)
2. Recursively expand each category into subcategories using LLM
3. Generate up to depth 5, resulting in ~80K leaf nodes

Usage:
    python generate_category_tree.py \
        --class_file class.txt \
        --output category_tree.json \
        --max_depth 5 \
        --concurrency 8
"""

import os
import sys
import json
import argparse
import asyncio
import time
from typing import List, Dict, Any, Tuple
from openai import AsyncOpenAI


# 25 root categories for web content
DEFAULT_ROOT_CATEGORIES = [
    "News & Media",
    "Business & Finance",
    "Technology & Computing",
    "Education & Learning",
    "Health & Fitness",
    "Travel & Tourism",
    "Food & Beverage",
    "Shopping & E-Commerce",
    "Entertainment & Media",
    "Sports & Recreation",
    "Real Estate & Housing",
    "Automotive & Transport",
    "Science & Environment",
    "Government & Politics",
    "Law & Legal Services",
    "Arts & Culture",
    "Home & Lifestyle",
    "Career & Employment",
    "Religion & Spirituality",
    "Nonprofit & Social Impact",
    "Gaming",
    "Fashion & Apparel",
    "Pets & Animals",
    "Events & Conferences",
    "Personal & Portfolio"
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate website category tree via LLM")
    parser.add_argument("--class_file", type=str, default=None,
                        help="Path to class.txt (root categories). If not provided, uses default 25 categories")
    parser.add_argument("--output", type=str, default="category_tree.json",
                        help="Output JSON file for the tree")
    parser.add_argument("--stats_output", type=str, default="category_tree_stats.json",
                        help="Output JSON for statistics")
    parser.add_argument("--cache_file", type=str, default="category_tree_cache.json",
                        help="Cache file to resume")
    parser.add_argument("--max_depth", type=int, default=5,
                        help="Maximum depth of the tree (root depth=0)")
    parser.add_argument("--api_key", type=str, default="EMPTY",
                        help="API key for LLM")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1",
                        help="Base URL for LLM API")
    parser.add_argument("--model", type=str, default="Qwen3-480b",
                        help="Model name")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Max tokens for each call")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Max concurrent LLM requests")
    parser.add_argument("--max_children", type=int, default=15,
                        help="Maximum children per node")
    return parser.parse_args()


def load_root_categories(class_file: str = None) -> List[str]:
    """Load root categories from file or use defaults."""
    if class_file and os.path.exists(class_file):
        classes = []
        with open(class_file, 'r', encoding='utf-8') as f:
            for line in f:
                name = line.strip()
                if not name:
                    continue
                if ":" in name and name.split(":", 1)[0].lstrip("L").isdigit():
                    name = name.split(":", 1)[1].strip()
                classes.append(name)
        return classes
    return DEFAULT_ROOT_CATEGORIES


def make_node(name: str, depth: int) -> Dict[str, Any]:
    """Create a tree node."""
    return {
        "name": name,
        "depth": depth,
        "children": [],
        "is_leaf": False
    }


def scaffold_tree(roots: List[str]) -> Dict[str, Any]:
    """Create initial tree structure."""
    return {
        "name": "ROOT",
        "depth": 0,
        "children": [make_node(r, 1) for r in roots],
        "is_leaf": False,
        "meta": {"created_at": int(time.time())}
    }


def save_json(obj: Any, path: str) -> None:
    """Save object to JSON file."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[save] {path}")


def normalize_path_key(path_parts: List[str]) -> str:
    """Create unique key from path."""
    return "/".join([p.replace("/", "-") for p in path_parts])


def build_subcategory_prompt(category_name: str, parent_path: List[str], max_children: int) -> str:
    """Build prompt for subcategory generation."""
    path_str = " > ".join(parent_path + [category_name])
    return (
        f"You are an expert in website taxonomy and classification. "
        f"Generate a fine-grained list of subcategories for the current category.\n\n"
        f"Current category path: {path_str}\n\n"
        f"Requirements:\n"
        f"- Output **only** a valid JSON object: {{\"subcategories\": [\"sub1\", \"sub2\", ...]}}\n"
        f"- Subcategories must be real, specific website content topics under the current category\n"
        f"- Do not repeat the parent category or output overly broad terms\n"
        f"- Subcategories must be in English, without explanations or numbering\n"
        f"- Generate between 5 and {max_children} subcategories; do not exceed {max_children}\n"
        f"- If the category cannot be subdivided further, output {{\"subcategories\": []}}\n"
        f"- Do not include any other text outside the JSON object\n"
    )


def parse_subcategories(response_text: str, max_children: int) -> List[str]:
    """Parse subcategories from LLM response."""
    if not response_text:
        return []

    text = response_text.strip()
    # Remove code fences
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("\n", 1)[0]
    text = text.strip()

    # Try JSON parsing
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and isinstance(obj.get("subcategories"), list):
            items = [str(x).strip() for x in obj["subcategories"] if str(x).strip()]
            return items[:max_children]
        if isinstance(obj, list):
            items = [str(x).strip() for x in obj if str(x).strip()]
            return items[:max_children]
    except Exception:
        pass

    # Fallback: parse as lines
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    candidates = []
    for ln in lines:
        ln = ln.lstrip("-* ")
        ln = ln.split(". ", 1)[-1] if ln[:2].isdigit() else ln
        if ln:
            candidates.append(ln)
    return candidates[:max_children]


async def fetch_subcategories(
    client: AsyncOpenAI,
    category_name: str,
    parent_path: List[str],
    args: argparse.Namespace
) -> List[str]:
    """Fetch subcategories from LLM."""
    prompt = build_subcategory_prompt(category_name, parent_path, args.max_children)
    max_retries = 3

    for attempt in range(1, max_retries + 1):
        try:
            response = await client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=args.max_tokens,
                temperature=args.temperature
            )

            if response and response.choices:
                text = response.choices[0].message.content
                items = parse_subcategories(text, args.max_children)
                path_str = " > ".join(parent_path + [category_name])
                print(f"[LLM] {path_str}: {len(items)} subcategories")
                return items

        except Exception as e:
            print(f"[LLM] Attempt {attempt} failed for {category_name}: {e}")

        await asyncio.sleep(min(3 * attempt, 6))

    return []


def load_cache(cache_path: str) -> Dict[str, Any]:
    """Load cache file."""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {"nodes": {}}
    return {"nodes": {}}


def save_cache(cache_obj: Dict[str, Any], cache_path: str) -> None:
    """Save cache file."""
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache_obj, f, ensure_ascii=False, indent=2)


def compute_tree_stats(tree: Dict[str, Any]) -> Dict[str, Any]:
    """Compute tree statistics."""
    depth_counts = {}
    leaf_count = 0
    total_nodes = 0
    max_depth_seen = 0
    branching_factors = []

    def dfs(node):
        nonlocal leaf_count, total_nodes, max_depth_seen
        total_nodes += 1
        d = node.get("depth", 0)
        max_depth_seen = max(max_depth_seen, d)
        depth_counts[d] = depth_counts.get(d, 0) + 1
        children = node.get("children", [])
        branching_factors.append(len(children))
        if not children or node.get("is_leaf", False):
            leaf_count += 1
        for ch in children:
            dfs(ch)

    dfs(tree)
    avg_branch = sum(branching_factors) / len(branching_factors) if branching_factors else 0.0

    return {
        "total_nodes": total_nodes,
        "leaf_count": leaf_count,
        "max_depth": max_depth_seen,
        "depth_counts": {str(k): v for k, v in sorted(depth_counts.items())},
        "avg_branching_factor": round(avg_branch, 3)
    }


def get_pending_nodes(tree: Dict[str, Any], max_depth: int) -> List[Tuple[Dict, List[str]]]:
    """Get nodes that need expansion."""
    pending = []

    def dfs(node, path):
        d = node.get("depth", 0)
        children = node.get("children", [])
        is_leaf = node.get("is_leaf", False)

        if d >= 1 and d < max_depth and not is_leaf and len(children) == 0:
            pending.append((node, path[:-1]))

        for ch in children:
            dfs(ch, path + [ch["name"]])

    dfs(tree, [])
    return pending


async def expand_node(
    node: Dict,
    path: List[str],
    cache: Dict,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    args: argparse.Namespace
) -> None:
    """Expand a single node."""
    key = normalize_path_key(path + [node["name"]])
    cached = cache.get("nodes", {}).get(key)

    if cached is not None:
        subcats = list(map(str, cached.get("children", [])))
    else:
        async with semaphore:
            subcats = await fetch_subcategories(client, node["name"], path, args)
        cache.setdefault("nodes", {})[key] = {
            "children": subcats,
            "is_leaf": len(subcats) == 0
        }

    if len(subcats) == 0:
        node["is_leaf"] = True
        node["children"] = []
    else:
        node["children"] = [make_node(sc, node["depth"] + 1) for sc in subcats]


async def expand_tree(
    tree: Dict,
    client: AsyncOpenAI,
    args: argparse.Namespace,
    cache_path: str,
    out_path: str,
    stats_path: str
) -> None:
    """Recursively expand entire tree."""
    cache = load_cache(cache_path)
    semaphore = asyncio.Semaphore(args.concurrency)
    wave = 1

    while True:
        pending = get_pending_nodes(tree, args.max_depth)
        if not pending:
            print("[expand] No more nodes to expand")
            break

        print(f"[expand] Wave {wave}: {len(pending)} nodes to expand")

        tasks = [
            expand_node(node, path, cache, client, semaphore, args)
            for node, path in pending
        ]
        await asyncio.gather(*tasks)

        # Save after each wave
        save_cache(cache, cache_path)
        save_json(tree, out_path)
        stats = compute_tree_stats(tree)
        save_json(stats, stats_path)
        print(f"[expand] Wave {wave} complete: {stats['total_nodes']} nodes, {stats['leaf_count']} leaves")

        wave += 1


def main():
    args = parse_args()

    print("=" * 60)
    print("Category Tree Generator")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Max Depth: {args.max_depth}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Max Children: {args.max_children}")
    print("=" * 60)

    # Load root categories
    roots = load_root_categories(args.class_file)
    print(f"\nLoaded {len(roots)} root categories")
    for i, r in enumerate(roots, 1):
        print(f"  {i:2d}. {r}")

    # Build initial tree
    tree = scaffold_tree(roots)
    save_json(tree, args.output)

    # Initialize client
    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=120.0
    )

    # Expand tree
    print("\nStarting tree expansion...")
    asyncio.run(expand_tree(
        tree, client, args,
        args.cache_file, args.output, args.stats_output
    ))

    # Final stats
    stats = compute_tree_stats(tree)
    save_json(tree, args.output)
    save_json(stats, args.stats_output)

    print("\n" + "=" * 60)
    print("Category Tree Generation Complete!")
    print("=" * 60)
    print(f"  Total nodes: {stats['total_nodes']:,}")
    print(f"  Leaf nodes: {stats['leaf_count']:,}")
    print(f"  Max depth: {stats['max_depth']}")
    print(f"  Avg branching: {stats['avg_branching_factor']}")
    print(f"  Output: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
