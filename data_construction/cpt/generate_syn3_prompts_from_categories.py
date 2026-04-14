"""
CPT Data Construction - Step 2: Generate Prompts from Category Tree

This script generates millions of HTML prompts from a hierarchical web category tree.
Based on the paper: 25 top-level categories → 80K leaf categories → 8M+ prompts

Pipeline:
1. Load category tree (25 top-level categories, 80K leaf nodes)
2. Load prompt templates (2000 templates from Qwen3-Coder)
3. For each leaf node, generate 100 prompts by filling template variables
4. Output: ~8M prompts for HTML code generation

Usage:
    python generate_prompts_from_categories.py \
        --category_tree category_tree.json \
        --templates prompt_templates.jsonl \
        --output_dir prompts_output/
"""

import json
import os
import random
import argparse
from typing import List, Dict, Any, Tuple
from multiprocessing import Process, Manager
import time


class CategoryPromptGenerator:
    """Generate prompts from category tree and templates."""

    def __init__(
        self,
        category_tree_path: str,
        template_path: str,
        output_dir: str,
        prompts_per_leaf: int = 100,
        seed: int = 42
    ):
        self.category_tree_path = category_tree_path
        self.template_path = template_path
        self.output_dir = output_dir
        self.prompts_per_leaf = prompts_per_leaf
        self.seed = seed

        os.makedirs(output_dir, exist_ok=True)

        # Initialize variable pools
        self.styles = self._init_styles()
        self.features = self._init_features()
        self.colors = self._init_colors()
        self.layouts = self._init_layouts()

    def _init_styles(self) -> List[str]:
        """Design style variations."""
        return [
            "minimalist", "modern", "clean", "elegant", "sleek",
            "refined", "streamlined", "polished", "professional", "balanced",
            "creative", "artistic", "expressive", "experimental", "playful",
            "innovative", "aesthetic", "bold", "dynamic", "trend-driven",
            "corporate", "luxury", "premium", "trustworthy", "formal",
            "approachable", "friendly", "casual", "youthful", "vibrant",
            "modern Asian", "Nordic", "Japanese simplicity", "European classic",
            "urban contemporary", "nature-inspired", "organic", "warm", "neutral", "calming",
            "Bootstrap style", "Tailwind style", "Ant Design style", "Material UI style"
        ]

    def _init_features(self) -> List[str]:
        """Feature specifications."""
        return [
            "responsive", "mobile-first", "adaptive layout", "fluid design",
            "animated", "interactive", "hover effects", "smooth transitions",
            "parallax scrolling", "scroll animations", "micro-interactions",
            "accessible", "ARIA compliant", "semantic HTML", "keyboard navigable",
            "lazy loaded", "optimized images", "progressive loading", "skeleton UI",
            "CSS Grid layout", "Flexbox layout", "CSS variables",
            "dark mode toggle", "SVG icons", "web fonts"
        ]

    def _init_colors(self) -> List[str]:
        """Color scheme variations."""
        return [
            "light theme", "dark theme", "high contrast", "low contrast",
            "warm tones", "cool tones", "earth tones", "pastel palette",
            "neon palette", "natural tones", "ocean-inspired", "sunset-inspired",
            "corporate blue and gray", "tech-inspired cyan", "eco green",
            "gradient-based palette", "glassmorphism colors", "duotone scheme"
        ]

    def _init_layouts(self) -> List[str]:
        """Layout type variations."""
        return [
            "single column", "two column", "three column", "multi-column",
            "sidebar layout", "split screen", "full-width", "boxed layout",
            "grid layout", "masonry layout", "card layout", "list layout",
            "F-pattern layout", "Z-pattern layout", "modular layout",
            "sticky header", "hero section layout"
        ]

    def load_category_tree(self) -> Dict[str, Any]:
        """Load category tree from JSON."""
        with open(self.category_tree_path, 'r') as f:
            return json.load(f)

    def load_templates(self) -> List[Dict[str, Any]]:
        """Load prompt templates."""
        templates = []
        with open(self.template_path, 'r') as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    if "prompt_template" in obj and "variables" in obj:
                        templates.append(obj)
        return templates

    def extract_leaf_paths(
        self,
        node: Dict[str, Any],
        current_path: List[str]
    ) -> List[Tuple[str, List[str]]]:
        """Recursively extract all leaf node paths."""
        paths = []

        if node.get('is_leaf', False) or not node.get('children'):
            path_str = ' > '.join(current_path[1:]) if len(current_path) > 1 else current_path[0]
            paths.append((path_str, current_path[1:] if len(current_path) > 1 else current_path))

        if 'children' in node and node['children']:
            for child in node['children']:
                child_name = child.get('name', child.get('category', ''))
                child_path = current_path + [child_name]
                paths.extend(self.extract_leaf_paths(child, child_path))

        return paths

    def fill_template(
        self,
        template_str: str,
        var_names: List[str],
        web_category: str,
        rng: random.Random
    ) -> str:
        """Fill template variables with random values."""
        prompt = template_str

        for var_name in var_names:
            placeholder = "{" + var_name + "}"

            if var_name == "web_category":
                value = web_category
            elif var_name == "color_scheme":
                value = rng.choice(self.colors)
            elif var_name == "design_style":
                value = rng.choice(self.styles)
            elif var_name == "features":
                value = rng.choice(self.features)
            elif var_name == "layout_type":
                value = rng.choice(self.layouts)
            else:
                continue

            prompt = prompt.replace(placeholder, value)

        return prompt

    def process_node(
        self,
        node: Dict[str, Any],
        node_index: int,
        templates: List[Dict[str, Any]],
        stats_queue
    ):
        """Process a single top-level node (runs in separate process)."""
        rng = random.Random(self.seed + node_index)
        node_name = node.get('name', node.get('category', f'Node_{node_index}'))

        # Extract leaf paths
        leaf_paths = self.extract_leaf_paths(node, ['ROOT', node_name])
        print(f"[Process {node_index}] Node: {node_name}, Leaves: {len(leaf_paths)}")

        # Output file
        node_name_safe = node_name.replace('/', '_').replace(' ', '_').replace('&', 'and')
        output_file = os.path.join(
            self.output_dir,
            f"prompts_node_{node_index:02d}_{node_name_safe}.jsonl"
        )

        # Generate prompts
        generated_count = 0
        with open(output_file, 'w') as f:
            for path_str, path_list in leaf_paths:
                for _ in range(self.prompts_per_leaf):
                    template_obj = rng.choice(templates)
                    prompt = self.fill_template(
                        template_obj['prompt_template'],
                        template_obj['variables'],
                        path_str,
                        rng
                    )

                    output_obj = {
                        'prompt': prompt,
                        'web_category': path_str,
                        'category_path': path_list
                    }
                    f.write(json.dumps(output_obj, ensure_ascii=False) + '\n')
                    generated_count += 1

        stats_queue.put({
            'node_index': node_index,
            'node_name': node_name,
            'leaf_count': len(leaf_paths),
            'generated_count': generated_count,
            'output_file': output_file
        })

        print(f"[Process {node_index}] Done! Generated {generated_count:,} prompts")

    def generate_all(self):
        """Main function: Generate all prompts using multiprocessing."""
        print("="*60)
        print("Generate Prompts from Category Tree")
        print("="*60)

        print("\n[1/4] Loading category tree...")
        tree = self.load_category_tree()

        print("[2/4] Loading templates...")
        templates = self.load_templates()
        print(f"      Loaded {len(templates)} templates")

        print("[3/4] Extracting top-level nodes...")
        if 'children' in tree:
            top_level_nodes = tree['children']
        else:
            top_level_nodes = [tree]
        print(f"      Found {len(top_level_nodes)} top-level nodes")

        print("[4/4] Starting multiprocess generation...\n")

        manager = Manager()
        stats_queue = manager.Queue()

        start_time = time.time()
        processes = []

        for i, node in enumerate(top_level_nodes):
            p = Process(
                target=self.process_node,
                args=(node, i, templates, stats_queue)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        elapsed = time.time() - start_time

        # Collect stats
        print("\n" + "="*60)
        print("Generation Complete!")
        print("="*60)

        all_stats = []
        while not stats_queue.empty():
            all_stats.append(stats_queue.get())

        all_stats.sort(key=lambda x: x['node_index'])

        total = 0
        for stat in all_stats:
            print(f"Node {stat['node_index']:2d} [{stat['node_name']}]: {stat['generated_count']:,} prompts")
            total += stat['generated_count']

        print("="*60)
        print(f"Total: {total:,} prompts")
        print(f"Time: {elapsed:.2f}s")
        print(f"Output: {os.path.abspath(self.output_dir)}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Generate prompts from category tree")

    parser.add_argument("--category_tree", type=str, required=True,
                        help="Category tree JSON file")
    parser.add_argument("--templates", type=str, required=True,
                        help="Prompt templates JSONL file")
    parser.add_argument("--output_dir", type=str, default="./prompts_output",
                        help="Output directory")
    parser.add_argument("--prompts_per_leaf", type=int, default=100,
                        help="Prompts per leaf node")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    generator = CategoryPromptGenerator(
        category_tree_path=args.category_tree,
        template_path=args.templates,
        output_dir=args.output_dir,
        prompts_per_leaf=args.prompts_per_leaf,
        seed=args.seed
    )

    generator.generate_all()


if __name__ == "__main__":
    main()
