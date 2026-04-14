"""
FrontCoder HTML Reward Function

Vision-grounded reward computation for front-end code generation RL training.

Reward Formula (from paper):
    R(y) = I_rep(y) * I_render(y) * (α * S_chk + β * S_sim + γ * S_len)

Where:
    - I_rep(y): Repetition indicator (0 or 1), detects degenerate repetition
    - I_render(y): Render indicator (0 or 1), checks if HTML renders successfully
    - S_chk: Checklist score (0-1), 20-item verification via VLM
    - S_sim: Similarity score = 0.5 * S_struct + 0.5 * S_sem
    - S_len: Length score with thresholds L_min=12K, L_max=16K
    - α=0.6, β=0.3, γ=0.1 (reward weights)

Requirements:
    - HTTP render service running (for sandboxed HTML rendering)
    - VLM service (Qwen2.5-VL-72B recommended) for checklist scoring
"""

import asyncio
import aiohttp
import json
import base64
import io
import os
import re
import time
import gc
import difflib
from typing import List, Optional, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from lxml import html as lxml_html
from openai import OpenAI


class HTMLRewardJudge:
    """
    HTML Reward Judge for RL training.

    Implements the composite reward function from the paper with:
    1. Repetition detection (I_rep)
    2. Render validation (I_render)
    3. Checklist scoring via VLM (S_chk)
    4. Structural/semantic similarity (S_sim)
    5. Length regularization (S_len)
    """

    def __init__(
        self,
        render_service_url: str = "http://localhost:8768",
        vlm_base_url: str = "http://localhost:8000/v1",
        vlm_api_key: str = "EMPTY",
        vlm_model: str = "Qwen2.5-VL-72B-Instruct",
        vlm_timeout: int = 120,
        max_concurrent_renders: int = 1024,
        max_concurrent_vlm: int = 1024,
        save_dir: Optional[str] = None,
        enable_save: bool = False
    ):
        """
        Initialize the HTML Reward Judge.

        Args:
            render_service_url: URL of the HTTP render service
            vlm_base_url: Base URL for VLM API (OpenAI-compatible)
            vlm_api_key: API key for VLM service
            vlm_model: VLM model name
            vlm_timeout: Timeout for VLM requests (seconds)
            max_concurrent_renders: Max concurrent render requests
            max_concurrent_vlm: Max concurrent VLM requests
            save_dir: Directory to save renders (optional)
            enable_save: Whether to save HTML/screenshots
        """
        # VLM client configuration
        self.vlm_client = OpenAI(
            base_url=vlm_base_url,
            api_key=vlm_api_key,
            timeout=vlm_timeout
        )
        self.vlm_model = vlm_model

        # Render service configuration
        self.render_service_url = render_service_url.rstrip('/')

        # Concurrency control
        self.max_concurrent_renders = max_concurrent_renders
        self.max_concurrent_vlm = max_concurrent_vlm
        self._render_semaphores = {}
        self._vlm_semaphores = {}
        self._http_sessions = {}

        # VLM thread pool
        self._vlm_thread_pool = ThreadPoolExecutor(
            max_workers=min(max_concurrent_vlm, 1024)
        )

        # Save configuration
        self.save_dir = save_dir
        self.enable_save = enable_save
        if enable_save and save_dir:
            os.makedirs(save_dir, exist_ok=True)

        print(f"HTMLRewardJudge initialized:")
        print(f"  Render service: {render_service_url}")
        print(f"  VLM service: {vlm_base_url}")
        print(f"  Max concurrent renders: {max_concurrent_renders}")
        print(f"  Max concurrent VLM: {max_concurrent_vlm}")

    @property
    def render_semaphore(self):
        """Get render semaphore for current event loop."""
        loop_id = id(asyncio.get_event_loop())
        if loop_id not in self._render_semaphores:
            self._render_semaphores[loop_id] = asyncio.Semaphore(
                self.max_concurrent_renders
            )
        return self._render_semaphores[loop_id]

    @property
    def vlm_semaphore(self):
        """Get VLM semaphore for current event loop."""
        loop_id = id(asyncio.get_event_loop())
        if loop_id not in self._vlm_semaphores:
            self._vlm_semaphores[loop_id] = asyncio.Semaphore(
                self.max_concurrent_vlm
            )
        return self._vlm_semaphores[loop_id]

    async def get_http_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session for current event loop."""
        loop_id = id(asyncio.get_event_loop())
        if loop_id not in self._http_sessions or self._http_sessions[loop_id].closed:
            connector = aiohttp.TCPConnector(
                limit=128,
                limit_per_host=1024,
                ttl_dns_cache=300
            )
            timeout = aiohttp.ClientTimeout(total=30, connect=5)
            self._http_sessions[loop_id] = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            )
        return self._http_sessions[loop_id]

    # ==================== HTML Extraction ====================

    def extract_html_from_code_block(self, content: str) -> str:
        """
        Extract HTML/SVG content from text (handles code blocks, tags, etc.)

        Args:
            content: Raw text containing HTML code

        Returns:
            Extracted HTML string or empty string if extraction fails
        """
        if not content or not content.strip():
            return ""

        # Method 1: Extract from ```html``` code blocks
        code_block_pattern = r'```(?:html|svg|xml)\s*\n(.*?)\n```'
        matches = re.findall(code_block_pattern, content, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[0].strip()

        # Method 2: Extract <!DOCTYPE html>...</html>
        doctype_pattern = r'(<!DOCTYPE\s+html[^>]*>.*?</html>)'
        matches = re.findall(doctype_pattern, content, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[0].strip()

        # Method 3: Extract <html>...</html>
        html_pattern = r'(<html[^>]*>.*?</html>)'
        matches = re.findall(html_pattern, content, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[0].strip()

        # Method 4: Extract <svg>...</svg> and wrap in HTML
        if '<svg' in content.lower():
            svg_pattern = r'(<svg[^>]*>.*?</svg>)'
            matches = re.findall(svg_pattern, content, re.DOTALL | re.IGNORECASE)
            if matches:
                return f'''<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;display:flex;justify-content:center;align-items:center;min-height:100vh;">
{matches[0]}
</body>
</html>'''

        # Method 5: Extract <body>...</body>
        body_pattern = r'<body[^>]*>(.*?)</body>'
        matches = re.findall(body_pattern, content, re.DOTALL | re.IGNORECASE)
        if matches:
            return f'''<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body>{matches[0].strip()}</body>
</html>'''

        return ""

    # ==================== Repetition Detection ====================

    def check_repetition(
        self,
        html_code: str,
        window_size: int = 8000,
        step_size: int = 4000,
        local_threshold: float = 0.65,
        jaccard_threshold: float = 0.85,
        ngram_size: int = 8
    ) -> int:
        """
        Check for degenerate repetition using sliding window n-gram analysis.

        Based on paper: Uses sliding window + n-gram method for efficient detection.

        Args:
            html_code: HTML code to check
            window_size: Window size in characters
            step_size: Step size for sliding window
            local_threshold: Threshold for within-window repetition
            jaccard_threshold: Threshold for between-window similarity
            ngram_size: Size of n-grams

        Returns:
            1 if no repetition detected, 0 if repetition detected
        """
        try:
            text = self.extract_html_from_code_block(html_code)
            if not text or len(text) < 100:
                return 1  # Too short to judge

            # Create sliding windows
            windows = []
            start = 0
            while start < len(text):
                end = min(start + window_size, len(text))
                window = text[start:end]
                if len(window) >= 100:
                    windows.append(window)
                start += step_size

            if not windows:
                return 1

            # Check each window for internal repetition
            window_ngram_sets = []
            for window in windows:
                ngrams = [window[i:i+ngram_size] for i in range(len(window) - ngram_size + 1)]
                if not ngrams:
                    window_ngram_sets.append(set())
                    continue

                unique_ngrams = set(ngrams)
                rep_ratio = 1.0 - len(unique_ngrams) / len(ngrams)

                if rep_ratio >= local_threshold:
                    return 0  # High internal repetition

                window_ngram_sets.append(unique_ngrams)

            # Check between-window similarity
            for i in range(len(window_ngram_sets)):
                for j in range(i + 1, min(i + 3, len(window_ngram_sets))):
                    set_i = window_ngram_sets[i]
                    set_j = window_ngram_sets[j]

                    if not set_i or not set_j:
                        continue

                    intersection = len(set_i & set_j)
                    union = len(set_i | set_j)
                    jaccard = intersection / union if union > 0 else 0

                    if jaccard >= jaccard_threshold:
                        return 0  # High between-window similarity

            return 1  # No repetition detected

        except Exception as e:
            print(f"Repetition check error: {e}")
            return 1  # Default to no repetition on error

    # ==================== Length Score ====================

    def check_length_control(self, html_code: str) -> float:
        """
        Calculate length score S_len based on paper formula.

        S_len(y) = {
            1.0               if L(y) <= L_min
            (L_max - L(y)) / (L_max - L_min)  if L_min < L(y) < L_max
            0.0               if L(y) >= L_max
        }

        Where L_min = 12K tokens, L_max = 16K tokens (paper specification)

        Args:
            html_code: HTML code to evaluate

        Returns:
            Length score between 0 and 1
        """
        try:
            text = self.extract_html_from_code_block(html_code)
            if not text:
                return 1.0

            # Estimate tokens (approximately 0.75 tokens per character)
            estimated_tokens = len(text) * 0.75

            # Paper thresholds
            L_MIN = 12000  # 12K tokens
            L_MAX = 16000  # 16K tokens

            if estimated_tokens <= L_MIN:
                return 1.0
            elif estimated_tokens >= L_MAX:
                return 0.0
            else:
                return (L_MAX - estimated_tokens) / (L_MAX - L_MIN)

        except Exception:
            return 1.0

    # ==================== Similarity Score ====================

    def calculate_html_similarity(
        self,
        generated_html: str,
        reference_html: str
    ) -> float:
        """
        Calculate HTML similarity score S_sim.

        S_sim = 0.5 * S_struct + 0.5 * S_sem

        Where:
        - S_struct: Structural similarity (DOM tree LCS ratio)
        - S_sem: Semantic similarity (semantic role matching)

        Args:
            generated_html: Generated HTML code
            reference_html: Reference HTML code

        Returns:
            Similarity score between 0 and 1
        """
        try:
            gen_html = self.extract_html_from_code_block(generated_html)
            ref_html = self.extract_html_from_code_block(reference_html)

            if not gen_html or not ref_html:
                return 0.0

            # Parse HTML
            gen_tree = lxml_html.fromstring(gen_html)
            ref_tree = lxml_html.fromstring(ref_html)

            # Calculate structural similarity
            s_struct = self._calculate_structure_similarity(gen_tree, ref_tree)

            # Calculate semantic similarity
            s_sem = self._calculate_semantic_similarity(gen_tree, ref_tree)

            # Combined score (paper formula)
            return 0.5 * s_struct + 0.5 * s_sem

        except Exception as e:
            print(f"Similarity calculation error: {e}")
            return 0.0

    def _calculate_structure_similarity(self, tree1, tree2) -> float:
        """Calculate structural similarity using LCS on DOM sequence."""
        seq1 = self._tree_to_sequence(tree1)
        seq2 = self._tree_to_sequence(tree2)

        if not seq1 and not seq2:
            return 1.0
        if not seq1 or not seq2:
            return 0.0

        # LCS length
        lcs_len = self._lcs_length(seq1, seq2)
        max_len = max(len(seq1), len(seq2))

        return lcs_len / max_len if max_len > 0 else 0.0

    def _tree_to_sequence(self, element, seq=None):
        """Convert DOM tree to inorder sequence of tag-attribute pairs."""
        if seq is None:
            seq = []

        try:
            tag = element.tag if not callable(element.tag) else str(element.tag())
            attrs = dict(element.attrib) if hasattr(element, 'attrib') else {}
            seq.append(f"{tag.lower()}_{sorted(attrs.items())}")

            for child in element:
                self._tree_to_sequence(child, seq)
        except:
            pass

        return seq

    def _lcs_length(self, seq1, seq2) -> int:
        """Calculate LCS length using dynamic programming."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]

    def _calculate_semantic_similarity(self, tree1, tree2) -> float:
        """Calculate semantic similarity using role matching."""
        roles = ['header', 'footer', 'nav', 'main', 'sidebar', 'ad']

        def get_role_elements(tree):
            elements = {}
            for role in roles:
                elements[role] = []
                for elem in tree.iter():
                    if hasattr(elem, 'tag') and role in str(elem.tag).lower():
                        elements[role].append(str(elem.tag))
                    if hasattr(elem, 'attrib'):
                        for attr_val in elem.attrib.values():
                            if role in str(attr_val).lower():
                                elements[role].append(str(elem.tag))
                                break
            return elements

        map1 = get_role_elements(tree1)
        map2 = get_role_elements(tree2)

        total_sim = 0
        for role in roles:
            e1 = map1.get(role, [])
            e2 = map2.get(role, [])
            if e1 and e2:
                matcher = difflib.SequenceMatcher(None, e1, e2)
                total_sim += matcher.ratio()
            elif not e1 and not e2:
                total_sim += 1.0

        return total_sim / len(roles)

    # ==================== VLM Checklist Scoring ====================

    async def call_vlm_checklist_judge(
        self,
        question: str,
        checklist: List[Dict],
        image_base64: Optional[str] = None,
        html_code: Optional[str] = None
    ) -> List[float]:
        """
        Call VLM to score 20-item checklist in a single request.

        Args:
            question: User requirement/question
            checklist: List of 20 checklist items with title/description/maxScore
            image_base64: Base64 encoded screenshot (optional)
            html_code: HTML code to evaluate

        Returns:
            List of 20 scores (0-5 each)
        """
        async with self.vlm_semaphore:
            # Build checklist text
            checklist_text = ""
            for i, item in enumerate(checklist[:20]):
                if isinstance(item, dict):
                    title = item.get('title', f'Item {i+1}')
                    desc = item.get('description', '')
                    max_score = item.get('maxScore', 5)
                    checklist_text += f"{i+1}. {title} (max {max_score})\n   {desc}\n\n"

            # Extract HTML
            extracted_html = self.extract_html_from_code_block(html_code) if html_code else ""
            if not extracted_html and html_code:
                extracted_html = html_code[:10000]  # Fallback to raw text

            # Build prompt
            if image_base64 and extracted_html:
                prompt = f"""You are an HTML/UI expert. Score the following HTML render based on the requirements and checklist.

User Requirement: {question}

HTML Code:
```html
{extracted_html}
```

Scoring Checklist (20 items, 0-5 points each):
{checklist_text}

Based on the rendered image and HTML code, score each item 0-5.

Output ONLY a JSON object with a "scores" array of 20 integers:
{{"scores": [score1, score2, ..., score20]}}"""

                content = [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                ]
            elif extracted_html:
                prompt = f"""You are an HTML/UI expert. Score the following HTML code based on the requirements and checklist.

User Requirement: {question}

HTML Code:
```html
{extracted_html}
```

Scoring Checklist (20 items, 0-5 points each):
{checklist_text}

Note: Rendering failed, score based on code only.

Output ONLY a JSON object with a "scores" array of 20 integers:
{{"scores": [score1, score2, ..., score20]}}"""

                content = [{"type": "text", "text": prompt}]
            else:
                return [0.0] * 20

            # Call VLM
            try:
                def call_vlm_sync():
                    response = self.vlm_client.chat.completions.create(
                        model=self.vlm_model,
                        messages=[{"role": "user", "content": content}],
                        temperature=0.1,
                        max_tokens=512
                    )
                    return response.choices[0].message.content.strip()

                loop = asyncio.get_event_loop()
                response_text = await loop.run_in_executor(
                    self._vlm_thread_pool, call_vlm_sync
                )

                return self._parse_json_scores(response_text)

            except Exception as e:
                print(f"VLM call error: {e}")
                return [0.0] * 20

    def _parse_json_scores(self, response_text: str) -> List[float]:
        """Parse JSON scores from VLM response."""
        try:
            # Clean markdown
            cleaned = response_text.strip()
            if cleaned.startswith('```'):
                cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
                cleaned = re.sub(r'\s*```$', '', cleaned)

            # Find JSON
            json_match = re.search(r'\{[^{}]*"scores"\s*:\s*\[[^\]]+\][^{}]*\}', cleaned)
            if json_match:
                data = json.loads(json_match.group())
                scores = data.get("scores", [])
                if len(scores) >= 20:
                    return [max(0.0, min(5.0, float(s))) for s in scores[:20]]

            # Fallback: find numbers
            numbers = re.findall(r'\b([0-5])\b', response_text)
            if len(numbers) >= 20:
                return [float(n) for n in numbers[:20]]

            return [0.0] * 20

        except Exception:
            return [0.0] * 20

    # ==================== HTML Rendering ====================

    async def render_html_to_image(
        self,
        html_code: str,
        width: int = 800,
        height: int = 600
    ) -> Optional[Image.Image]:
        """
        Render HTML to image using HTTP render service.

        Args:
            html_code: HTML code to render
            width: Viewport width
            height: Viewport height

        Returns:
            PIL Image or None if rendering fails
        """
        async with self.render_semaphore:
            try:
                extracted_html = self.extract_html_from_code_block(html_code)
                if not extracted_html:
                    return None

                # Save HTML to temp file
                import tempfile
                with tempfile.NamedTemporaryFile(
                    mode='w', suffix='.html', delete=False
                ) as f:
                    f.write(extracted_html)
                    html_path = f.name

                screenshot_path = html_path.replace('.html', '.png')

                # Call render service
                session = await self.get_http_session()
                async with session.post(
                    f"{self.render_service_url}/render",
                    json={
                        "html_filepath": html_path,
                        "screenshot_filepath": screenshot_path,
                        "width": width,
                        "height": height,
                        "timeout": 20000
                    }
                ) as response:
                    if response.status != 200:
                        return None

                    result = await response.json()
                    if not result.get('success'):
                        return None

                # Load image
                if os.path.exists(screenshot_path):
                    image = Image.open(screenshot_path)
                    # Cleanup
                    os.unlink(html_path)
                    os.unlink(screenshot_path)
                    return image

                return None

            except Exception as e:
                print(f"Render error: {e}")
                return None

    def image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format='PNG', optimize=True)
        return base64.b64encode(buffer.getvalue()).decode()


# ==================== Main Reward Function ====================

# Global judge instance
_html_judge = None


def get_html_judge():
    """Get or create global HTMLRewardJudge instance."""
    global _html_judge
    if _html_judge is None:
        _html_judge = HTMLRewardJudge(
            render_service_url=os.environ.get(
                "RENDER_SERVICE_URL", "http://localhost:8768"
            ),
            vlm_base_url=os.environ.get(
                "VLM_BASE_URL", "http://localhost:8000/v1"
            ),
            vlm_api_key=os.environ.get("VLM_API_KEY", "EMPTY"),
            vlm_model=os.environ.get("VLM_MODEL", "Qwen2.5-VL-72B-Instruct"),
            max_concurrent_renders=1024,
            max_concurrent_vlm=1024
        )
    return _html_judge


def compute_html_reward_batch(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[str],
    extra_infos: Optional[List[Dict]] = None,
    **kwargs
) -> List[float]:
    """
    Batch reward computation for HTML generation RL training.

    Reward Formula:
        R(y) = I_rep(y) * I_render(y) * (α * S_chk + β * S_sim + γ * S_len)

    Where:
        - α = 0.6 (checklist weight)
        - β = 0.3 (similarity weight)
        - γ = 0.1 (length weight)

    Args:
        data_sources: List of data source identifiers
        solution_strs: List of generated HTML codes
        ground_truths: List of reference HTML codes
        extra_infos: List of dicts with 'question' and 'checklist'

    Returns:
        List of reward scores
    """
    judge = get_html_judge()
    batch_size = len(data_sources)

    if extra_infos is None:
        extra_infos = [{}] * batch_size

    print(f"\n{'='*60}")
    print(f"Computing rewards for batch of {batch_size} samples")
    print(f"Formula: R = I_rep * I_render * (0.6*S_chk + 0.3*S_sim + 0.1*S_len)")
    print(f"{'='*60}\n")

    async def compute_rewards_async():
        # Phase 1: Preprocessing (repetition check + HTML extraction)
        print("[Phase 1/3] Preprocessing (repetition + extraction)...")
        sample_infos = []
        render_indices = []

        for i in range(batch_size):
            solution = solution_strs[i]
            extra = extra_infos[i]

            # Parse question
            question = extra.get("question", "")
            if isinstance(question, list) and question:
                question = question[0].get("content", str(question[0])) \
                    if isinstance(question[0], dict) else str(question[0])

            # Parse checklist
            checklist = extra.get("checklist", [])
            if isinstance(checklist, str):
                try:
                    checklist = json.loads(checklist)
                except:
                    checklist = []

            # Repetition check
            i_rep = judge.check_repetition(solution)

            # HTML extraction
            extracted = judge.extract_html_from_code_block(solution)
            i_extract = 1 if extracted else 0

            sample_infos.append({
                'solution': solution,
                'extracted': extracted,
                'question': question,
                'checklist': checklist,
                'i_rep': i_rep,
                'i_extract': i_extract
            })

            if i_rep == 1 and i_extract == 1:
                render_indices.append(i)

        rep_pass = sum(1 for s in sample_infos if s['i_rep'] == 1)
        print(f"  Repetition check: {rep_pass}/{batch_size} passed")
        print(f"  Samples to render: {len(render_indices)}")

        # Phase 2: Rendering
        print(f"\n[Phase 2/3] Rendering {len(render_indices)} samples...")
        render_results = {}

        async def render_single(idx):
            html = sample_infos[idx]['extracted']
            image = await judge.render_html_to_image(html)
            return idx, image

        if render_indices:
            tasks = [render_single(i) for i in render_indices]
            results = await asyncio.gather(*tasks)
            for idx, image in results:
                render_results[idx] = image

        render_success = sum(1 for img in render_results.values() if img)
        print(f"  Render success: {render_success}/{len(render_indices)}")

        # Phase 3: VLM Scoring
        vlm_indices = [i for i in range(batch_size) if sample_infos[i]['i_rep'] == 1]
        print(f"\n[Phase 3/3] VLM scoring {len(vlm_indices)} samples...")

        async def score_single(idx):
            info = sample_infos[idx]
            image = render_results.get(idx)
            image_b64 = judge.image_to_base64(image) if image else None

            scores = await judge.call_vlm_checklist_judge(
                question=info['question'],
                checklist=info['checklist'],
                image_base64=image_b64,
                html_code=info['extracted'] or info['solution']
            )
            return idx, scores

        vlm_scores = {}
        if vlm_indices:
            tasks = [score_single(i) for i in vlm_indices]
            results = await asyncio.gather(*tasks)
            for idx, scores in results:
                vlm_scores[idx] = scores

        # Phase 4: Compute final rewards
        print(f"\n[Computing final rewards...]")
        rewards = []
        alpha, beta, gamma = 0.6, 0.3, 0.1

        for i in range(batch_size):
            info = sample_infos[i]

            if info['i_rep'] == 0:
                # Failed repetition check
                rewards.append(0.0)
                continue

            # Get scores
            scores = vlm_scores.get(i, [0.0] * 20)
            s_chk = sum(scores) / 100.0  # Normalize to 0-1

            # Similarity score
            gt = ground_truths[i] if i < len(ground_truths) else ""
            s_sim = judge.calculate_html_similarity(info['solution'], gt) if gt else 0.0

            # Length score
            s_len = judge.check_length_control(info['solution'])

            # Render indicator
            i_render = 1 if i in render_results and render_results[i] else 0

            # Final reward
            if info['i_extract'] == 0 or i_render == 0:
                # No valid render, reduce but don't zero
                reward = alpha * s_chk + gamma * s_len
            else:
                reward = alpha * s_chk + beta * s_sim + gamma * s_len

            rewards.append(reward)

        return rewards

    # Run async computation
    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                lambda: asyncio.run(compute_rewards_async())
            )
            rewards = future.result(timeout=600)
    except RuntimeError:
        rewards = asyncio.run(compute_rewards_async())

    # Summary
    print(f"\n{'='*60}")
    print(f"Reward computation complete!")
    print(f"  Mean reward: {sum(rewards)/len(rewards):.4f}")
    print(f"  Min/Max: {min(rewards):.4f} / {max(rewards):.4f}")
    print(f"{'='*60}\n")

    # Cleanup
    gc.collect()

    return rewards
