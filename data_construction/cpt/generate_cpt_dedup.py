"""
CPT Data Construction - MinHash Deduplication

This script implements MinHash-based deduplication for CPT data.
Removes near-duplicate HTML code responses to ensure data quality.

Algorithm:
1. Generate k MinHash signatures for each response
2. Use LSH (Locality Sensitive Hashing) for candidate pair detection
3. Compute exact Jaccard similarity for candidates
4. Remove duplicates above threshold

Usage:
    python generate_cpt_dedup.py \
        --input cpt_raw.jsonl \
        --output cpt_deduped.jsonl \
        --threshold 0.7 \
        --num_perm 128
"""

import json
import argparse
import hashlib
from typing import List, Set, Dict, Tuple
from collections import defaultdict
from tqdm import tqdm
import numpy as np


def get_shingles(text: str, k: int = 5) -> Set[str]:
    """
    Generate k-shingles (character n-grams) from text.

    Args:
        text: Input text
        k: Shingle size

    Returns:
        Set of k-shingles
    """
    text = text.lower().strip()
    if len(text) < k:
        return {text}
    return {text[i:i+k] for i in range(len(text) - k + 1)}


def minhash_signature(shingles: Set[str], num_perm: int = 128, seed: int = 42) -> np.ndarray:
    """
    Compute MinHash signature for a set of shingles.

    Args:
        shingles: Set of shingles
        num_perm: Number of permutations
        seed: Random seed

    Returns:
        MinHash signature array
    """
    np.random.seed(seed)

    # Generate hash functions (using a*x + b mod p)
    max_hash = 2**32 - 1
    prime = 4294967311  # Large prime

    a_values = np.random.randint(1, max_hash, num_perm)
    b_values = np.random.randint(0, max_hash, num_perm)

    signature = np.full(num_perm, max_hash, dtype=np.uint64)

    for shingle in shingles:
        # Hash the shingle
        h = int(hashlib.md5(shingle.encode()).hexdigest(), 16) % max_hash

        # Compute all hash values
        hash_values = (a_values * h + b_values) % prime

        # Update signature with minimum
        signature = np.minimum(signature, hash_values)

    return signature


def lsh_buckets(signature: np.ndarray, num_bands: int) -> List[Tuple[int, int]]:
    """
    Generate LSH bucket assignments using banding technique.

    Args:
        signature: MinHash signature
        num_bands: Number of bands

    Returns:
        List of (band_id, bucket_hash) tuples
    """
    rows_per_band = len(signature) // num_bands
    buckets = []

    for band_id in range(num_bands):
        start = band_id * rows_per_band
        end = start + rows_per_band
        band = signature[start:end]

        # Hash the band to get bucket
        bucket_hash = hash(tuple(band))
        buckets.append((band_id, bucket_hash))

    return buckets


def jaccard_similarity(set1: Set, set2: Set) -> float:
    """Compute exact Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def deduplicate_minhash(
    records: List[Dict],
    text_key: str = "response",
    threshold: float = 0.7,
    num_perm: int = 128,
    num_bands: int = 16,
    shingle_k: int = 5
) -> List[Dict]:
    """
    Deduplicate records using MinHash LSH.

    Args:
        records: List of records with text field
        text_key: Key for text field
        threshold: Jaccard similarity threshold
        num_perm: Number of permutations for MinHash
        num_bands: Number of bands for LSH
        shingle_k: Size of shingles

    Returns:
        Deduplicated list of records
    """
    print(f"Starting MinHash deduplication...")
    print(f"  Records: {len(records):,}")
    print(f"  Threshold: {threshold}")
    print(f"  Permutations: {num_perm}")
    print(f"  Bands: {num_bands}")

    # Step 1: Generate shingles and signatures
    print("\nStep 1: Generating MinHash signatures...")
    shingles_list = []
    signatures = []

    for record in tqdm(records, desc="Computing signatures"):
        text = record.get(text_key, "")
        shingles = get_shingles(text, shingle_k)
        signature = minhash_signature(shingles, num_perm)

        shingles_list.append(shingles)
        signatures.append(signature)

    # Step 2: LSH bucketing
    print("\nStep 2: LSH bucketing for candidate pairs...")
    bucket_to_indices = defaultdict(list)

    for idx, sig in enumerate(tqdm(signatures, desc="Bucketing")):
        buckets = lsh_buckets(sig, num_bands)
        for band_id, bucket_hash in buckets:
            bucket_to_indices[(band_id, bucket_hash)].append(idx)

    # Step 3: Find candidate pairs
    print("\nStep 3: Finding candidate pairs...")
    candidate_pairs = set()

    for indices in bucket_to_indices.values():
        if len(indices) > 1:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    if indices[i] < indices[j]:
                        candidate_pairs.add((indices[i], indices[j]))
                    else:
                        candidate_pairs.add((indices[j], indices[i]))

    print(f"  Candidate pairs: {len(candidate_pairs):,}")

    # Step 4: Verify with exact Jaccard similarity
    print("\nStep 4: Verifying duplicates...")
    duplicates = set()

    for i, j in tqdm(candidate_pairs, desc="Verifying"):
        if i in duplicates or j in duplicates:
            continue

        sim = jaccard_similarity(shingles_list[i], shingles_list[j])
        if sim >= threshold:
            # Keep the one with longer text
            text_i = records[i].get(text_key, "")
            text_j = records[j].get(text_key, "")

            if len(text_j) > len(text_i):
                duplicates.add(i)
            else:
                duplicates.add(j)

    # Step 5: Filter out duplicates
    print(f"\nStep 5: Filtering duplicates...")
    print(f"  Duplicates found: {len(duplicates):,}")

    deduped = [r for i, r in enumerate(records) if i not in duplicates]

    print(f"  Original: {len(records):,}")
    print(f"  Deduplicated: {len(deduped):,}")
    print(f"  Removed: {len(records) - len(deduped):,} ({100*(len(records)-len(deduped))/len(records):.2f}%)")

    return deduped


def main():
    parser = argparse.ArgumentParser(description="CPT MinHash Deduplication")

    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True,
                        help="Output deduplicated JSONL file")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Jaccard similarity threshold")
    parser.add_argument("--num_perm", type=int, default=128,
                        help="Number of MinHash permutations")
    parser.add_argument("--num_bands", type=int, default=16,
                        help="Number of LSH bands")
    parser.add_argument("--shingle_k", type=int, default=5,
                        help="Shingle size")
    parser.add_argument("--text_key", type=str, default="response",
                        help="Key for text field (default: response)")

    args = parser.parse_args()

    print(f"{'='*60}")
    print("CPT MinHash Deduplication")
    print(f"{'='*60}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Threshold: {args.threshold}")
    print(f"Text key: {args.text_key}")
    print(f"{'='*60}\n")

    # Load data
    print("Loading data...")
    records = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    obj = json.loads(line)
                    records.append(obj)
                except json.JSONDecodeError as e:
                    print(f"Warning: Line {line_num} parse error: {e}")
                    continue

    print(f"Loaded {len(records):,} records")

    # Deduplicate
    deduped_records = deduplicate_minhash(
        records,
        text_key=args.text_key,
        threshold=args.threshold,
        num_perm=args.num_perm,
        num_bands=args.num_bands,
        shingle_k=args.shingle_k
    )

    # Save output
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        for record in deduped_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"\n{'='*60}")
    print("Deduplication Complete!")
    print(f"  Input: {len(records):,}")
    print(f"  Output: {len(deduped_records):,}")
    print(f"  Saved to: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
