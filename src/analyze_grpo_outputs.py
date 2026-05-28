"""
Analyze GRPO model eval outputs for:
1. Tag diversity — count unique XML tags
2. Multi-block <think> prevalence — how many completions use >1 block
3. Transition token frequency — are reward hacking or genuine concise reasoning?
4. Reward hack rate — multi-block completions as fraction of total

Usage:
  python src/analyze_grpo_outputs.py outputs.jsonl
"""
import re
import json
import sys
from collections import Counter

TRANSITION_TOKENS = ["wait", "hmm", "but", "thinking", "actually", "let me check"]

def extract_all_tags(text):
    return re.findall(r"</?(\w+)>", text)

def count_think_blocks(text):
    return text.count("<think>")

def count_transition_tokens(text):
    text_lower = text.lower()
    return sum(text_lower.count(tok) for tok in TRANSITION_TOKENS)

def analyze_completion(comp_text, idx):
    tags = extract_all_tags(comp_text)
    unique_tags = set(tags)
    think_blocks = count_think_blocks(comp_text)
    trans_count = count_transition_tokens(comp_text)

    # Extract answer and check for \boxed
    has_boxed = "\\boxed{" in comp_text
    has_think_tag = "<think>" in comp_text
    has_invented = any(t for t in unique_tags if t.lower() not in ["think", "/think"])

    return {
        "idx": idx,
        "think_blocks": think_blocks,
        "unique_tags": sorted(unique_tags),
        "invented_tags": [t for t in unique_tags if t.lower() not in ["think", "/think"]],
        "transition_tokens": trans_count,
        "has_boxed": has_boxed,
        "chars": len(comp_text),
        "preview": comp_text[:120].replace("\n", " | ")
    }

def analyze_file(path):
    with open(path) as f:
        lines = [json.loads(line) for line in f if line.strip()]

    results = []
    for i, entry in enumerate(lines):
        if "completion" in entry:
            results.append(analyze_completion(entry["completion"], i))
        elif "output" in entry:
            results.append(analyze_completion(entry["output"], i))
        elif "generated" in entry:
            results.append(analyze_completion(entry["generated"], i))
        else:
            # Assume JSONL has a text field or raw string
            for key in ["text", "response", "answer"]:
                if key in entry:
                    results.append(analyze_completion(entry[key], i))
                    break

    total = len(results)
    if total == 0:
        print("No completions found. Expected JSONL with 'completion', 'output', 'generated', 'text', 'response', or 'answer' keys.")
        return

    # Aggregate stats
    multi_block = [r for r in results if r["think_blocks"] > 1]
    single_block = [r for r in results if r["think_blocks"] == 1]
    no_think = [r for r in results if r["think_blocks"] == 0]
    with_invented = [r for r in results if r["invented_tags"]]
    all_invented_tags = [t for r in results for t in r["invented_tags"]]

    print("=" * 60)
    print("GRPO EVAL OUTPUT TAG DIVERSITY ANALYSIS")
    print("=" * 60)
    print(f"Total completions analyzed: {total}")
    print()
    print("--- Think Block Usage ---")
    print(f"  Single block (correct):     {len(single_block):4d} ({100*len(single_block)/total:.1f}%)")
    print(f"  Multi-block (reward hack):  {len(multi_block):4d} ({100*len(multi_block)/total:.1f}%)")
    print(f"  No think tag:               {len(no_think):4d} ({100*len(no_think)/total:.1f}%)")
    print(f"  Reward hack rate:           {100*len(multi_block)/total:.1f}%")
    print()
    print("--- Tag Diversity ---")
    all_tags = Counter()
    for r in results:
        all_tags.update(r["unique_tags"])
    print(f"  Total unique tag types: {len(all_tags)}")
    if all_invented_tags:
        print(f"  Invented tags found: {len(with_invented)}/{total} completions")
        print(f"  Unique invented tags: {sorted(set(all_invented_tags))}")
        print(f"  Invented tag occurrences: {Counter(all_invented_tags).most_common()}")
    else:
        print("  No invented tags detected.")
    print()
    print("--- Transition Token Stats ---")
    trans_counts = [r["transition_tokens"] for r in results]
    avg_trans = sum(trans_counts) / total if total else 0
    print(f"  Mean transition tokens: {avg_trans:.2f}")
    print(f"  Max transition tokens: {max(trans_counts)}")
    print(f"  Min transition tokens: {min(trans_counts)}")
    print()
    print("--- Multi-Block Details ---")
    if multi_block:
        multi_block.sort(key=lambda r: -r["think_blocks"])
        for r in multi_block[:10]:
            print(f"  #{r['idx']}: {r['think_blocks']} blocks, {r['transition_tokens']} trans, {r['chars']} chars")
            print(f"    tags: {r['unique_tags']}")
            print(f"    preview: {r['preview']}")
    print()
    print("--- Invented Tag Details ---")
    for r in with_invented:
        print(f"  #{r['idx']}: tags={r['invented_tags']}")
        print(f"    preview: {r['preview']}")
        print()

    # Interpretation
    print("--- Interpretation ---")
    if all_invented_tags:
        if len(set(all_invented_tags)) >= 2:
            print("  MULTIPLE invented tags found -> supports Schema Generalization (Interp A)")
            print("  GRPO trained abstract XML schema, not specific token strings.")
        else:
            print("  SINGLE invented tag type -> supports Format Hallucination (Interp B)")
            print("  Model lacks stable tag grounding, produces noise under distribution shift.")
    print()
    if len(multi_block) / total > 0.2:
        print("  HIGH reward hack rate (>{:.0f}%) -> conciseness gain is substantially confounded.".format(20))
    else:
        print(f"  LOW reward hack rate ({len(multi_block)}/{total}) -> multi-block is an edge case, not a systematic exploit.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_grpo_outputs.py <outputs.jsonl>")
        sys.exit(1)
    analyze_file(sys.argv[1])
