import sys
import json
import os
from collections import Counter

def load_answers(path):
    answers = {}
    if path.endswith('.jsonl'):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                d = json.loads(line)
                qid = d.get('qa_id') or d.get('id')
                ans = d.get('graph_correct') or d.get('correct') or d.get('pred_text')
                if qid:
                    answers[qid] = str(ans).strip().lower() if ans else ""
    else:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for d in data:
                qid = d.get('qa_id') or d.get('id')
                ans = d.get('graph_correct') or d.get('correct')
                if qid:
                    answers[qid] = str(ans).strip().lower() if ans else ""
    return answers

def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_submissions.py <file1> <file2>")
        sys.exit(1)

    f1 = sys.argv[1]
    f2 = sys.argv[2]

    print(f"Loading {f1}...")
    ans1 = load_answers(f1)
    print(f"Loaded {len(ans1)} answers.")

    print(f"Loading {f2}...")
    ans2 = load_answers(f2)
    print(f"Loaded {len(ans2)} answers.")

    common_ids = set(ans1.keys()).intersection(set(ans2.keys()))
    total = len(common_ids)

    if total == 0:
        print("No common IDs found between the two files.")
        sys.exit(0)

    matches = sum(1 for qid in common_ids if ans1[qid] == ans2[qid])
    diffs = total - matches

    print(f"\n--- Comparison Results ---")
    print(f"Comparing: {os.path.basename(f1)} vs {os.path.basename(f2)}")
    print(f"Common IDs: {total}")
    print(f"Matches: {matches} ({matches/total*100:.2f}%)")
    print(f"Differences: {diffs} ({diffs/total*100:.2f}%)")

    c1 = Counter([ans1[q] for q in common_ids])
    c2 = Counter([ans2[q] for q in common_ids])

    print(f"\nAnswer Distribution in {os.path.basename(f1)}:")
    for k, v in c1.most_common():
        print(f"  {k}: {v} ({v/total*100:.2f}%)")

    print(f"\nAnswer Distribution in {os.path.basename(f2)}:")
    for k, v in c2.most_common():
        print(f"  {k}: {v} ({v/total*100:.2f}%)")

if __name__ == "__main__":
    main()
