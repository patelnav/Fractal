from humaneval_harness import load_humaneval_dataset

def list_structural_problems():
    problems = load_humaneval_dataset()
    structural_keywords = ["nested", "grid", "matrix", "common", "sort", "filter", "traverse"]
    
    print(f"Total problems: {len(problems)}")
    
    selected = []
    
    # We want a mix of:
    # 1. Simple linear (control)
    # 2. Single loop (control)
    # 3. Nested loop (failure case)
    # 4. Loop + If (failure case)
    
    # Manually picking some known structures based on ID if possible, or just printing potential candidates
    
    candidates = [
        "HumanEval/0",  # has_close_elements (Loop + If)
        "HumanEval/1",  # separate_paren_groups (Nested logic)
        "HumanEval/10", # make_palindrome (Loop)
        "HumanEval/11", # string_xor (Loop)
        "HumanEval/12", # longest (Loop + If)
        "HumanEval/26", # remove_duplicates (Loop + count)
        "HumanEval/29", # filter_by_prefix (Loop + If)
        "HumanEval/32", # poly (Math, nested?)
        "HumanEval/33", # sort_third (List manipulation)
        "HumanEval/37", # sort_even (List manipulation)
        "HumanEval/39", # prime_fib (Loop + Loop)
        "HumanEval/40", # triples_sum_to_zero (Triple nested loop potentially)
        "HumanEval/43", # pairs_sum_to_zero (Nested loop)
        "HumanEval/46", # fib4 (Loop)
        "HumanEval/129" # minPath (Grid/Search - Hard)
    ]
    
    print("Selected Representative Set:")
    found = 0
    for p in problems:
        if p['task_id'] in candidates:
            print(f"  {p['task_id']}: {p['entry_point']}")
            found += 1
            
    if found < len(candidates):
        print(f"Warning: Only found {found} of {len(candidates)} candidates. Dataset might be incomplete.")

if __name__ == "__main__":
    list_structural_problems()
