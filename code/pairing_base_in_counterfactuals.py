import os
import json
import itertools
from collections import defaultdict

INPUT_DIR = os.path.join("dataset", "variations")
OUTPUT_FILE = os.path.join("dataset", "base_in_counterfactual_pairs.json")

def parse_variation_filename(filename):
    """
    Parses the filename into its core identity and its variation suffix.
    Expected format: Race_Gender_Age_AttributeKey_AttributeValue.png
    Example: 'Indian_Male_Young_expr_smile.png' 
    Returns: ( ('Indian', 'Male', 'Young'), 'expr_smile' )
    """
    name_clean = filename.replace(".png", "")
    parts = name_clean.split("_")
    
    # Must be a variation image (at least 5 parts)
    if len(parts) < 5:
        return None, None
        
    identity = tuple(parts[:3]) # (Race, Gender, Age)
    variation_suffix = "_".join(parts[3:]) # e.g., 'expr_smile'
    
    return identity, variation_suffix

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Directory '{INPUT_DIR}' not found.")
        return

    print(f"Scanning {INPUT_DIR}...")
    
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".png")]
    files.sort()
    
    # Group images by their exact variation suffix
    # Format: { "expr_smile": [ ("path/to/img1.png", ("Indian", "Male", "Young")), ... ] }
    images_by_variation = defaultdict(list)
    
    count_valid = 0
    count_skipped = 0
    
    for f in files:
        identity, variation_suffix = parse_variation_filename(f)
        if identity and variation_suffix:
            filepath = os.path.join(INPUT_DIR, f).replace("\\", "/")
            images_by_variation[variation_suffix].append((filepath, identity))
            count_valid += 1
        else:
            count_skipped += 1

    print(f"Processed {len(files)} files.")
    print(f" - Included (Variations): {count_valid}")
    print(f" - Skipped (Base Images / Invalid): {count_skipped}")

    valid_pairs = []
    category_counts = {"Race": 0, "Gender": 0, "Age": 0}

    print("\nGenerating cross-identity pairs (varying exactly 1 identity attribute)...")
    
    # Iterate through each specific variation (e.g., all people with 'cult_bandana')
    for variation_suffix, items in images_by_variation.items():
        
        if len(items) < 2:
            continue
            
        # Generate all possible pairs within this specific variation group
        all_possible_pairs = itertools.combinations(items, 2)
        
        for item_a, item_b in all_possible_pairs:
            path_a, id_a = item_a
            path_b, id_b = item_b
            
            # Count how many base identity attributes differ
            differences = [i for i in range(3) if id_a[i] != id_b[i]]
            
            # Keep pair ONLY if exactly 1 base identity attribute is different
            if len(differences) == 1:
                valid_pairs.append([path_a, path_b])
                
                # Track which identity attribute varied
                diff_index = differences[0]
                if diff_index == 0:
                    category_counts["Race"] += 1
                elif diff_index == 1:
                    category_counts["Gender"] += 1
                elif diff_index == 2:
                    category_counts["Age"] += 1

    print(f"\nTotal valid cross-identity pairs generated: {len(valid_pairs)}")
    print("Breakdown of varying base attributes:")
    print(f" - Differ only by Race:   {category_counts['Race']}")
    print(f" - Differ only by Gender: {category_counts['Gender']}")
    print(f" - Differ only by Age:    {category_counts['Age']}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    print(f"\nSaving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(valid_pairs, f, indent=2)

    print("Done.")

if __name__ == "__main__":
    main()