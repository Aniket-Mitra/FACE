import os
import json
import itertools
from collections import defaultdict

INPUT_DIR = os.path.join("dataset", "variations")
OUTPUT_FILE = os.path.join("dataset", "counterfactual_image_pairs.json")

def get_grouping_prefix(filename):
    """
    Parses filename to extract the Base Identity and the Attribute Type.
    Expected format: Race_Gender_Age_AttributeKey_AttributeValue.png
    Example: 'Indian_Male_Young_expr_smile.png' -> 'Indian_Male_Young_expr'
    """
    name_clean = filename.replace(".png", "")
    parts = name_clean.split("_")

    # Requires at least 5 parts to be a valid counterfactual variation
    if len(parts) < 5:
        return None

    # Isolate the core grouping prefix: Race_Gender_Age_AttributeKey
    prefix = "_".join(parts[:4])
    return prefix

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Directory '{INPUT_DIR}' not found.")
        return

    images_by_prefix = defaultdict(list)

    print(f"Scanning {INPUT_DIR}...")

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".png")]
    files.sort()

    count_valid = 0
    count_skipped = 0

    for f in files:
        prefix = get_grouping_prefix(f)

        if prefix:
            filepath = os.path.join(INPUT_DIR, f).replace("\\", "/")
            images_by_prefix[prefix].append(filepath)
            count_valid += 1
        else:
            count_skipped += 1

    print(f"Processed {len(files)} files.")
    print(f" - Included (Variations): {count_valid}")
    print(f" - Skipped (Base Images / Invalid): {count_skipped}")

    all_pairs = []

    print("\nGenerating strictly controlled pairs (varying exactly 1 attribute value)...")

    for prefix, image_list in images_by_prefix.items():
        # Generate all combinations of 2 strictly within this specific identity + attribute group
        if len(image_list) > 1:
            pairs_in_group = list(itertools.combinations(image_list, 2))
            all_pairs.extend(pairs_in_group)

    print(f"\nTotal strictly isolated pairs generated: {len(all_pairs)}")
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    print(f"Saving to {OUTPUT_FILE}...")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_pairs, f, indent=2)

    print("Done.")

if __name__ == "__main__":
    main()