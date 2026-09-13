import os
import json
import itertools

INPUT_DIR = os.path.join("dataset", "base_images")
OUTPUT_FILE = os.path.join("dataset", "base_image_pairs.json")

def parse_base_filename(filename):
    """
    Parses the base image filename into its core demographic attributes.
    Expected format: Race_Gender_Age.png (e.g., 'Indian_Male_Young.png')
    Returns a tuple: (Race, Gender, Age)
    """
    name_clean = filename.replace(".png", "")
    parts = name_clean.split("_")
    
    if len(parts) != 3:
        # Skip files that don't match the strict tripartite base format
        return None
        
    return tuple(parts)

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Directory '{INPUT_DIR}' not found.")
        return

    print(f"Scanning {INPUT_DIR}...")
    
    # Get all PNG files
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".png")]
    files.sort()
    
    # Parse attributes for all files
    # Dictionary mapping filepath -> (Race, Gender, Age)
    image_attributes = {}
    for f in files:
        attrs = parse_base_filename(f)
        if attrs:
            filepath = os.path.join(INPUT_DIR, f).replace("\\", "/")
            image_attributes[filepath] = attrs

    print(f"Successfully parsed {len(image_attributes)} base images.")
    
    valid_pairs = []
    category_counts = {"Race": 0, "Gender": 0, "Age": 0}

    print("\nGenerating isolated pairs (varying exactly 1 attribute)...")
    
    # Generate all possible pairs of base images
    all_possible_pairs = itertools.combinations(image_attributes.keys(), 2)
    
    for img_a, img_b in all_possible_pairs:
        attrs_a = image_attributes[img_a]
        attrs_b = image_attributes[img_b]
        
        # Count how many attributes differ between the two images
        differences = [i for i in range(3) if attrs_a[i] != attrs_b[i]]
        
        # Only keep the pair if EXACTLY 1 attribute is different
        if len(differences) == 1:
            valid_pairs.append([img_a, img_b])
            
            # Track which attribute varied for analytics
            diff_index = differences[0]
            if diff_index == 0:
                category_counts["Race"] += 1
            elif diff_index == 1:
                category_counts["Gender"] += 1
            elif diff_index == 2:
                category_counts["Age"] += 1

    print(f"\nTotal valid pairs generated: {len(valid_pairs)}")
    print("Breakdown of varying attributes:")
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