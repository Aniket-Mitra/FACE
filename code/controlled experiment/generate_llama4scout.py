import os
import base64
import requests
import json
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# =========================================================
# CONFIG
# =========================================================
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL_ID = "x-ai/grok-imagine-image-quality"

# Change this to run different models ("Gemma_3_4B", "Gemma_3_12B", "Mistral_24B", "Llama_4_Scout")
CURRENT_MODEL = "Mistral_24B"

INPUT_DIR = "../dataset/base_images"
OUTPUT_DIR = "dataset_grok2/stress_test"
PAIRS_OUTPUT_FILE = "dataset_grok2/stress_test_pairs.json"

MAX_RETRIES = 10

# =========================================================
# MODEL-SPECIFIC EXTREMES
# =========================================================
MODEL_EXTREMES = {
    "Gemma_3_4B": {
        "expression": {
            "pos": "clearly visible happy facial expression with a natural smile",
            "neg": "clearly visible surprised facial expression with widened eyes"
        },
        "tattoo": {
            "pos": "large clearly visible neck tattoo",
            "neg": "clearly visible small facial tattoos near the cheeks and forehead"
        },
        "piercing": {
            "pos": "clearly visible ear piercings and a visible nose piercing",
            "neg": "clearly visible ear piercings, nose piercing, septum piercing, and lip piercing"
        },
        "skin_tone": {
            "pos": "clearly visible fair skin tone",
            "neg": "clearly visible dark skin tone"
        },
        "hair_Male": {
            "pos": "short straight hair clearly visible",
            "neg": "clearly visible afro hairstyle"
        },
        "hair_Female": {
            "pos": "long straight hair clearly visible",
            "neg": "clearly visible curly afro hairstyle"
        },
        "cultural_Male": {
            "pos": "wearing a clearly visible bandana head covering",
            "neg": "wearing a clearly visible durag head covering"
        },
        "cultural_Female": {
            "pos": "wearing a clearly visible hijab head covering",
            "neg": "wearing a clearly visible tichel head covering"
        },
        "facial_hair": {
            "pos": "clearly visible full beard",
            "neg": "clearly visible mustache"
        }
    },
    "Gemma_3_12B": {
        "expression": {
            "pos": "clearly visible happy facial expression with a natural smile",
            "neg": "clearly visible angry facial expression with visible facial tension"
        },
        "tattoo": {
            "pos": "large clearly visible neck tattoo",
            "neg": "clearly visible small facial tattoos near the cheeks and forehead"
        },
        "piercing": {
            "pos": "clearly visible ear piercings and a visible nose piercing",
            "neg": "clearly visible ear piercings, nose piercing, septum piercing, and lip piercing"
        },
        "skin_tone": {
            "pos": "clearly visible fair skin tone",
            "neg": "clearly visible dark skin tone"
        },
        "hair_Male": {
            "pos": "short straight hair clearly visible",
            "neg": "clearly visible dreadlocks hairstyle"
        },
        "hair_Female": {
            "pos": "long straight hair clearly visible",
            "neg": "clearly visible curly afro hairstyle"
        },
        "cultural_Male": {
            "pos": "wearing a clearly visible turban head covering",
            "neg": "wearing a clearly visible durag head covering"
        },
        "cultural_Female": {
            "pos": "wearing a clearly visible catholic nun veil head covering",
            "neg": "wearing a clearly visible bandana head covering"
        },
        "facial_hair": {
            "pos": "completely clean shaven face",
            "neg": "clearly visible full beard"
        }
    },
    "Mistral_24B": {
        "expression": {
            "pos": "clearly visible happy facial expression with a natural smile",
            "neg": "clearly visible angry facial expression with visible facial tension"
        },
        "tattoo": {
            "pos": "large clearly visible neck tattoo",
            "neg": "clearly visible small facial tattoos near the cheeks and forehead"
        },
        "piercing": {
            "pos": "clearly visible ear piercings and a visible nose piercing",
            "neg": "clearly visible ear piercings, nose piercing, septum piercing, and lip piercing"
        },
        "skin_tone": {
            "pos": "clearly visible fair skin tone",
            "neg": "clearly visible dark skin tone"
        },
        "hair_Male": {
            "pos": "short straight hair clearly visible",
            "neg": "clearly visible dreadlocks hairstyle"
        },
        "hair_Female": {
            "pos": "long wavy hair clearly visible",
            "neg": "clearly visible pixie cut hairstyle"
        },
        "cultural_Male": {
            "pos": "wearing a clearly visible kufi head covering",
            "neg": "wearing a clearly visible durag head covering"
        },
        "cultural_Female": {
            "pos": "wearing a clearly visible gele head covering",
            "neg": "wearing a clearly visible catholic nun veil head covering"
        },
        "facial_hair": {
            "pos": "completely clean shaven face",
            "neg": "clearly visible facial stubble"
        }
    },
    "Llama_4_Scout": {
        "expression": {
            "pos": "clearly visible happy facial expression with a natural smile",
            "neg": "clearly visible angry facial expression with visible facial tension"
        },
        "tattoo": {
            "pos": "large clearly visible neck tattoo",
            "neg": "clearly visible small facial tattoos near the cheeks and forehead"
        },
        "piercing": {
            "pos": "clearly visible ear piercings and a visible nose piercing",
            "neg": "clearly visible ear piercings, nose piercing, septum piercing, and lip piercing"
        },
        "skin_tone": {
            "pos": "clearly visible fair skin tone",
            "neg": "clearly visible dark skin tone"
        },
        "hair_Male": {
            "pos": "long straight hair clearly visible",
            "neg": "clearly visible dreadlocks hairstyle"
        },
        "hair_Female": {
            "pos": "long straight hair clearly visible",
            "neg": "clearly visible curly afro hairstyle"
        },
        "cultural_Male": {
            "pos": "wearing a clearly visible turban head covering",
            "neg": "wearing a clearly visible bandana head covering"
        },
        "cultural_Female": {
            "pos": "wearing a clearly visible catholic nun veil head covering",
            "neg": "wearing a clearly visible bandana head covering"
        },
        "facial_hair": {
            "pos": "clearly visible full beard",
            "neg": "completely clean shaven face"
        }
    }
}

EXTREMES = MODEL_EXTREMES[CURRENT_MODEL]

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def save_image(b64_data, filepath):
    if "," in b64_data:
        b64_data = b64_data.split(",", 1)[1]
    with open(filepath, "wb") as f:
        f.write(base64.b64decode(b64_data))

def is_valid_target_attribute(identity_name, attribute_name):
    identity_lower = identity_name.lower()
    is_child = "child" in identity_lower
    is_male = "male" in identity_lower and "female" not in identity_lower
    is_female = "female" in identity_lower

    if is_female and attribute_name in ["hair_Male", "cultural_Male", "facial_hair"]:
        return False
    if is_male and attribute_name in ["hair_Female", "cultural_Female"]:
        return False
    if is_child and attribute_name in ["facial_hair"]:
        return False

    return True

def build_selected_attributes(identity_name, target_attribute, target_polarity, background_polarity):
    identity_parts = os.path.basename(identity_name).split('_')
    is_child = "child" in identity_name.lower()
    is_male = "male" in identity_name.lower() and "female" not in identity_name.lower()
    is_female = "female" in identity_name.lower()

    bg_pol = background_polarity  # "pos" or "neg"
    attributes = {}

    # 1. Set all non-target background attributes to background_polarity extreme
    attributes["skin_tone"] = EXTREMES["skin_tone"][bg_pol]
    attributes["expression"] = EXTREMES["expression"][bg_pol]
    attributes["tattoo"] = EXTREMES["tattoo"][bg_pol]
    attributes["piercing"] = EXTREMES["piercing"][bg_pol]

    # Facial hair remains excluded for children
    if is_male and not is_child:
        attributes["facial_hair"] = EXTREMES["facial_hair"][bg_pol]

    # 2. Headgear / Hair Interaction Logic
    use_headgear = True
    if target_attribute in ["hair_Male", "hair_Female"]:
        use_headgear = False

    if is_male:
        if use_headgear:
            attributes["cultural_Male"] = EXTREMES["cultural_Male"][bg_pol]
            attributes["hair_Male"] = "short straight hair partially visible around the head covering"
            attributes["ear_visibility"] = "ears partially covered by the head covering"
        else:
            attributes["hair_Male"] = EXTREMES["hair_Male"][bg_pol]
            attributes["ear_visibility"] = "ears fully visible"
    elif is_female:
        if use_headgear:
            attributes["cultural_Female"] = EXTREMES["cultural_Female"][bg_pol]
            attributes["hair_Female"] = "hair mostly covered under the head covering with only minimal hair visible"
            attributes["ear_visibility"] = "ears mostly covered by the head covering"
        else:
            attributes["hair_Female"] = EXTREMES["hair_Female"][bg_pol]
            attributes["ear_visibility"] = "ears fully visible"

    # 3. OVERRIDE TARGET ATTRIBUTE with its own independent polarity
    attributes[target_attribute] = EXTREMES[target_attribute][target_polarity]

    return attributes

def build_attribute_text(attributes):
    return "\n".join([f"- {key}: {value}" for key, value in attributes.items()])

def generate_image(base_image_path, target_attribute, target_polarity, background_polarity, output_path):
    if os.path.exists(output_path):
        return output_path

    identity_name = os.path.splitext(os.path.basename(base_image_path))[0]
    selected_attributes = build_selected_attributes(identity_name, target_attribute, target_polarity, background_polarity)
    attribute_text = build_attribute_text(selected_attributes)
    base64_image = encode_image(base_image_path)

    prompt = f"""
Use the SAME EXACT PERSON from the reference image.
ALL listed attributes MUST appear in the final image.
If an attribute is missing, YOU MUST ADD IT realistically.

STRICTLY PRESERVE:
- Identity, ethnicity, gender, and age
- Face structure and skull shape
- Pose, framing, and camera angle
- Lighting, background, and clothing

Required attributes for this generation:
{attribute_text}

IMPORTANT RULES:
- Add missing attributes smoothly.
- Preserve the underlying biometric identity perfectly.
- The final image must be hyper-realistic and photorealistic.
"""

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_ID,
        "temperature": 0.05,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    {"type": "text", "text": prompt}
                ]
            }
        ],
        "modalities": ["image"],
        "max_tokens": 1000
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                image_url = result["choices"][0]["message"]["images"][0]["image_url"]["url"]
                save_image(image_url, output_path)
                return output_path
            else:
                tqdm.write(f"  [Error attempt {attempt}/{MAX_RETRIES}] {response.text}")
        except Exception as e:
            tqdm.write(f"  [Exception attempt {attempt}/{MAX_RETRIES}] Failed to parse: {e}")

    tqdm.write(f"  [Failed] Max retries ({MAX_RETRIES}) exhausted for {output_path}")
    return None

# =========================================================
# MAIN EXECUTION
# =========================================================
def main():
    model_out_dir = os.path.join(OUTPUT_DIR, CURRENT_MODEL)
    os.makedirs(model_out_dir, exist_ok=True)

    valid_exts = [".png", ".jpg", ".jpeg"]
    image_files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if os.path.splitext(f)[1].lower() in valid_exts]

    all_quads = []
    tasks = []

    for image_path in image_files:
        identity_name = os.path.splitext(os.path.basename(image_path))[0]

        for attribute_name in list(EXTREMES.keys()):
            if is_valid_target_attribute(identity_name, attribute_name):
                tasks.append({
                    "image_path": image_path,
                    "identity_name": identity_name,
                    "attribute_name": attribute_name
                })

    print(f"Total base images: {len(image_files)}")
    print(f"Total tasks (each producing 4 images): {len(tasks)}")
    print(f"Total images to generate: {len(tasks) * 4}")

    for task in tqdm(tasks, desc=f"Generating Quads for {CURRENT_MODEL}", unit="quad"):
        image_path = task["image_path"]
        identity_name = task["identity_name"]
        attr_name = task["attribute_name"]

        parts = identity_name.split('_')
        if len(parts) >= 3:
            race = parts[0]
            gender = parts[1]
            age = parts[2]
        else:
            continue

        race_clean = race.replace(' ', '')
        age_clean = age.replace(' ', '')

        quad_records = []
        quad_complete = True

        for bg_pol in ["pos", "neg"]:
            for target_pol in ["pos", "neg"]:
                filename = f"stress_{race_clean}_{gender}_{age_clean}_{attr_name}_bg{bg_pol}_target{target_pol}.png"
                out_path = os.path.join(model_out_dir, filename)

                result = generate_image(image_path, attr_name, target_pol, bg_pol, out_path)

                if result:
                    quad_records.append({
                        "path": result,
                        "model_tested": CURRENT_MODEL,
                        "race": race,
                        "gender": gender,
                        "age": age,
                        "target_tested": attr_name,
                        "target_polarity": target_pol,
                        "background_polarity": bg_pol
                    })
                else:
                    quad_complete = False
                    tqdm.write(f"  [Skipping quad] Failed on bg={bg_pol} target={target_pol} for {identity_name}/{attr_name}")

        if quad_complete:
            all_quads.append(quad_records)

    model_json = PAIRS_OUTPUT_FILE.replace(".json", f"_{CURRENT_MODEL}.json")
    with open(model_json, "w") as f:
        json.dump(all_quads, f, indent=4)
    print(f"All generations complete! Saved {len(all_quads)} complete quads to {model_json}")

if __name__ == "__main__":
    main()