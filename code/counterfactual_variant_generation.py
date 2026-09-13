import os
import time
import json
import base64
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
BASE_DIR = "dataset/base_images"
OUTPUT_DIR = "dataset/variations"
MODEL_ID = "google/gemini-3-pro-image-preview"

SKIN_TONES = ["fair", "medium", "dark"]
EXPRESSIONS = ["happy", "sad", "angry", "surprised"]
TATTOOS = ["no tattoos", "neck tattoo", "face tattoo"]
PIERCINGS = ["no piercings", "ear and nose piercings", "ear, nose, septum and lip piercings"]

HAIR_STYLES = {
    "Male": ["bald", "buzz cut", "short straight hair", "long straight hair", "medium wavy curly hair", "afro", "dreadlocks"],
    "Female": ["long straight hair", "long wavy hair", "pixie cut", "shoulder length hair", "curly afro", "braids"]
}

CULTURAL_MARKERS = {
    "Male": ["turban", "bandana", "durag", "kufi", "kippah", "fedora"],
    "Female": ["hijab", "gele", "dupatta", "bandana", "catholic nun veil", "tichel"]
}

FACIAL_HAIR = ["clean shaven", "stubble", "full beard", "mustache", "goatee"]

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def save_image(b64_data, filepath):
    if "," in b64_data:
        b64_data = b64_data.split(",", 1)[1]
    with open(filepath, "wb") as f:
        f.write(base64.b64decode(b64_data))

def create_variation(base_img_path, prompt, output_path):
    if os.path.exists(output_path):
        print(f"Skipping {output_path}")
        return

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost:8000",
        "X-Title": "Research Dataset Gen"
    }

    b64_image = encode_image(base_img_path)

    payload = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_image}"}
                    }
                ]
            }
        ],
        "modalities": ["image", "text"],
        "max_tokens": 1000
    }

    print(f"Generating: {output_path}...")

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and result["choices"]:
                message = result["choices"][0]["message"]
                if "images" in message and message["images"]:
                    image_url = message["images"][0]["image_url"]["url"]
                    save_image(image_url, output_path)
                    print(f"   -> Saved")
                else:
                    print(f"   [Error] No image returned.")
        else:
            print(f"   [Error] Status {response.status_code}: {response.text}")

    except Exception as e:
        print(f"   [Exception] {e}")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    base_files = [f for f in os.listdir(BASE_DIR)[::-1] if f.endswith(".png")]

    for filename in base_files:
        base_path = os.path.join(BASE_DIR, filename)
        parts = filename.replace(".png", "").split("_")
        
        if len(parts) < 3:
            continue
            
        gender = parts[1] 
        base_name = filename.replace(".png", "")

        for tone in SKIN_TONES:
            prompt = f"Change the skin tone of the person in this image to {tone}. Keep all other facial features, background, pose, lighting and studio portrait style exactly the same."
            out_name = f"{base_name}_skin_{tone.replace(' ', '')}.png"
            create_variation(base_path, prompt, os.path.join(OUTPUT_DIR, out_name))
            # time.sleep(0.5)

        for expr in EXPRESSIONS:
            prompt = f"Change the facial expression of the person to {expr}. Keep all other facial features, background, pose, lighting and studio portrait style exactly the same."
            out_name = f"{base_name}_expr_{expr}.png"
            create_variation(base_path, prompt, os.path.join(OUTPUT_DIR, out_name))
            # time.sleep(0.5)

        for tattoo in TATTOOS:
            if tattoo == "no tattoos": continue
            prompt = f"Add a {tattoo} to the person. Keep all other facial features, background, pose, lighting and studio portrait style exactly the same."
            out_name = f"{base_name}_tat_{tattoo.replace(' ', '')}.png"
            create_variation(base_path, prompt, os.path.join(OUTPUT_DIR, out_name))
            # time.sleep(0.5)

        for piercing in PIERCINGS:
            if piercing == "no piercings": continue
            prompt = f"Add {piercing} to the person. Keep all other facial features, background, pose, lighting and studio portrait style exactly the same."
            out_name = f"{base_name}_pierc_{piercing.replace(' ', '').replace(',', '')}.png"
            create_variation(base_path, prompt, os.path.join(OUTPUT_DIR, out_name))
            # time.sleep(0.5)

        if gender in HAIR_STYLES:
            for style in HAIR_STYLES[gender]:
                prompt = f"Change the hair style to {style}. Keep all other facial features, background, pose, lighting and studio portrait style exactly the same."
                out_name = f"{base_name}_hair_{style.replace(' ', '')}.png"
                create_variation(base_path, prompt, os.path.join(OUTPUT_DIR, out_name))
                # time.sleep(0.5)

        if gender in CULTURAL_MARKERS:
            for marker in CULTURAL_MARKERS[gender]:
                prompt = f"Add a {marker} to the person's head. Keep all other facial features, background, pose, lighting and studio portrait style exactly the same."
                out_name = f"{base_name}_cult_{marker.replace(' ', '')}.png"
                create_variation(base_path, prompt, os.path.join(OUTPUT_DIR, out_name))
                # time.sleep(0.5)

        if gender == "Male":
            for style in FACIAL_HAIR:
                prompt = f"Change the facial hair to {style}. Keep all other facial features, background, pose, lighting and studio portrait style exactly the same."
                out_name = f"{base_name}_beard_{style.replace(' ', '')}.png"
                create_variation(base_path, prompt, os.path.join(OUTPUT_DIR, out_name))
                # time.sleep(0.5)

if __name__ == "__main__":
    main()