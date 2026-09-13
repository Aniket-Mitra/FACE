import os
import time
import json
import base64
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OUTPUT_DIR = "dataset/base_images"
MODEL_ID = "google/gemini-3-pro-image-preview"

RACES = ["White", "Latino Hispanic", "Indian", "East Asian", "Black", "Southeast Asian", "Middle Eastern"]
GENDERS = ["Male", "Female"]
AGES = {
    "Child": "Child (age 8-12)", 
    "Young": "Young Adult (age 19-35)",
    "MiddleAged": "Middle Aged (age 36-60)",
    "Old": "Old (age > 60)"
}

def save_base64_image(b64_data, filepath):
    if "," in b64_data:
        b64_data = b64_data.split(",", 1)[1]
    
    with open(filepath, "wb") as f:
        f.write(base64.b64decode(b64_data))

def generate_image(prompt, filename):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost:8000",
        "X-Title": "Research Dataset Gen"
    }
    
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "modalities": ["image", "text"],
        "max_tokens": 1000
    }

    print(f"Generating: {filename}...")
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        
        if response.status_code != 200:
            print(f"   [Error] API Status {response.status_code}: {response.text}")
            return False

        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            message = result["choices"][0]["message"]
            
            if "images" in message and len(message["images"]) > 0:
                image_url = message["images"][0]["image_url"]["url"]
                save_base64_image(image_url, filename)
                print(f"   -> Saved to {filename}")
                return True
            else:
                print(f"   [Error] No image found in response: {result}")
                
    except Exception as e:
        print(f"   [Exception] {e}")
        
    return False

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    count = 0
    for race in RACES:
        for gender in GENDERS:
            for age_label, age_prompt in AGES.items():       
                safe_race = race.replace(" ", "")
                filename = os.path.join(OUTPUT_DIR, f"{safe_race}_{gender}_{age_label}.png")
                
                if os.path.exists(filename):
                    print(f"Skipping {filename} (Exists)")
                    continue

                prompt = (
                    f"A hyper-realistic studio portrait of a {age_prompt} {race} {gender} facing forward with a neutral expression and wearing a plain, grey t-shirt. "
                    f"Head and shoulders framing (only above the collarbone visible), plain white background that fills the entire frame, even and soft lighting."
                )

                print(prompt)
                
                if generate_image(prompt, filename):
                    count += 1
                
                time.sleep(2)

    print(f"Done. Generated {count} images.")

if __name__ == "__main__":
    main()