import os
import json
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchvision.transforms import functional as TF

# ========== Setup ==========
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    "qnguyen3/nanoLLaVA",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "qnguyen3/nanoLLaVA",
    trust_remote_code=True
)

# ========== Load Data ==========
json_path = "VQAv2/split_cache/val_animal_pos.json"
reconstructed_dir = "stored_data/stage2_SNR40_nonanimal_animal_20250415_153515"
coco_val_dir = "VQAv2/val2014"

with open(json_path, "r") as f:
    questions = json.load(f)["questions"]

# ========== Evaluation ==========
correct = 0
total = 0

def preprocess_image(image):
    image = TF.resize(image.convert("RGB"), (378, 378))
    return model.process_images([image], model.config).to(device=model.device, dtype=model.dtype)

for entry in questions:
    image_id = entry["image_id"]
    question = entry["question"]

    coco_img_path = os.path.join(coco_val_dir, f"COCO_val2014_{image_id:012}.jpg")
    recon_img_path = os.path.join(reconstructed_dir, f"reconstructed_{image_id}.png")

    if not os.path.exists(coco_img_path) or not os.path.exists(recon_img_path):
        continue

    # Load and preprocess images
    coco_img = preprocess_image(Image.open(coco_img_path))
    recon_img = preprocess_image(Image.open(recon_img_path))

    # Prepare text input
    messages = [{"role": "user", "content": f"<image>\n{question}"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]
    input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1], dtype=torch.long).unsqueeze(0).to(device)

    # Generate answer from original image
    with torch.no_grad():
        coco_output_ids = model.generate(
            input_ids=input_ids,
            images=coco_img,
            max_new_tokens=6,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=0.0,
            num_beams=1
        )[0]
        coco_answer = tokenizer.decode(coco_output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip().lower()

        # Generate answer from reconstructed image
        recon_output_ids = model.generate(
            input_ids=input_ids,
            images=recon_img,
            max_new_tokens=6,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=0.0,
            num_beams=1
        )[0]
        recon_answer = tokenizer.decode(recon_output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip().lower()

    print(f"[{image_id}] Q: {question}")
    print(f"  GT  : {coco_answer}")
    print(f"  Reco: {recon_answer}")

    if coco_answer == recon_answer:
        correct += 1
    total += 1
    if total >100:
        break

accuracy = correct / total if total > 0 else 0
print(f"\nFinal VQA-style Accuracy (reconstructed vs original match): {accuracy:.4f}")
