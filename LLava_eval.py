import json
import base64
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

# Load NanoLLaVA
model_id = "qnguyen3/nanoLLaVA"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(model_id).to(device)

# VQA-style accuracy function
def vqa_accuracy(pred: str, gt: str) -> float:
    return 1.0 if pred.strip().lower() == gt.strip().lower() else 0.0

# Load JSONL file
with open("generated_outputs.jsonl", "r") as f:
    entries = [json.loads(line) for line in f]

results = []
for entry in tqdm(entries):
    image_bytes = base64.b64decode(entry["image_base64"])
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    question = entry["question"]
    answer = entry["answer"]

    inputs = processor(text=question, images=image, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=20)
    pred = processor.batch_decode(output, skip_special_tokens=True)[0]

    acc = vqa_accuracy(pred, answer)
    results.append({
        "image_id": entry["image_id"],
        "question": question,
        "predicted_answer": pred,
        "expected_answer": answer,
        "accuracy": acc,
    })

# Print average accuracy
avg_acc = sum(r["accuracy"] for r in results) / len(results)
print(f"Average VQA-style accuracy: {avg_acc:.4f}")
