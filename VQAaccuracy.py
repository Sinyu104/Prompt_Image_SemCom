import os
import warnings
import torch
import json
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

# ========== Suppress logging & warnings ==========
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
warnings.filterwarnings("ignore")

# ========== Device setup ==========
torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

# ========== Model & tokenizer ==========
model = AutoModelForCausalLM.from_pretrained(
    "qnguyen3/nanoLLaVA",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(
    "qnguyen3/nanoLLaVA",
    trust_remote_code=True
)


# ========== Paths & questions ==========
json_path         = "VQAv2/split_cache/val_animal_pos.json"
coco_dir          = "VQAv2/val2014"
recon_dir         = "stored_data/stage2_SNR40_nonanimal_animal_20250415_153515"
with open(json_path, "r") as f:
    questions = json.load(f)["questions"]

# ========== Helper functions ==========
def make_input_ids(question: str):
    # build the chat template around a single <image> tag
    messages = [{"role": "user", "content": f"<image>\n{question}"}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # split on the literal "<image>" and tokenize each part
    text_chunks = [tokenizer(chunk).input_ids for chunk in text.split("<image>")]

    # insert the true image-token ID between the two text chunks
    all_ids = text_chunks[0] + [-200] + text_chunks[1]

    # return as a batch of size 1
    return torch.tensor(all_ids, dtype=torch.long).unsqueeze(0)

def preprocess(image_path: str):
    # load image (PIL will auto‐handle most modes)
    image = Image.open(image_path)
    # let the model’s own process_images do resize/convert/normalize
    return model.process_images([image], model.config).to(dtype=model.dtype)

# ========== Inference & evaluation ==========
correct, total = 0, 0

for entry in questions:
    img_id   = entry["image_id"]
    question = entry["question"]

    # file names
    coco_path = os.path.join(coco_dir, f"COCO_val2014_{img_id:012}.jpg")
    recon_path= os.path.join(recon_dir, f"reconstructed_{img_id}.png")
    if not (os.path.exists(coco_path) and os.path.exists(recon_path)):
        continue

    # prepare inputs
    input_ids  = make_input_ids(question).to(model.device)
    coco_img   = preprocess(coco_path)
    recon_img  = preprocess(recon_path)

    # generate on COCO
    out_ids = model.generate(
        input_ids,
        images=coco_img,
        max_new_tokens=10,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
        do_sample=False,
    )[0]
    coco_ans = tokenizer.decode(out_ids[input_ids.shape[1]:], skip_special_tokens=True).strip().lower()

    # generate on reconstructed
    out_ids = model.generate(
        input_ids,
        images=recon_img,
        max_new_tokens=10,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
        do_sample=False,
    )[0]
    recon_ans = tokenizer.decode(out_ids[input_ids.shape[1]:], skip_special_tokens=True).strip().lower()

    print(f"[{img_id}] Q: {question}")
    print(f"  COCO: {coco_ans}")
    print(f"  Reco: {recon_ans}\n")

    correct += (coco_ans == recon_ans)
    total   += 1
    if total >= 10:
        break

print(f"Final VQA-style accuracy (match rate): {correct/total:.4f}")
