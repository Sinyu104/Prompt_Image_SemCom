import os
import warnings
import torch
import json
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import re
import string
from difflib import SequenceMatcher

# ========== Suppress logging & warnings ==========
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
warnings.filterwarnings("ignore")

# ========== Device setup ==========
torch.set_default_device("cuda:1" if torch.cuda.is_available() else "cpu")

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
recon_dir         = "stored_data/stage2_SNR20_nonanimal_animal_20250426_034417"
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
    image = Image.open(image_path).convert("RGB")
    # let the model’s own process_images do resize/convert/normalize
    return model.process_images([image], model.config).to(dtype=model.dtype)

# ========== QA matching utilities ==========
_WORD2NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12
}

def extract_number(ans: str):
    """Return the first integer found (digit or word), else None."""
    m = re.search(r"\b(\d+)\b", ans)
    if m:
        return int(m.group(1))
    for w, v in _WORD2NUM.items():
        if re.search(rf"\b{w}\b", ans.lower()):
            return v
    return None

def normalize(ans: str) -> str:
    """Lowercase, strip punctuation & articles, collapse spaces."""
    s = ans.lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", "", s)
    return " ".join(s.split())

def question_type(q: str) -> str:
    """Rudimentary Q‐type classifier."""
    ql = q.lower().strip()
    if ql.startswith("how many"):
        return "count"
    if ql.split()[0] in {"is","are","do","does","was","were","can","could"}:
        return "yesno"
    if ql.startswith("where"):
        return "location"
    return "open"

def is_match_by_type(qtype: str, a1: str, a2: str) -> bool:
    n1, n2 = normalize(a1), normalize(a2)
    if qtype == "count":
        num1, num2 = extract_number(a1), extract_number(a2)
        return (num1 is not None and num2 is not None and num1 == num2)

    if qtype == "yesno":
        # Grab the first word of each answer (if it exists)
        t1 = n1.split()[0] if n1.split() else ""
        t2 = n2.split()[0] if n2.split() else ""
        # Count as correct if they both start with the same "yes" or "no"
        return (t1 == t2)

    if qtype == "location":
        # substring or fuzzy
        return (n1 in n2) or (n2 in n1) or (SequenceMatcher(None, n1, n2).ratio() >= 0.8)

    # open‐ended: exact or fuzzy
    
    ratio = SequenceMatcher(None, n1, n2).ratio()
    if n1 == n2:
        return True
    return SequenceMatcher(None, n1, n2).ratio() >= 0.85

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
    # compare via question‐type dispatch
    qtype = question_type(question)
    match = is_match_by_type(qtype, coco_ans, recon_ans)

    print(f"[{img_id}] Q: {question}")
    print(f"  COCO ({qtype}): {coco_ans}")
    print(f"  Reco:          {recon_ans}")
    print(f"  → match? {match}\n")

    correct += int(match)
    total   += 1
    if total >= 100:
        break

print(f"Final VQA-style accuracy (match rate): {correct/total:.4f}")
