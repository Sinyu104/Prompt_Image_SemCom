import json
import os
import logging
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

###############################################################################
# Helper to Generate Split Configurations
###############################################################################
def get_split_configs(train_category: str, val_category: str):
    """
    Given the train_category and val_category strings, return configuration dictionaries
    for training and validation filtering.
    Allowed pairs:
      - ("nonanimal", "animal")
      - ("nonhuman", "human")
      - ("outdoor", "indoor")
      - ("indoor", "outdoor")
      
    The 'non' prefix indicates filtering out images containing that category.
    """
    allowed_pairs = [
        ("nonanimal", "animal"),
        ("nonhuman", "human"),
        ("outdoor", "indoor"),
        ("indoor", "outdoor")
    ]
    if (train_category, val_category) not in allowed_pairs:
        raise ValueError(f"Allowed split pairs: {allowed_pairs}. Provided: {(train_category, val_category)}")
    
    def make_config(cat: str):
        if cat.startswith("non"):
            # Remove "non" prefix and set negation flag
            return {"category": cat[3:].lower(), "negation": True}
        else:
            return {"category": cat.lower(), "negation": False}
    
    return make_config(train_category), make_config(val_category)

###############################################################################
# Image Categorizer with Category Mapping
###############################################################################
class ImageCategorizer:
    def __init__(self, instances_file):
        """
        Load the instances file (e.g. in COCO format) and build a mapping from image_id to the 
        set of object category names present in that image.
        """
        with open(instances_file, 'r') as f:
            instances = json.load(f)
        
        # Define broad category mapping (all strings should be lowercase)
        self.category_mapping = {
            'animal': [
                'dog', 'cat', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 
                'zebra', 'giraffe'
            ],
            'human': [
                'person', 'people', 'man', 'woman', 'child', 'baby', 'guy', 'girl', 'boy'
            ],
            'indoor': [
                'couch', 'bed', 'toilet', 'tv', 'mouse', 'keyboard', 'remote', 
                'cell phone', 'microwave', 'oven', 'refrigerator', 'sink', 'toaster',
                'laptop', 'computer', 'chair', 'table', 'sofa', 'desk'
            ],
            'outdoor': [
                'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
                'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter'
            ]
        }
        
        # Build mapping from image_id to set of category names (all lowercase)
        self.image_categories = {}
        cat_id_to_name = {}
        if 'categories' in instances:
            for cat in instances['categories']:
                cat_id_to_name[cat['id']] = cat['name'].lower()
        
        if 'annotations' in instances:
            for ann in instances['annotations']:
                image_id = ann['image_id']
                category_id = ann.get('category_id')
                if category_id is not None:
                    cat_name = cat_id_to_name.get(category_id, "")
                    if image_id not in self.image_categories:
                        self.image_categories[image_id] = set()
                    self.image_categories[image_id].add(cat_name)
    
    
    def matches_config(self, image_id, config: dict):
        """
        Check if the image (by image_id) meets the filtering criteria.
        """
        specific_classes = set(self.category_mapping.get(config["category"], []))
        image_cats = self.image_categories.get(image_id, set())
        has_match = bool(image_cats.intersection(specific_classes))
        return not has_match if config["negation"] else has_match

###############################################################################
# VQAv2Dataset with Updated Filtering Logic
###############################################################################
class VQAv2Dataset(torch.utils.data.Dataset):
    def __init__(self, questions_file, annotations_file, image_dir, instances_file, processor, split='train', split_config=None):
        super().__init__()
        
        logger = logging.getLogger('training')
        is_main_process = int(os.environ.get("LOCAL_RANK", -1)) == 0
        
        self.split = split
        self.processor = processor
        self.image_dir = image_dir
        
        # The cache directory is based on the directory of the questions file.
        self.cache_dir = os.path.join(os.path.dirname(questions_file), "split_cache")
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        # Load questions and annotations (only main process loads first in distributed training)
        logger = logging.getLogger('training')
        is_main_process = int(os.environ.get("LOCAL_RANK", -1)) == 0
        
        if is_main_process:
            logger.info(f"Loading {split} dataset...")
            with open(questions_file, 'r') as f:
                self.questions = json.load(f)
            with open(annotations_file, 'r') as f:
                self.annotations = json.load(f)['annotations']
            self.categorizer = ImageCategorizer(instances_file)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        if not is_main_process:
            with open(questions_file, 'r') as f:
                self.questions = json.load(f)
            with open(annotations_file, 'r') as f:
                self.annotations = json.load(f)['annotations']
            self.categorizer = ImageCategorizer(instances_file)
        
        # If a split configuration is provided, load or create the split.
        if split_config is not None:
            self.load_or_create_split(split_config)
        else:
            # No filtering configuration provided, so perform random subsampling.
            # This branch is used when train_category and val_category are None.
            max_samples = 20000 if self.split == 'train' else 2000
            total_questions = len(self.questions['questions'])
            if total_questions > max_samples:
                # Distributed handling: only main process selects indices,
                # then the indices are broadcasted to all other processes.
                device = torch.device(f'cuda:{int(os.environ.get("LOCAL_RANK", 0))}')
                if is_main_process:
                    indices = list(np.random.choice(total_questions, max_samples, replace=False))
                else:
                    indices = None
                if torch.distributed.is_initialized():
                    if is_main_process:
                        num_indices = torch.tensor(len(indices), dtype=torch.long, device=device)
                    else:
                        num_indices = torch.tensor(0, dtype=torch.long, device=device)
                    torch.distributed.broadcast(num_indices, src=0)
                    if is_main_process:
                        indices_tensor = torch.tensor(indices, dtype=torch.long, device=device)
                    else:
                        indices_tensor = torch.zeros(num_indices.item(), dtype=torch.long, device=device)
                    torch.distributed.broadcast(indices_tensor, src=0)
                    indices = indices_tensor.cpu().tolist()
                # Subsample the questions and annotations.
                self.questions['questions'] = [self.questions['questions'][i] for i in indices]
                if self.annotations:
                    self.annotations = [self.annotations[i] for i in indices]
        
        logger.info(f"Loaded {len(self.questions['questions'])} questions for split: {split}")
    
    def get_cache_path(self, split_config: dict) -> str:
        """
        Generate a cache file path based on the split configuration.
        For example, for split 'train' and config {'category': 'animal', 'negation': True},
        a possible name is 'train_animal_neg.json'.
        """
        suffix = "neg" if split_config["negation"] else "pos"
        cache_name = f"{self.split}_{split_config['category']}_{suffix}.json"
        return os.path.join(self.cache_dir, cache_name)

    def load_or_create_split(self, split_config: dict):
        """
        Load the filtered (split) dataset from cache if available.
        Otherwise, filter the dataset and save the result to the cache.
        """
        cache_path = self.get_cache_path(split_config)
        logger = logging.getLogger('training')
        is_main_process = int(os.environ.get("LOCAL_RANK", -1)) == 0
        
        if os.path.exists(cache_path):
            if is_main_process:
                logger.info(f"Loading split dataset from cache: {cache_path}")
            try:
                with open(cache_path, 'r') as f:
                    cache_data = json.load(f)
                    # Update the questions and annotations from the cached split.
                    self.questions['questions'] = cache_data['questions']
                    self.annotations = cache_data['annotations']
                return
            except (json.JSONDecodeError, OSError) as e:
                if is_main_process:
                    logger.warning(f"Failed to load cache file {cache_path}: {e}. Recreating split...")
                try:
                    os.remove(cache_path)
                except OSError:
                    pass

        # If cache does not exist, filter the dataset and then save.
        self.filter_dataset(split_config)
        if is_main_process:
            logger.info(f"Saving split dataset to cache: {cache_path}")
            cache_data = {
                'questions': self.questions['questions'],
                'annotations': self.annotations,
                'split_config': split_config
            }
            temp_path = cache_path + '.tmp'
            try:
                with open(temp_path, 'w') as f:
                    json.dump(cache_data, f)
                os.replace(temp_path, cache_path)
            except Exception as e:
                logger.error(f"Failed to save cache file: {e}")
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def filter_dataset(self, split_config):
        is_main_process = int(os.environ.get("LOCAL_RANK", -1)) == 0
        device = torch.device(f'cuda:{int(os.environ.get("LOCAL_RANK", 0))}')
        
        filtered_indices = []
        max_samples = 20000 if self.split == 'train' else 2000
        
        if is_main_process:
            total_questions = len(self.questions['questions'])
            pbar = tqdm(total=total_questions, 
                        desc=f"Filtering {self.split} dataset for {split_config['category']} (negation={split_config['negation']})")
            for idx, question in enumerate(self.questions['questions']):
                image_id = question['image_id']
                if not self.categorizer.matches_config(image_id, split_config):
                    pbar.update(1)
                    continue
                filtered_indices.append(idx)
                pbar.update(1)
            pbar.close()
            
            if len(filtered_indices) > max_samples:
                filtered_indices = list(np.random.choice(filtered_indices, max_samples, replace=False))
        
        if torch.distributed.is_initialized():
            if is_main_process:
                num_indices = torch.tensor(len(filtered_indices), dtype=torch.long, device=device)
            else:
                num_indices = torch.tensor(0, dtype=torch.long, device=device)
            torch.distributed.broadcast(num_indices, src=0)
            if is_main_process:
                indices_tensor = torch.tensor(filtered_indices, dtype=torch.long, device=device)
            else:
                indices_tensor = torch.zeros(num_indices.item(), dtype=torch.long, device=device)
            torch.distributed.broadcast(indices_tensor, src=0)
            filtered_indices = indices_tensor.cpu().tolist()
        
        self.questions['questions'] = [self.questions['questions'][i] for i in filtered_indices]
        if self.annotations:
            self.annotations = [self.annotations[i] for i in filtered_indices]

    def get_answer(self, idx):
        if self.annotations:
            return self.annotations[idx]['multiple_choice_answer']
        return None
    
    def __len__(self):
        return len(self.questions['questions'])
    
    def __getitem__(self, idx):
        question_data = self.questions['questions'][idx]
        image_id = question_data['image_id']
        question = question_data['question']
        
        image_path = os.path.join(
            self.image_dir,
            f'COCO_{self.split}2014_{str(image_id).zfill(12)}.jpg'
        )
        image = Image.open(image_path).convert('RGB')
        image = self.processor(image)
        
        answer = self.get_answer(idx) if self.annotations else None
        
        return {
            'image': image,
            'question': question,
            'answer_text': answer,
            'image_id': image_id,
        }
    
    def get_captions(self, image_id: int):
        captions = []
        for q in self.questions['questions']:
            if q['image_id'] == image_id:
                captions.append(q['question'])
        return captions

###############################################################################
# DataLoader Creation with New Split Options
###############################################################################
def create_vqa_dataloaders(args):
    logger = logging.getLogger('training')
    is_main_process = int(os.environ.get("LOCAL_RANK", -1)) == 0
    
    transform = transforms.Compose([
        transforms.Resize((352, 352)),
        transforms.ToTensor(),
    ])
    
    # If both categories are "None", then use the full dataset without filtering.
    if args.train_category != "None" and args.val_category != "None":
        train_config, val_config = get_split_configs(args.train_category, args.val_category)
        logger.info(f"Train config: {train_config} - Val config: {val_config}")
    else:
        logger.info("No filtering will be applied; using full dataset.")
        train_config, val_config = None, None
    
    train_dataset = VQAv2Dataset(
        questions_file=os.path.join(args.data_dir, "v2_OpenEnded_mscoco_train2014_questions.json"),
        annotations_file=os.path.join(args.data_dir, "v2_mscoco_train2014_annotations.json"),
        image_dir=os.path.join(args.data_dir, "train2014"),
        instances_file=os.path.join(args.data_dir, "instances_train2014.json"),
        processor=transform,
        split='train',
        split_config=train_config
    )
    
    val_dataset = VQAv2Dataset(
        questions_file=os.path.join(args.data_dir, "v2_OpenEnded_mscoco_val2014_questions.json"),
        annotations_file=os.path.join(args.data_dir, "v2_mscoco_val2014_annotations.json"),
        image_dir=os.path.join(args.data_dir, "val2014"),
        instances_file=os.path.join(args.data_dir, "instances_val2014.json"),
        processor=transform,
        split='val',
        split_config=val_config
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    
    return train_dataloader, val_dataloader


def collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    questions = [item['question'] for item in batch]
    answers = [item['answer_text'] for item in batch]
    image_ids = [item['image_id'] for item in batch]
    
    return {
        'image': images,
        'question': questions,
        'answer_text': answers,
        'image_id': image_ids,
    }