import os
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import numpy as np
from PIL import Image
import requests
import torchvision.transforms as transforms
from io import BytesIO
import skimage.io as io
import matplotlib.pyplot as plt
import json
from collections import Counter
from torchvision import transforms
import logging
from typing import List, Dict
from dataset_utils import ImageCategorizer, DatasetSplitter
from tqdm import tqdm

class COCOSegmentationDataset(Dataset):
    def __init__(self, annotation_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Root directory of COCO dataset containing image folders
            annotation_dir (str): Directory containing annotation files
            split (str): 'train' or 'val'
            transform (callable, optional): Optional transform to be applied on images
        """
        is_main_process = int(os.environ.get("LOCAL_RANK", -1)) == 0
        self.split = split
        self.transform = transform if transform else transforms.ToTensor()
        
        # Set up file paths
        caption_file = os.path.join(annotation_dir, f'captions_{split}2017.json')
        instance_file = os.path.join(annotation_dir, f'instances_{split}2017.json')
        
        # Only load COCO on main process
        if is_main_process:
            self.coco_caps = COCO(caption_file)
            self.coco_instances = COCO(instance_file)
        
        # Wait for main process to load
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        
        # Other processes can now use the loaded COCO data
        if not is_main_process:
            self.coco_caps = COCO(caption_file)
            self.coco_instances = COCO(instance_file)
        
        # Get all image IDs that have both captions and instance annotations
        self.image_ids = list(set(self.coco_caps.getImgIds()) & 
                            set(self.coco_instances.getImgIds()))
        
        # Pre-fetch category names
        self.categories = {cat['id']: cat['name'] 
                         for cat in self.coco_instances.loadCats(
                             self.coco_instances.getCatIds())}

    def __len__(self):
        return len(self.image_ids)

    def get_captions(self, img_id):
        """Get all captions for an image."""
        ann_ids = self.coco_caps.getAnnIds(imgIds=img_id)
        anns = self.coco_caps.loadAnns(ann_ids)
        return [ann['caption'] for ann in anns]

    def get_instance_masks(self, img_id, img_size):
        """Get instance segmentation masks for an image."""
        ann_ids = self.coco_instances.getAnnIds(imgIds=img_id)
        anns = self.coco_instances.loadAnns(ann_ids)
        
        # Initialize empty mask
        masks = []
        categories = []
        
        for ann in anns:
            mask = self.coco_instances.annToMask(ann)
            category = self.categories[ann['category_id']]
            
            # Resize mask to match image size if necessary
            if mask.shape != img_size:
                mask = Image.fromarray(mask)
                mask = mask.resize((img_size[1], img_size[0]), Image.NEAREST)
                mask = np.array(mask)
            
            masks.append(mask)
            categories.append(category)
        
        return masks, categories

    def load_image_from_url(self, url):
        """Load image from URL."""
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content)).convert('RGB')
            return image
        except Exception as e:
            print(f"Error loading image from URL: {e}")
            return None

    def __getitem__(self, idx):
        """
        Returns:
            dict: {
                'image': tensor of shape [3, H, W],
                'masks': list of binary masks,
                'categories': list of category names,
                'captions': list of caption strings,
                'image_id': COCO image ID
            }
        """
        img_id = self.image_ids[idx]
        
        # Load image
        img_info = self.coco_caps.loadImgs(img_id)[0]
        image = self.load_image_from_url(img_info['coco_url'])
       
        
        if image is None:
            # Return None or skip this sample if image loading fails
            return None
        
        # Get image size for mask generation
        img_size = image.size[::-1]  # (H, W)
        
        # Apply transforms to image
        image = self.transform(image)
        
        # Get captions
        captions = self.get_captions(img_id)
        
        # Get instance masks and categories
        masks, categories = self.get_instance_masks(img_id, img_size)
        
        # Convert masks to tensor
        if masks:
            masks = torch.tensor(np.stack(masks))
        else:
            masks = torch.zeros((0, img_size[0], img_size[1]))
        
        return {
            'image': image,
            'masks': masks,
            'categories': categories,
            'captions': captions,
            'image_id': img_id
        }

def get_coco_dataloader(annotation_dir, split='train', 
                       batch_size=1, num_workers=4, shuffle=True):
    """
    Create a DataLoader for the COCO dataset.
    
    Args:
        root_dir (str): Root directory of COCO dataset
        annotation_dir (str): Directory containing annotation files
        split (str): 'train' or 'val'
        batch_size (int): Batch size
        num_workers (int): Number of workers for data loading
        shuffle (bool): Whether to shuffle the dataset
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((352, 352)),  # Resize to fixed size
        transforms.ToTensor(),
    ])
    
    # Create dataset
    dataset = COCOSegmentationDataset(
        annotation_dir=annotation_dir,
        split=split,
        transform=transform
    )

    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return dataloader


class VQAv2Dataset(Dataset):
    def __init__(self, questions_file, annotations_file, image_dir, instances_file, processor, split='train', 
                 split_config: Dict = None):
        super().__init__()
        
        logger = logging.getLogger('training')
        is_main_process = int(os.environ.get("LOCAL_RANK", -1)) == 0
        
        self.split = split
        self.processor = processor
        self.image_dir = image_dir
        self.splitter = DatasetSplitter()
        self.cache_dir = os.path.join(os.path.dirname(questions_file), "split_cache")
        
        # Load questions and annotations only on main process first
        if is_main_process:
            logger.info(f"Loading {split} dataset...")
            if split_config:
                logger.info(f"Using split config: {split_config}")
            else:
                logger.info("Using full dataset (no splitting)")
            
            # Create cache directory if it doesn't exist
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
            
            # Load questions and annotations
            with open(questions_file, 'r') as f:
                self.questions = json.load(f)
            with open(annotations_file, 'r') as f:
                self.annotations = json.load(f)['annotations']
                
            # Initialize categorizer
            self.categorizer = ImageCategorizer(instances_file)
        
        # Wait for main process to load
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        
        # Other processes load after main process is done
        if not is_main_process:
            with open(questions_file, 'r') as f:
                self.questions = json.load(f)
            with open(annotations_file, 'r') as f:
                self.annotations = json.load(f)['annotations']
            self.categorizer = ImageCategorizer(instances_file)
        
        # Apply splitting if config is provided
        if split_config:
            self.load_or_create_split(split_config, questions_file, annotations_file)
            
        if is_main_process:
            logger.info(f"Loaded {len(self.questions['questions'])} questions")

    def get_cache_path(self, split_config: Dict) -> str:
        """Generate cache file path based on split configuration"""
        cache_name = f"{self.split}_{split_config['split_type']}_{split_config['level2']}_{split_config['category']}.json"
        return os.path.join(self.cache_dir, cache_name)
    
    def load_or_create_split(self, split_config: Dict, questions_file: str, annotations_file: str):
        """Load split dataset from cache or create and cache it"""
        cache_path = self.get_cache_path(split_config)
        is_main_process = int(os.environ.get("LOCAL_RANK", -1)) == 0
        logger = logging.getLogger('training')
        
        # Try to load from cache
        if os.path.exists(cache_path):
            if is_main_process:
                logger.info(f"Loading split dataset from cache: {cache_path}")
            try:
                with open(cache_path, 'r') as f:
                    cached_data = json.load(f)
                    self.questions['questions'] = cached_data['questions']
                    self.annotations = cached_data['annotations']
                    # Convert criteria back to set if loading from cache
                    if 'split_config' in cached_data and 'criteria' in cached_data['split_config']:
                        cached_data['split_config']['criteria'] = set(cached_data['split_config']['criteria'])
                return
            except (json.JSONDecodeError, OSError) as e:
                if is_main_process:
                    logger.warning(f"Failed to load cache file {cache_path}: {e}")
                    # Delete corrupted cache file
                    try:
                        os.remove(cache_path)
                    except OSError:
                        pass
        
        # If not cached, create split
        if is_main_process:
            logger.info("Split dataset not found in cache, creating new split...")
        
        # Filter dataset
        self.filter_dataset(split_config)
        
        # Save to cache if main process
        if is_main_process:
            # Convert sets to lists in split_config for JSON serialization
            json_safe_config = split_config.copy()
            if 'criteria' in json_safe_config:
                json_safe_config['criteria'] = list(json_safe_config['criteria'])
            
            cache_data = {
                'questions': self.questions['questions'],
                'annotations': self.annotations,
                'split_config': json_safe_config
            }
            try:
                # Write to temporary file first
                temp_path = cache_path + '.tmp'
                with open(temp_path, 'w') as f:
                    json.dump(cache_data, f)
                # Then rename to final path (atomic operation)
                os.replace(temp_path, cache_path)
                logger.info(f"Saved split dataset to cache: {cache_path}")
            except Exception as e:
                logger.error(f"Failed to save cache file: {e}")
                # Clean up temp file if it exists
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def filter_dataset(self, split_config):
        """Filter dataset based on the provided split configuration"""
        is_main_process = int(os.environ.get("LOCAL_RANK", -1)) == 0
        device = torch.device(f'cuda:{int(os.environ.get("LOCAL_RANK", 0))}')
        
        filtered_indices = []
        max_samples = 20000 if self.split == 'train' else 2000
        
        # Only main process does the filtering
        if is_main_process:
            # Get total number of questions
            total_questions = len(self.questions['questions'])
            
            # Create progress bar
            pbar = tqdm(total=total_questions, 
                      desc=f"Filtering {self.split} dataset (max {max_samples}) for {split_config['category']} category")
            
            for idx, question in enumerate(self.questions['questions']):
                image_id = question['image_id']
                answer = self.annotations[idx]['multiple_choice_answer']
                matches = self.categorizer.categorize_sample(
                    image_id,
                    question['question'],
                    answer,
                    split_config['criteria']
                )
                
                if not matches:
                    pbar.update(1)
                    continue
                
                filtered_indices.append(idx)
                pbar.update(1)
            
            pbar.close()
            
            # Randomly sample if we have more than max_samples
            if len(filtered_indices) > max_samples:
                filtered_indices = list(np.random.choice(filtered_indices, max_samples, replace=False))

        # Broadcast filtered indices from main process to all processes
        if torch.distributed.is_initialized():
            # First broadcast the number of indices
            if is_main_process:
                num_indices = torch.tensor(len(filtered_indices), dtype=torch.long, device=device)
            else:
                num_indices = torch.tensor(0, dtype=torch.long, device=device)
            torch.distributed.broadcast(num_indices, src=0)
            
            # Convert to tensor for broadcasting
            if is_main_process:
                indices_tensor = torch.tensor(filtered_indices, dtype=torch.long, device=device)
            else:
                indices_tensor = torch.zeros(num_indices.item(), dtype=torch.long, device=device)
            
            # Broadcast from rank 0 to all processes
            torch.distributed.broadcast(indices_tensor, src=0)
            filtered_indices = indices_tensor.cpu().tolist()
        
        # Update questions and annotations
        self.questions['questions'] = [self.questions['questions'][i] for i in filtered_indices]
        if self.annotations:
            self.annotations = [self.annotations[i] for i in filtered_indices]

    def get_answer(self, idx):
        """Get answer for a given question index"""
        if self.annotations:
            return self.annotations[idx]['multiple_choice_answer']
        return None
    
    def __len__(self):
        return len(self.questions['questions'])
    
    def __getitem__(self, idx):
        # Get question data
        question_data = self.questions['questions'][idx]
        image_id = question_data['image_id']
        question = question_data['question']
        
        # Load and preprocess image
        image_path = os.path.join(
            self.image_dir,
            f'COCO_{self.split}2014_{str(image_id).zfill(12)}.jpg'
        )
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms (assuming 'processor' includes ToTensor)
        image = self.processor(image)
        
        # Get answer if available
        answer = self.get_answer(idx) if self.annotations else None
        
        return {
            'image': image,  # Already a tensor scaled to [0,1]
            'question': question,
            'answer_text': answer,
            'image_id': image_id,
            'target_mask': torch.zeros((352, 352))  # Placeholder mask
        }

    def get_captions(self, image_id: int) -> List[str]:
        """Get captions for an image from annotations"""
        # For VQA-v2, we'll use the question as a caption since we don't have direct access to COCO captions
        captions = []
        for q in self.questions['questions']:
            if q['image_id'] == image_id:
                captions.append(q['question'])
        return captions

def create_vqa_dataloaders(args):
    """Create DataLoader for the VQA dataset."""
    logger = logging.getLogger('training')
    is_main_process = int(os.environ.get("LOCAL_RANK", -1)) == 0
    
    transform = transforms.Compose([
        transforms.Resize((352, 352)),
        transforms.ToTensor(),
    ])
    
    # Create split config for all processes
    split_config = None
    if all([args.split_type, args.split_level2, args.train_category, args.val_category]):
        splitter = DatasetSplitter()
        split_config = splitter.get_split_config(
            args.split_type,
            args.split_level2,
            args.train_category,
            args.val_category
        )
    
    # Create datasets on all processes
    train_dataset = VQAv2Dataset(
        questions_file=os.path.join(args.data_dir, "v2_OpenEnded_mscoco_train2014_questions.json"),
        annotations_file=os.path.join(args.data_dir, "v2_mscoco_train2014_annotations.json"),
        image_dir=os.path.join(args.data_dir, "train2014"),
        instances_file=os.path.join(args.data_dir, "instances_train2014.json"),
        processor=transform,
        split='train',
        split_config=split_config.get('train') if split_config else None
    )
    
    val_dataset = VQAv2Dataset(
        questions_file=os.path.join(args.data_dir, "v2_OpenEnded_mscoco_val2014_questions.json"),
        annotations_file=os.path.join(args.data_dir, "v2_mscoco_val2014_annotations.json"),
        image_dir=os.path.join(args.data_dir, "val2014"),
        instances_file=os.path.join(args.data_dir, "instances_val2014.json"),
        processor=transform,
        split='val',
        split_config=split_config.get('val') if split_config else None
    )

    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    return train_dataloader, val_dataloader

def collate_fn(batch):
    """
    Custom collate function for the dataloader
    """
    # Collect all items from batch
    images = torch.stack([item['image'] for item in batch])
    questions = [item['question'] for item in batch]
    answers = [item['answer_text'] for item in batch]
    image_ids = [item['image_id'] for item in batch]
    target_masks = torch.stack([item['target_mask'] for item in batch])
    
    return {
        'image': images,
        'question': questions,
        'answer_text': answers,
        'image_id': image_ids,
        'target_mask': target_masks
    }
