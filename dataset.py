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

class COCOSegmentationDataset(Dataset):
    def __init__(self, annotation_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Root directory of COCO dataset containing image folders
            annotation_dir (str): Directory containing annotation files
            split (str): 'train' or 'val'
            transform (callable, optional): Optional transform to be applied on images
        """
        self.split = split
        self.transform = transform if transform else transforms.ToTensor()
        
        # Set up file paths
        caption_file = os.path.join(annotation_dir, f'captions_{split}2017.json')
        instance_file = os.path.join(annotation_dir, f'instances_{split}2017.json')
        
        # Initialize COCO API for captions and instances
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
    def __init__(self, questions_file, annotations_file, image_dir, processor, split='train'):
        super().__init__()
        
        logger = logging.getLogger('training')
        # Only log on main process
        is_main_process = int(os.environ.get("LOCAL_RANK", -1)) == 0
        
        if is_main_process:
            logger.info(f"Loading {split} questions...")
        with open(questions_file, 'r') as f:
            self.questions = json.load(f)
            
        if is_main_process:
            logger.info(f"Loading {split} annotations...")
        if annotations_file:
            with open(annotations_file, 'r') as f:
                self.annotations = json.load(f)['annotations']
        else:
            self.annotations = None
            
        self.image_dir = image_dir
        self.split = split
        self.processor = processor
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

def create_vqa_dataloaders(args):
    """
    Create DataLoader for the VQA dataset.
    """
    logger = logging.getLogger('training')
    # Only log on main process
    is_main_process = int(os.environ.get("LOCAL_RANK", -1)) == 0
    
    # Get world size and local rank for distributed training
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((352, 352)),  # Resize to fixed size
        transforms.ToTensor(),
    ])
    
    # Create datasets
    if is_main_process:
        logger.info("Creating training dataset...")
    train_dataset = VQAv2Dataset(
        questions_file=os.path.join(args.data_dir, "v2_OpenEnded_mscoco_train2014_questions.json"),
        annotations_file=os.path.join(args.data_dir, "v2_mscoco_train2014_annotations.json"),
        image_dir=os.path.join(args.data_dir, "train2014"),
        processor=transform,
        split='train'
    )
    
    if is_main_process:
        logger.info("Creating validation dataset...")
    val_dataset = VQAv2Dataset(
        questions_file=os.path.join(args.data_dir, "v2_OpenEnded_mscoco_val2014_questions.json"),
        annotations_file=os.path.join(args.data_dir, "v2_mscoco_val2014_annotations.json"),
        image_dir=os.path.join(args.data_dir, "val2014"),
        processor=transform,
        split='val'
    )
    
    # Create distributed samplers
    train_sampler = torch.utils.data.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True
    )
    
    val_sampler = torch.utils.data.DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=False
    )
    
    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    return train_dataloader, val_dataloader, train_sampler, val_sampler
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
