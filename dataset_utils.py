from collections import defaultdict
import json
from typing import List, Dict, Set
import re
from pycocotools.coco import COCO
import logging
import os
import torch
import torch.distributed

class ImageCategorizer:
    def __init__(self, instances_file: str):
        is_main_process = int(os.environ.get("LOCAL_RANK", -1)) == 0
        # Cache for image categories
        self.image_categories: Dict[int, Set[str]] = defaultdict(set)
        self.splitter = DatasetSplitter()  # Use the hierarchical splitter instead
        # Only load COCO on main process
        if is_main_process:
            self.coco = COCO(instances_file)
        
        # Wait for main process to load
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        
        # Other processes can now use the loaded COCO data
        if not is_main_process:
            self.coco = COCO(instances_file)
        
        # Map our categories to COCO categories
        self.category_mapping = {
            'animal': [
                'dog', 'cat', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 
                'zebra', 'giraffe'
            ],
            'human': ['person'],
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
        
        # Create reverse mapping from COCO category names to our categories
        self.coco_to_our_categories = {}
        for our_cat, coco_cats in self.category_mapping.items():
            for coco_cat in coco_cats:
                self.coco_to_our_categories[coco_cat] = our_cat
        
        # Add question-related keywords for each category
        self.question_keywords = {
            'animal': ['animal', 'dog', 'cat', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'],
            'human': ['person', 'people', 'man', 'woman', 'child', 'baby', 'guy', 'girl', 'boy'],
            'indoor': ['room', 'house', 'kitchen', 'bathroom', 'bedroom', 'living room', 'furniture'],
            'outdoor': ['street', 'park', 'road', 'sky', 'building', 'car', 'traffic']
        }
    
    def analyze_sample(self, captions: List[str], question: str = None, split_type: str = 'image_based') -> Dict[str, any]:
        """Analyze both image content and question type"""
        categories = self.splitter.split_hierarchy[split_type]
        analysis = {
            'categories': self.categorize_sample(captions, question, categories)
        }
        return analysis
    
    def categorize_image(self, captions: List[str], question: str = None) -> Set[str]:
        """Categorize image based on captions and question"""
        categories = set()
        
        # Combine all text for analysis
        all_text = ' '.join(captions + ([question] if question else [])).lower()
        
        # Check each category
        for category, keywords in self.categories.items():
            if any(keyword in all_text for keyword in keywords):
                categories.add(category)
        
        return categories
    
    def get_image_categories(self, image_id: int, captions: List[str], question: str = None) -> Set[str]:
        """Get or compute image categories"""
        if image_id not in self.image_categories:
            self.image_categories[image_id] = self.categorize_image(captions, question)
        return self.image_categories[image_id]

    def categorize_sample(self, image_id: int, question: str, answer: str, criteria: Set[str]) -> bool:
        """Check if question or answer matches criteria"""
        
        # Check both question and answer content
        answer = answer.lower()
        question = question.lower()
        answer_categories = set()
        question_categories = set()
        
        for category, keywords in self.category_mapping.items():
            if any(keyword in answer for keyword in keywords):
                answer_categories.add(category)
            if any(keyword in question for keyword in keywords):
                question_categories.add(category)
        
        # Determine target category from criteria
        target_category = None
        for cat in ['animal', 'human', 'indoor', 'outdoor']:
            if cat == 'animal' and any(animal in criteria for animal in self.category_mapping['animal']):
                target_category = cat
                break
            elif cat == 'human' and any(human in criteria for human in self.category_mapping['human']):
                target_category = cat
                break
            elif cat == 'indoor' and any(indoor in criteria for indoor in self.category_mapping['indoor']):
                target_category = cat
                break
            elif cat == 'outdoor' and any(outdoor in criteria for outdoor in self.category_mapping['outdoor']):
                target_category = cat
                break
        
        # Check if either question or answer matches the target category
        return target_category in answer_categories or target_category in question_categories

class DatasetSplitter:
    def __init__(self):
        # Level 1: Split Type
        self.split_hierarchy = {
            'image_based': {
                'scene': {  # Level 2: Scene Type
                    'indoor': {'room', 'couch', 'bed', 'toilet', 'tv', 'mouse', 'keyboard', 'remote','cellphone','microwave','oven','refrigerator','sink','toaster','shower'},
                    'outdoor': {'bicycle', 'car', 'truck', 'bus', 'motorcycle', 'train', 'airplane', 'boat', 'sky', 'traffic light', 'stop sign', 'parking meter', 'fire hydrant'}
                },
                'subject': {  # Level 2: Subject Type
                    'animal': {'dog', 'cat', 'bird', 'horse', 'sheep', 'cow', 'bear', 'elephant', 'zebra', 'giraffe'},
                    'human': {'person', 'people', 'man', 'woman', 'child', 'group', 'human'},
                    'object': {'car', 'chair', 'table', 'phone', 'book', 'computer'}
                }
            },
            'question_based': {
                'reasoning': {  # Level 2: Reasoning Type
                    'simple': {
                        'patterns': [r'what color', r'how many', r'where is'],
                        'max_length': 8
                    },
                    'complex': {
                        'patterns': [r'why', r'how does', r'what would happen'],
                        'indicators': ['because', 'if', 'when', 'while']
                    }
                },
                'type': {  # Level 2: Question Type
                    'counting': {
                        'patterns': [r'how many\s+', r'count\s+'],
                        'keywords': {'many', 'count', 'number'}
                    },
                    'descriptive': {
                        'patterns': [r'what (is|are)', r'describe'],
                        'keywords': {'what', 'describe', 'look'}
                    },
                    'action': {
                        'patterns': [r'what.+doing', r'action'],
                        'keywords': {'doing', 'action', 'activity'}
                    },
                    'yes_no': {
                        'patterns': [r'^is', r'^are', r'^does'],
                        'keywords': {'is', 'are', 'does'}
                    }
                }
            }
        }

    def get_split_config(self, split_type: str, level2: str, train_category: str, val_category: str):
        """Get configuration for dataset splitting"""
        if split_type not in self.split_hierarchy:
            raise ValueError(f"Invalid split type. Choose from: {list(self.split_hierarchy.keys())}")
        
        if level2 not in self.split_hierarchy[split_type]:
            raise ValueError(f"Invalid level2 for {split_type}. Choose from: {list(self.split_hierarchy[split_type].keys())}")
            
        categories = self.split_hierarchy[split_type][level2]
        if train_category not in categories or val_category not in categories:
            raise ValueError(f"Invalid categories. Choose from: {list(categories.keys())}")
        
        return {
            'train': {
                'split_type': split_type,
                'level2': level2,
                'category': train_category,
                'criteria': categories[train_category]
            },
            'val': {
                'split_type': split_type,
                'level2': level2,
                'category': val_category,
                'criteria': categories[val_category]
            }
        } 