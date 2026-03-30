"""
dataset.py — PyTorch Dataset classes for FER-2013 and CK+
These classes load images and labels for training/testing.
"""
 
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from preprocess import get_train_transforms, get_test_transforms

class FERDataset(Dataset):
    """
    Dataset class for FER-2013.
    Expects folder structure:
        data/fer2013/train/angry/img1.jpg
        data/fer2013/train/happy/img2.jpg
        etc.
    """
    
    EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    
    def __init__(self, root_dir, split='train'):
        """
        Args:
            root_dir: Path to fer2013/ folder
            split: 'train' or 'test'
        """
        self.root_dir = Path(root_dir) / split
        self.split = split
        self.transform = get_train_transforms() if split == 'train' else get_test_transforms()
                
        # Build list of (image_path, label) pairs
        self.samples = []
        for label_idx, emotion in enumerate(self.EMOTIONS):
            emotion_dir = self.root_dir / emotion
            if not emotion_dir.exists():
                print(f"⚠️  Warning: {emotion_dir} not found")
                continue
            
            for img_file in emotion_dir.iterdir():
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(img_file), label_idx))
        
        print(f"✅ Loaded {len(self.samples)} images for FER-2013 {split}")
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Apply transforms
        img_tensor = self.transform(img)
        
        return img_tensor, torch.tensor(label, dtype=torch.long)
class CKPlusDataset(Dataset):
    """
    Dataset class for CK+ dataset.
    Same structure as FERDataset but with 6 emotion classes.
    """
    
    EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise"]
    
    def __init__(self, root_dir, split='train'):
        self.root_dir = Path(root_dir) / split
        self.split = split
        self.transform = get_train_transforms() if split == 'train' else get_test_transforms()
        self.samples = []
        for label_idx, emotion in enumerate(self.EMOTIONS):
            emotion_dir = self.root_dir / emotion
            if not emotion_dir.exists():
                continue
            for img_file in emotion_dir.iterdir():
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(img_file), label_idx))
        
        print(f"✅ Loaded {len(self.samples)} images for CK+ {split}")
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_tensor = self.transform(img)
        return img_tensor, torch.tensor(label, dtype=torch.long)
def get_dataloaders(dataset_name='fer2013', batch_size=32):
    """
    Creates train and test DataLoaders ready for training.
    
    Returns:
        train_loader, test_loader, num_classes
    """
    import yaml
    with open('config.yaml') as f:
        cfg = yaml.safe_load(f)
    
    if dataset_name == 'fer2013':
        train_ds = FERDataset(cfg['data']['fer2013_path'], 'train')
        test_ds  = FERDataset(cfg['data']['fer2013_path'], 'test')
        num_classes = cfg['data']['num_classes_fer']
    else:
        train_ds = CKPlusDataset(cfg['data']['ckplus_path'], 'train')
        test_ds  = CKPlusDataset(cfg['data']['ckplus_path'], 'test')
        num_classes = cfg['data']['num_classes_ck']
    
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True,          # Shuffle training data every epoch
        num_workers=2,         # Parallel data loading
        pin_memory=True        # Faster GPU transfer
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False,         # No shuffle for test
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, test_loader, num_classes
    
