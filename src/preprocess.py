import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
# ─── Define emotion label mappings ───────────────────────────────────────────
EMOTIONS_FER = {
    0: "angry",   1: "disgust", 2: "fear",    3: "happy",
    4: "sad",     5: "surprise", 6: "neutral"
}
EMOTIONS_CK = {
    0: "angry",   1: "disgust", 2: "fear",
    3: "happy",   4: "sad",     5: "surprise"
}


def get_train_transforms():
    """
    Augmentation applied during TRAINING only.
    These create variations so the model learns to be robust.
    """
    return transforms.Compose([
        # Step 1: Resize to 48x48 using bicubic interpolation (smoother than bilinear)
        transforms.Resize((48, 48), interpolation=transforms.InterpolationMode.BICUBIC),
 
        # Step 2: Random horizontal flip (50% chance) — faces look similar mirrored
        transforms.RandomHorizontalFlip(p=0.5),
 
        # Step 3: Random rotation ±15 degrees — handles head tilts
        transforms.RandomRotation(degrees=15),
 
        # Step 4: Brightness and contrast variation ±20%
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
 
        # Step 5: Random translation ±10% of image dimensions
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
 
        # Step 6: Convert PIL Image to tensor (values become 0.0 to 1.0 automatically)
        transforms.ToTensor(),
 
        # Step 7: Normalize using ImageNet mean/std (standard practice)
        # For grayscale images converted to RGB:
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


# ─── Test/Validation transforms (NO augmentation) ────────────────────────────
def get_test_transforms():
    """
    Transforms applied during TESTING and INFERENCE.
    No random augmentation — we want consistent results.
    """
    return transforms.Compose([
        transforms.Resize((48, 48), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

# ─── Preprocessing function for single image ─────────────────────────────────
def preprocess_image(image_path, mode='test'):
    """
    Load and preprocess a single image from file path.
    
    Args:
        image_path: Path to image file
        mode: 'train' (with augmentation) or 'test' (without)
    
    Returns:
        tensor: Shape [1, 3, 48, 48] — batch of 1 image
    """
    # Load image using PIL
    img = Image.open(image_path)
    
    # Convert grayscale to RGB (model expects 3 channels)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Apply transforms
    if mode == 'train':
        transform = get_train_transforms()
    else:
        transform = get_test_transforms()
    
    # Apply transform and add batch dimension
    tensor = transform(img).unsqueeze(0)  # [1, 3, 48, 48]
    return tensor


# ─── Preprocess from numpy array (for webcam input) ──────────────────────────
def preprocess_frame(frame_bgr):
    """
    Preprocess a frame captured from webcam (BGR numpy array from OpenCV).
    
    Args:
        frame_bgr: numpy array in BGR format (from cv2.VideoCapture)
    
    Returns:
        tensor: Shape [1, 3, 48, 48]
    """
    # OpenCV uses BGR, PIL uses RGB — convert
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    img = Image.fromarray(frame_rgb)
    
    # Apply test transforms
    transform = get_test_transforms()
    tensor = transform(img).unsqueeze(0)
    return tensor


# ─── Denormalize for visualization ───────────────────────────────────────────
def denormalize(tensor):
    """
    Reverse normalization for displaying images.
    Used when you want to show the preprocessed image visually.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean
