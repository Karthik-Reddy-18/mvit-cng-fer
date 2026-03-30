"""
evaluate.py — Full evaluation with metrics, confusion matrix, and plots
"""
 
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from pathlib import Path
import json
import yaml
 
import sys, os
sys.path.append(os.path.dirname(__file__))
from model import MViTCnG
from dataset import get_dataloaders
 
# Emotion labels for display
EMOTIONS_FER = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
EMOTIONS_CK  = ["angry", "disgust", "fear", "happy", "sad", "surprise"]
 
def load_trained_model(checkpoint_path, num_classes, device):
    """Load a saved model checkpoint."""
    with open('config.yaml') as f:
        cfg = yaml.safe_load(f)
    
    model = MViTCnG(
        num_classes=num_classes,
        image_size=cfg['data']['image_size'],
        embed_dim=256,
        num_heads=cfg['model']['num_heads'],
        num_layers=6,
        mlp_ratio=cfg['model']['mlp_ratio'],
        dropout=0.0,  # No dropout during evaluation
        contrastive_dim=cfg['model']['contrastive_dim']
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded from {checkpoint_path}")
    print(f"   Best accuracy during training: {checkpoint['best_acc']:.2f}%")
    return model
 
@torch.no_grad()
def get_predictions(model, dataloader, device):
    """Run model on entire dataset and collect predictions."""
    all_preds = []
    all_labels = []
    all_probs = []
    
    for images, labels in dataloader:
        images = images.to(device)
        logits, _ = model(images)
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)
 
def compute_metrics(y_true, y_pred, emotion_names):
    """Compute all evaluation metrics."""
    metrics = {
        'accuracy':  accuracy_score(y_true, y_pred) * 100,
        'precision': precision_score(y_true, y_pred, average='weighted') * 100,
        'recall':    recall_score(y_true, y_pred, average='weighted') * 100,
        'f1':        f1_score(y_true, y_pred, average='weighted') * 100
    }
    
    print("\n" + "=" * 50)
    print("📊 EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"  Precision: {metrics['precision']:.2f}%")
    print(f"  Recall:    {metrics['recall']:.2f}%")
    print(f"  F1-Score:  {metrics['f1']:.2f}%")
    print("\n" + classification_report(y_true, y_pred, target_names=emotion_names))
    
    return metrics
 
def plot_confusion_matrix(y_true, y_pred, emotion_names, save_path):
    """Generate and save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized, annot=True, fmt='.2f', cmap='Blues',
        xticklabels=emotion_names, yticklabels=emotion_names,
        linewidths=0.5, linecolor='gray'
    )
    plt.title('Confusion Matrix (Normalized)', fontsize=15, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved to: {save_path}")
 
def plot_training_curves(history_path, save_dir):
    """Plot accuracy and loss curves from training history."""
    with open(history_path) as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_acc']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    ax1.plot(epochs, history['train_acc'], 'b-', label='Training', linewidth=2)
    ax1.plot(epochs, history['test_acc'],  'r-', label='Testing',  linewidth=2)
    ax1.set_title('Model Accuracy vs Epochs', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 100])
    
    # Loss plot
    ax2.plot(epochs, history['train_loss'], 'b-', label='Training', linewidth=2)
    ax2.plot(epochs, history['test_loss'],  'r-', label='Testing',  linewidth=2)
    ax2.set_title('Model Loss vs Epochs', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Training curves saved to: {save_dir}/training_curves.png")
 
def plot_per_class_metrics(y_true, y_pred, emotion_names, save_path):
    """Bar chart showing per-class F1 scores."""
    f1_per_class = f1_score(y_true, y_pred, average=None) * 100
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(emotion_names)))
    bars = plt.bar(emotion_names, f1_per_class, color=colors, edgecolor='navy', linewidth=1.2)
    
    # Add value labels on bars
    for bar, val in zip(bars, f1_per_class):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.title('F1-Score per Emotion Class', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('F1-Score (%)')
    plt.xlabel('Emotion')
    plt.ylim([0, 110])
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Per-class metrics saved to: {save_path}")
 
def run_full_evaluation(dataset_name='fer2013'):
    """Run complete evaluation pipeline."""
    with open('config.yaml') as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if dataset_name == 'fer2013':
        num_classes = cfg['data']['num_classes_fer']
        emotion_names = EMOTIONS_FER
    else:
        num_classes = cfg['data']['num_classes_ck']
        emotion_names = EMOTIONS_CK
    
    # Load model and data
    model = load_trained_model(
        f"models/mvit_cng_{dataset_name}.pth",
        num_classes, device
    )
    _, test_loader, _ = get_dataloaders(dataset_name, batch_size=64)
    
    # Get predictions
    y_true, y_pred, y_probs = get_predictions(model, test_loader, device)
    
    # Compute and print metrics
    metrics = compute_metrics(y_true, y_pred, emotion_names)
    
    # Generate plots
    save_dir = "outputs/plots"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    plot_confusion_matrix(y_true, y_pred, emotion_names,
                          f"{save_dir}/confusion_matrix_{dataset_name}.png")
    
    plot_per_class_metrics(y_true, y_pred, emotion_names,
                           f"{save_dir}/per_class_f1_{dataset_name}.png")
    
    history_path = f"outputs/logs/history_{dataset_name}.json"
    if Path(history_path).exists():
        plot_training_curves(history_path, save_dir)
    
    return metrics
 
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='fer2013', choices=['fer2013', 'ckplus'])
    args = parser.parse_args()
    run_full_evaluation(args.dataset)
