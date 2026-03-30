
"""
train.py — Complete training loop for MViT-CnG
"""
 
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
import os
import json
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
 
# Import our modules (all in src/ folder)
import sys
sys.path.append(os.path.dirname(__file__))
 
from model import MViTCnG, count_parameters
from contrastive_loss import CombinedLoss
from dataset import get_dataloaders
 
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """Run one complete pass through the training data."""
    model.train()  # Set to training mode (enables dropout, etc.)
    
    total_loss = 0.0
    total_ce = 0.0
    total_nce = 0.0
    correct = 0
    total_samples = 0
    
    # tqdm creates a nice progress bar
    loop = tqdm(dataloader, desc="Training", leave=False)
    
    for images, labels in loop:
        # Move data to GPU if available
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass: feed images through model
        logits, contrastive_features = model(images)
        
        # Compute combined loss
        loss, ce_val, nce_val = criterion(logits, contrastive_features, labels)
        
        # Backward pass: compute gradients
        optimizer.zero_grad()   # Clear old gradients
        loss.backward()         # Compute new gradients
        
        # Gradient clipping: prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update model weights
        optimizer.step()
        
        # Track statistics
        total_loss += loss.item()
        total_ce += ce_val
        total_nce += nce_val
        
        # Calculate accuracy
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)
        
        # Update progress bar display
        loop.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.0 * correct / total_samples:.2f}%'
        })
    
    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'ce_loss': total_ce / n,
        'nce_loss': total_nce / n,
        'accuracy': 100.0 * correct / total_samples
    }
 
@torch.no_grad()  # Decorator: disables gradient computation (saves memory + speed)
def evaluate(model, dataloader, criterion, device):
    """Evaluate model on test/validation set."""
    model.eval()  # Set to evaluation mode (disables dropout)
    
    total_loss = 0.0
    correct = 0
    total_samples = 0
    
    loop = tqdm(dataloader, desc="Evaluating", leave=False)
    
    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)
        
        logits, contrastive_features = model(images)
        loss, _, _ = criterion(logits, contrastive_features, labels)
        
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': 100.0 * correct / total_samples
    }
 
def train(dataset_name='fer2013'):
    """Main training function."""
    
    # Load config
    with open('config.yaml') as f:
        cfg = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Create output directories
    Path(cfg['training']['save_dir']).mkdir(exist_ok=True)
    Path(cfg['training']['log_dir']).mkdir(parents=True, exist_ok=True)
    Path('outputs/plots').mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n📁 Loading datasets...")
    train_loader, test_loader, num_classes = get_dataloaders(
        dataset_name=dataset_name,
        batch_size=cfg['training']['batch_size']
    )
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Test batches:  {len(test_loader)}")
    
    # Build model
    print("\n🧠 Building MViT-CnG model...")
    model = MViTCnG(
        num_classes=num_classes,
        image_size=cfg['data']['image_size'],
        embed_dim=256,
        num_heads=cfg['model']['num_heads'],
        num_layers=6,
        mlp_ratio=cfg['model']['mlp_ratio'],
        dropout=cfg['model']['dropout'],
        contrastive_dim=cfg['model']['contrastive_dim']
    ).to(device)
    
    count_parameters(model)
    
    # Setup loss function
    criterion = CombinedLoss(
        num_classes=num_classes,
        temperature=0.5,
        alpha=0.5
    )
    
    # Setup optimizer (Adam)
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg['training']['learning_rate'],
        weight_decay=1e-4  # L2 regularization
    )
    
    # Learning rate scheduler: gradually reduces LR over training
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg['training']['epochs'],
        eta_min=1e-6
    )
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [],  'test_acc': []
    }
    
    best_acc = 0.0
    
    print(f"\n🚀 Starting training for {cfg['training']['epochs']} epochs...")
    print("=" * 60)
    
    for epoch in range(1, cfg['training']['epochs'] + 1):
        print(f"\nEpoch {epoch}/{cfg['training']['epochs']}")
        
        # Train
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # Evaluate
        test_metrics = evaluate(model, test_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['test_loss'].append(test_metrics['loss'])
        history['test_acc'].append(test_metrics['accuracy'])
        
        print(f"  Train — Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.2f}%")
        print(f"  Test  — Loss: {test_metrics['loss']:.4f} | Acc: {test_metrics['accuracy']:.2f}%")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
            save_path = f"{cfg['training']['save_dir']}mvit_cng_{dataset_name}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'config': cfg
            }, save_path)
            print(f"  ✅ Best model saved! Accuracy: {best_acc:.2f}%")
    
    # Save training history
    log_path = f"{cfg['training']['log_dir']}history_{dataset_name}.json"
    with open(log_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n🎉 Training complete!")
    print(f"   Best Test Accuracy: {best_acc:.2f}%")
    print(f"   Model saved to: {save_path}")
    
    return history
 
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='fer2013', choices=['fer2013', 'ckplus'])
    args = parser.parse_args()
    train(dataset_name=args.dataset)
