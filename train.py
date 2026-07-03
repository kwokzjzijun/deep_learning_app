import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import time
import json
from model import create_model

class UricAcidDataset(Dataset):
    """Dataset class for uric acid test strip images"""
    
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.samples = []
        
        # Load all image paths and labels
        for label_folder in os.listdir(data_path):
            label_path = os.path.join(data_path, label_folder)
            if os.path.isdir(label_path):
                try:
                    label = float(label_folder)  # Convert folder name to concentration
                    for img_file in os.listdir(label_path):
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(label_path, img_file)
                            self.samples.append((img_path, label))
                except ValueError:
                    continue
        
        print(f"Loaded {len(self.samples)} samples from {data_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)

def get_transforms():
    """Define image transforms for training and validation"""
    
    # No additional augmentation since data is already augmented
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_data_loaders(train_path, val_path, test_path, batch_size=128):
    """Create data loaders for training, validation, and testing"""
    
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    train_dataset = UricAcidDataset(train_path, transform=train_transform)
    val_dataset = UricAcidDataset(val_path, transform=val_transform)
    test_dataset = UricAcidDataset(test_path, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=15, min_delta=0.1, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

def train_epoch(model, train_loader, criterion, optimizer, device, scheduler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    predictions = []
    targets = []
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.squeeze(), target)
        
        loss.backward()
        # Gradient clipping to prevent gradient explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        predictions.extend(output.squeeze().detach().cpu().numpy())
        targets.extend(target.detach().cpu().numpy())
        
        if batch_idx % 10 == 0:
            print(f'Train Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    if scheduler:
        scheduler.step()
    
    avg_loss = total_loss / len(train_loader)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    
    return avg_loss, mae, r2

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output.squeeze(), target)
            
            total_loss += loss.item()
            predictions.extend(output.squeeze().cpu().numpy())
            targets.extend(target.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    
    return avg_loss, mae, r2

def test_model(model, test_loader, device):
    """Test the model and return predictions"""
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            predictions.extend(output.squeeze().cpu().numpy())
            targets.extend(target.cpu().numpy())
    
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    
    return predictions, targets, mae, r2

def plot_training_history(train_losses, val_losses, train_maes, val_maes):
    """Create SCI-style training plots"""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss plot
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # MAE plot (acting as accuracy metric)
    ax2.plot(epochs, train_maes, 'b-', linewidth=2, label='Training MAE')
    ax2.plot(epochs, val_maes, 'r-', linewidth=2, label='Validation MAE')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Mean Absolute Error (ppm)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy (MAE)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Configuration
    config = {
        'data_path': '0724_Enlarged',
        'batch_size': 128,
        'learning_rate': 0.001,
        'num_epochs': 100,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'model_save_path': 'best_uric_acid_model.pth'
    }
    
    print(f"Using device: {config['device']}")
    device = torch.device(config['device'])
    
    # Data paths
    train_path = os.path.join(config['data_path'], 'train')
    val_path = os.path.join(config['data_path'], 'val')
    test_path = os.path.join(config['data_path'], 'test')
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_path, val_path, test_path, config['batch_size']
    )
    
    # Create model
    model = create_model(num_classes=1, dropout_rate=0.5)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], 
                           weight_decay=0.05)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=15, min_delta=0.05)
    
    # Training history
    train_losses, val_losses = [], []
    train_maes, val_maes = [], []
    
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(config['num_epochs']):
        print(f'\nEpoch {epoch+1}/{config["num_epochs"]}')
        print('-' * 50)
        
        # Train
        train_loss, train_mae, train_r2 = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_mae, val_r2 = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Record history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_maes.append(train_mae)
        val_maes.append(val_mae)
        
        print(f'Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, Train R²: {train_r2:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, Val R²: {val_r2:.4f}')
        
        # Early stopping check
        if early_stopping(val_loss, model):
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    training_time = time.time() - start_time
    print(f'\nTraining completed in {training_time/60:.2f} minutes')
    
    # Save model
    torch.save(model.state_dict(), config['model_save_path'])
    print(f"Model saved to {config['model_save_path']}")
    
    # Test the model
    print("\nTesting model...")
    predictions, targets, test_mae, test_r2 = test_model(model, test_loader, device)
    
    print(f'Test MAE: {test_mae:.4f} ppm')
    print(f'Test R²: {test_r2:.4f}')
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_maes, val_maes)
    
    # Save training results
    results = {
        'config': config,
        'training_time_minutes': float(training_time/60),  # 转换为Python float
        'final_epoch': len(train_losses),
        'best_val_loss': float(min(val_losses)),  # 转换为Python float
        'best_val_mae': float(min(val_maes)),     # 转换为Python float
        'test_mae': float(test_mae),              # 转换为Python float
        'test_r2': float(test_r2),                # 转换为Python float
        'train_losses': [float(x) for x in train_losses],  # 转换列表中的所有元素
        'val_losses': [float(x) for x in val_losses],      # 转换列表中的所有元素
        'train_maes': [float(x) for x in train_maes],      # 转换列表中的所有元素
        'val_maes': [float(x) for x in val_maes]           # 转换列表中的所有元素
    }
    
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nTraining results saved to training_results.json")
    
    # Example prediction
    if predictions and targets:
        sample_idx = np.random.randint(0, len(predictions))
        predicted_concentration = float(predictions[sample_idx])  # 转换为Python float
        actual_concentration = float(targets[sample_idx])         # 转换为Python float
        
        print(f"\nExample prediction:")
        print(f"Predicted: {predicted_concentration:.1f} ppm")
        print(f"Actual: {actual_concentration:.1f} ppm")
        print(f"Error: {abs(predicted_concentration - actual_concentration):.1f} ppm")

if __name__ == "__main__":
    main()