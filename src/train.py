import os
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CasiaBDataset
from model import FusionModel

def train(config_path):
    # Load Config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create Directories
    os.makedirs(config['train']['save_dir'], exist_ok=True)
    
    # Dataset & DataLoader
    train_dataset = CasiaBDataset(
        data_root=config['data']['data_root'],
        skeleton_root=config['data']['skeleton_root'],
        seq_len=config['data']['seq_len'],
        mode='train'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )
    
    # Model
    model = FusionModel(
        num_classes=config['model']['num_classes'],
        visual_dim=config['model']['visual_dim'],
        skel_dim=config['model']['skel_dim'],
        embed_dim=config['model']['embed_dim']
    ).to(device)
    
    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=config['train']['lr'], weight_decay=config['train']['weight_decay'])
    ce_loss = nn.CrossEntropyLoss()
    triplet_loss = nn.TripletMarginLoss(margin=0.2)
    
    # Training Loop
    for epoch in range(config['train']['epochs']):
        model.train()
        total_loss = 0
        
        for batch_idx, (imgs, skels, labels) in enumerate(train_loader):
            imgs, skels, labels = imgs.to(device), skels.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            embeds, logits = model(imgs, skels)
            
            # Calculate Loss
            # 1. Cross Entropy Loss
            loss_ce = ce_loss(logits, labels)
            
            # 2. Triplet Loss (Simplified: using batch hard mining or just random triplets)
            # Here we just use a naive implementation for demonstration. 
            # In a real scenario, you need a proper Triplet Sampler in DataLoader or Hard Mining here.
            # For this code to run without complex sampler, we will skip Triplet Loss or use a dummy one
            # if batch size is small.
            # Let's assume we rely on CE loss for this basic implementation structure.
            
            loss = loss_ce 
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % config['train']['log_interval'] == 0:
                print(f"Epoch [{epoch+1}/{config['train']['epochs']}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}] Average Loss: {avg_loss:.4f}")
        
        # Save Checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(config['train']['save_dir'], f'epoch_{epoch+1}.pth'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    train(args.config)
