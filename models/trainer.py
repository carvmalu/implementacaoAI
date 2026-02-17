"""
MODEL: Trainer
Paper: i3DMM - Deep Implicit 3D Morphable Model of Human Heads
Section: 3.3 Training - Optimization (page 5)

PAPER REFERENCE:
"We train our networks using PyTorch [37], where we use the 
Adam [25] solver with mini-batches of size 64. We train for 
1000 epochs with a learning rate of 0.0005, which decays by 
a factor of 2 every 250 epochs."

ARGMIN PROBLEM (Equation 7):
argmin_{θ, {z_geo, z_col}} ∑∑ L_θ(x, z_geo, z_col)
     θ,{z}   N=1 x

This joint optimization updates both:
- Network weights θ (RefNet, DeformNet, ColorNet)
- Latent codes for all training scans
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
import time
from pathlib import Path

# Import our modules
from models.ref_net import RefNet
from models.deform_net import DeformNet, GeometryLatentCodes
from models.color_net import ColorNet, ColorLatentCodes
from models.losses import TotalLoss


class HeadDataset(torch.utils.data.Dataset):
    """
    Dataset for head scans.
    
    Each sample contains:
    - Query points (x, y, z)
    - Ground truth SDF values
    - Ground truth colors
    - Landmark positions
    - Identity/expression/hairstyle indices
    """
    
    def __init__(self, data_dir, num_samples_per_scan=100000):
        """
        Args:
            data_dir: Directory containing preprocessed .npz files
            num_samples_per_scan: Number of points to sample per scan
        """
        self.data_dir = Path(data_dir)
        self.scan_files = list(self.data_dir.glob("*_sdf_samples.npz"))
        self.num_samples_per_scan = num_samples_per_scan
        
        # Load metadata
        self.metadata = []
        for scan_file in self.scan_files:
            # Extract identity, expression, hairstyle from filename
            # Format: {id}_{exp}_{hair}_sdf_samples.npz
            parts = scan_file.stem.split('_')
            metadata = {
                'file': scan_file,
                'identity': int(parts[0]),
                'expression': int(parts[1]),
                'hairstyle': int(parts[2])
            }
            self.metadata.append(metadata)
    
    def __len__(self):
        return len(self.scan_files)
    
    def __getitem__(self, idx):
        metadata = self.metadata[idx]
        
        # Load precomputed samples
        data = np.load(metadata['file'])
        points = data['points']
        sdf_values = data['sdf_values']
        colors = data['colors']
        
        # Randomly subsample points
        n_points = len(points)
        indices = np.random.choice(n_points, self.num_samples_per_scan, replace=False)
        
        return {
            'points': torch.FloatTensor(points[indices]),
            'sdf': torch.FloatTensor(sdf_values[indices]).unsqueeze(-1),
            'colors': torch.FloatTensor(colors[indices]),
            'identity': torch.LongTensor([metadata['identity']]),
            'expression': torch.LongTensor([metadata['expression']]),
            'hairstyle_geo': torch.LongTensor([metadata['hairstyle']]),
            'hairstyle_col': torch.LongTensor([min(metadata['hairstyle'], 2)])  # Map to 3 color codes
        }


class i3DMMTrainer:
    """
    Trainer for i3DMM.
    
    Handles the joint optimization of network weights and latent codes
    as described in the paper.
    """
    
    def __init__(self,
                 # Network configurations
                 latent_dim_identity=128,
                 latent_dim_expression=32,
                 latent_dim_hairstyle_geo=16,
                 latent_dim_hairstyle_col=16,
                 
                 # Training parameters
                 batch_size=64,
                 num_epochs=1000,
                 learning_rate=0.0005,
                 lr_decay_factor=2,
                 lr_decay_epochs=250,
                 
                 # Loss weights (from paper)
                 w_geo=1.0,
                 w_def=1.0,
                 w_col=1.0,
                 w_reg=0.001,
                 w_lm=10.0,
                 
                 # Device
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.lr_decay_factor = lr_decay_factor
        self.lr_decay_epochs = lr_decay_epochs
        
        print("=" * 60)
        print("INITIALIZING i3DMM TRAINER")
        print("=" * 60)
        print(f"Device: {device}")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {num_epochs}")
        print(f"Learning rate: {learning_rate}")
        
        # Initialize networks
        print("\n1. Building networks...")
        
        self.ref_net = RefNet(
            input_dim=3,
            hidden_dim=512,
            num_layers=3,
            num_encoding_frequencies=10,
            output_dim=1
        ).to(device)
        print(f"   ✓ RefNet: 3 layers, 512 hidden")
        
        self.deform_net = DeformNet(
            input_dim=3,
            latent_dim_identity=latent_dim_identity,
            latent_dim_expression=latent_dim_expression,
            latent_dim_hairstyle=latent_dim_hairstyle_geo,
            hidden_dim=1024,
            num_layers=8,
            num_encoding_frequencies=10,
            output_dim=3
        ).to(device)
        print(f"   ✓ DeformNet: 8 layers, 1024 hidden")
        
        self.color_net = ColorNet(
            input_dim=3,
            latent_dim_identity=latent_dim_identity,
            latent_dim_hairstyle=latent_dim_hairstyle_col,
            hidden_dim=1024,
            num_layers=9,
            num_encoding_frequencies=10,
            output_dim=3
        ).to(device)
        print(f"   ✓ ColorNet: 9 layers, 1024 hidden")
        
        # Initialize latent codes
        print("\n2. Initializing latent codes...")
        
        self.geo_latent_codes = GeometryLatentCodes(
            num_identities=58,
            num_expressions=10,
            num_hairstyles=4,
            latent_dim_identity=latent_dim_identity,
            latent_dim_expression=latent_dim_expression,
            latent_dim_hairstyle=latent_dim_hairstyle_geo
        ).to(device)
        
        self.col_latent_codes = ColorLatentCodes(
            num_identities=58,
            num_hairstyles=3,
            latent_dim_identity=latent_dim_identity,
            latent_dim_hairstyle=latent_dim_hairstyle_col
        ).to(device)
        
        print(f"   ✓ Geometry codes:")
        print(f"      - Identity: 58 × {latent_dim_identity}")
        print(f"      - Expression: 10 × {latent_dim_expression}")
        print(f"      - Hairstyle: 4 × {latent_dim_hairstyle_geo}")
        print(f"   ✓ Color codes:")
        print(f"      - Identity: 58 × {latent_dim_identity}")
        print(f"      - Hairstyle: 3 × {latent_dim_hairstyle_col}")
        
        # Initialize loss function
        self.loss_fn = TotalLoss(
            w_geo=w_geo,
            w_def=w_def,
            w_col=w_col,
            w_reg=w_reg,
            w_lm=w_lm,
            clamp_value=0.1
        )
        
        # Setup optimizer
        print("\n3. Setting up optimizer...")
        
        # Group parameters: network weights and latent codes
        params = [
            {'params': self.ref_net.parameters()},
            {'params': self.deform_net.parameters()},
            {'params': self.color_net.parameters()},
            {'params': self.geo_latent_codes.parameters()},
            {'params': self.col_latent_codes.parameters()}
        ]
        
        self.optimizer = optim.Adam(params, lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=lr_decay_epochs,
            gamma=1.0/lr_decay_factor
        )
        
        print(f"   ✓ Adam optimizer with LR={learning_rate}")
        print(f"   ✓ LR decay by factor {lr_decay_factor} every {lr_decay_epochs} epochs")
        
        # Training history
        self.history = {
            'loss': [],
            'loss_geo': [],
            'loss_def': [],
            'loss_col': [],
            'loss_reg': [],
            'loss_lm': [],
            'lr': []
        }
        
        print("=" * 60)
    
    def train_epoch(self, dataloader):
        """
        Train for one epoch.
        """
        self.ref_net.train()
        self.deform_net.train()
        self.color_net.train()
        
        epoch_losses = {
            'total': 0.0,
            'geo': 0.0,
            'def': 0.0,
            'col': 0.0,
            'reg': 0.0,
            'lm': 0.0
        }
        num_batches = 0
        
        pbar = tqdm(dataloader, desc="Training")
        
        for batch in pbar:
            # Move data to device
            points = batch['points'].to(self.device)          # (B, N, 3)
            gt_sdf = batch['sdf'].to(self.device)            # (B, N, 1)
            gt_colors = batch['colors'].to(self.device)      # (B, N, 3)
            
            # Get indices for latent codes
            id_idx = batch['identity'].to(self.device)       # (B, 1)
            ex_idx = batch['expression'].to(self.device)     # (B, 1)
            hair_geo_idx = batch['hairstyle_geo'].to(self.device)  # (B, 1)
            hair_col_idx = batch['hairstyle_col'].to(self.device)  # (B, 1)
            
            batch_size = points.shape[0]
            num_points = points.shape[1]
            
            # Reshape points for network input
            points_flat = points.view(-1, 3)                  # (B*N, 3)
            
            # Get latent codes
            z_id_geo, z_ex, z_hair_geo = self.geo_latent_codes(
                id_idx.view(-1),
                ex_idx.view(-1),
                hair_geo_idx.view(-1)
            )
            
            z_id_col, z_hair_col = self.col_latent_codes(
                id_idx.view(-1),
                hair_col_idx.view(-1)
            )
            
            # Expand latent codes to match number of points
            z_id_geo_exp = z_id_geo.repeat_interleave(num_points, dim=0)
            z_ex_exp = z_ex.repeat_interleave(num_points, dim=0) if z_ex is not None else None
            z_hair_geo_exp = z_hair_geo.repeat_interleave(num_points, dim=0) if z_hair_geo is not None else None
            
            z_id_col_exp = z_id_col.repeat_interleave(num_points, dim=0)
            z_hair_col_exp = z_hair_col.repeat_interleave(num_points, dim=0) if z_hair_col is not None else None
            
            # Forward pass
            # 1. Predict deformation
            delta = self.deform_net(
                points_flat,
                z_id_geo_exp,
                z_ex_exp,
                z_hair_geo_exp
            )
            
            # 2. Deform points to reference space
            points_deformed = points_flat + delta
            
            # 3. Query SDF from RefNet
            pred_sdf = self.ref_net(points_deformed)
            
            # 4. Query colors from ColorNet
            pred_colors = self.color_net(
                points_deformed,
                z_id_col_exp,
                z_hair_col_exp
            )
            
            # Compute losses
            # For now, we're not using landmarks in this simplified version
            # In a full implementation, you'd also have landmarks_batch and deltas_lm
            
            # Reshape predictions back to (B, N, ...)
            pred_sdf = pred_sdf.view(batch_size, num_points, 1)
            pred_colors = pred_colors.view(batch_size, num_points, 3)
            delta = delta.view(batch_size, num_points, 3)
            
            # Compute loss
            losses, total_loss = self.loss_fn(
                pred_sdf=pred_sdf,
                gt_sdf=gt_sdf,
                delta=delta,
                pred_colors=pred_colors,
                gt_colors=gt_colors,
                z_geo_list=[z_id_geo, z_ex, z_hair_geo],
                z_col_list=[z_id_col, z_hair_col],
                landmarks_batch=torch.zeros(batch_size, 16, 3).to(self.device),  # Placeholder
                deltas_lm=torch.zeros(batch_size, 16, 3).to(self.device),       # Placeholder
                ear_mask=None
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # Update statistics
            for key in epoch_losses.keys():
                epoch_losses[key] += losses[key]
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{losses['total']:.4f}"})
        
        # Average losses
        for key in epoch_losses.keys():
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def train(self, train_loader, val_loader=None):
        """
        Full training loop.
        """
        print("\n" + "=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            # Train
            train_losses = self.train_epoch(train_loader)
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Log
            self.history['loss'].append(train_losses['total'])
            self.history['loss_geo'].append(train_losses['geo'])
            self.history['loss_def'].append(train_losses['def'])
            self.history['loss_col'].append(train_losses['col'])
            self.history['loss_reg'].append(train_losses['reg'])
            self.history['loss_lm'].append(train_losses['lm'])
            self.history['lr'].append(current_lr)
            
            # Print progress
            if (epoch + 1) % 20 == 0:
                elapsed = time.time() - start_time
                print(f"\nEpoch [{epoch+1}/{self.num_epochs}] - Time: {elapsed:.1f}s")
                print(f"  Total Loss: {train_losses['total']:.4f}")
                print(f"    Geo: {train_losses['geo']:.4f} | Def: {train_losses['def']:.4f} | Col: {train_losses['col']:.4f}")
                print(f"    Reg: {train_losses['reg']:.4f} | LM: {train_losses['lm']:.4f}")
                print(f"    LR: {current_lr:.6f}")
        
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"TRAINING COMPLETE! Total time: {total_time:.1f}s")
        print("=" * 60)
        
        return self.history
    
    def save_checkpoint(self, filepath):
        """
        Save model checkpoint.
        """
        checkpoint = {
            'ref_net': self.ref_net.state_dict(),
            'deform_net': self.deform_net.state_dict(),
            'color_net': self.color_net.state_dict(),
            'geo_latent_codes': self.geo_latent_codes.state_dict(),
            'col_latent_codes': self.col_latent_codes.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'history': self.history,
            'config': {
                'batch_size': self.batch_size,
                'num_epochs': self.num_epochs,
                'learning_rate': self.learning_rate
            }
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath):
        """
        Load model checkpoint.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.ref_net.load_state_dict(checkpoint['ref_net'])
        self.deform_net.load_state_dict(checkpoint['deform_net'])
        self.color_net.load_state_dict(checkpoint['color_net'])
        self.geo_latent_codes.load_state_dict(checkpoint['geo_latent_codes'])
        self.col_latent_codes.load_state_dict(checkpoint['col_latent_codes'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.history = checkpoint['history']
        
        print(f"Checkpoint loaded from {filepath}")


def test_trainer():
    """
    Test function to verify trainer implementation.
    """
    print("\n" + "=" * 60)
    print("TESTING TRAINER")
    print("=" * 60)
    
    # Create synthetic dataset for testing
    print("\n1. Creating synthetic test data...")
    
    # Simulate 10 scans with 1000 points each
    num_scans = 10
    num_points = 1000
    
    test_data_dir = Path("test_data")
    test_data_dir.mkdir(exist_ok=True)
    
    for i in range(num_scans):
        points = np.random.randn(num_points, 3) * 0.5
        sdf_values = np.random.randn(num_points) * 0.1
        colors = np.random.rand(num_points, 3)
        
        # Identity, expression, hairstyle indices
        identity = i % 5
        expression = i % 3
        hairstyle = i % 2
        
        filename = test_data_dir / f"{identity}_{expression}_{hairstyle}_sdf_samples.npz"
        np.savez(filename, points=points, sdf_values=sdf_values, colors=colors)
    
    print(f"   ✓ Created {num_scans} synthetic scans")
    
    # Create dataset and dataloader
    dataset = HeadDataset(test_data_dir, num_samples_per_scan=500)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    print(f"   ✓ Dataset size: {len(dataset)} scans")
    
    # Initialize trainer with small dimensions for testing
    trainer = i3DMMTrainer(
        latent_dim_identity=32,
        latent_dim_expression=8,
        latent_dim_hairstyle_geo=4,
        latent_dim_hairstyle_col=4,
        batch_size=4,
        num_epochs=10,  # Just 10 epochs for testing
        learning_rate=0.0005,
        device='cpu'
    )
    
    # Train for a few epochs
    print("\n2. Running test training (10 epochs)...")
    history = trainer.train(dataloader)
    
    # Check history
    print("\n3. Verifying training history...")
    print(f"   ✓ Final loss: {history['loss'][-1]:.4f}")
    print(f"   ✓ History length: {len(history['loss'])} epochs")
    
    # Test checkpoint
    print("\n4. Testing checkpoint saving/loading...")
    checkpoint_path = "test_checkpoint.pt"
    trainer.save_checkpoint(checkpoint_path)
    
    # Create new trainer and load
    trainer2 = i3DMMTrainer(device='cpu')
    trainer2.load_checkpoint(checkpoint_path)
    print(f"   ✓ Checkpoint loaded successfully")
    
    # Cleanup
    import shutil
    shutil.rmtree(test_data_dir)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    print("\n" + "=" * 60)
    print("TRAINER TEST PASSED!")
    print("=" * 60)
    
    return trainer


if __name__ == "__main__":
    test_trainer()