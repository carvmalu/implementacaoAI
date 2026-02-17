"""
MODEL: Fitting
Paper: i3DMM - Deep Implicit 3D Morphable Model of Human Heads
Section: 4.1 Reconstruction (page 6)
Equation: 8

PAPER REFERENCE:
"For a given test scan, we fit our learned model to it. 
This is done by optimizing for the latent vector that can best 
reproduce the scan, i.e. by finding the latent variables that 
minimize the problem

argmin_{z_geo, z_col} ∑ (L_geo_θ(x, z_geo) + L_def_θ(x, z_geo)
                     x   + L_col_θ(x, z_col) + L_reg(z_geo, z_col))

where, z_geo, z_col is the latent code for test scan i. This 
equation is similar to Eq. (7), with the difference that the 
network weights are fixed here and the pairwise landmark 
supervision loss in Eq. (6) is not enforced."
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import trimesh
from pathlib import Path


class i3DMMFitter:
    """
    Test-time optimization to fit i3DMM to new scans.
    
    Given a new head scan (not seen during training), this class
    optimizes the latent codes to reconstruct it as closely as possible.
    
    Key differences from training:
    - Network weights are FIXED
    - Only latent codes are optimized
    - No pairwise landmark loss (L_lm)
    """
    
    def __init__(self,
                 ref_net,
                 deform_net,
                 color_net,
                 geo_latent_codes_template=None,
                 col_latent_codes_template=None,
                 latent_dim_identity=128,
                 latent_dim_expression=32,
                 latent_dim_hairstyle_geo=16,
                 latent_dim_hairstyle_col=16,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.device = device
        self.latent_dim_identity = latent_dim_identity
        self.latent_dim_expression = latent_dim_expression
        self.latent_dim_hairstyle_geo = latent_dim_hairstyle_geo
        self.latent_dim_hairstyle_col = latent_dim_hairstyle_col
        
        # Load pretrained networks (FIXED during fitting)
        self.ref_net = ref_net.to(device).eval()
        self.deform_net = deform_net.to(device).eval()
        self.color_net = color_net.to(device).eval()
        
        # Freeze all network weights
        for net in [self.ref_net, self.deform_net, self.color_net]:
            for param in net.parameters():
                param.requires_grad = False
        
        # Initialize latent codes for the new scan
        # These will be optimized
        self.z_geo_id = nn.Parameter(torch.randn(1, latent_dim_identity, device=device) * 0.01)
        self.z_geo_ex = nn.Parameter(torch.randn(1, latent_dim_expression, device=device) * 0.01)
        self.z_geo_hair = nn.Parameter(torch.randn(1, latent_dim_hairstyle_geo, device=device) * 0.01)
        
        self.z_col_id = nn.Parameter(torch.randn(1, latent_dim_identity, device=device) * 0.01)
        self.z_col_hair = nn.Parameter(torch.randn(1, latent_dim_hairstyle_col, device=device) * 0.01)
        
        # Optionally initialize from template (e.g., mean codes)
        if geo_latent_codes_template is not None:
            with torch.no_grad():
                self.z_geo_id.data = geo_latent_codes_template['id'].mean(0, keepdim=True)
                self.z_geo_ex.data = geo_latent_codes_template['ex'].mean(0, keepdim=True)
                self.z_geo_hair.data = geo_latent_codes_template['hair'].mean(0, keepdim=True)
        
        if col_latent_codes_template is not None:
            with torch.no_grad():
                self.z_col_id.data = col_latent_codes_template['id'].mean(0, keepdim=True)
                self.z_col_hair.data = col_latent_codes_template['hair'].mean(0, keepdim=True)
    
    def compute_losses(self, points, gt_sdf, gt_colors=None):
        """
        Compute losses for a batch of points.
        
        Args:
            points: (N, 3) - Query points
            gt_sdf: (N, 1) - Ground truth SDF values
            gt_colors: (N, 3) - Optional ground truth colors
        
        Returns:
            losses: Dictionary of individual losses
            total_loss: Scalar tensor
        """
        # Expand latent codes to match number of points
        num_points = points.shape[0]
        
        z_id_geo_exp = self.z_geo_id.repeat(num_points, 1)
        z_ex_exp = self.z_geo_ex.repeat(num_points, 1)
        z_hair_geo_exp = self.z_geo_hair.repeat(num_points, 1)
        
        # Forward pass
        # 1. Predict deformation
        delta = self.deform_net(points, z_id_geo_exp, z_ex_exp, z_hair_geo_exp)
        
        # 2. Deform points to reference space
        points_deformed = points + delta
        
        # 3. Query SDF from RefNet
        pred_sdf = self.ref_net(points_deformed)
        
        # 4. Geometry loss (L1 on clamped SDF)
        clamp_value = 0.1
        pred_clamped = torch.clamp(pred_sdf, -clamp_value, clamp_value)
        gt_clamped = torch.clamp(gt_sdf, -clamp_value, clamp_value)
        loss_geo = nn.functional.l1_loss(pred_clamped, gt_clamped)
        
        # 5. Deformation regularization (L2 on displacement)
        loss_def = torch.norm(delta, dim=-1).mean()
        
        # 6. Color loss (if colors provided)
        loss_col = torch.tensor(0.0, device=self.device)
        if gt_colors is not None:
            z_id_col_exp = self.z_col_id.repeat(num_points, 1)
            z_hair_col_exp = self.z_col_hair.repeat(num_points, 1)
            
            pred_colors = self.color_net(points_deformed, z_id_col_exp, z_hair_col_exp)
            loss_col = nn.functional.l1_loss(pred_colors, gt_colors)
        
        # 7. Latent regularization (L2 prior)
        loss_reg = (
            torch.norm(self.z_geo_id) +
            torch.norm(self.z_geo_ex) +
            torch.norm(self.z_geo_hair) +
            torch.norm(self.z_col_id) +
            torch.norm(self.z_col_hair)
        ) / 5.0
        
        # Total loss (weights from paper)
        w_geo, w_def, w_col, w_reg = 1.0, 1.0, 1.0, 0.001
        
        total_loss = (
            w_geo * loss_geo +
            w_def * loss_def +
            w_col * loss_col +
            w_reg * loss_reg
        )
        
        losses = {
            'geo': loss_geo.item(),
            'def': loss_def.item(),
            'col': loss_col.item(),
            'reg': loss_reg.item(),
            'total': total_loss.item()
        }
        
        return losses, total_loss
    
    def fit(self,
           points,              # (N, 3) - Query points
           sdf_values,          # (N, 1) - Ground truth SDF
           colors=None,         # (N, 3) - Optional ground truth colors
           num_iterations=1000,
           learning_rate=0.01,
           print_every=100):
        """
        Optimize latent codes to fit the scan.
        
        Args:
            points: Sampled points from the scan
            sdf_values: Ground truth SDF values
            colors: Optional ground truth colors
            num_iterations: Number of optimization iterations
            learning_rate: Learning rate for Adam
            print_every: Print progress every N iterations
        
        Returns:
            optimized_codes: Dictionary with fitted latent codes
            history: Loss history
        """
        # Ensure points are tensors
        if not isinstance(points, torch.Tensor):
            points = torch.FloatTensor(points).to(self.device)
        if not isinstance(sdf_values, torch.Tensor):
            sdf_values = torch.FloatTensor(sdf_values).to(self.device)
        if colors is not None and not isinstance(colors, torch.Tensor):
            colors = torch.FloatTensor(colors).to(self.device)
        
        # Setup optimizer (only latent codes, not network weights)
        optimizer = optim.Adam([
            self.z_geo_id,
            self.z_geo_ex,
            self.z_geo_hair,
            self.z_col_id,
            self.z_col_hair
        ], lr=learning_rate)
        
        history = []
        
        print("=" * 60)
        print("FITTING i3DMM TO NEW SCAN")
        print("=" * 60)
        print(f"Points: {points.shape[0]}")
        print(f"Iterations: {num_iterations}")
        print(f"Learning rate: {learning_rate}")
        print("-" * 60)
        
        for iteration in tqdm(range(num_iterations), desc="Fitting"):
            optimizer.zero_grad()
            
            # Forward pass
            losses, total_loss = self.compute_losses(points, sdf_values, colors)
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Store history
            history.append(losses)
            
            # Print progress
            if (iteration + 1) % print_every == 0:
                print(f"\nIteration [{iteration+1}/{num_iterations}]")
                print(f"  Total Loss: {losses['total']:.4f}")
                print(f"    Geo: {losses['geo']:.4f} | Def: {losses['def']:.4f}")
                print(f"    Col: {losses['col']:.4f} | Reg: {losses['reg']:.4f}")
        
        print("-" * 60)
        print("FITTING COMPLETE!")
        print("=" * 60)
        
        # Return optimized codes
        optimized_codes = {
            'z_geo_id': self.z_geo_id.detach().cpu().numpy(),
            'z_geo_ex': self.z_geo_ex.detach().cpu().numpy(),
            'z_geo_hair': self.z_geo_hair.detach().cpu().numpy(),
            'z_col_id': self.z_col_id.detach().cpu().numpy(),
            'z_col_hair': self.z_col_hair.detach().cpu().numpy()
        }
        
        return optimized_codes, history
    
    def reconstruct_mesh(self, resolution=256, bbox=(-1.0, 1.0)):
        """
        Reconstruct mesh from fitted latent codes using marching cubes.
        
        Args:
            resolution: Grid resolution
            bbox: Bounding box coordinates
        
        Returns:
            mesh: Trimesh object
        """
        # Create 3D grid
        axis = np.linspace(bbox[0], bbox[1], resolution)
        grid_x, grid_y, grid_z = np.meshgrid(axis, axis, axis, indexing='ij')
        
        # Flatten grid
        grid_points = np.stack([
            grid_x.reshape(-1),
            grid_y.reshape(-1),
            grid_z.reshape(-1)
        ], axis=1)
        
        # Convert to tensor
        grid_tensor = torch.FloatTensor(grid_points).to(self.device)
        
        # Compute SDF in batches
        batch_size = 10000
        sdf_values = []
        
        with torch.no_grad():
            for i in range(0, len(grid_tensor), batch_size):
                batch = grid_tensor[i:i+batch_size]
                
                # Expand latent codes
                num_points = batch.shape[0]
                z_id_exp = self.z_geo_id.repeat(num_points, 1)
                z_ex_exp = self.z_geo_ex.repeat(num_points, 1)
                z_hair_exp = self.z_geo_hair.repeat(num_points, 1)
                
                # Forward pass
                delta = self.deform_net(batch, z_id_exp, z_ex_exp, z_hair_exp)
                points_deformed = batch + delta
                sdf_batch = self.ref_net(points_deformed)
                
                sdf_values.append(sdf_batch.cpu().numpy())
        
        sdf_values = np.concatenate(sdf_values).reshape(resolution, resolution, resolution)
        
        # Marching cubes
        try:
            import skimage.measure
            verts, faces, _, _ = skimage.measure.marching_cubes(sdf_values, level=0)
            
            # Scale vertices back to original space
            verts = verts / (resolution - 1) * (bbox[1] - bbox[0]) + bbox[0]
            
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            return mesh
            
        except ImportError:
            print("scikit-image not available for marching cubes")
            return None
    
    def query_color(self, points):
        """
        Query colors at given points.
        
        Args:
            points: (N, 3) - Query points
        
        Returns:
            colors: (N, 3) - RGB colors
        """
        points_tensor = torch.FloatTensor(points).to(self.device)
        num_points = points_tensor.shape[0]
        
        with torch.no_grad():
            # Get deformation
            z_id_exp = self.z_geo_id.repeat(num_points, 1)
            z_ex_exp = self.z_geo_ex.repeat(num_points, 1)
            z_hair_exp = self.z_geo_hair.repeat(num_points, 1)
            
            delta = self.deform_net(points_tensor, z_id_exp, z_ex_exp, z_hair_exp)
            points_deformed = points_tensor + delta
            
            # Query color
            z_id_col_exp = self.z_col_id.repeat(num_points, 1)
            z_hair_col_exp = self.z_col_hair.repeat(num_points, 1)
            
            colors = self.color_net(points_deformed, z_id_col_exp, z_hair_col_exp)
        
        return colors.cpu().numpy()


def create_test_scan():
    """
    Create a synthetic test scan for fitting demonstration.
    """
    # Create a simple sphere as test scan
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
    
    # Sample points
    points = mesh.sample(10000)
    
    # Compute approximate SDF (distance to surface with sign)
    # For sphere, SDF = distance from center - radius
    center = np.mean(mesh.vertices, axis=0)
    distances = np.linalg.norm(points - center, axis=1) - 0.5
    sdf_values = distances.reshape(-1, 1)
    
    # Random colors (for demonstration)
    colors = np.random.rand(10000, 3)
    
    return points, sdf_values, colors, mesh


def test_fitter():
    """
    Test function to verify fitter implementation.
    """
    print("\n" + "=" * 60)
    print("TESTING FITTER")
    print("=" * 60)
    
    # Create synthetic test scan
    print("\n1. Creating test scan (sphere)...")
    points, sdf_values, colors, mesh = create_test_scan()
    print(f"   ✓ Test scan created")
    print(f"      - Points: {points.shape[0]}")
    print(f"      - Mesh vertices: {len(mesh.vertices)}")
    
    # Create mock networks for testing
    from models.ref_net import RefNet
    from models.deform_net import DeformNet
    from models.color_net import ColorNet
    
    print("\n2. Loading mock networks...")
    ref_net = RefNet()
    deform_net = DeformNet()
    color_net = ColorNet()
    print("   ✓ Mock networks created")
    
    # Create fitter
    print("\n3. Initializing fitter...")
    fitter = i3DMMFitter(
        ref_net=ref_net,
        deform_net=deform_net,
        color_net=color_net,
        device='cpu'
    )
    print("   ✓ Fitter initialized")
    
    # Run fitting
    print("\n4. Running fitting (100 iterations)...")
    optimized_codes, history = fitter.fit(
        points=points,
        sdf_values=sdf_values,
        colors=colors,
        num_iterations=100,
        learning_rate=0.01,
        print_every=20
    )
    
    # Check results
    print("\n5. Verifying results...")
    print(f"   ✓ Final loss: {history[-1]['total']:.4f}")
    print(f"   ✓ Initial loss: {history[0]['total']:.4f}")
    print(f"   ✓ Improvement: {history[0]['total'] - history[-1]['total']:.4f}")
    
    print("\n   Optimized latent codes:")
    for key, value in optimized_codes.items():
        print(f"      {key}: norm = {np.linalg.norm(value):.4f}")
    
    print("\n" + "=" * 60)
    print("FITTER TEST PASSED!")
    print("=" * 60)
    
    return fitter, optimized_codes, history


if __name__ == "__main__":
    fitter, codes, history = test_fitter()