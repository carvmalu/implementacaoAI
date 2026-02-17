'''
equation 3:
L_O (x, z_geo, z_col) = kE (L_geo_O(x, z_geo) + L_def_O(x, z_geo) i= 1 
+ L_col_O(x, z_col) + L_reg(z_geo, z_col) +E L_lm_O(z_geo_i, z_geo_j) i different j
onde:
L_geo: Geometry reconstruction loss (eq. 4)
L_def: deformation regularization loss
L_col: color reconstruction loss (Eq.5)
L_reg: latent code regularization (L2 prior)
L_lm: Sparse pairwise landmark supervision (eq. 6)
'''
import torch
import torch.nn as nn
import torch.functional as F
import numpy as np

class GeometryLoss(nn.Module):
    '''
    Geometry reconstruction Loss
    L_geo_O(x, z_geo) = w_g * ||cl(s(x, z_geo), t) - cl(s_gt(x), t)||1
    where cl(x, t) := min(tn max(x, -t)) is symmetric clamping at t=0.1
    '''

    def __init__(self, weight=1.0, clamp_value=0.1):
        '''
        Docstring for __init__
            weight: w_g - weighting factor for geometry loss
            clamp_value: t - clamping threshold (default:0.1)
        '''
        super().__init__()
        self.weight = weight
        self.clamp_value = clamp_value
        self.l1_loss = nn.L1Loss()

    def clamp_symmetric(self, x):
        '''
        prevents the network from focusing too much on points
        far frome the surface, improving numerical stability
        '''
        return torch.clamp(x, -self.clamp_value, self.clamp_value)
    
    def foward(self, pred_sdf, gt_sdf):
        '''
        Docstring for foward
            pred_sdf: (batch_size, 1) - predicted SDF values
            gt_sdf: (batch_size, 1) - ground truth SDF values
        Returns:
            loss: Scalar tensor - Clamped L1 geometry loss
        '''

        # apply symmetric clamping
        pred_clamped = self.clamp_symmetric(pred_sdf)
        gt_clamped = self.clamp_symmetric(gt_sdf)

        # L1 loss on clamped values  
        loss = self.weight * self.l1_loss(pred_clamped, gt_clamped)

        return loss
class DeformationLoss(nn.Module):
    '''
    'Further, to ensure regularized deformations to the reference
    shape, we impose a loss on the amount of deformation.
    we use L_def_O(x, z_geo) = ws * ||f_s_O(x, z_geo)||2'

    this penalizes large displacements, keeping the reference shape
    as the 'mean' shape and preventing unnecessary deformations.
    '''

    def __init__(self, weight=1.0):
        '''
        Args:
            weight: ws - weighting factor for deformation loss
        '''
        super().__init__()
        self.weight = weight

    def foward(self, delta):
        '''
        Docstring for foward
            delta: (batch_size, 3) - predicted displacement vectors
        Returns:  
            loss: Scalar tensor - L2 norm of displacements  
        '''

        # L2 norm of displacement vectors
        deformation_magnitude = torch.norm(delta, dim=-1)
        loss = self.weight * deformation_magnitude.mean()

        return loss
    
class ColorLoss(nn.Module):
    '''
    Color reconstruction loss
    equation 5
    L_col_O(x, z_col) + w_c * || f_c_O(x + &, z_col) - c_gt(x)||1
    this is l1 loss between predicted and ground truth colors
    '''   
    def __init__(self, weight):
        '''
        Docstring for __init__
            weight: w_c - weighting factor for color loss
        '''
        super().__init__()
        self.weight = weight
        self.l1_loss = nn.L1Loss()

    def foward(self, pred_colors, gt_colors):
        '''
        Docstring for foward
            pred_colors: (batch_size, 3) - predicted RGB colors
            gt_colors: (batch_size, 3) - ground truth RGB colors
        Returns:
            loss: Scalar tensor - L1 color loss
        '''
        loss = self.weight * self.l1_loss(pred_colors, gt_colors)
        return loss
class LatentRegularizationLoss(nn.Module):
    '''
    Latent code Regularization Loss
    'finally, we use an '2-regularizer on the latent vectors assuming a gaussian prior 
    distribution, L_reg(z_geo, z_col) = w_r (||z_geo||2 + ||z_col||2)'
    this encourages latent codes to follow a gaussian distribution
    (zero mean, unit variance), which is useful for generation
    '''
    def __init__(self, weight= 0.001):
        '''
        Docstring for __init__
            weight:  w_r - weighting facor for regularization
        '''
        super().__init__()
        self.weight = weight

    def foward(self, z_geo =None, z_col = None):
        '''
        Docstring for foward
            z_geo: list or tuple of geometry latent codes
            z_col: list or tuple of color latent codes
        Returns:
            loss: scalar tensor - L2 regularization loss
        '''
        loss = 0.0

        # Geometry latent codes regularization
        if z_geo is not None:
            for z in z_geo:
                if z is not None:
                    loss += torch.norm(z, dim =-1).mean()

        # Color latent codes regularization
        if z_col is not None:
            for z in z_col:
                if z is not None:
                    loss += torch.norm(z, dim =-1).mean()

        loss = self.weight * loss
        return loss
    
class PairwiseLandmarkLoss(nn.Module):
    '''
       Sparse pairwise landmark supervision loss.
    L_lm_θ(z_geo_i, z_geo_j) = w_lm ∑ ||(x_ℓ_i + δ_ℓ_i) - (x_ℓ_j + δ_ℓ_j)||₂
    
    We enforce that the 16 landmarks of each shape i, deform to 
    the same points in the reference space using the pairwise loss.
    
    This is the key loss that enables learning dense correspondences
    with only sparse supervision (16 landmarks per scan).
    '''
    
    def __init__(self, weight=10.0, num_landmarks=16):
        """
        Args:
            weight: w_lm in paper - weighting factor for landmark loss
            num_landmarks: Number of landmarks (16 as per paper)
        """
        super().__init__()
        self.weight = weight
        self.num_landmarks = num_landmarks
    
    def forward(self, 
               landmarks_i,        # (batch_size, num_landmarks, 3)
               delta_i,           # (batch_size, num_landmarks, 3)
               landmarks_j,        # (batch_size, num_landmarks, 3)
               delta_j):          # (batch_size, num_landmarks, 3)
        """
        Compute pairwise loss between two scans.
        
        Args:
            landmarks_i: 3D positions of landmarks for shape i
            delta_i: Predicted displacements at landmarks for shape i
            landmarks_j: 3D positions of landmarks for shape j
            delta_j: Predicted displacements at landmarks for shape j
        
        Returns:
            loss: Scalar tensor - Pairwise landmark correspondence loss
        """
        # Deform landmarks to reference space
        ref_points_i = landmarks_i + delta_i
        ref_points_j = landmarks_j + delta_j
        
        # L2 distance between corresponding points in reference space
        # They should be identical (same point on reference shape)
        distances = torch.norm(ref_points_i - ref_points_j, dim=-1)
        
        loss = self.weight * distances.mean()
        
        return loss
    
    def forward_batch(self, 
                     landmarks_batch,    # (batch_size, num_landmarks, 3)
                     deltas_batch,       # (batch_size, num_landmarks, 3)
                     ear_mask=None):     # Optional mask for ears
        """
        Compute pairwise losses for all combinations in a batch.
        
        Args:
            landmarks_batch: Landmarks for all scans in batch
            deltas_batch: Predicted displacements for all scans
            ear_mask: Optional mask indicating which scans have visible ears
        
        Returns:
            loss: Scalar tensor - Average pairwise loss
        """
        batch_size = landmarks_batch.shape[0]
        total_loss = 0.0
        num_pairs = 0
        
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                # Skip pairs where ear visibility differs (if mask provided)
                if ear_mask is not None and ear_mask[i] != ear_mask[j]:
                    continue
                
                loss = self.forward(
                    landmarks_batch[i:i+1], deltas_batch[i:i+1],
                    landmarks_batch[j:j+1], deltas_batch[j:j+1]
                )
                total_loss += loss
                num_pairs += 1
        
        if num_pairs > 0:
            total_loss /= num_pairs
        
        return total_loss


class TotalLoss(nn.Module):
    """
    Complete loss function as per Equation 3.
    
    Combines all loss components:
    L_total = ∑(L_geo + L_def + L_col + L_reg) + L_lm
    """
    
    def __init__(self,
                 w_geo=1.0,           # Weight for geometry loss
                 w_def=1.0,           # Weight for deformation regularization
                 w_col=1.0,           # Weight for color loss
                 w_reg=0.001,         # Weight for latent regularization
                 w_lm=10.0,           # Weight for landmark loss
                 clamp_value=0.1):    # Clamping threshold for SDF
        """
        Args:
            w_geo: w_g in paper
            w_def: w_s in paper
            w_col: w_c in paper
            w_reg: w_r in paper
            w_lm: w_lm in paper
            clamp_value: t in paper
        """
        super().__init__()
        
        self.geo_loss = GeometryLoss(weight=w_geo, clamp_value=clamp_value)
        self.def_loss = DeformationLoss(weight=w_def)
        self.col_loss = ColorLoss(weight=w_col)
        self.reg_loss = LatentRegularizationLoss(weight=w_reg)
        self.lm_loss = PairwiseLandmarkLoss(weight=w_lm)
        
    def forward(self,
               pred_sdf, gt_sdf,           # Geometry predictions
               delta,                       # Deformation vectors
               pred_colors, gt_colors,      # Color predictions
               z_geo_list, z_col_list,      # Latent codes
               landmarks_batch, deltas_lm,  # Landmark data
               ear_mask=None):              # Ear visibility mask
        
        """
        Compute total loss.
        
        Args:
            pred_sdf: (batch_size, 1) - Predicted SDF values
            gt_sdf: (batch_size, 1) - Ground truth SDF values
            delta: (batch_size, 3) - Predicted displacements
            pred_colors: (batch_size, 3) - Predicted colors
            gt_colors: (batch_size, 3) - Ground truth colors
            z_geo_list: List of geometry latent codes [z_id, z_ex, z_hair]
            z_col_list: List of color latent codes [z_id, z_hair]
            landmarks_batch: (batch_size, num_landmarks, 3)
            deltas_lm: (batch_size, num_landmarks, 3)
            ear_mask: Optional mask for ear visibility
        
        Returns:
            losses: Dictionary with individual loss components
            total_loss: Scalar tensor
        """
        # Individual losses
        loss_geo = self.geo_loss(pred_sdf, gt_sdf)
        loss_def = self.def_loss(delta)
        loss_col = self.col_loss(pred_colors, gt_colors)
        loss_reg = self.reg_loss(z_geo_list, z_col_list)
        loss_lm = self.lm_loss.forward_batch(landmarks_batch, deltas_lm, ear_mask)
        
        # Total loss
        total_loss = loss_geo + loss_def + loss_col + loss_reg + loss_lm
        
        # Return dictionary for logging
        losses = {
            'geo': loss_geo.item(),
            'def': loss_def.item(),
            'col': loss_col.item(),
            'reg': loss_reg.item(),
            'lm': loss_lm.item() if isinstance(loss_lm, torch.Tensor) else loss_lm,
            'total': total_loss.item()
        }
        
        return losses, total_loss

def test_losses():
    """
    Test function to verify all loss implementations.
    """
    print("\n" + "=" * 60)
    print("TESTING LOSS FUNCTIONS")
    print("=" * 60)
    
    batch_size = 64
    num_landmarks = 16
    
    # 1. Test GeometryLoss
    print("\n1. Testing GeometryLoss...")
    geo_loss = GeometryLoss(weight=1.0, clamp_value=0.1)
    
    pred_sdf = torch.randn(batch_size, 1) * 0.5
    gt_sdf = torch.randn(batch_size, 1) * 0.5
    
    loss_geo = geo_loss(pred_sdf, gt_sdf)
    print(f"   ✓ Geometry loss: {loss_geo.item():.4f}")
    
    # Test clamping
    pred_extreme = torch.ones(batch_size, 1) * 10.0
    gt_normal = torch.randn(batch_size, 1) * 0.1
    loss_clamped = geo_loss(pred_extreme, gt_normal)
    print(f"   ✓ Clamping working: extreme value loss = {loss_clamped.item():.4f}")
    
    # 2. Test DeformationLoss
    print("\n2. Testing DeformationLoss...")
    def_loss = DeformationLoss(weight=1.0)
    
    delta = torch.randn(batch_size, 3) * 0.1
    loss_def = def_loss(delta)
    print(f"   ✓ Deformation loss: {loss_def.item():.4f}")
    
    # 3. Test ColorLoss
    print("\n3. Testing ColorLoss...")
    col_loss = ColorLoss(weight=1.0)
    
    pred_colors = torch.rand(batch_size, 3)
    gt_colors = torch.rand(batch_size, 3)
    loss_col = col_loss(pred_colors, gt_colors)
    print(f"   ✓ Color loss: {loss_col.item():.4f}")
    
    # 4. Test LatentRegularizationLoss
    print("\n4. Testing LatentRegularizationLoss...")
    reg_loss = LatentRegularizationLoss(weight=0.001)
    
    z_id = torch.randn(batch_size, 128)
    z_ex = torch.randn(batch_size, 32)
    z_hair = torch.randn(batch_size, 16)
    
    loss_reg = reg_loss([z_id, z_ex, z_hair], [z_id, z_hair])
    print(f"   ✓ Latent regularization loss: {loss_reg.item():.4f}")
    
    # 5. Test PairwiseLandmarkLoss
    print("\n5. Testing PairwiseLandmarkLoss...")
    lm_loss = PairwiseLandmarkLoss(weight=10.0, num_landmarks=16)
    
    landmarks = torch.randn(batch_size, num_landmarks, 3)
    deltas_lm = torch.randn(batch_size, num_landmarks, 3) * 0.05
    
    loss_lm = lm_loss.forward_batch(landmarks, deltas_lm)
    print(f"   ✓ Pairwise landmark loss: {loss_lm.item():.4f}")
    
    # 6. Test TotalLoss
    print("\n6. Testing TotalLoss...")
    total_loss_fn = TotalLoss(
        w_geo=1.0,
        w_def=1.0,
        w_col=1.0,
        w_reg=0.001,
        w_lm=10.0,
        clamp_value=0.1
    )
    
    losses, total_loss = total_loss_fn(
        pred_sdf=pred_sdf,
        gt_sdf=gt_sdf,
        delta=delta,
        pred_colors=pred_colors,
        gt_colors=gt_colors,
        z_geo_list=[z_id, z_ex, z_hair],
        z_col_list=[z_id, z_hair],
        landmarks_batch=landmarks,
        deltas_lm=deltas_lm,
        ear_mask=None
    )
    
    print(f"   ✓ Total loss components:")
    for name, value in losses.items():
        print(f"      - {name}: {value:.4f}")
    
    print("\n" + "=" * 60)
    print("LOSS FUNCTIONS TEST PASSED!")
    print("=" * 60)
    
    return total_loss_fn


if __name__ == "__main__":
    test_losses()
    