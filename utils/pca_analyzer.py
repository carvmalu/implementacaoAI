"""
MODEL: Applications
Paper: Section 4.5 - Semantic Head Editing (page 7)
       Section 4.6 - One-shot Annotation Transfer (page 8)

APPLICATIONS:
1. Semantic Head Editing - Change identity, expression, hairstyle independently
2. Texture Transfer - Transfer colors between scans
3. Annotation Transfer - Transfer segmentation masks and landmarks
"""

import torch
import numpy as np
import trimesh
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class SemanticEditor:
    """
    Semantic editing of head attributes.
    
    Paper: "Due to the semantic disentanglement in our model, we can 
           selectively change semantic components of a head."
    
    Enables independent editing of:
    - Identity geometry
    - Expression geometry
    - Hairstyle geometry
    - Identity color
    - Hairstyle color
    """
    
    def __init__(self, ref_net, deform_net, color_net, 
                 geo_latent_codes, col_latent_codes):
        """
        Args:
            ref_net: Reference Shape Network
            deform_net: Shape Deformation Network
            color_net: Color Network
            geo_latent_codes: Trained geometry latent codes (all identities)
            col_latent_codes: Trained color latent codes (all identities)
        """
        self.ref_net = ref_net.eval()
        self.deform_net = deform_net.eval()
        self.color_net = color_net.eval()
        
        self.geo_codes = geo_latent_codes
        self.col_codes = col_latent_codes
        
        # Precompute PCA for each latent space
        self.pca_models = {}
        self._compute_pca()
    
    def _compute_pca(self):
        """
        Compute PCA on latent spaces for meaningful edits.
        
        Paper: "To obtain the edited latent codes, we first find the 
               principle components of variation in each latent-subspace 
               by running PCA on the training latent spaces."
        """
        # Get all latent codes
        all_geo_id = self.geo_codes.identity_codes.weight.detach().cpu().numpy()
        all_geo_ex = self.geo_codes.expression_codes.weight.detach().cpu().numpy()
        all_geo_hair = self.geo_codes.hairstyle_codes.weight.detach().cpu().numpy()
        
        all_col_id = self.col_codes.identity_codes.weight.detach().cpu().numpy()
        all_col_hair = self.col_codes.hairstyle_codes.weight.detach().cpu().numpy()
        
        # Compute PCA
        self.pca_models['geo_id'] = PCA(n_components=10)
        self.pca_models['geo_id'].fit(all_geo_id)
        
        self.pca_models['geo_ex'] = PCA(n_components=5)
        self.pca_models['geo_ex'].fit(all_geo_ex)
        
        self.pca_models['geo_hair'] = PCA(n_components=2)
        self.pca_models['geo_hair'].fit(all_geo_hair)
        
        self.pca_models['col_id'] = PCA(n_components=10)
        self.pca_models['col_id'].fit(all_col_id)
        
        self.pca_models['col_hair'] = PCA(n_components=2)
        self.pca_models['col_hair'].fit(all_col_hair)
    
    def edit_identity(self, z_geo_id, z_col_id, 
                     geo_direction=0, col_direction=0, 
                     geo_strength=1.0, col_strength=1.0):
        """
        Edit identity (both geometry and color).
        
        Args:
            z_geo_id: Original geometry identity code
            z_col_id: Original color identity code
            geo_direction: Which PC direction to move (0-9)
            col_direction: Which PC direction to move (0-9)
            geo_strength: How much to move (in standard deviations)
            col_strength: How much to move (in standard deviations)
        
        Returns:
            Edited latent codes
        """
        z_geo_edited = z_geo_id.clone()
        z_col_edited = z_col_id.clone()
        
        # Edit geometry identity
        if geo_direction >= 0:
            pc = self.pca_models['geo_id'].components_[geo_direction]
            pc_tensor = torch.FloatTensor(pc).to(z_geo_id.device)
            
            # Move along PC direction
            std = np.sqrt(self.pca_models['geo_id'].explained_variance_[geo_direction])
            z_geo_edited = z_geo_id + geo_strength * std * pc_tensor
        
        # Edit color identity
        if col_direction >= 0:
            pc = self.pca_models['col_id'].components_[col_direction]
            pc_tensor = torch.FloatTensor(pc).to(z_col_id.device)
            
            std = np.sqrt(self.pca_models['col_id'].explained_variance_[col_direction])
            z_col_edited = z_col_id + col_strength * std * pc_tensor
        
        return z_geo_edited, z_col_edited
    
    def edit_expression(self, z_geo_ex, direction=0, strength=1.0):
        """
        Edit expression (geometry only).
        
        Args:
            z_geo_ex: Original expression code
            direction: Which PC direction to move (0-4)
            strength: How much to move
        
        Returns:
            Edited expression code
        """
        if direction >= 0:
            pc = self.pca_models['geo_ex'].components_[direction]
            pc_tensor = torch.FloatTensor(pc).to(z_geo_ex.device)
            
            std = np.sqrt(self.pca_models['geo_ex'].explained_variance_[direction])
            z_edited = z_geo_ex + strength * std * pc_tensor
        else:
            z_edited = z_geo_ex
        
        return z_edited
    
    def edit_hairstyle(self, z_geo_hair, z_col_hair,
                      geo_direction=0, col_direction=0,
                      geo_strength=1.0, col_strength=1.0):
        """
        Edit hairstyle (both geometry and color).
        
        Args:
            z_geo_hair: Original geometry hairstyle code
            z_col_hair: Original color hairstyle code
            geo_direction: Which PC direction to move (0-1)
            col_direction: Which PC direction to move (0-1)
            geo_strength: How much to move
            col_strength: How much to move
        
        Returns:
            Edited hairstyle codes
        """
        z_geo_edited = z_geo_hair.clone()
        z_col_edited = z_col_hair.clone()
        
        # Edit geometry hairstyle
        if geo_direction >= 0:
            pc = self.pca_models['geo_hair'].components_[geo_direction]
            pc_tensor = torch.FloatTensor(pc).to(z_geo_hair.device)
            
            std = np.sqrt(self.pca_models['geo_hair'].explained_variance_[geo_direction])
            z_geo_edited = z_geo_hair + geo_strength * std * pc_tensor
        
        # Edit color hairstyle
        if col_direction >= 0:
            pc = self.pca_models['col_hair'].components_[col_direction]
            pc_tensor = torch.FloatTensor(pc).to(z_col_hair.device)
            
            std = np.sqrt(self.pca_models['col_hair'].explained_variance_[col_direction])
            z_col_edited = z_col_hair + col_strength * std * pc_tensor
        
        return z_geo_edited, z_col_edited
    
    def reconstruct_head(self, z_geo_id, z_geo_ex, z_geo_hair,
                         z_col_id, z_col_hair, resolution=128):
        """
        Reconstruct mesh from latent codes.
        """
        device = z_geo_id.device
        
        # Create grid
        axis = np.linspace(-1, 1, resolution)
        grid_x, grid_y, grid_z = np.meshgrid(axis, axis, axis, indexing='ij')
        grid_points = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=1)
        grid_tensor = torch.FloatTensor(grid_points).to(device)
        
        # Process in batches
        batch_size = 10000
        sdf_values = []
        
        with torch.no_grad():
            for i in range(0, len(grid_tensor), batch_size):
                batch = grid_tensor[i:i+batch_size]
                n_points = batch.shape[0]
                
                # Expand latent codes
                z_id_exp = z_geo_id.repeat(n_points, 1)
                z_ex_exp = z_geo_ex.repeat(n_points, 1)
                z_hair_exp = z_geo_hair.repeat(n_points, 1)
                
                # Forward pass
                delta = self.deform_net(batch, z_id_exp, z_ex_exp, z_hair_exp)
                points_deformed = batch + delta
                sdf = self.ref_net(points_deformed)
                sdf_values.append(sdf.cpu().numpy())
        
        sdf_values = np.concatenate(sdf_values).reshape(resolution, resolution, resolution)
        
        # Marching cubes
        try:
            import skimage.measure
            verts, faces, _, _ = skimage.measure.marching_cubes(sdf_values, level=0)
            
            # Scale back
            verts = verts / (resolution - 1) * 2 - 1
            
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            return mesh
        except ImportError:
            print("scikit-image not available")
            return None


class TextureTransfer:
    """
    Transfer texture between scans using dense correspondences.
    
    Paper: "We demonstrate these correspondences in Fig. 6, where 
           the color is transferred from one scan to the other."
    """
    
    def __init__(self, ref_net, deform_net, color_net):
        self.ref_net = ref_net
        self.deform_net = deform_net
        self.color_net = color_net
    
    def transfer_texture(self, source_mesh, target_mesh,
                        source_z_geo, source_z_col,
                        target_z_geo, target_z_col,
                        num_points=10000):
        """
        Transfer texture from source to target mesh.
        
        Steps:
        1. Sample points on source mesh
        2. Deform to reference space using source codes
        3. Get colors in reference space using source color codes
        4. Find corresponding points on target mesh
        5. Assign colors
        """
        device = next(self.deform_net.parameters()).device
        
        # Sample points on source mesh
        source_points = source_mesh.sample(num_points)
        source_tensor = torch.FloatTensor(source_points).to(device)
        
        with torch.no_grad():
            # Step 1: Deform source points to reference space
            n_points = source_tensor.shape[0]
            
            z_id_exp = source_z_geo['id'].repeat(n_points, 1)
            z_ex_exp = source_z_geo.get('ex', torch.zeros(1, 32).to(device)).repeat(n_points, 1)
            z_hair_exp = source_z_geo.get('hair', torch.zeros(1, 16).to(device)).repeat(n_points, 1)
            
            delta_source = self.deform_net(source_tensor, z_id_exp, z_ex_exp, z_hair_exp)
            ref_points = source_tensor + delta_source
            
            # Step 2: Get colors in reference space
            z_id_col_exp = source_z_col['id'].repeat(n_points, 1)
            z_hair_col_exp = source_z_col.get('hair', torch.zeros(1, 16).to(device)).repeat(n_points, 1)
            
            colors_ref = self.color_net(ref_points, z_id_col_exp, z_hair_col_exp)
            colors_ref = colors_ref.cpu().numpy()
        
        # Step 3: Find closest points on target mesh
        from scipy.spatial import KDTree
        target_tree = KDTree(target_mesh.vertices)
        
        # For each source point, find nearest target vertex
        _, indices = target_tree.query(source_points)
        
        # Assign colors to target mesh
        target_colors = np.zeros((len(target_mesh.vertices), 3))
        
        # Average colors for vertices that get multiple assignments
        from collections import defaultdict
        color_sums = defaultdict(lambda: np.zeros(3))
        color_counts = defaultdict(int)
        
        for src_idx, tgt_idx in enumerate(indices):
            color_sums[tgt_idx] += colors_ref[src_idx]
            color_counts[tgt_idx] += 1
        
        for tgt_idx in color_sums.keys():
            target_colors[tgt_idx] = color_sums[tgt_idx] / color_counts[tgt_idx]
        
        # Create new mesh with transferred colors
        textured_mesh = trimesh.Trimesh(
            vertices=target_mesh.vertices,
            faces=target_mesh.faces,
            vertex_colors=target_colors
        )
        
        return textured_mesh


class AnnotationTransfer:
    """
    Transfer annotations (segmentation, landmarks) between scans.
    
    Paper: "As i3DMM can predict dense correspondences among head scans,
           it also enables us to transfer annotations across different
           reconstructions."
    """
    
    def __init__(self, deform_net):
        self.deform_net = deform_net
    
    def transfer_landmarks(self, source_landmarks, source_z_geo, target_z_geo):
        """
        Transfer landmarks from source to target scan.
        
        Args:
            source_landmarks: (L, 3) - Landmarks on source scan
            source_z_geo: Source geometry codes
            target_z_geo: Target geometry codes
        
        Returns:
            target_landmarks: (L, 3) - Landmarks on target scan
        """
        device = next(self.deform_net.parameters()).device
        
        source_tensor = torch.FloatTensor(source_landmarks).to(device)
        n_landmarks = source_tensor.shape[0]
        
        with torch.no_grad():
            # Step 1: Deform source landmarks to reference space
            z_id_exp = source_z_geo['id'].repeat(n_landmarks, 1)
            z_ex_exp = source_z_geo.get('ex', torch.zeros(1, 32).to(device)).repeat(n_landmarks, 1)
            z_hair_exp = source_z_geo.get('hair', torch.zeros(1, 16).to(device)).repeat(n_landmarks, 1)
            
            delta_source = self.deform_net(source_tensor, z_id_exp, z_ex_exp, z_hair_exp)
            ref_points = source_tensor + delta_source
            
            # Step 2: Find points on target that deform to same reference points
            # This is a root-finding problem: find p such that p + δ_target(p) = ref_points
            
            target_landmarks = []
            for i in range(n_landmarks):
                p = ref_points[i:i+1].clone().detach().requires_grad_(True)
                optimizer = torch.optim.Adam([p], lr=0.01)
                
                for _ in range(100):
                    optimizer.zero_grad()
                    
                    z_id_exp = target_z_geo['id'].repeat(1, 1)
                    z_ex_exp = target_z_geo.get('ex', torch.zeros(1, 32).to(device))
                    z_hair_exp = target_z_geo.get('hair', torch.zeros(1, 16).to(device))
                    
                    delta_p = self.deform_net(p, z_id_exp, z_ex_exp, z_hair_exp)
                    p_ref = p + delta_p
                    
                    loss = torch.norm(p_ref - ref_points[i:i+1])
                    loss.backward()
                    optimizer.step()
                
                target_landmarks.append(p.detach().cpu().numpy())
        
        return np.vstack(target_landmarks)


def test_applications():
    """
    Test function for applications.
    """
    print("\n" + "=" * 60)
    print("TESTING APPLICATIONS")
    print("=" * 60)
    
    # Mock networks and codes
    from models.ref_net import RefNet
    from models.deform_net import DeformNet
    from models.color_net import ColorNet
    
    ref_net = RefNet()
    deform_net = DeformNet()
    color_net = ColorNet()
    
    # Mock latent codes
    class MockLatentCodes:
        def __init__(self):
            self.identity_codes = type('obj', (), {'weight': torch.randn(58, 128)})()
            self.expression_codes = type('obj', (), {'weight': torch.randn(10, 32)})()
            self.hairstyle_codes = type('obj', (), {'weight': torch.randn(4, 16)})()
    
    geo_codes = MockLatentCodes()
    col_codes = MockLatentCodes()
    
    # Test SemanticEditor
    print("\n1. Testing SemanticEditor...")
    editor = SemanticEditor(ref_net, deform_net, color_net, geo_codes, col_codes)
    
    z_geo_id = torch.randn(1, 128)
    z_col_id = torch.randn(1, 128)
    z_geo_ex = torch.randn(1, 32)
    z_geo_hair = torch.randn(1, 16)
    z_col_hair = torch.randn(1, 16)
    
    # Test identity edit
    z_id_edited, z_col_edited = editor.edit_identity(z_geo_id, z_col_id, 0, 0, 1.0, 1.0)
    print(f"   ✓ Identity edit - Geo diff: {torch.norm(z_id_edited - z_geo_id).item():.4f}")
    print(f"   ✓ Identity edit - Col diff: {torch.norm(z_col_edited - z_col_id).item():.4f}")
    
    # Test expression edit
    z_ex_edited = editor.edit_expression(z_geo_ex, 0, 1.0)
    print(f"   ✓ Expression edit - Diff: {torch.norm(z_ex_edited - z_geo_ex).item():.4f}")
    
    # Test hairstyle edit
    z_hair_edited, z_col_hair_edited = editor.edit_hairstyle(z_geo_hair, z_col_hair, 0, 0, 1.0, 1.0)
    print(f"   ✓ Hairstyle edit - Geo diff: {torch.norm(z_hair_edited - z_geo_hair).item():.4f}")
    print(f"   ✓ Hairstyle edit - Col diff: {torch.norm(z_col_hair_edited - z_col_hair).item():.4f}")
    
    print("\n" + "=" * 60)
    print("APPLICATIONS TEST PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_applications()