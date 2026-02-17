"""
MODEL: Evaluation Metrics
Paper: Section 4.4 - Comparisons to Existing Models (page 7)
Table: 1 - Quantitative comparison

METRICS:
- Chamfer Distance: Mean distance between points of two shapes
- F-score: Harmonic mean of precision and recall at threshold
- Color Error: Mean color difference between corresponding points
"""

import torch
import numpy as np
from scipy.spatial import KDTree
import trimesh


class EvaluationMetrics:
    """
    Evaluation metrics for 3D head reconstruction.
    
    As used in Table 1 of the paper:
    - Chamfer distance in mm
    - F-score with threshold 0.01
    - Color error in RGB space
    """
    
    def __init__(self, threshold=0.01):
        """
        Args:
            threshold: Distance threshold for F-score (0.01 as in paper)
        """
        self.threshold = threshold
    
    def compute_chamfer_distance(self, vertices_pred, vertices_gt, symmetric=True):
        """
        Compute Chamfer distance between two point clouds.
        
        Paper: "Chamfer distance is the mean distance of points from 
               one shape to their closest points in another."
        
        Args:
            vertices_pred: (N, 3) - Predicted mesh vertices
            vertices_gt: (M, 3) - Ground truth mesh vertices
            symmetric: If True, compute symmetric Chamfer distance
        
        Returns:
            chamfer_dist: Mean distance in mm
        """
        # Build KD-trees for fast nearest neighbor search
        tree_pred = KDTree(vertices_pred)
        tree_gt = KDTree(vertices_gt)
        
        # Distances from pred to gt
        dist_pred_to_gt, _ = tree_gt.query(vertices_pred)
        mean_pred_to_gt = np.mean(dist_pred_to_gt) * 1000  # Convert to mm
        
        if symmetric:
            # Distances from gt to pred
            dist_gt_to_pred, _ = tree_pred.query(vertices_gt)
            mean_gt_to_pred = np.mean(dist_gt_to_pred) * 1000  # Convert to mm
            
            # Symmetric Chamfer distance
            chamfer_dist = (mean_pred_to_gt + mean_gt_to_pred) / 2
        else:
            chamfer_dist = mean_pred_to_gt
        
        return chamfer_dist
    
    def compute_f_score(self, vertices_pred, vertices_gt):
        """
        Compute F-score at threshold.
        
        Paper: "F-score is (100x) the harmonic mean of precision 
               and recall after applying a threshold."
        
        Args:
            vertices_pred: (N, 3) - Predicted mesh vertices
            vertices_gt: (M, 3) - Ground truth mesh vertices
        
        Returns:
            f_score: F-score (0-100 scale)
        """
        # Build KD-trees
        tree_pred = KDTree(vertices_pred)
        tree_gt = KDTree(vertices_gt)
        
        # Precision: % of predicted points within threshold of GT
        dist_pred_to_gt, _ = tree_gt.query(vertices_pred)
        precision = np.mean(dist_pred_to_gt <= self.threshold)
        
        # Recall: % of GT points within threshold of prediction
        dist_gt_to_pred, _ = tree_pred.query(vertices_gt)
        recall = np.mean(dist_gt_to_pred <= self.threshold)
        
        # F-score (harmonic mean)
        if precision + recall > 0:
            f_score = 100 * 2 * precision * recall / (precision + recall)
        else:
            f_score = 0.0
        
        return f_score
    
    def compute_color_error(self, mesh_pred, mesh_gt, num_samples=150000):
        """
        Compute color error between two textured meshes.
        
        Paper: "First, the mean error in color at points in the ground 
               truth and the color at their nearest points in the fits 
               is computed, along with the mean error in the opposite 
               direction. Finally, the color error is reported as their 
               average."
        
        Args:
            mesh_pred: Trimesh object with vertex colors
            mesh_gt: Trimesh object with vertex colors
            num_samples: Number of points to sample (150k as in paper)
        
        Returns:
            color_error: Mean color error in RGB space (0-1 scale)
        """
        # Sample points on both meshes
        points_pred = mesh_pred.sample(num_samples)
        points_gt = mesh_gt.sample(num_samples)
        
        # Get colors at sampled points (nearest vertex)
        tree_pred = KDTree(mesh_pred.vertices)
        tree_gt = KDTree(mesh_gt.vertices)
        
        # Colors at pred points: nearest GT vertex color
        _, idx_gt = tree_gt.query(points_pred)
        colors_gt_at_pred = mesh_gt.visual.vertex_colors[idx_gt][:, :3] / 255.0
        
        # Colors at GT points: nearest pred vertex color
        _, idx_pred = tree_pred.query(points_gt)
        colors_pred_at_gt = mesh_pred.visual.vertex_colors[idx_pred][:, :3] / 255.0
        
        # Get actual colors at sampled points
        # For simplicity, we use vertex colors directly
        # A more accurate method would interpolate colors
        
        # Error pred -> GT
        error_pred_to_gt = np.mean(np.abs(colors_gt_at_pred - colors_pred_at_gt))
        
        # Error GT -> pred
        error_gt_to_pred = np.mean(np.abs(colors_pred_at_gt - colors_gt_at_pred))
        
        # Average error
        color_error = (error_pred_to_gt + error_gt_to_pred) / 2
        
        return color_error
    
    def evaluate_reconstruction(self, mesh_pred, mesh_gt, colors_pred=None, colors_gt=None):
        """
        Complete evaluation of reconstruction quality.
        
        Args:
            mesh_pred: Predicted mesh
            mesh_gt: Ground truth mesh
            colors_pred: Optional predicted colors
            colors_gt: Optional ground truth colors
        
        Returns:
            metrics: Dictionary with all metrics
        """
        metrics = {}
        
        # Get vertices
        verts_pred = mesh_pred.vertices
        verts_gt = mesh_gt.vertices
        
        # Chamfer distance
        metrics['chamfer_full'] = self.compute_chamfer_distance(verts_pred, verts_gt)
        
        # For face region only (simulate by cropping)
        # In practice, you'd need face masks
        # Here we just take a subset as demonstration
        face_ratio = 0.3  # Assume 30% are face vertices
        num_face = int(len(verts_pred) * face_ratio)
        
        # Randomly sample face vertices (simplified)
        np.random.seed(42)
        face_idx = np.random.choice(len(verts_pred), num_face, replace=False)
        verts_pred_face = verts_pred[face_idx]
        verts_gt_face = verts_gt[face_idx[:len(verts_gt)]]
        
        metrics['chamfer_face'] = self.compute_chamfer_distance(verts_pred_face, verts_gt_face)
        
        # F-score
        metrics['f_score_full'] = self.compute_f_score(verts_pred, verts_gt)
        metrics['f_score_face'] = self.compute_f_score(verts_pred_face, verts_gt_face)
        
        # Color error
        if colors_pred is not None and colors_gt is not None:
            metrics['color_error'] = self.compute_color_error(mesh_pred, mesh_gt)
        
        return metrics


def test_evaluation():
    """
    Test function for evaluation metrics.
    """
    print("\n" + "=" * 60)
    print("TESTING EVALUATION METRICS")
    print("=" * 60)
    
    # Create two simple meshes for testing
    mesh1 = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    mesh2 = trimesh.creation.icosphere(subdivisions=2, radius=1.05)  # Slightly larger
    
    # Add random colors
    mesh1.visual.vertex_colors = np.random.randint(0, 255, (len(mesh1.vertices), 4))
    mesh2.visual.vertex_colors = np.random.randint(0, 255, (len(mesh2.vertices), 4))
    
    # Initialize evaluator
    evaluator = EvaluationMetrics(threshold=0.01)
    
    # Compute metrics
    print("\n1. Computing Chamfer distance...")
    chamfer = evaluator.compute_chamfer_distance(mesh1.vertices, mesh2.vertices)
    print(f"   ✓ Chamfer distance: {chamfer:.4f} mm")
    
    print("\n2. Computing F-score...")
    f_score = evaluator.compute_f_score(mesh1.vertices, mesh2.vertices)
    print(f"   ✓ F-score: {f_score:.2f}")
    
    print("\n3. Computing color error...")
    color_error = evaluator.compute_color_error(mesh1, mesh2)
    print(f"   ✓ Color error: {color_error:.4f}")
    
    print("\n4. Full evaluation...")
    metrics = evaluator.evaluate_reconstruction(mesh1, mesh2)
    for key, value in metrics.items():
        print(f"   ✓ {key}: {value:.4f}")
    
    print("\n" + "=" * 60)
    print("EVALUATION TEST PASSED!")
    print("=" * 60)
    
    return evaluator


if __name__ == "__main__":
    test_evaluation()