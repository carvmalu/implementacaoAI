import numpy as np
import trimesh
from scipy.spatial import KDTree

class EducationalSDFSampler:
    """
    Amostragem de pontos para treinamento SDF
    Estratégias: uniforme (25%) + landmark-based (75%)
    """
    
    def __init__(self, config):
        self.config = config
        self.num_samples = config.NUM_SAMPLES
        self.uniform_ratio = config.UNIFORM_RATIO
        self.sphere_radius = config.LANDMARK_SPHERE_RADIUS
        self.perturb_std = config.PERTURBATION_STD
    
    def compute_sdf(self, mesh, query_points):
        """
        Computa SDF simplificado para pontos de query
        """
        # 1. Encontra ponto mais próximo na superfície
        vertices = mesh.vertices
        tree = KDTree(vertices)
        distances, indices = tree.query(query_points)
        
        # 2. Determina sinal (interior/exterior)
        # Simplificação: usa convex hull para determinar interior
        try:
            hull = mesh.convex_hull
            inside = hull.contains(query_points)
            signs = np.where(inside, -1.0, 1.0)
        except:
            # Fallback: assume que centro é interior
            center = np.mean(vertices, axis=0)
            to_center = query_points - center
            center_dists = np.linalg.norm(to_center, axis=1)
            max_dist = np.max(np.linalg.norm(vertices - center, axis=1))
            signs = np.where(center_dists < max_dist, -1.0, 1.0)
        
        sdf_values = signs * distances
        return sdf_values
    
    def sample_uniform(self, mesh, n_points):
        """Amostragem uniforme na superfície"""
        points = mesh.sample(n_points)
        return points
    
    def sample_landmark_regions(self, mesh, landmarks, n_points):
        """Amostragem baseada em landmarks"""
        all_points = []
        n_landmarks = len(landmarks)
        
        if n_landmarks == 0:
            return np.array([])
        
        n_per_landmark = n_points // n_landmarks
        
        for name, center in landmarks.items():
            # Pontos em esfera ao redor do landmark
            phi = np.random.uniform(0, 2*np.pi, n_per_landmark)
            costheta = np.random.uniform(-1, 1, n_per_landmark)
            theta = np.arccos(costheta)
            r = self.sphere_radius * np.random.uniform(0.8, 1.2, n_per_landmark)
            
            x = r * np.sin(theta) * np.cos(phi) + center[0]
            y = r * np.sin(theta) * np.sin(phi) + center[1]
            z = r * np.cos(theta) + center[2]
            
            points = np.column_stack([x, y, z])
            all_points.append(points)
        
        return np.vstack(all_points)
    
    def sample_simple(self, mesh, landmarks):
        """Pipeline completo de amostragem"""
        print("\n[SDF SAMPLING]")
        print("=" * 40)
        
        # 1. Amostragem uniforme
        n_uniform = int(self.num_samples * self.uniform_ratio)
        uniform_points = self.sample_uniform(mesh, n_uniform)
        print(f"  ✓ Amostragem uniforme: {len(uniform_points)} pontos")
        
        # 2. Amostragem baseada em landmarks
        n_landmark = self.num_samples - n_uniform
        landmark_points = self.sample_landmark_regions(mesh, landmarks, n_landmark)
        print(f"  ✓ Amostragem landmark-based: {len(landmark_points)} pontos")
        
        # 3. Combina pontos
        all_points = np.vstack([uniform_points, landmark_points])
        
        # 4. Perturbação Gaussiana
        bbox = mesh.bounds
        bbox_size = np.max(bbox[1] - bbox[0])
        noise_std = self.perturb_std * bbox_size
        
        perturbed_points = all_points + np.random.randn(*all_points.shape) * noise_std
        print(f"  ✓ Perturbação aplicada (σ={noise_std:.4f})")
        
        # 5. Calcula SDF
        sdf_values = self.compute_sdf(mesh, perturbed_points)
        print(f"  ✓ SDF calculado para {len(perturbed_points)} pontos")
        print(f"    - Min: {sdf_values.min():.4f}")
        print(f"    - Max: {sdf_values.max():.4f}")
        print(f"    - Mean: {sdf_values.mean():.4f}")
        print(f"    - Interior (SDF<0): {np.sum(sdf_values < 0)} pontos")
        print(f"    - Exterior (SDF>0): {np.sum(sdf_values > 0)} pontos")
        
        # 6. Cores simplificadas (educacional)
        colors = np.zeros((len(perturbed_points), 3))
        colors[:, 0] = (perturbed_points[:, 0] - bbox[0, 0]) / bbox_size  # R
        colors[:, 1] = (perturbed_points[:, 1] - bbox[0, 1]) / bbox_size  # G
        colors[:, 2] = (perturbed_points[:, 2] - bbox[0, 2]) / bbox_size  # B
        
        return perturbed_points, sdf_values, colors