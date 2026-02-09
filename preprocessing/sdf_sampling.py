# preprocessing/sdf_sampling.py
import numpy as np
import trimesh
from scipy.spatial import KDTree

class SDFSampler:
    """Amostragem de pontos para treino SDF"""
   
    def __init__(self, config):
        self.config = config
   
    def compute_sdf(self, mesh, query_points):
        """
        Computa SDF simplificado para pontos de query
        """
        # Encontra o ponto mais próximo na superfície
        vertices = mesh.vertices
        tree = KDTree(vertices)
       
        distances, indices = tree.query(query_points)
       
        # Determina sinal (simplificado - assume orientação consistente)
        # Para educativo: assume interior se dentro do convex hull
        hull = mesh.convex_hull
        signs = np.where(hull.contains(query_points), -1, 1)
       
        sdf_values = signs * distances
       
        return sdf_values
   
    def sample_training_points(self, mesh, landmarks):
        """
        Implementa a estratégia de amostragem do paper:
        1. Amostragem uniforme
        2. Amostragem baseada em landmarks
        3. Perturbação gaussiana
        """
        n_total = self.config.NUM_SAMPLES
        n_uniform = int(n_total * self.config.UNIFORM_RATIO)
        n_landmark = n_total - n_uniform
       
        # 1. Amostragem uniforme na superfície
        uniform_points = mesh.sample(n_uniform)
       
        # 2. Amostragem baseada em landmarks
        landmark_points = []
        for landmark_name, position in landmarks.items():
            # Esfera ao redor do landmark
            n_per_landmark = n_landmark // len(landmarks)
           
            # Amostra pontos em esfera
            phi = np.random.uniform(0, 2*np.pi, n_per_landmark)
            costheta = np.random.uniform(-1, 1, n_per_landmark)
            theta = np.arccos(costheta)
            r = 0.05  # Raio pequeno
           
            # Coordenadas esféricas para cartesianas
            x = r * np.sin(theta) * np.cos(phi) + position[0]
            y = r * np.sin(theta) * np.sin(phi) + position[1]
            z = r * np.cos(theta) + position[2]
           
            landmark_points.append(np.column_stack([x, y, z]))
       
        landmark_points = np.vstack(landmark_points)
       
        # Combina os pontos
        all_points = np.vstack([uniform_points, landmark_points])
       
        # 3. Perturbação gaussiana
        bbox = mesh.bounds
        bbox_size = np.max(bbox[1] - bbox[0])
        noise_std = 0.005 * bbox_size
       
        perturbed_points = all_points + np.random.randn(*all_points.shape) * noise_std
       
        # 4. Calcula SDF para todos os pontos
        sdf_values = self.compute_sdf(mesh, perturbed_points)
       
        # 5. Obtém cores (simplificado)
        # Para educativo: cores constantes ou baseadas em posição
        colors = np.zeros((len(perturbed_points), 3))
        colors[:, 0] = (perturbed_points[:, 0] - bbox[0, 0]) / bbox_size  # R
        colors[:, 1] = (perturbed_points[:, 1] - bbox[0, 1]) / bbox_size  # G
        colors[:, 2] = (perturbed_points[:, 2] - bbox[0, 2]) / bbox_size  # B
       
        return perturbed_points, sdf_values, colors