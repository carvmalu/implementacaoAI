import numpy as np
import trimesh

class EducationalAligner:
    """
    Alinhamento rígido baseado em landmarks
    Usa Procrustes (SVD) - método correto segundo o artigo
    """
    
    def __init__(self):
        self.use_scale = False  # Artigo não usa escala
        
    def procrustes_analysis(self, source_points, target_points):
        """
        Procrustes analysis com SVD
        Args:
            source_points: (N,3) - pontos da malha fonte
            target_points: (N,3) - pontos do template
        Returns:
            R: (3,3) - matriz de rotação
            t: (3,) - vetor de translação
        """
        # 1. Centraliza os pontos
        source_mean = np.mean(source_points, axis=0)
        target_mean = np.mean(target_points, axis=0)
        
        source_centered = source_points - source_mean
        target_centered = target_points - target_mean
        
        # 2. SVD para rotação ótima
        H = source_centered.T @ target_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # 3. Corrige reflexão (garante rotação própria)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # 4. Translação
        t = target_mean - R @ source_mean
        
        return R, t
    
    def align_to_template(self, mesh, landmarks, template_landmarks):
        """
        Alinha malha ao template usando landmarks correspondentes
        """
        # 1. Extrai pontos correspondentes (apenas landmarks comuns)
        common_points = []
        source_pts = []
        target_pts = []
        
        for name in template_landmarks.keys():
            if name in landmarks:
                common_points.append(name)
                source_pts.append(landmarks[name])
                target_pts.append(template_landmarks[name])
        
        if len(common_points) < 3:
            print(f"  ⚠️ Apenas {len(common_points)} landmarks comuns, insuficiente para alinhamento")
            return mesh
        
        source_pts = np.array(source_pts)
        target_pts = np.array(target_pts)
        
        # 2. Calcula transformação
        R, t = self.procrustes_analysis(source_pts, target_pts)
        
        # 3. Aplica a todos os vértices
        vertices = mesh.vertices
        aligned_vertices = vertices @ R.T + t
        
        # 4. Atualiza landmarks
        aligned_landmarks = {}
        for name, pt in landmarks.items():
            aligned_landmarks[name] = pt @ R.T + t
        
        aligned_mesh = trimesh.Trimesh(
            vertices=aligned_vertices,
            faces=mesh.faces.copy()
        )
        
        print(f"  ✓ Alinhamento concluído: rotação R, translação t")
        print(f"    - Landmarks usados: {len(common_points)}")
        print(f"    - RMSE: {np.sqrt(np.mean(np.linalg.norm(source_pts @ R.T + t - target_pts, axis=1)**2)):.4f}")
        
        return aligned_mesh, aligned_landmarks
    
    def simple_pipeline(self, mesh, landmarks, template_landmarks):
        """Pipeline simplificado de alinhamento"""
        print("\n[RIGID ALIGNMENT]")
        print("=" * 40)
        
        aligned_mesh, aligned_landmarks = self.align_to_template(
            mesh, landmarks, template_landmarks
        )
        
        return aligned_mesh, aligned_landmarks