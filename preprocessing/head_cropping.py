import numpy as np
import trimesh

class EducationalHeadCropper:
    """
    Crop da cabeça baseado no artigo:
    - Vetor v: centro dos olhos → ponta do nariz
    - Plano: p + 0.5v com normal v
    - Mantém lado com landmarks faciais
    """
    
    def __init__(self):
        self.eye_landmarks = ['left_eye', 'right_eye']
        self.nose_landmark = 'nose_tip'
        self.chin_landmark = 'chin'
    
    def crop_simple(self, mesh, landmarks):
        """Aplica crop baseado em landmarks"""
        print("  Aplicando head cropping...")
        
        # 1. Centro dos olhos
        left_eye = landmarks.get('left_eye')
        right_eye = landmarks.get('right_eye')
        
        if left_eye is None or right_eye is None:
            print("  ⚠️ Landmarks dos olhos não encontrados, usando bounds")
            return self._crop_by_bounds(mesh)
        
        eye_centroid = (left_eye + right_eye) / 2
        
        # 2. Ponta do nariz
        nose_tip = landmarks.get('nose_tip')
        if nose_tip is None:
            print("  ⚠️ Landmark do nariz não encontrado, usando crop por bounds")
            return self._crop_by_bounds(mesh)
        
        # 3. Vetor v: centro dos olhos → ponta do nariz
        v = nose_tip - eye_centroid
        v_norm = v / (np.linalg.norm(v) + 1e-8)
        
        # 4. Ponto do queixo
        chin = landmarks.get('chin')
        if chin is None:
            print("  ⚠️ Landmark do queixo não encontrado, usando nose_tip")
            chin = nose_tip
        
        # 5. Ponto no plano: p + 0.5v
        plane_point = chin + 0.5 * v
        
        # 6. Distância dos vértices ao plano
        vertices = mesh.vertices
        distances = np.dot(vertices - plane_point, v_norm)
        
        # 7. Mantém lado positivo (onde estão os landmarks faciais)
        keep_mask = distances > -0.02  # Margem pequena
        
        if np.sum(keep_mask) < 100:  # Menos de 100 vértices
            print("  ⚠️ Crop muito agressivo, ajustando...")
            # Fallback: mantém 80% dos vértices
            threshold = np.percentile(distances, 20)
            keep_mask = distances > threshold
        
        # 8. Cria nova malha
        cropped_vertices = vertices[keep_mask]
        cropped_faces = []
        
        # Reindexa faces
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(np.where(keep_mask)[0])}
        
        for face in mesh.faces:
            if all(idx in vertex_map for idx in face):
                new_face = [vertex_map[idx] for idx in face]
                cropped_faces.append(new_face)
        
        if len(cropped_faces) == 0:
            print("  ⚠️ Nenhuma face mantida, usando fallback")
            return self._crop_by_percentile(mesh)
        
        cropped_mesh = trimesh.Trimesh(
            vertices=cropped_vertices,
            faces=np.array(cropped_faces)
        )
        
        print(f"  ✓ Crop concluído: {len(cropped_vertices)} vértices, {len(cropped_faces)} faces")
        return cropped_mesh
    
    def _crop_by_bounds(self, mesh):
        """Fallback: crop baseado em bounds"""
        bounds = mesh.bounds
        y_min = bounds[0, 1]
        y_max = bounds[1, 1]
        
        # Mantém apenas a metade superior
        y_threshold = y_min + 0.6 * (y_max - y_min)
        vertices = mesh.vertices
        keep_mask = vertices[:, 1] > y_threshold
        
        return self._create_mesh_from_mask(mesh, keep_mask)
    
    def _crop_by_percentile(self, mesh):
        """Fallback: crop baseado em percentil Y"""
        vertices = mesh.vertices
        y_values = vertices[:, 1]
        y_threshold = np.percentile(y_values, 30)  # Mantém 70% superiores
        keep_mask = y_values > y_threshold
        
        return self._create_mesh_from_mask(mesh, keep_mask)
    
    def _create_mesh_from_mask(self, mesh, keep_mask):
        """Cria nova malha a partir de máscara de vértices"""
        vertices = mesh.vertices[keep_mask]
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(np.where(keep_mask)[0])}
        
        faces = []
        for face in mesh.faces:
            if all(idx in vertex_map for idx in face):
                faces.append([vertex_map[idx] for idx in face])
        
        if len(faces) == 0:
            # Fallback extremo: retorna convex hull
            hull = trimesh.creation.icosphere()
            return hull
        
        return trimesh.Trimesh(vertices=vertices, faces=np.array(faces))