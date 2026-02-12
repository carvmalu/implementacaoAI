import numpy as np
import trimesh
from scipy.spatial import KDTree
import json

class EducationalLandmarkDetector:
    """
    Detector de landmarks educacional
    Baseado no artigo: dlib (2D) + transferência 3D + correção manual
    Versão simplificada para fins educacionais
    """
    
    def __init__(self, config):
        self.config = config
        self.landmark_names = config.ALL_LANDMARKS
        
    def detect_simple(self, mesh):
        """
        Detecta landmarks usando heurísticas geométricas
        (Simulação do pipeline dlib + transferência 3D)
        """
        vertices = mesh.vertices
        landmarks = {}
        
        print("  Detectando landmarks geométricos...")
        
        # 1. Queixo (ponto mais baixo)
        chin_idx = np.argmin(vertices[:, 1])
        landmarks['chin'] = vertices[chin_idx]
        
        # 2. Ponta do nariz (ponto mais à frente, assumindo Z como frente)
        nose_idx = np.argmax(vertices[:, 2])
        landmarks['nose_tip'] = vertices[nose_idx]
        
        # 3. Olhos (pontos mais altos à esquerda/direita do nariz)
        nose_x = vertices[nose_idx, 0]
        y_threshold = np.percentile(vertices[:, 1], 75)  # Top 25% em Y
        
        # Filtra pontos acima do threshold
        top_points = vertices[vertices[:, 1] > y_threshold]
        
        if len(top_points) > 0:
            # Olho esquerdo (menor X)
            left_eye_idx = np.argmin(top_points[:, 0])
            landmarks['left_eye'] = top_points[left_eye_idx]
            
            # Olho direito (maior X)
            right_eye_idx = np.argmax(top_points[:, 0])
            landmarks['right_eye'] = top_points[right_eye_idx]
        
        # 4. Boca (pontos médios na região inferior do nariz)
        nose_y = vertices[nose_idx, 1]
        mouth_region = vertices[
            (vertices[:, 1] < nose_y) & 
            (vertices[:, 1] > nose_y - 0.1) &
            (np.abs(vertices[:, 0]) < 0.1)
        ]
        
        if len(mouth_region) > 0:
            mouth_left_idx = np.argmin(mouth_region[:, 0])
            mouth_right_idx = np.argmax(mouth_region[:, 0])
            landmarks['mouth_left'] = mouth_region[mouth_left_idx]
            landmarks['mouth_right'] = mouth_region[mouth_right_idx]
        else:
            # Fallback
            landmarks['mouth_left'] = np.array([-0.05, nose_y - 0.05, 0.15])
            landmarks['mouth_right'] = np.array([0.05, nose_y - 0.05, 0.15])
        
        # 5. Orelhas (simulação - pontos laterais)
        left_ear_region = vertices[vertices[:, 0] < -0.15]
        right_ear_region = vertices[vertices[:, 0] > 0.15]
        
        if len(left_ear_region) > 0:
            ear_left_top_idx = np.argmax(left_ear_region[:, 1])
            ear_left_bottom_idx = np.argmin(left_ear_region[:, 1])
            landmarks['ear_left_top'] = left_ear_region[ear_left_top_idx]
            landmarks['ear_left_bottom'] = left_ear_region[ear_left_bottom_idx]
        
        if len(right_ear_region) > 0:
            ear_right_top_idx = np.argmax(right_ear_region[:, 1])
            ear_right_bottom_idx = np.argmin(right_ear_region[:, 1])
            landmarks['ear_right_top'] = right_ear_region[ear_right_top_idx]
            landmarks['ear_right_bottom'] = right_ear_region[ear_right_bottom_idx]
        
        # Preenche landmarks faltantes com posições do template
        for name in self.landmark_names:
            if name not in landmarks:
                print(f"  ⚠️ Landmark {name} não detectado, usando template")
                landmarks[name] = self.config.TEMPLATE_LANDMARKS[name].copy()
        
        print(f"  ✓ {len(landmarks)} landmarks detectados")
        return landmarks
    
    def save_landmarks(self, landmarks, output_path):
        """Salva landmarks em JSON"""
        landmarks_dict = {}
        for name, point in landmarks.items():
            landmarks_dict[name] = {
                'x': float(point[0]),
                'y': float(point[1]),
                'z': float(point[2])
            }
        
        with open(output_path, 'w') as f:
            json.dump(landmarks_dict, f, indent=2)
        
        print(f"  ✓ Landmarks salvos em: {output_path}")
        return output_path
    
    def load_landmarks(self, input_path):
        """Carrega landmarks de JSON"""
        with open(input_path, 'r') as f:
            landmarks_dict = json.load(f)
        
        landmarks = {}
        for name, coords in landmarks_dict.items():
            landmarks[name] = np.array([coords['x'], coords['y'], coords['z']])
        
        return landmarks