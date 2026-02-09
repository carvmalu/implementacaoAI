# preprocessing
import numpy as np
import trimesh
from scipy.spatial import KDTree

class LandmarkDetection:
    '''Detector de landmarks simplificado para fins educacionais'''

    def __init__(self, config):
        self.config = config
    
    def detect_from_mesh(self, mesh):
        '''
        Detecta landmarks simplificados baseado em heuristicas geométricas
        para educativo: não usamos dlib, usamos posições relativas
        '''

        vertices = mesh.vertices

        # heuristicas simples para landmarks
        landmarks = {}

        # 1. ponto mais baixo = queixo aproximado
        chin_idx = np.argmin(vertices[:,1]) # y minimo
        landmarks['chin'] = vertices[chin_idx]

        # 2. ponto mais a frente = ponta do nariz aproximado
        nose_idx = np.argmax(vertices[:,2])
        landmarks['nose_tip'] = vertices[nose_idx]

        # 3. Olhos: ponto mais altos à esquerda/direita da matriz
        nose_x = vertices[nose_idx, 0]
        left_mask = vertices[:,0] < nose_x
        right_mask = vertices[:, 0] > nose_x

        if np.any(left_mask):
            left_eye_idx = np.argmax(vertices[left_mask, 1]) # y maximo a esquerda
            landmarks['left_eye'] = vertices[left_mask][left_eye_idx]
        if np.any(right_mask):
            right_eye_idx = np.argmax(vertices[right_mask,1]) # y maximo a direita
            landmarks['right_eye'] = vertices[right_mask][right_eye_idx]
        return landmarks
    
    def save_landmarks(self, landmarks, output_path):
        'salva landmarks para visualização'
        import json

        # converte arrays numpy pata listas
        landmarks_dict = {k:v.tolist() for k,v in landmarks.items()}

        with open(output_path, 'w') as f:
            json.dump(landmarks_dict, f, indent=2) 

