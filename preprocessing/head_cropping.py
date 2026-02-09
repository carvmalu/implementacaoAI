# preprocessing / head_cropping.py
import numpy as np
import trimesh
from scipy.spatial import ConvexHull

class HeadCropper:
    '''Crop da cabeça baseado em landmarks'''

    def crop_head(self, mesh, landmarks):
        '''
        Implementa o cropping descrito no paper:
        1. Vetor v: centro dos olhos -> ponta de nariz
        2. Plano em p + 0.5 * v com normal v
        3. Mantém lado com landmarks faciais
        '''
        # Calcula centro dos olhos 
        eye_centroid = (landmarks['left_eye'] + landmarks['right_eye'])/2

        # Vetor v: centro dos olhos -> ponta do nariz
        v = landmarks['nose_tip'] - eye_centroid
        v = v / np.linalg.norm(v) # normaliza

        # ponto p: queixo
        p = landmarks['chin']

        # ponto no plano: p + 0.5*v
        plane_point = p + 0.5* v

        # calcula distância de cada vértice do plano 
        vertices = mesh.vertices
        distances = np.dot(vertices - plane_point, v)

        # mantém vértices do lado positivo (onde estão os landmarks)
        keep_mask = distances > 0 

        if not np.any(keep_mask):
            # fallback: mantém 80% dos vértices mais próximos dos landmarks
            landmarks_points = np.array(list(landmarks.values()))
            distances_to_landmarks = np.min(np.linalg.norm(vertices[:, np.newaxis] - landmarks_points, axis=2), axis =1) 
            keep_mask = distances_to_landmarks < np.percentile(distances_to_landmarks, 80)

        # cria nova malha apenas com vértices mantidos
        # (simplificação - na prática precisaria reindexar faces)
        cropped_vertices = vertices[keep_mask]

        # cria convex hull para ter uma superficie fechada
        hull = ConvexHull(cropped_vertices)
        cropped_mesh = trimesh.Trimesh(vertices= cropped_vertices[hull.vertices], faces = hull.simplices)

        return cropped_mesh