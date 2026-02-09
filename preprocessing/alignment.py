import numpy as np
import trimesh 
from scipy.spatial.transform import rotation
from scipy.linalg import orthogonal_procrustes
import networkx as nx

class RigidAligner:
    '''Alinhamento rígido usando Transformation Sychronization'''

    def pairwise_procrustes(self, source_points, target_points):
        '''
        Procrustes entre dois conjuntos de pontos
        Return: R (rotação), t (translação)
        '''
        # centraliza 
        source_mean = np.mean(source_points, axis=0)
        target_mean = np.mean(target_points, axis=0)

        source_centered = source_points - source_mean
        target_centered = target_points - target_mean

        # SVD para rotação ótima
        H = source_centered.T @ target_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # corrige reflexão se necessário
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # translação
        t = target_mean - R @ source_mean
        return R, t
    def transformation_synchronization(self, pairwise_transforms):
        '''
        Transformation synchronization [2,3]
        encontra transformações consistentes globalmente 
        '''
        n = len(pairwise_transforms)

        # constrói matriz de rotações 3n x 3n
        R_block = np.zeros((3*n, 3*n))

        for i in range(n):
            for j in range(n):
                if i != j and pairwise_transforms[i][j] is not None:
                    R_ij, _ = pairwise_transforms[i][j]
                    R_block[3*i:3*1+3,3*j:3*j+3]
        
        # SVD para encontrar rotações absolutas
        U, _, _ = np.linalg.svd(R_block)
        R_absolute = U[:, :3]

        return R_absolute
    
    def align_all_meshes(self, meshes, all_landmarks):
        '''
        Alinha TODAS as malhas simultaneamente usando transformation synchronization
        '''
        n = len(meshes)

        # 1. calcula transformações par a par
        pairwise_transforms = [[None] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i != j:
                    # extrai landmarks correspondentes
                    source_landmarks = np.array(list(all_landmarks[i].values()))
                    target_landmarks = np.array(list(all_landmarks[j].values()))

                    # calcula transformação i -> j
                    R_ij, t_ij = self.pairwise_procrustes(source_landmarks, target_landmarks)
                    pairwise_transforms[i][j] = (R_ij, t_ij)

            # 2. transformation synchronization
            R_absolute = self.transformation_synchronization(pairwise_transforms)

            # 3. aplica transformações absolutas
            aligned_meshes = []
            for i in range(n):
                # extrai a rotação absoluta para este scan
                R_i = R_absolute[3*i: 3*i + 3,:3]

                # para translação usa media das translações relativas
                translations = []
                for j in range(n):
                    if i != j and pairwise_transforms[i][j] is not None:
                        _,  t_ij = pairwise_transforms[i][j]
                        translations.append(t_ij)
                t_i = np.mean(translations, axis =0) if translations else np.zeros(3)

                # aplica transformações
                vertices = meshes[i].vertices
                aligned_vertices = vertices @ R_i.T + t_i
                aligned_mesh =  trimesh.Trimesh(vertices = aligned_vertices, faces = meshes[i].faces)
                aligned_meshes.append(aligned_mesh)

        return aligned_meshes
    def alignt_to_mean_shape(self, mesh, landmarks, mean_landmarks):
        '''
        Versãp simplificada: alinha uma malha a uma forma média
        '''
        # converte landmarks para arrays
        source_points = np.array(list(landmarks.values()))
        target_points = np.array(list(mean_landmarks.values()))

        # procurustes direto
        R, t = self.pairwise_procrustes(source_points, target_points)

        # aplica
        aligned_vertices = mesh.vertices @ R.T + t

        return trimesh.Trimesh(vertices = aligned_vertices, faces= mesh.faces)
    




