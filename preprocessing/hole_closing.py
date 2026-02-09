import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from scipy.interpolate import griddata

class HoleCloser:
    '''
    Fecha buracos na malha para torna-lá watertight
    Necessário para representação SDF
    '''

    def close_neck_hole_simple(self, mesh, y_threshold=None):
        '''
        Fecha buracos no pescoço (simplificado)
        Baseado no artigo: extrude boundary vertices para menor coordenada y 
        '''
        vertices = mesh.vertices
        faces = mesh.faces

        # 1. encontra vértices de boundary(bordas)
        # Usa análise de edges para encontrar boundary 
        edges = mesh.edges_sorted
        unique_edges, counts = np.unique(edges, axis =0, return_counts=True)
        boundary_edges = unique_edges[counts == 1]

        if len(boundary_edges) == 0:
            print('No boundary edges found - mesh might already by watertight')
            return mesh

        # 2. Identifica boundary vertices
        boundary_vertex_indices = np.unique(boundary_edges.flatten())
        boundary_vertices = vertcies[boundary_vertex_indices]

        # 3. encontra a menor coordenada y entre os boundary vertices
        if y_threshold is None:
            y_threshold = np.min(boundary_vertices[:,1]) - 0.01

        # 4. projeta boundary vertoices para o plano y= y_threshold
        new_vertices = vertices.copy()
        new_vertices[boundary_vertex_indices, 1] = y_threshold

        # 5. cria face de fechamento(tapa buraco)
        # para simplicidade, vou criar uma face convexa
        if len(boundary_vertices) >= 3:
            # ordena boundary vertices angularmente em torno do centro
            boundary_center = np.mean(boundary_vertices, axis =0)
            boundary_relative = boundary_vertices - boundary_center

            # calcula o ângulo no plano XZ (ignora y)
            angles = np.arctan2(boundary_relative[:,2], boundary_relative[:, 0])
            sorted_indices = np.argsort(angles)

            # cria faces triangulares para fechar
            new_faces =[]
            num_boundary = len(boundary_vertex_indices)

            # cria triângulos convergindo para um ponto central adicional
            center_vertex = np.array([boundary_center[0], y_threshold, boundary_center[2]])

            # adiciona vértice central
            new_vertices = np.vstack([new_vertices, center_vertex])
            center_idx = len(new_vertices) - 1

            # cria faces triangulares
            for i in range(num_boundary):
                v1 = boundary_vertex_indices[sorted_indices[i]] 
                v2 = boundary_vertex_indices[sorted_indices[(i+1) % num_boundary]]
                new_faces.append(v1,v2,center_idx)
            # adiciona novas faces às existentes
            all_faces = np.vstack([faces, new_faces])
            closed_mesh = trimesh.Trimesh(vertices=new_vertices, faces= faces)
        else:
            # fallback: simplesmente usa a malha original
            closed_mesh = trimesh.Trimesh(vertices= new_vertices, faces= faces)
        return closed_mesh
    
    def make_watertight_convex_hull(self, mesh):
        '''
        Método alternativo: usa convex hull para garantir watertight 
        mais simples, mas menos preciso
        '''
        vertices = mesh.vertices

        # calcula convex hull
        hull = ConvexHull(vertices)

        # cria malha do convex hull
        hull_mesh = trimesh.Trimesh(vertices=vertices[hull.vertices], faces= hull.simplices)

        return hull_mesh

    def close_holes_trimesh(self,mesh):
        '''
        Usa função do trimesh para fechar buracos 
        '''
        # processa a malha para fechar buracos
        processed = mesh.process()

        # tentar fechar buracos
        if processed.is_watertight:
            return processed
        else: 
            # se não for watertight, tenta fill_holes
            try:
                filled = processed.fill_holes()
                return filled
            except:
                # fallback para convex hull
                return self.make_watertight_convex_hull(mesh)

    def create_flat_patch(self, boundary_vertices, normal=(0, -1, 0)):
        '''
        Cria um patch plano para fechar o buraco
        '''
        # projeta boundary vertices em um plano
        from scipy.spatial import Delaunay

        # encontra plano de melhor ajuste
        centroid = np.mean(boundary_vertices, axis = 0)

        # usa normal fornecida ou calcula do plano
        if normal == (0, -1, 0):
            # plano horizontal (para pescoço)
            plane_normal = np.array([0, -1, 0])
        else:
            # calcula normal do plano via PCA
            centered = boundary_vertices - centroid
            _, _, Vt = np.linalg.svd(centered)
            plane_normal = Vt[2] # menor componente singular

            # garante que aponta para baixo
            if plane_normal[1] > 0:
                plane_normal = -plane_normal

        # Projeta vértices no plano
        projected = []
        for v in boundary_vertices:
            # projeção no plano
            t = np.dot(plane_normal, centroid - v) / np.dot(plane_normal, plane_normal)
            proj_v = v + t * plane_normal
            projected.append(proj_v)

        projected = np.array(projected)

        # triangula no plano 2D 
        # projeto para 2D ignorando a dimensão da normal
        if abs(plane_normal[2]) > abs(plane_normal[0]) and abs(plane_normal[2]) > abs(plane_normal[1]):
            # Normal principalmente em Z, usa projeção XY 
            coords_2d = projected[:, :2]
        elif abs(plane_normal[1]) > abs(plane_normal[0]):
            # Normal principalmente em Y, usa projeção XZ
            coords_2d = projected[:, [0,2]]
        else:
            # Normal principalmente em X, usa projeção YZ
            coords_2d = projected[:, 1:]

        # delaunay triangulation
        try:
            tri = Delaunay(coords_2d)
            patch_faces = tri.simplices

            # verifica orientação
            for i, face in enumerate(patch_faces):
                v0, v1, v2 = projected[face]
                normal_face = np.cross(v1- v0, v2-v0)
                if np.dot(normal_face, plane_normal) < 0:
                    patch_faces[i] = face[::-1] # inverte orientação
            return projected, patch_faces
        except:
            # fallback: triângulo simples se daulanay falhar
            if len(projected) >= 3:
                center = np.mean(projected, axis = 0)
                projected = np.vstack([projected, center])

                patch_faces = []
                n = len(boundary_vertices)
                center_idx = n

                for i in range(n): 
                    patch_faces.append([i, (i+ 1) % n, center_idx])
                return projected, np.array(patch_faces)

            else:
                return boundary_vertices, np.array([])
        
    def close_with_flat_patch(self, mesh, normal=(0, -1, 0)):
        '''
        Implementação do método descritono artigo:
        use a flat patch to close the hole at the neck 
        '''
        # 1. Identifica boundary vertices
        edges = mesh.edges_sorted
        unique_edges, counts = np.unique(edges, axis = 0, return_counts =True)
        boundary_edges = unique_edges[counts == 1]

        if len(boundary_edges) == 0:
            return mesh
        
        boundary_vertex_indices = np.unique(boundary_edges.flatten())
        boundary_vertices = mesh.vertices[boundary_vertex_indices]

        # 2. Cria patch plano
        patch_vertices, patch_faces = self.create_flat_patch(boundary_vertices, normal)

        if len(patch_faces) == 0:
            return mesh

        # 3. Combina com malha original 
        # Mapeia índices dos boundary vertices para as novas
        vertex_map = {idx: i for i, idx in enumerate(boundary_vertex_indices)}

        # atualiza vértices dos boundary
        new_vertices = mesh.vertices.copy()
        for i, idx in enumerate(boundary_vertex_indices):
            if i < len(patch_vertices):
                new_vertices[idx] = patch_vertices[i]
        
        # adiciona novos vértices do patch (se houver vértices adicionais)
        if len(patch_vertices) > len(boundary_vertex_indices):
            extra_vertices = patch_vertices[len(boundary_vertex_indices):]
            new_vertices = np.vstack([new_vertices, extra_vertices])

            # Atualiza indices das faces do patch 
            for face in patch_faces:
                for j in range(3):
                    if face[j] >= len(boundary_vertex_indices):
                        face[j] = len(mesh.vertices) + (face[j] - len(boundary_vertex_indices))
                    else:
                        face[j] = boundary_vertex_indices[face[j]]
        else:
            # apenas atualiza índices
            for face in patch_faces:
                face[:] = [boundary_vertex_indices[f] for f in face]
        
        # 4. combina faces
        all_faces = np.vstack([mesh.faces, patch_faces])

        closed_mesh = trimesh.Trimesh(vertices= new_vertices, faces = all_faces)

        return closed_mesh
        