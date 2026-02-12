import numpy as np
import trimesh
from scipy.spatial import ConvexHull

class EducationalHoleCloser:
    """
    Fechamento de buracos para malhas watertight
    Baseado no artigo: patch plano no pescoço após alinhamento
    """
    
    def __init__(self):
        self.verbose = True
    
    def find_boundary_vertices(self, mesh):
        """Encontra vértices de boundary (borda)"""
        edges = mesh.edges_sorted
        unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
        boundary_edges = unique_edges[counts == 1]
        
        if len(boundary_edges) == 0:
            return np.array([])
        
        boundary_vertices = np.unique(boundary_edges.flatten())
        return boundary_vertices
    
    def identify_neck_boundary(self, mesh, boundary_vertices, y_threshold=None):
        """
        Identifica boundary do pescoço (parte inferior da cabeça)
        Assumes que após alinhamento, eixo Y é vertical
        """
        vertices = mesh.vertices
        
        if y_threshold is None:
            # Menor Y entre os vértices de boundary
            y_threshold = np.min(vertices[boundary_vertices, 1]) + 0.02
        
        # Boundary vertices na parte inferior
        neck_mask = vertices[boundary_vertices, 1] < y_threshold
        neck_boundary = boundary_vertices[neck_mask]
        
        return neck_boundary, y_threshold
    
    def close_neck_with_flat_patch(self, mesh):
        """
        Fecha buraco do pescoço com patch plano
        Método descrito no artigo: extrud boundary vertices para y_min
        """
        print("  Fechando buraco do pescoço...")
        
        # 1. Encontra boundary vertices
        boundary_vertices = self.find_boundary_vertices(mesh)
        
        if len(boundary_vertices) == 0:
            print("  ✓ Malha já é watertight")
            return mesh
        
        # 2. Identifica boundary do pescoço
        neck_boundary, y_min = self.identify_neck_boundary(mesh, boundary_vertices)
        
        if len(neck_boundary) < 3:
            print("  ⚠️ Poucos vértices para fechar, usando convex hull")
            return self.make_watertight_convex_hull(mesh)
        
        # 3. Cria patch plano
        vertices = mesh.vertices.copy()
        faces = mesh.faces.copy()
        
        # Projeta boundary vertices para y_min
        for idx in neck_boundary:
            vertices[idx, 1] = y_min
        
        # 4. Adiciona ponto central para triangulação
        center = np.mean(vertices[neck_boundary], axis=0)
        center[1] = y_min  # Garante que está no plano
        center_idx = len(vertices)
        vertices = np.vstack([vertices, center])
        
        # 5. Triangula o patch
        n = len(neck_boundary)
        new_faces = []
        
        # Ordena vértices angularmente
        boundary_pts = vertices[neck_boundary]
        angles = np.arctan2(boundary_pts[:, 2] - center[2], 
                           boundary_pts[:, 0] - center[0])
        sorted_idx = neck_boundary[np.argsort(angles)]
        
        # Cria faces triangulares
        for i in range(n):
            v1 = sorted_idx[i]
            v2 = sorted_idx[(i + 1) % n]
            new_faces.append([v1, v2, center_idx])
        
        # 6. Adiciona novas faces
        all_faces = np.vstack([faces, new_faces])
        
        closed_mesh = trimesh.Trimesh(vertices=vertices, faces=all_faces)
        
        print(f"  ✓ Buraco fechado: {len(new_faces)} novas faces")
        print(f"    - Watertight: {closed_mesh.is_watertight}")
        print(f"    - Volume: {closed_mesh.volume:.4f}")
        
        return closed_mesh
    
    def make_watertight_convex_hull(self, mesh):
        """Fallback: usa convex hull para watertight"""
        print("  Usando convex hull como fallback...")
        vertices = mesh.vertices
        hull = ConvexHull(vertices)
        
        hull_mesh = trimesh.Trimesh(
            vertices=vertices[hull.vertices],
            faces=hull.simplices
        )
        
        return hull_mesh
    
    def simple_pipeline(self, mesh):
        """Pipeline completo de hole closing"""
        print("\n[HOLE CLOSING]")
        print("=" * 40)
        
        print(f"  Estado inicial:")
        print(f"    - Watertight: {mesh.is_watertight}")
        print(f"    - Volume: {mesh.volume:.4f}")
        print(f"    - Buracos: {len(mesh.holes) if hasattr(mesh, 'holes') else '?'}")
        
        if mesh.is_watertight:
            print("  ✓ Malha já é watertight")
            return mesh
        
        # Tenta fechar pescoço
        try:
            closed = self.close_neck_with_flat_patch(mesh)
            if closed.is_watertight:
                return closed
        except Exception as e:
            print(f"  ⚠️ Erro no fechamento: {e}")
        
        # Fallback
        print("  ⚠️ Usando convex hull como fallback final")
        return self.make_watertight_convex_hull(mesh)