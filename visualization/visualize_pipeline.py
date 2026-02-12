import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go
import trimesh
from pathlib import Path
import json

class PipelineVisualizer:
    """Visualizador completo do pipeline i3DMM"""
    
    def __init__(self, output_dir="visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def plot_mesh_comparison(self, original_mesh, processed_mesh, 
                           landmarks=None, title="Mesh Comparison"):
        """Compara malha original vs processada"""
        fig = plt.figure(figsize=(15, 6))
        
        # Original
        ax1 = fig.add_subplot(121, projection='3d')
        self._plot_mesh_3d(ax1, original_mesh, color='lightblue', alpha=0.6)
        if landmarks:
            self._plot_landmarks(ax1, landmarks, color='red', size=50)
        ax1.set_title("Original Mesh")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        
        # Processada
        ax2 = fig.add_subplot(122, projection='3d')
        self._plot_mesh_3d(ax2, processed_mesh, color='lightgreen', alpha=0.6)
        if landmarks:
            self._plot_landmarks(ax2, landmarks, color='blue', size=50)
        ax2.set_title("Processed Mesh")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        filename = self.output_dir / f"{title.replace(' ', '_').lower()}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()
        
        return filename
    
    def plot_pipeline_stages(self, stages, titles):
        """Plota estágios do pipeline"""
        n_stages = len(stages)
        fig = plt.figure(figsize=(5*n_stages, 5))
        
        for i, (stage_mesh, title) in enumerate(zip(stages, titles)):
            ax = fig.add_subplot(1, n_stages, i+1, projection='3d')
            
            if stage_mesh is not None:
                self._plot_mesh_3d(ax, stage_mesh, color='lightblue', alpha=0.7)
            
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.view_init(elev=20, azim=-60)
        
        plt.suptitle("Pipeline Processing Stages", fontsize=14, y=1.05)
        plt.tight_layout()
        
        filename = self.output_dir / "pipeline_stages.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()
        
        return filename
    
    def plot_sdf_samples(self, points, sdf_values, mesh=None, title="SDF Sampling"):
        """Visualiza pontos SDF"""
        fig = plt.figure(figsize=(14, 6))
        
        # 3D Scatter
        ax1 = fig.add_subplot(121, projection='3d')
        norm_sdf = (sdf_values - sdf_values.min()) / (sdf_values.max() - sdf_values.min() + 1e-8)
        colors = plt.cm.RdBu_r(norm_sdf)
        
        ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c=colors, s=1, alpha=0.6, edgecolors='none')
        
        if mesh is not None:
            self._plot_mesh_wireframe(ax1, mesh, color='black', alpha=0.2)
        
        ax1.set_title("SDF Samples (Red: interior, Blue: exterior)")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        
        # Histograma
        ax2 = fig.add_subplot(122)
        ax2.hist(sdf_values, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax2.axvline(x=0, color='red', linestyle='--', label='Surface (SDF=0)')
        ax2.set_title("SDF Value Distribution")
        ax2.set_xlabel("SDF Value")
        ax2.set_ylabel("Frequency")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        filename = self.output_dir / f"{title.replace(' ', '_').lower()}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()
        
        return filename
    
    def plot_landmark_regions(self, mesh, landmarks, sphere_radius=0.05):
        """Visualiza regiões de landmarks"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Mesh
        self._plot_mesh_3d(ax, mesh, color='lightgray', alpha=0.3)
        
        # Landmarks
        landmark_points = np.array(list(landmarks.values()))
        landmark_names = list(landmarks.keys())
        
        scatter = ax.scatter(landmark_points[:, 0], landmark_points[:, 1], landmark_points[:, 2],
                           c='red', s=100, alpha=0.8, edgecolors='black', zorder=10)
        
        # Esferas ao redor dos landmarks
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        
        for i, (name, center) in enumerate(landmarks.items()):
            # Esfera
            x = sphere_radius * np.outer(np.cos(u), np.sin(v)) + center[0]
            y = sphere_radius * np.outer(np.sin(u), np.sin(v)) + center[1]
            z = sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
            
            ax.plot_surface(x, y, z, color='blue', alpha=0.1, zorder=5)
            
            # Label
            ax.text(center[0], center[1], center[2] + sphere_radius*1.2,
                   name, fontsize=8, ha='center', zorder=20)
        
        ax.set_title("Landmark Regions for Sampling")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        
        # Ajusta limites
        all_points = np.vstack([mesh.vertices, landmark_points])
        max_range = np.ptp(all_points, axis=0).max() / 2.0
        mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
        mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
        mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        
        filename = self.output_dir / "landmark_regions.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()
        
        return filename
    
    def plot_alignment_quality(self, source_landmarks, target_landmarks, 
                              aligned_landmarks=None, title="Alignment Quality"):
        """Visualiza qualidade do alinhamento"""
        fig = plt.figure(figsize=(12, 5))
        
        # Antes do alinhamento
        ax1 = fig.add_subplot(121, projection='3d')
        
        source_pts = np.array(list(source_landmarks.values()))
        target_pts = np.array(list(target_landmarks.values()))
        
        ax1.scatter(source_pts[:, 0], source_pts[:, 1], source_pts[:, 2],
                   c='red', s=50, label='Source', alpha=0.7)
        ax1.scatter(target_pts[:, 0], target_pts[:, 1], target_pts[:, 2],
                   c='blue', s=50, label='Target', alpha=0.7)
        
        # Linhas conectando correspondências
        for s, t in zip(source_pts, target_pts):
            ax1.plot([s[0], t[0]], [s[1], t[1]], [s[2], t[2]], 
                    'k-', alpha=0.3, linewidth=0.5)
        
        ax1.set_title("Before Alignment")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.legend()
        
        # Depois do alinhamento
        ax2 = fig.add_subplot(122, projection='3d')
        
        if aligned_landmarks is not None:
            aligned_pts = np.array(list(aligned_landmarks.values()))
            
            ax2.scatter(aligned_pts[:, 0], aligned_pts[:, 1], aligned_pts[:, 2],
                       c='green', s=50, label='Aligned', alpha=0.7)
            ax2.scatter(target_pts[:, 0], target_pts[:, 1], target_pts[:, 2],
                       c='blue', s=50, label='Target', alpha=0.7)
            
            for a, t in zip(aligned_pts, target_pts):
                ax2.plot([a[0], t[0]], [a[1], t[1]], [a[2], t[2]], 
                        'g-', alpha=0.3, linewidth=0.5)
        
        ax2.set_title("After Alignment")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")
        ax2.legend()
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        filename = self.output_dir / "alignment_quality.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()
        
        return filename
    
    def plot_hole_closing(self, original_mesh, closed_mesh, title="Hole Closing"):
        """Visualiza fechamento de buracos"""
        fig = plt.figure(figsize=(12, 5))
        
        # Antes
        ax1 = fig.add_subplot(121, projection='3d')
        self._plot_mesh_3d(ax1, original_mesh, color='salmon', alpha=0.7)
        ax1.set_title(f"Before - Watertight: {original_mesh.is_watertight}")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        
        # Depois
        ax2 = fig.add_subplot(122, projection='3d')
        self._plot_mesh_3d(ax2, closed_mesh, color='lightgreen', alpha=0.7)
        ax2.set_title(f"After - Watertight: {closed_mesh.is_watertight}")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        filename = self.output_dir / "hole_closing.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()
        
        return filename
    
    def create_interactive_plotly(self, mesh, points=None, sdf_values=None,
                                landmarks=None, title="3D Visualization"):
        """Visualização interativa com Plotly"""
        fig = go.Figure()
        
        # Mesh
        if mesh is not None:
            vertices = mesh.vertices
            faces = mesh.faces
            
            fig.add_trace(go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                opacity=0.5,
                color='lightblue',
                name='Mesh',
                hoverinfo='none'
            ))
        
        # Pontos SDF
        if points is not None and sdf_values is not None:
            norm_sdf = (sdf_values - sdf_values.min()) / (sdf_values.max() - sdf_values.min() + 1e-8)
            
            fig.add_trace(go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=norm_sdf,
                    colorscale='RdBu_r',
                    opacity=0.6,
                    showscale=True,
                    colorbar=dict(title="SDF Value")
                ),
                name='SDF Samples'
            ))
        
        # Landmarks
        if landmarks is not None:
            landmark_points = np.array(list(landmarks.values()))
            landmark_names = list(landmarks.keys())
            
            fig.add_trace(go.Scatter3d(
                x=landmark_points[:, 0],
                y=landmark_points[:, 1],
                z=landmark_points[:, 2],
                mode='markers+text',
                marker=dict(size=8, color='red', symbol='diamond'),
                text=landmark_names,
                textposition="top center",
                name='Landmarks',
                hoverinfo='text'
            ))
        
        # Layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            showlegend=True,
            width=900,
            height=600,
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        # Salva HTML
        filename = self.output_dir / f"{title.replace(' ', '_').lower()}.html"
        fig.write_html(str(filename))
        
        return filename
    
    def generate_report(self, mesh_info, processing_times, output_files):
        """Gera relatório textual do pipeline"""
        report = """
╔══════════════════════════════════════════════════════════════╗
║                 i3DMM EDUCATIONAL PIPELINE                   ║
║                         RELATÓRIO                            ║
╚══════════════════════════════════════════════════════════════╝

▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
📊 INFORMAÇÕES DA MALHA
▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
"""
        for key, value in mesh_info.items():
            report += f"  {key:<20}: {value}\n"
        
        report += """
▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
⏱️  TEMPOS DE PROCESSAMENTO
▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
"""
        for stage, time_val in processing_times.items():
            report += f"  {stage:<25}: {time_val:.2f} seconds\n"
        
        report += """
▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
💾 ARQUIVOS GERADOS
▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
"""
        for file_type, filepath in output_files.items():
            report += f"  {file_type:<25}: {Path(filepath).name}\n"
            report += f"  {' ':<25}  📁 {Path(filepath).parent}/\n"
        
        if 'sdf_samples' in output_files:
            try:
                data = np.load(output_files['sdf_samples'])
                sdf_values = data['sdf_values']
                
                report += """
▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
📈 ESTATÍSTICAS SDF
▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
"""
                report += f"  Total samples        : {len(sdf_values):,}\n"
                report += f"  SDF min              : {sdf_values.min():.4f}\n"
                report += f"  SDF max              : {sdf_values.max():.4f}\n"
                report += f"  SDF mean             : {sdf_values.mean():.4f}\n"
                report += f"  SDF std              : {sdf_values.std():.4f}\n"
                report += f"  Points interior (<0) : {np.sum(sdf_values < 0):,} ({np.mean(sdf_values < 0)*100:.1f}%)\n"
                report += f"  Points exterior (>0) : {np.sum(sdf_values > 0):,} ({np.mean(sdf_values > 0)*100:.1f}%)\n"
                report += f"  Points surface (=0)  : {np.sum(np.abs(sdf_values) < 0.001):,} ({np.mean(np.abs(sdf_values) < 0.001)*100:.1f}%)\n"
            except Exception as e:
                report += f"  SDF statistics not available: {e}\n"
        
        report += """
▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
✅ PIPELINE CONCLUÍDO COM SUCESSO!
▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
"""
        
        # Salva relatório
        report_file = self.output_dir / "pipeline_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        return report_file
    
    def _plot_mesh_3d(self, ax, mesh, color='lightblue', alpha=0.6):
        """Plota malha 3D em matplotlib"""
        vertices = mesh.vertices
        faces = mesh.faces
        
        mesh_collection = Poly3DCollection(vertices[faces], 
                                          alpha=alpha, 
                                          facecolors=color,
                                          edgecolors='darkgray',
                                          linewidths=0.5)
        ax.add_collection3d(mesh_collection)
        
        # Ajusta limites
        ax.auto_scale_xyz(vertices[:, 0], vertices[:, 1], vertices[:, 2])
    
    def _plot_mesh_wireframe(self, ax, mesh, color='black', alpha=0.3, max_faces=100):
        """Plota wireframe da malha"""
        vertices = mesh.vertices
        faces = mesh.faces[:min(len(mesh.faces), max_faces)]
        
        for face in faces:
            triangle = vertices[face]
            ax.plot3D(triangle[:, 0], triangle[:, 1], triangle[:, 2], 
                     color=color, alpha=alpha, linewidth=0.5)
    
    def _plot_landmarks(self, ax, landmarks, color='red', size=50):
        """Plota landmarks"""
        for name, point in landmarks.items():
            ax.scatter(point[0], point[1], point[2], 
                      c=color, s=size, marker='o', 
                      edgecolors='black', linewidths=0.5,
                      alpha=0.8, label=name if 'chin' in name else "")
            ax.text(point[0], point[1], point[2] + 0.02, 
                   name, fontsize=8, ha='center')

# Função de conveniência para visualização rápida
def quick_visualize(mesh, points=None, sdf_values=None, landmarks=None):
    """Visualização rápida para debugging"""
    viz = PipelineVisualizer(output_dir="quick_viz")
    
    # Visualizações básicas
    if landmarks:
        viz.plot_landmark_regions(mesh, landmarks)
    
    if points is not None and sdf_values is not None:
        viz.plot_sdf_samples(points, sdf_values, mesh)
    
    # Interativo
    viz.create_interactive_plotly(mesh, points, sdf_values, landmarks, 
                                 "Quick Visualization")
    
    print(f"✓ Visualizações salvas em: {viz.output_dir}")