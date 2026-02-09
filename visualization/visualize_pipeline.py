import numpy as np
import matplotlib.plyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import Poly3DCollection
import plotly.graph_objects as go
import trimesh
import os
from pathlib import path

class PipelineVisualizer:
    ' Visualizador para todas as etapas do pipeline'
    def __init__(self, output_dir = 'visualizations'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok = True)
    
    def plot_mesh_comparison(self, original_mesh, processed_mesh, landmarks = None, title= 'Mesh comparison'):
        '''
        Compara malha original vs processada lado a lado
        '''
        fig = plt.fig(figzise= (15,6))

        # plot original
        ax1 = fig.add_subplot(121, projection ='3d')
        self._plot_mesh_3d(ax1, original_mesh, color = 'lightblue', alph = 0.6)
        if landmarks:
            self._plot_landmarks(ax1, landmarks, color='red', size=50)
        ax1.set_title('Original mesh')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        # plot processado
        ax2 = fig.add_subplot(122, projection= '3d')
        self._plot_mesh_3d(ax2, processed_mesh, color= 'lightgreen', alpha = 0.6)
        if landmarks:
            self._plot_landmarks(ax2, landmarks, color='blue', size=50)
        ax2.set_title('Processed mesh')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

         plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        # Salva
        filename = self.output_dir / f"{title.replace(' ', '_').lower()}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()
        
        return filename
    
    def plot_sdf_samples(self, points, sdf_values, mesh=None, 
                        title="SDF Sampling"):
        """
        Visualiza pontos amostrados com cores baseadas no SDF
        """
        fig = plt.figure(figsize=(14, 6))
        
        # Subplot 1: Pontos coloridos por SDF
        ax1 = fig.add_subplot(121, projection='3d')
        
        # Normaliza SDF para escala de cores
        norm_sdf = (sdf_values - sdf_values.min()) / (sdf_values.max() - sdf_values.min())
        colors = plt.cm.RdBu_r(norm_sdf)  # Vermelho (negativo) -> Azul (positivo)
        
        scatter = ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                            c=colors, s=1, alpha=0.6, edgecolors='none')
        
        if mesh is not None:
            self._plot_mesh_wireframe(ax1, mesh, color='black', alpha=0.2)
        
        ax1.set_title("SDF Samples (Red: interior, Blue: exterior)")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        
        # Subplot 2: Histograma dos valores SDF
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
        """
        Mostra regiões ao redor dos landmarks para amostragem
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plota malha
        self._plot_mesh_3d(ax, mesh, color='lightgray', alpha=0.3)
        
        # Plota landmarks
        landmark_points = np.array(list(landmarks.values()))
        landmark_names = list(landmarks.keys())
        
        ax.scatter(landmark_points[:, 0], landmark_points[:, 1], landmark_points[:, 2],
                  c='red', s=100, alpha=0.8, edgecolors='black')
        
        # Adiciona esferas ao redor dos landmarks
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        
        for i, (name, center) in enumerate(landmarks.items()):
            # Cria esfera
            x = sphere_radius * np.outer(np.cos(u), np.sin(v)) + center[0]
            y = sphere_radius * np.outer(np.sin(u), np.sin(v)) + center[1]
            z = sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
            
            ax.plot_surface(x, y, z, color='blue', alpha=0.1)
            
            # Adiciona label
            ax.text(center[0], center[1], center[2] + sphere_radius*1.2,
                   name, fontsize=9, ha='center')
        
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
    
    def plot_pipeline_stages(self, stages, titles):
        """
        Plota múltiplos estágios do pipeline em uma figura
        """
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
            
            # Configuração de view consistente
            ax.view_init(elev=20, azim=-60)
        
        plt.suptitle("Pipeline Processing Stages", fontsize=14, y=1.05)
        plt.tight_layout()
        
        filename = self.output_dir / "pipeline_stages.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()
        
        return filename
    
    def create_interactive_plotly(self, mesh, points=None, sdf_values=None,
                                landmarks=None, title="3D Visualization"):
        """
        Cria visualização interativa com Plotly
        """
        fig = go.Figure()
        
        # Adiciona malha
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
                name='Mesh'
            ))
        
        # Adiciona pontos SDF se fornecidos
        if points is not None and sdf_values is not None:
            # Normaliza SDF para cores
            norm_sdf = (sdf_values - sdf_values.min()) / (sdf_values.max() - sdf_values.min())
            
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
        
        # Adiciona landmarks
        if landmarks is not None:
            landmark_points = np.array(list(landmarks.values()))
            landmark_names = list(landmarks.keys())
            
            fig.add_trace(go.Scatter3d(
                x=landmark_points[:, 0],
                y=landmark_points[:, 1],
                z=landmark_points[:, 2],
                mode='markers+text',
                marker=dict(size=8, color='red'),
                text=landmark_names,
                textposition="top center",
                name='Landmarks'
            ))
        
        # Configura layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            showlegend=True,
            width=800,
            height=600
        )
        
        # Salva como HTML interativo
        filename = self.output_dir / f"{title.replace(' ', '_').lower()}.html"
        fig.write_html(str(filename))
        
        return filename
    
    def _plot_mesh_3d(self, ax, mesh, color='lightblue', alpha=0.6):
        """Plota malha 3D em um axes matplotlib"""
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Cria collection de polígonos
        mesh_collection = Poly3DCollection(vertices[faces], 
                                          alpha=alpha, 
                                          facecolors=color,
                                          edgecolors='darkgray',
                                          linewidths=0.5)
        ax.add_collection3d(mesh_collection)
        
        # Ajusta limites
        ax.auto_scale_xyz(vertices[:, 0], vertices[:, 1], vertices[:, 2])
    
    def _plot_mesh_wireframe(self, ax, mesh, color='black', alpha=0.3):
        """Plota wireframe da malha"""
        vertices = mesh.vertices
        faces = mesh.faces
        
        for face in faces[:100]:  # Limita para performance
            triangle = vertices[face]
            ax.plot(triangle[:, 0], triangle[:, 1], triangle[:, 2], 
                   color=color, alpha=alpha, linewidth=0.5)
    
    def _plot_landmarks(self, ax, landmarks, color='red', size=50):
        """Plota landmarks"""
        for name, point in landmarks.items():
            ax.scatter(point[0], point[1], point[2], 
                      c=color, s=size, marker='o', label=name if 'chin' in name else "")
    
    def generate_report(self, mesh_info, processing_times, output_files):
        """
        Gera relatório textual do pipeline
        """
        report = """
        ========================================
        i3DMM EDUCATIONAL PIPELINE - RELATÓRIO
        ========================================
        
        INFORMACOES DA MALHA:
        ---------------------
        """
        
        for key, value in mesh_info.items():
            report += f"{key}: {value}\n"
        
        report += """
        
        TEMPOS DE PROCESSAMENTO:
        -----------------------
        """
        
        for stage, time in processing_times.items():
            report += f"{stage}: {time:.2f} seconds\n"
        
        report += """
        
        ARQUIVOS GERADOS:
        -----------------
        """
        
        for file_type, filepath in output_files.items():
            report += f"{file_type}: {filepath}\n"
        
        report += """
        
        ESTATISTICAS SDF:
        -----------------
        """
        
        if 'sdf_samples' in output_files:
            try:
                data = np.load(output_files['sdf_samples'])
                sdf_values = data['sdf_values']
                
                report += f"Total samples: {len(sdf_values):,}\n"
                report += f"SDF min: {sdf_values.min():.4f}\n"
                report += f"SDF max: {sdf_values.max():.4f}\n"
                report += f"SDF mean: {sdf_values.mean():.4f}\n"
                report += f"SDF std: {sdf_values.std():.4f}\n"
                report += f"Points inside (SDF < 0): {np.sum(sdf_values < 0):,} ({np.mean(sdf_values < 0)*100:.1f}%)\n"
                report += f"Points outside (SDF > 0): {np.sum(sdf_values > 0):,} ({np.mean(sdf_values > 0)*100:.1f}%)\n"
            except:
                report += "SDF statistics not available\n"
        
        # Salva relatório
        report_file = self.output_dir / "pipeline_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(report)
        return report_file

    # Função de conveniência para uso rápido
    def quick_visualize(mesh, points=None, sdf_values=None, landmarks=None):
        """
        Visualização rápida para debugging
        """
        visualizer = PipelineVisualizer(output_dir="quick_visualizations")
        
        # Plota malha básica
        visualizer.plot_mesh_comparison(mesh, mesh, landmarks, "Quick Visualization")
        
        # Plota SDF se disponível
        if points is not None and sdf_values is not None:
            visualizer.plot_sdf_samples(points, sdf_values, mesh, "Quick SDF Visualization")
        
        # Plota interativo se disponível
        if landmarks is not None:
            visualizer.create_interactive_plotly(mesh, points, sdf_values, landmarks, "Interactive 3D View")