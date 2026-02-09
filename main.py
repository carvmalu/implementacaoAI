# main_pipeline.py
import os
import numpy as np
import trimesh
from config import Config
from preprocessing.landmark_detection import LandmarkDetector
from preprocessing.head_cropping import HeadCropper
from preprocessing.alignment import RigidAligner
from preprocessing.sdf_sampling import SDFSampler
from visualization.visualize_pipeline import PipelineVisualizer

class EducationalPipeline:
    """Pipeline completo de pré-processamento"""
    
    def __init__(self):
        self.config = Config()
        self.config.setup_directories()
        
        self.landmark_detector = LandmarkDetector(self.config)
        self.head_cropper = HeadCropper()
        self.aligner = RigidAligner()
        self.sampler = SDFSampler(self.config)
        self.visualizer = PipelineVisualizer()
        
        # Template simplificado (poderia carregar de um arquivo)
        self.template_landmarks = {
            'left_eye': np.array([-0.1, 0.2, 0.0]),
            'right_eye': np.array([0.1, 0.2, 0.0]),
            'nose_tip': np.array([0.0, 0.1, 0.2]),
            'chin': np.array([0.0, -0.2, 0.0])
        }
    
    def process_mesh(self, mesh_path, output_prefix):
        """Processa uma malha completa"""
        print(f"\n=== Processando: {mesh_path} ===")
        
        # 1. Carrega malha
        mesh = trimesh.load(mesh_path)
        print(f"1. Malha carregada: {len(mesh.vertices)} vértices")
        
        # 2. Detecta landmarks
        landmarks = self.landmark_detector.detect_from_mesh(mesh)
        print(f"2. Landmarks detectados: {len(landmarks)}")
        
        # 3. Crop da cabeça
        cropped_mesh = self.head_cropper.crop_head(mesh, landmarks)
        print(f"3. Crop realizado: {len(cropped_mesh.vertices)} vértices restantes")
        
        # 4. Alinhamento rígido
        aligned_mesh = self.aligner.align_to_template(
            cropped_mesh, landmarks, self.template_landmarks
        )
        print("4. Alinhamento concluído")
        
        # 5. Amostragem SDF
        points, sdf_values, colors = self.sampler.sample_training_points(
            aligned_mesh, landmarks
        )
        print(f"5. Amostragem: {len(points)} pontos coletados")
        
        # 6. Salva resultados
        self.save_results(
            output_prefix, mesh, cropped_mesh, aligned_mesh,
            landmarks, points, sdf_values, colors
        )
        
        # 7. Visualiza
        self.visualizer.visualize_pipeline(
            mesh, cropped_mesh, aligned_mesh,
            landmarks, points, sdf_values
        )
        
        return aligned_mesh, points, sdf_values, colors
    
    def save_results(self, prefix, original_mesh, cropped_mesh, 
                    aligned_mesh, landmarks, points, sdf_values, colors):
        """Salva todos os resultados do pipeline"""
        
        # Salva malhas
        cropped_mesh.export(f"{prefix}_cropped.obj")
        aligned_mesh.export(f"{prefix}_aligned.obj")
        
        # Salva landmarks
        self.landmark_detector.save_landmarks(
            landmarks, f"{prefix}_landmarks.json"
        )
        
        # Salva pontos SDF
        np.savez(
            f"{prefix}_sdf_samples.npz",
            points=points,
            sdf_values=sdf_values,
            colors=colors
        )
        
        print(f"Resultados salvos com prefixo: {prefix}")

# Execução do pipeline
if __name__ == "__main__":
    # Configuração
    pipeline = EducationalPipeline()
    
    # Exemplo: processa uma malha de teste
    # (assumindo que você tem uma malha .obj ou .ply)
    test_mesh_path = "data/raw_scans/sample_head.obj"
    
    if os.path.exists(test_mesh_path):
        output_prefix = "data/processed/sample_processed"
        pipeline.process_mesh(test_mesh_path, output_prefix)
    else:
        print("Criando malha de exemplo para demonstração...")
        
        # Cria uma malha esférica simplificada para teste
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        test_path = "data/raw_scans/test_sphere.obj"
        mesh.export(test_path)
        
        output_prefix = "data/processed/test_sphere"
        pipeline.process_mesh(test_path, output_prefix)