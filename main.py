#!/usr/bin/env python3
"""
i3DMM Educational Pipeline - Main Script
Implementação educacional do pipeline completo do artigo:
"i3DMM: Deep Implicit 3D Morphable Model of Human Heads"

Autores originais: Tarun Yenamandra, Ayush Tewari, et al.
Implementação educacional: Adaptado para fins de aprendizado
"""

import time
import numpy as np
import trimesh
from pathlib import Path
import json
import sys

# Configuração
from config import Config

# Módulos de pré-processamento
from preprocessing.landmark_detection import EducationalLandmarkDetector
from preprocessing.head_cropping import EducationalHeadCropper
from preprocessing.alignment import EducationalAligner
from preprocessing.hole_closing import EducationalHoleCloser
from preprocessing.sdf_sampling import EducationalSDFSampler

# Visualização
from visualization.visualize_pipeline import PipelineVisualizer

class i3DMMEducationalPipeline:
    """
    Pipeline principal do i3DMM - Versão Educacional
    
    Implementa sequência completa do artigo:
    1. Landmark Detection
    2. Head Cropping  
    3. Rigid Alignment (Procrustes + Template)
    4. Hole Closing (Watertight)
    5. SDF Sampling (Uniforme + Landmark-based)
    
    Todas as etapas são visualizadas e documentadas.
    """
    
    def __init__(self):
        """Inicializa pipeline com todos os módulos"""
        print("\n" + "=" * 60)
        print("   i3DMM EDUCATIONAL PIPELINE")
        print("   Deep Implicit 3D Morphable Model of Human Heads")
        print("=" * 60)
        
        # Configuração
        self.config = Config()
        self.config.setup_directories()
        
        # Inicializa módulos
        print("\n🔧 Inicializando módulos...")
        self.landmark_detector = EducationalLandmarkDetector(self.config)
        self.head_cropper = EducationalHeadCropper()
        self.aligner = EducationalAligner()
        self.hole_closer = EducationalHoleCloser()
        self.sampler = EducationalSDFSampler(self.config)
        self.visualizer = PipelineVisualizer(output_dir=str(self.config.VIZ_DIR))
        
        # Template landmarks (forma média)
        self.template_landmarks = self.config.TEMPLATE_LANDMARKS
        
        # Métricas
        self.processing_times = {}
        self.output_files = {}
        self.mesh_info = {}
        
        print("  ✓ Landmark Detector")
        print("  ✓ Head Cropper")
        print("  ✓ Rigid Aligner (Procrustes)")
        print("  ✓ Hole Closer")
        print("  ✓ SDF Sampler")
        print("  ✓ Visualizer")
        print("\n📁 Diretórios:")
        print(f"  Raw scans:  {self.config.RAW_SCANS_DIR}")
        print(f"  Processed:  {self.config.PROCESSED_DIR}")
        print(f"  Samples:    {self.config.SAMPLES_DIR}")
        print(f"  Visuals:    {self.config.VIZ_DIR}")
        print("=" * 60)
    
    def run_full_pipeline(self, mesh_path, save_intermediate=True):
        """
        Executa pipeline completo em uma malha
        
        Args:
            mesh_path: Caminho para arquivo .obj ou .ply
            save_intermediate: Salva resultados intermediários
            
        Returns:
            aligned_mesh: Malha alinhada e processada
            landmarks: Landmarks detectados e alinhados
            points: Pontos amostrados para SDF
            sdf_values: Valores SDF correspondentes
            colors: Cores associadas aos pontos
        """
        mesh_path = Path(mesh_path)
        print(f"\n{'='*60}")
        print(f"🚀 EXECUTANDO PIPELINE COMPLETO")
        print(f"📄 Arquivo: {mesh_path.name}")
        print(f"{'='*60}")
        
        pipeline_start = time.time()
        
        # ------------------------------------------------------
        # 1. CARREGAR MALHA
        # ------------------------------------------------------
        print("\n[1/6] 📥 Carregando malha...")
        load_start = time.time()
        
        mesh = trimesh.load(str(mesh_path))
        
        self.mesh_info = {
            "Arquivo": mesh_path.name,
            "Vértices": f"{len(mesh.vertices):,}",
            "Faces": f"{len(mesh.faces):,}",
            "Watertight": f"{mesh.is_watertight}",
            "Volume": f"{mesh.volume:.4f}",
            "Bounds": f"[{mesh.bounds[0,0]:.2f}, {mesh.bounds[1,0]:.2f}] x "
                     f"[{mesh.bounds[0,1]:.2f}, {mesh.bounds[1,1]:.2f}] x "
                     f"[{mesh.bounds[0,2]:.2f}, {mesh.bounds[1,2]:.2f}]"
        }
        
        print(f"  ✓ Vértices: {self.mesh_info['Vértices']}")
        print(f"  ✓ Faces: {self.mesh_info['Faces']}")
        print(f"  ✓ Volume: {self.mesh_info['Volume']}")
        
        self.processing_times["Load Mesh"] = time.time() - load_start
        
        # ------------------------------------------------------
        # 2. DETECTAR LANDMARKS
        # ------------------------------------------------------
        print("\n[2/6] 🔍 Detectando landmarks (16 pontos)...")
        landmark_start = time.time()
        
        landmarks = self.landmark_detector.detect_simple(mesh)
        
        self.processing_times["Landmark Detection"] = time.time() - landmark_start
        print(f"  ✓ {len(landmarks)} landmarks detectados em {self.processing_times['Landmark Detection']:.2f}s")
        
        # Salva landmarks originais
        if save_intermediate:
            orig_landmark_path = self.config.PROCESSED_DIR / f"{mesh_path.stem}_landmarks_original.json"
            self.landmark_detector.save_landmarks(landmarks, str(orig_landmark_path))
            self.output_files["Original Landmarks"] = str(orig_landmark_path)
        
        # ------------------------------------------------------
        # 3. HEAD CROPPING
        # ------------------------------------------------------
        print("\n[3/6] ✂️  Aplicando head cropping...")
        crop_start = time.time()
        
        cropped_mesh = self.head_cropper.crop_simple(mesh, landmarks)
        
        self.processing_times["Head Cropping"] = time.time() - crop_start
        print(f"  ✓ Crop concluído: {len(cropped_mesh.vertices)} vértices, {len(cropped_mesh.faces)} faces")
        
        # Salva malha cropada
        if save_intermediate:
            crop_path = self.config.PROCESSED_DIR / f"{mesh_path.stem}_cropped.obj"
            cropped_mesh.export(str(crop_path))
            self.output_files["Cropped Mesh"] = str(crop_path)
        
        # ------------------------------------------------------
        # 4. RIGID ALIGNMENT (PROCRUSTES + TEMPLATE)
        # ------------------------------------------------------
        print("\n[4/6] 📐 Alinhando ao template (Procrustes)...")
        align_start = time.time()
        
        # PASSO CRÍTICO: Passar template_landmarks!
        aligned_mesh, aligned_landmarks = self.aligner.simple_pipeline(
            cropped_mesh, 
            landmarks, 
            self.template_landmarks  # ← CORRETO! Template é passado aqui
        )
        
        self.processing_times["Rigid Alignment"] = time.time() - align_start
        print(f"  ✓ Alinhamento concluído em {self.processing_times['Rigid Alignment']:.2f}s")
        
        # Visualiza qualidade do alinhamento
        align_viz_path = self.visualizer.plot_alignment_quality(
            landmarks, self.template_landmarks, aligned_landmarks,
            f"Alignment Quality - {mesh_path.stem}"
        )
        self.output_files["Alignment Visualization"] = str(align_viz_path)
        
        # Salva malha alinhada
        if save_intermediate:
            align_path = self.config.PROCESSED_DIR / f"{mesh_path.stem}_aligned.obj"
            aligned_mesh.export(str(align_path))
            self.output_files["Aligned Mesh"] = str(align_path)
            
            align_landmark_path = self.config.PROCESSED_DIR / f"{mesh_path.stem}_landmarks_aligned.json"
            self.landmark_detector.save_landmarks(aligned_landmarks, str(align_landmark_path))
            self.output_files["Aligned Landmarks"] = str(align_landmark_path)
        
        # ------------------------------------------------------
        # 5. HOLE CLOSING (WATERTIGHT)
        # ------------------------------------------------------
        print("\n[5/6] 🔧 Fechando buracos (watertight)...")
        hole_start = time.time()
        
        watertight_mesh = self.hole_closer.simple_pipeline(aligned_mesh)
        
        self.processing_times["Hole Closing"] = time.time() - hole_start
        
        # Visualiza hole closing
        hole_viz_path = self.visualizer.plot_hole_closing(
            aligned_mesh, watertight_mesh,
            f"Hole Closing - {mesh_path.stem}"
        )
        self.output_files["Hole Closing Visualization"] = str(hole_viz_path)
        
        # Salva malha watertight
        if save_intermediate:
            watertight_path = self.config.PROCESSED_DIR / f"{mesh_path.stem}_watertight.obj"
            watertight_mesh.export(str(watertight_path))
            self.output_files["Watertight Mesh"] = str(watertight_path)
        
        # ------------------------------------------------------
        # 6. SDF SAMPLING
        # ------------------------------------------------------
        print("\n[6/6] 🎯 Amostrando pontos SDF...")
        sample_start = time.time()
        
        points, sdf_values, colors = self.sampler.sample_simple(
            watertight_mesh, aligned_landmarks  # Usa landmarks alinhados
        )
        
        self.processing_times["SDF Sampling"] = time.time() - sample_start
        print(f"  ✓ {len(points):,} pontos amostrados em {self.processing_times['SDF Sampling']:.2f}s")
        
        # ------------------------------------------------------
        # 7. VISUALIZAÇÕES FINAIS
        # ------------------------------------------------------
        print("\n🎨 Gerando visualizações finais...")
        viz_start = time.time()
        
        # Comparação malha original vs processada
        self.visualizer.plot_mesh_comparison(
            mesh, watertight_mesh, aligned_landmarks,
            f"Original vs Processed - {mesh_path.stem}"
        )
        
        # Estágios do pipeline
        stages = [mesh, cropped_mesh, aligned_mesh, watertight_mesh]
        stage_titles = ["Original", "After Cropping", "After Alignment", "After Hole Closing"]
        self.visualizer.plot_pipeline_stages(stages, stage_titles)
        
        # Regiões de landmarks
        self.visualizer.plot_landmark_regions(watertight_mesh, aligned_landmarks)
        
        # Amostras SDF
        self.visualizer.plot_sdf_samples(
            points, sdf_values, watertight_mesh,
            f"SDF Sampling - {mesh_path.stem}"
        )
        
        # Visualização interativa
        interactive_file = self.visualizer.create_interactive_plotly(
            watertight_mesh, points, sdf_values, aligned_landmarks,
            f"i3DMM Pipeline - {mesh_path.stem}"
        )
        self.output_files["Interactive 3D"] = str(interactive_file)
        
        self.processing_times["Visualization"] = time.time() - viz_start
        
        # ------------------------------------------------------
        # 8. SALVAR RESULTADOS
        # ------------------------------------------------------
        print("\n💾 Salvando resultados finais...")
        save_start = time.time()
        
        # Pontos SDF
        samples_path = self.config.SAMPLES_DIR / f"{mesh_path.stem}_sdf_samples.npz"
        np.savez(
            str(samples_path),
            points=points,
            sdf_values=sdf_values,
            colors=colors
        )
        self.output_files["SDF Samples"] = str(samples_path)
        
        # Configuração do pipeline
        config_path = self.config.PROCESSED_DIR / f"{mesh_path.stem}_config.json"
        config_dict = {
            "mesh_file": mesh_path.name,
            "num_vertices": len(watertight_mesh.vertices),
            "num_faces": len(watertight_mesh.faces),
            "num_samples": len(points),
            "sdf_min": float(sdf_values.min()),
            "sdf_max": float(sdf_values.max()),
            "sdf_mean": float(sdf_values.mean()),
            "sdf_std": float(sdf_values.std()),
            "processing_times": {k: round(v, 2) for k, v in self.processing_times.items()}
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        self.output_files["Configuration"] = str(config_path)
        
        self.processing_times["Save Results"] = time.time() - save_start
        
        # ------------------------------------------------------
        # 9. TEMPO TOTAL
        # ------------------------------------------------------
        self.processing_times["Total Pipeline"] = time.time() - pipeline_start
        
        print(f"\n✅ Pipeline concluído em {self.processing_times['Total Pipeline']:.2f} segundos!")
        
        # ------------------------------------------------------
        # 10. RELATÓRIO FINAL
        # ------------------------------------------------------
        report_file = self.visualizer.generate_report(
            self.mesh_info, self.processing_times, self.output_files
        )
        self.output_files["Report"] = str(report_file)
        
        print(f"\n📊 Relatório salvo em: {report_file}")
        print(f"🎯 Visualizações salvas em: {self.config.VIZ_DIR}")
        print("=" * 60)
        
        return watertight_mesh, aligned_landmarks, points, sdf_values, colors
    
    def process_multiple_scans(self, scan_dir=None, pattern="*.obj"):
        """
        Processa múltiplos scans em lote
        
        Args:
            scan_dir: Diretório com os scans (default: raw_scans)
            pattern: Padrão de arquivos (*.obj, *.ply)
        """
        if scan_dir is None:
            scan_dir = self.config.RAW_SCANS_DIR
        else:
            scan_dir = Path(scan_dir)
        
        mesh_files = list(scan_dir.glob(pattern))
        
        if not mesh_files:
            print(f"⚠️ Nenhum arquivo {pattern} encontrado em {scan_dir}")
            return
        
        print(f"\n📚 Processando {len(mesh_files)} scans em lote...")
        print("-" * 40)
        
        results = {}
        for i, mesh_file in enumerate(mesh_files, 1):
            print(f"\n[{i}/{len(mesh_files)}] Processando: {mesh_file.name}")
            try:
                result = self.run_full_pipeline(mesh_file, save_intermediate=True)
                results[mesh_file.stem] = {
                    "success": True,
                    "mesh": result[0],
                    "num_samples": len(result[2])
                }
            except Exception as e:
                print(f"❌ Erro ao processar {mesh_file.name}: {e}")
                results[mesh_file.stem] = {"success": False, "error": str(e)}
        
        # Relatório resumido do lote
        successful = sum(1 for r in results.values() if r.get("success", False))
        failed = len(results) - successful
        
        print("\n" + "=" * 60)
        print("📊 RELATÓRIO DO LOTE")
        print("=" * 60)
        print(f"Total de scans: {len(results)}")
        print(f"✅ Sucesso: {successful}")
        print(f"❌ Falhas: {failed}")
        print("=" * 60)
        
        return results
    
    def create_example_mesh(self):
        """Cria malha de exemplo (icosaedro) para testes"""
        print("\n🔨 Criando malha de exemplo...")
        
        # Cria icosaedro como cabeça simplificada
        example_mesh = trimesh.creation.icosahedron()
        
        # Escala e translação para posição mais realista
        example_mesh.vertices *= 0.5  # Escala
        example_mesh.vertices[:, 1] += 0.3  # Move para cima
        
        # Salva
        example_path = self.config.RAW_SCANS_DIR / "example_head.obj"
        example_mesh.export(str(example_path))
        
        print(f"  ✓ Malha de exemplo criada: {example_path}")
        print(f"    - Vértices: {len(example_mesh.vertices)}")
        print(f"    - Faces: {len(example_mesh.faces)}")
        
        return example_path

def main():
    """
    Função principal - Ponto de entrada do pipeline
    """
    print("\n" + "=" * 60)
    print("   i3DMM - DEEP IMPLICIT 3D MORPHABLE MODEL")
    print("   Implementação Educacional")
    print("   Baseado no artigo de Yenamandra, Tewari et al. (2021)")
    print("=" * 60)
    
    # Inicializa pipeline
    pipeline = i3DMMEducationalPipeline()
    
    # Menu interativo simples
    print("\n📋 OPÇÕES:")
    print("  1. Processar um scan específico")
    print("  2. Processar todos os scans na pasta raw/")
    print("  3. Criar malha de exemplo e processar")
    print("  4. Sair")
    
    try:
        choice = input("\n👉 Escolha uma opção (1-4): ").strip()
        
        if choice == "1":
            # Processar arquivo específico
            mesh_files = list(pipeline.config.RAW_SCANS_DIR.glob("*.obj")) + \
                        list(pipeline.config.RAW_SCANS_DIR.glob("*.ply"))
            
            if not mesh_files:
                print("⚠️ Nenhum arquivo .obj ou .ply encontrado.")
                print("   Criando malha de exemplo...")
                mesh_path = pipeline.create_example_mesh()
            else:
                print("\nArquivos disponíveis:")
                for i, f in enumerate(mesh_files, 1):
                    print(f"  {i}. {f.name}")
                
                idx = int(input(f"\nEscolha um arquivo (1-{len(mesh_files)}): ")) - 1
                mesh_path = mesh_files[idx]
            
            pipeline.run_full_pipeline(mesh_path)
            
        elif choice == "2":
            # Processar todos os scans
            pipeline.process_multiple_scans()
            
        elif choice == "3":
            # Criar exemplo e processar
            mesh_path = pipeline.create_example_mesh()
            pipeline.run_full_pipeline(mesh_path)
            
        else:
            print("👋 Saindo...")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrompido pelo usuário.")
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("   ✅ PIPELINE FINALIZADO")
    print("=" * 60)

if __name__ == "__main__":
    main()