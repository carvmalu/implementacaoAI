import os
import numpy as np
from pathlib import Path

class Config:
    """Configurações do pipeline educacional i3DMM"""
    
    # Diretórios
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    RAW_SCANS_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"
    SAMPLES_DIR = DATA_DIR / "samples"
    VIZ_DIR = BASE_DIR / "visualizations"
    
    # Landmarks - 16 pontos (8 face + 8 orelhas)
    FACIAL_LANDMARKS = [
        'left_eye', 'right_eye',
        'nose_tip',
        'mouth_left', 'mouth_right',
        'chin'
    ]
    
    EAR_LANDMARKS = [
        'ear_left_top', 'ear_left_bottom',
        'ear_right_top', 'ear_right_bottom'
    ]
    
    ALL_LANDMARKS = FACIAL_LANDMARKS + EAR_LANDMARKS
    
    # Template landmarks (forma média - baseado no artigo)
    TEMPLATE_LANDMARKS = {
        # Face
        'left_eye': np.array([-0.10, 0.20, 0.05]),
        'right_eye': np.array([0.10, 0.20, 0.05]),
        'nose_tip': np.array([0.00, 0.10, 0.20]),
        'mouth_left': np.array([-0.05, 0.00, 0.15]),
        'mouth_right': np.array([0.05, 0.00, 0.15]),
        'chin': np.array([0.00, -0.20, 0.10]),
        # Orelhas
        'ear_left_top': np.array([-0.25, 0.10, 0.00]),
        'ear_left_bottom': np.array([-0.25, -0.10, 0.00]),
        'ear_right_top': np.array([0.25, 0.10, 0.00]),
        'ear_right_bottom': np.array([0.25, -0.10, 0.00])
    }
    
    # Parâmetros de amostragem SDF
    NUM_SAMPLES = 100000
    UNIFORM_RATIO = 0.25  # 25% uniforme, 75% landmark-based
    LANDMARK_SPHERE_RADIUS = 0.05
    PERTURBATION_STD = 0.005  # 0.005 × bbox
    
    # Parâmetros de visualização
    PLOT_DPI = 150
    INTERACTIVE = True
    
    @classmethod
    def setup_directories(cls):
        """Cria estrutura de diretórios"""
        dirs = [
            cls.RAW_SCANS_DIR,
            cls.PROCESSED_DIR,
            cls.SAMPLES_DIR,
            cls.VIZ_DIR
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Diretórios criados em: {cls.DATA_DIR.parent}")