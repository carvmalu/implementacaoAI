import os 
from pathlib import Path

class Config:

    # paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / 'data'
    RAW_SCANS_DIR = DATA_DIR / 'raw_scans' 
    PROCESSED_DIR = DATA_DIR / 'processed'
    SAMPLES_DIR = DATA_DIR / 'samples'

    # landmark settings (simplificado)
    LANDMARK_INDICES = {
            'left_eye': 36, #dlib indice aproximado
            'right_eye': 45,
            'nose_tip': 30,
            'mouth_left': 48,
            'mouth_right': 54,
            'chin': 8  

    }

    # SDF sampling
    NUM_SAMPLES = 100000
    UNIFORM_RATIO = 0.25 # 25% uniform, 75% landmark-based

    @classmethod
    def setup_directories(cls):
        ''' Cria estrutura de diretórios'''
        dirs = [cls.RAW_SCANS_DIR, cls.PROCESSED_DIR, cls.SAMPLES_DIR]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

