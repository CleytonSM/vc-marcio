import cv2
import numpy as np
import os

def load_image(filepath):
    """Carrega uma imagem e a converte para RGB."""
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError(f"Could not load image at {filepath}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def to_grayscale(image):
    """Converte uma imagem RGB para tons de cinza."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def normalize_lighting(gray_img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Aplica CLAHE (Equalização de Histograma Adaptativa Limitada por Contraste)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray_img)

def apply_blur(gray_img, kernel_size=(5, 5)):
    """Aplica Desfoque Gaussiano para reduzir o ruído."""
    return cv2.GaussianBlur(gray_img, kernel_size, 0)

def preprocess_pipeline(filepath, debug=False, output_dir="results"):
    """Executa todo o pipeline de pré-processamento."""
    img = load_image(filepath)
    gray = to_grayscale(img)
    norm = normalize_lighting(gray)
    blurred = apply_blur(norm)
    
    if debug:
        from utils.visualization import save_image
        base_name = os.path.basename(filepath).split('.')[0]
        save_image(gray, f"{base_name}_01_grayscale.jpg", output_dir)
        save_image(norm, f"{base_name}_02_clahe.jpg", output_dir)
        save_image(blurred, f"{base_name}_03_blurred.jpg", output_dir)
        
    return img, blurred
