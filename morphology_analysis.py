import cv2
import numpy as np
from skimage.morphology import skeletonize

def analyze_morphology(gray_img, mask):
    """
    Aplica operações morfológicas centrais para analisar a degradação do padrão.
    """
    # Aplica a máscara à imagem para isolar o solado
    sole_img = cv2.bitwise_and(gray_img, gray_img, mask=mask)
    
    # 1. Gradiente Morfológico (Bordas dos padrões)
    # Gradiente = Dilatação - Erosão. Destaca as bordas onde a altura/cor muda.
    kernel_grad = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gradient = cv2.morphologyEx(sole_img, cv2.MORPH_GRADIENT, kernel_grad)
    
    # Limpa o gradiente aplicando a máscara
    gradient = cv2.bitwise_and(gradient, gradient, mask=mask)

    # Limiariza o gradiente para obter bordas binárias do padrão do solado
    # Usamos limiarização adaptativa pois a iluminação pode variar
    binary_edges = cv2.adaptiveThreshold(sole_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 15, 5)
    
    binary_edges = cv2.bitwise_and(binary_edges, binary_edges, mask=mask)
    
    # Abertura nas bordas binárias para remover pequenos ruídos
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary_edges = cv2.morphologyEx(binary_edges, cv2.MORPH_OPEN, kernel_small)

    # 2. Simula o desgaste com Erosão
    # Áreas muito desgastadas desaparecerão após uma leve erosão.
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded_tread = cv2.erode(binary_edges, kernel_erode, iterations=1)
    
    # 3. Esqueletização
    # Obtém a estrutura esquelética do padrão do solado
    # skimage espera um array booleano (True/False ou 1/0)
    bool_tread = binary_edges > 0
    skeleton = skeletonize(bool_tread)
    skeleton_img = (skeleton * 255).astype(np.uint8)
    
    return {
        'gradient': gradient,
        'binary_tread': binary_edges,
        'eroded_tread': eroded_tread,
        'skeleton': skeleton_img
    }
