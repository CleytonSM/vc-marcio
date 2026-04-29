import cv2
import numpy as np

def segment_sole(preprocessed_img):
    """
    Segmenta o solado do sapato do fundo.
    Assume bom contraste com o fundo da imagem.
    """
    # Usa limiarização de Otsu
    ret, thresh = cv2.threshold(preprocessed_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Heurística simples para garantir que o solado seja branco (255) e o fundo preto (0):
    # Verifica os cantos. Se forem em sua maioria brancos, inverte a imagem.
    h, w = thresh.shape
    corners = [
        thresh[0:10, 0:10],
        thresh[0:10, w-10:w],
        thresh[h-10:h, 0:10],
        thresh[h-10:h, w-10:w]
    ]
    white_pixels = sum(np.sum(c == 255) for c in corners)
    total_pixels = 400 # 4 * (10*10)
    
    if white_pixels > total_pixels * 0.5:
        # Fundo é branco, então inverte
        thresh = cv2.bitwise_not(thresh)
        
    # Operações Morfológicas para limpar a máscara
    # 1. Abertura para remover pequenos ruídos (ex: pontos no fundo)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)
    
    # 2. Fechamento para preencher buracos dentro do solado
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
    
    # Mantém apenas o maior componente conectado
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
    
    if num_labels > 1:
        # O rótulo 0 é o fundo
        # Encontra o maior componente excluindo o fundo
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.zeros_like(closed)
        mask[labels == largest_label] = 255
    else:
        mask = closed

    # Fechamento final para garantir bordas suaves
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    return mask
