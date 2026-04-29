import cv2
import numpy as np

def extract_features(morph_results, mask):
    """
    Extrai características numéricas da análise morfológica.
    """
    total_area = cv2.countNonZero(mask)
    if total_area == 0:
        return None
        
    binary_tread = morph_results['binary_tread']
    eroded_tread = morph_results['eroded_tread']
    skeleton = morph_results['skeleton']
    
    # 1. Densidade das Bordas: proporção da área do padrão em relação à área total do solado
    tread_area = cv2.countNonZero(binary_tread)
    edge_density = tread_area / total_area
    
    # 2. Métrica de Simulação de Desgaste: Quanto do padrão sobrevive à erosão
    # (Proporção menor = mais gasto, porque padrões finos/gastos desaparecem mais rápido)
    surviving_tread = cv2.countNonZero(eroded_tread)
    if tread_area > 0:
        wear_resistance_ratio = surviving_tread / tread_area
    else:
        wear_resistance_ratio = 0.0

    # 3. Proporção do Comprimento do Esqueleto: relação entre o comprimento do esqueleto e a área total
    skeleton_length = cv2.countNonZero(skeleton)
    skeleton_density = skeleton_length / total_area
    
    # 4. Conectividade: Número de componentes conectados no padrão
    # Um solado muito gasto pode ter padrões quebrados, aumentando a contagem de CC, 
    # ou áreas totalmente lisas, diminuindo a contagem de CC.
    num_labels, labels = cv2.connectedComponents(binary_tread, connectivity=8)
    # Normaliza pela área total (multiplicado por um fator para facilitar a leitura)
    connectivity_density = num_labels / total_area * 1000 
    
    # 5. Porcentagem Lisa: Áreas sem padrão significativo
    # Dilata o padrão levemente, e o inverso é a região lisa
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    dilated_tread = cv2.dilate(binary_tread, kernel, iterations=1)
    flat_mask = cv2.bitwise_and(cv2.bitwise_not(dilated_tread), mask)
    flat_area = cv2.countNonZero(flat_mask)
    flatness_percentage = flat_area / total_area

    features = {
        'edge_density': edge_density,
        'wear_resistance_ratio': wear_resistance_ratio,
        'skeleton_density': skeleton_density,
        'connectivity_density': connectivity_density,
        'flatness_percentage': flatness_percentage
    }
    return features
