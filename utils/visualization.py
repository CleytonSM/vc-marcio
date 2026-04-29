import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_image(image, filename, output_dir="results"):
    ensure_dir(output_dir)
    filepath = os.path.join(output_dir, filename)
    if len(image.shape) == 2:
        # Tons de cinza
        cv2.imwrite(filepath, image)
    else:
        # RGB
        cv2.imwrite(filepath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return filepath

def plot_comparison(original, processed, title_orig="Original", title_proc="Processed"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    if len(original.shape) == 2:
        axes[0].imshow(original, cmap='gray')
    else:
        axes[0].imshow(original)
    axes[0].set_title(title_orig)
    axes[0].axis('off')

    if len(processed.shape) == 2:
        axes[1].imshow(processed, cmap='gray')
    else:
        axes[1].imshow(processed)
    axes[1].set_title(title_proc)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

def annotate_image(image, score, classification):
    annotated = image.copy()
    if len(annotated.shape) == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2RGB)
    
    # Configurações de texto
    font = cv2.FONT_HERSHEY_SIMPLEX
    text1 = f"Pontuacao: {score:.1f}/100"
    text2 = f"Classe: {classification}"
    
    cv2.putText(annotated, text1, (30, 50), font, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(annotated, text2, (30, 90), font, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
    
    return annotated

def create_dashboard(original, preprocessed, mask, gradient, skeleton, final, features, score, classification):
    target_size = (300, 300)
    
    def prepare(img):
        resized = cv2.resize(img, target_size)
        if len(resized.shape) == 2:
            return cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        else:
            return cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
            
    orig_bgr = prepare(original)
    prep_bgr = prepare(preprocessed)
    mask_bgr = prepare(mask)
    grad_bgr = prepare(gradient)
    skel_bgr = prepare(skeleton)
    final_bgr = prepare(final)
    
    def add_title(img, title):
        cv2.putText(img, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return img
        
    orig_bgr = add_title(orig_bgr, "Original")
    prep_bgr = add_title(prep_bgr, "Pre-processado")
    mask_bgr = add_title(mask_bgr, "Mascara")
    grad_bgr = add_title(grad_bgr, "Gradiente")
    skel_bgr = add_title(skel_bgr, "Esqueleto")
    final_bgr = add_title(final_bgr, "Final Anotado")
    
    row1 = np.hstack([orig_bgr, prep_bgr, mask_bgr])
    row2 = np.hstack([grad_bgr, skel_bgr, final_bgr])
    grid = np.vstack([row1, row2])
    
    panel_width = 350
    panel = np.zeros((grid.shape[0], panel_width, 3), dtype=np.uint8)
    
    y = 40
    cv2.putText(panel, "PAINEL DE ANALISE", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    y += 40
    cv2.putText(panel, f"Pontuacao: {score:.1f}/100", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y += 30
    cv2.putText(panel, f"Classe: {classification}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y += 50
    cv2.putText(panel, "Caracteristicas:", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    if features:
        translations = {
            'edge_density': 'Densidade das Bordas',
            'wear_resistance_ratio': 'Taxa de Resistencia',
            'skeleton_density': 'Densidade do Esqueleto',
            'connectivity_density': 'Densidade de Conexao',
            'flatness_percentage': 'Porcentagem Lisa'
        }
        for k, v in features.items():
            y += 30
            formatted_key = translations.get(k, k.replace('_', ' ').title())
            cv2.putText(panel, f"{formatted_key}:", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            y += 20
            cv2.putText(panel, f"{v:.4f}", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
    dashboard = np.hstack([grid, panel])
    return dashboard
