import argparse
import os
import glob
import cv2
from preprocessing import preprocess_pipeline, to_grayscale
from segmentation import segment_sole
from morphology_analysis import analyze_morphology
from feature_extraction import extract_features
from classifier import WearClassifier
from utils.visualization import save_image, annotate_image, ensure_dir, create_dashboard

def process_image(filepath, classifier, debug=False, show=False, output_dir="results"):
    print(f"Processando: {filepath}")
    base_name = os.path.basename(filepath).split('.')[0]
    
    try:
        # 1. Pré-processamento
        original_img, preprocessed_img = preprocess_pipeline(filepath, debug, output_dir)
        
        # 2. Segmentação
        mask = segment_sole(preprocessed_img)
        if debug:
            save_image(mask, f"{base_name}_04_mask.jpg", output_dir)
            
        # 3. Análise Morfológica
        gray_orig = to_grayscale(original_img)
        morph_results = analyze_morphology(gray_orig, mask)
        
        if debug:
            save_image(morph_results['gradient'], f"{base_name}_05_morph_gradient.jpg", output_dir)
            save_image(morph_results['binary_tread'], f"{base_name}_06_binary_tread.jpg", output_dir)
            save_image(morph_results['skeleton'], f"{base_name}_07_skeleton.jpg", output_dir)
            
        # 4. Extração de Características
        features = extract_features(morph_results, mask)
        if debug:
            print(f"Características para {base_name}:")
            for k, v in features.items():
                print(f"  {k}: {v:.4f}")
            
        # 5. Pontuação e Classificação do Desgaste
        score = classifier.calculate_score(features)
        classification = classifier.classify(score)
        
        # 6. Visualização Final
        final_img = annotate_image(original_img, score, classification)
        save_image(final_img, f"{base_name}_08_final.jpg", output_dir)
        
        print(f"Resultado para {base_name} - Pontuação: {score:.1f}, Desgaste: {classification}")
        
        if show:
            dashboard = create_dashboard(
                original_img, preprocessed_img, mask, morph_results['gradient'], 
                morph_results['skeleton'], final_img, features, score, classification
            )
            cv2.imshow(f"Painel: {base_name}", dashboard)
            print("Pressione qualquer tecla na janela do painel para fechá-la e continuar...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        return score, classification
        
    except Exception as e:
        print(f"Erro processando {filepath}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Shoe Sole Wear Detection Pipeline")
    parser.add_argument("--input", type=str, help="Caminho para um arquivo de imagem ou diretório")
    parser.add_argument("--output", type=str, default="results", help="Diretório para salvar as imagens de saída")
    parser.add_argument("--debug", action="store_true", help="Salvar etapas intermediárias")
    parser.add_argument("--batch", action="store_true", help="Processar todas as imagens de um diretório")
    parser.add_argument("--show", action="store_true", help="Mostrar a imagem final anotada em uma janela")
    
    args = parser.parse_args()
    
    if not args.input:
        parser.print_help()
        return

    classifier = WearClassifier()
    ensure_dir(args.output)
    
    if args.batch:
        if not os.path.isdir(args.input):
            print(f"Erro: {args.input} não é um diretório. Use --input apontando para uma pasta em modo --batch.")
            return
            
        # Busca por jfif, jpg, jpeg, png
        extensions = ['*.[jJ][pP][gG]', '*.[jJ][pP][eE][gG]', '*.[jJ][fF][iI][fF]', '*.[pP][nN][gG]']
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(args.input, ext)))
            
        print(f"Encontradas {len(files)} imagens para processar no modo batch.")
        for f in files:
            process_image(f, classifier, args.debug, args.show, args.output)
            print("-" * 40)
            
    else:
        if os.path.isdir(args.input):
            print(f"Erro: {args.input} é um diretório. Use a flag --batch para processar pastas.")
            return
        if not os.path.exists(args.input):
            print(f"Erro: {args.input} não existe.")
            return
            
        process_image(args.input, classifier, args.debug, args.show, args.output)

if __name__ == "__main__":
    main()
