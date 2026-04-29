class WearClassifier:
    def __init__(self, low_thresh=30, high_thresh=70):
        """
        Inicializa o classificador com limiares configuráveis.
        """
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh
        
    def calculate_score(self, features):
        """
        Calcula uma pontuação de desgaste de 0 a 100 com base nas características extraídas.
        Uma pontuação maior significa mais desgaste.
        """
        if not features:
            return 100.0, "DESCONHECIDO"

        # Heurísticas para pontuação de desgaste:
        # - Maior porcentagem_lisa -> mais desgastado
        # - Menor taxa_de_resistencia -> mais desgastado
        # - Menor densidade_do_esqueleto -> mais desgastado
        
        # Normaliza e combina (esses pesos são heurísticos e precisam ser ajustados)
        # 1. Porcentagem Lisa: 0 a 1 -> 0 a 100
        f_flat = min(features['flatness_percentage'] * 150, 100) # scale up a bit
        
        # 2. Resistência ao desgaste: 1 a 0 -> 0 a 100
        f_resist = (1.0 - features['wear_resistance_ratio']) * 100
        
        # 3. Densidade do esqueleto: Geralmente em torno de 0.05. Menor significa mais gasto.
        # Densidade máxima esperada ~ 0.1. Então 0.1 -> desgaste 0, 0.0 -> desgaste 100
        f_skel = max(0, 100 - (features['skeleton_density'] * 1000))
        
        # Soma ponderada para a pontuação final
        # Dá mais peso à porcentagem lisa e resistência ao desgaste
        score = (f_flat * 0.4) + (f_resist * 0.4) + (f_skel * 0.2)
        score = max(0, min(100, score)) # limita entre 0 e 100
        
        return score
        
    def classify(self, score):
        if score < self.low_thresh:
            return "BAIXO"
        elif score < self.high_thresh:
            return "MEDIO"
        else:
            return "ALTO"
