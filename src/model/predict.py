# Script para fazer predicoes usando o modelo treinado
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class PreditorProjetos:
    '''Classe para fazer predicoes de sucesso de projetos'''
    
    def __init__(self):
        '''Inicializa o preditor carregando o modelo'''
        self.modelo = None
        self.scaler = None
        self.label_encoders = None
        self.feature_names = None
        self._carregar_modelo()
    
    def _carregar_modelo(self):
        '''Carrega o modelo e componentes salvos'''
        try:
            self.modelo = joblib.load('models/modelo_projetos.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            self.label_encoders = joblib.load('models/label_encoders.pkl')
            self.feature_names = joblib.load('models/feature_names.pkl')
            print("✅ Modelo carregado com sucesso!")
        except FileNotFoundError:
            print("❌ Erro: Modelo não encontrado. Execute train.py primeiro!")
            raise
    
    def preparar_entrada(self, dados_projeto):
        '''
        Prepara os dados de entrada para predicao
        
        Args:
            dados_projeto (dict): Dicionario com os dados do projeto
            
        Returns:
            pd.DataFrame: DataFrame pronto para predicao
        '''
        # Calcular features derivadas
        start_date = pd.to_datetime(dados_projeto['start_date'])
        end_date = pd.to_datetime(dados_projeto['end_date'])
        duracao_dias = (end_date - start_date).days
        
        roi = (dados_projeto['project_benefit'] - dados_projeto['project_cost']) / dados_projeto['project_cost']
        custo_por_dia = dados_projeto['project_cost'] / duracao_dias
        
        # Criar DataFrame com todas as features
        features = {
            'Project Cost': dados_projeto['project_cost'],
            'Project Benefit': dados_projeto['project_benefit'],
            'Completion': dados_projeto.get('completion', 0.5),  # Default 50% se nao informado
            'Year': dados_projeto.get('year', datetime.now().year),
            'Month': dados_projeto.get('month', datetime.now().month),
            'Duracao_Dias': duracao_dias,
            'ROI': roi,
            'Custo_Por_Dia': custo_por_dia
        }
        
        # Adicionar features categoricas codificadas
        for cat_feature in ['Project Type', 'Region', 'Department', 'Complexity', 'Phase']:
            if cat_feature in self.label_encoders:
                valor = dados_projeto.get(cat_feature.lower().replace(' ', '_'), 'Unknown')
                try:
                    # Tentar codificar o valor
                    features[cat_feature] = self.label_encoders[cat_feature].transform([valor])[0]
                except ValueError:
                    # Se valor nao conhecido, usar a moda
                    classes = self.label_encoders[cat_feature].classes_
                    features[cat_feature] = self.label_encoders[cat_feature].transform([classes[0]])[0]
                    print(f"⚠️  Valor '{valor}' não conhecido para {cat_feature}. Usando valor padrão.")
        
        # Criar DataFrame com ordem correta de features
        df = pd.DataFrame([features])[self.feature_names]
        
        return df
    
    def prever(self, dados_projeto):
        '''
        Faz a predicao de sucesso do projeto
        
        Args:
            dados_projeto (dict): Dicionario com os dados do projeto
            
        Returns:
            dict: Dicionario com predicao e probabilidades
        '''
        # Preparar dados
        X = self.preparar_entrada(dados_projeto)
        
        # Normalizar se necessario (para Logistic Regression)
        if hasattr(self.modelo, 'coef_'):  # E Logistic Regression
            X_scaled = self.scaler.transform(X)
            predicao = self.modelo.predict(X_scaled)[0]
            probabilidades = self.modelo.predict_proba(X_scaled)[0]
        else:  # Random Forest
            predicao = self.modelo.predict(X)[0]
            probabilidades = self.modelo.predict_proba(X)[0]
        
        # Calcular metricas adicionais
        roi = (dados_projeto['project_benefit'] - dados_projeto['project_cost']) / dados_projeto['project_cost']
        
        resultado = {
            'sucesso': bool(predicao),
            'probabilidade_sucesso': float(probabilidades[1]),
            'probabilidade_fracasso': float(probabilidades[0]),
            'confianca': float(max(probabilidades)),
            'roi_esperado': float(roi),
            'recomendacoes': self._gerar_recomendacoes(dados_projeto, probabilidades[1], roi)
        }
        
        return resultado
    
    def _gerar_recomendacoes(self, dados_projeto, prob_sucesso, roi):
        '''Gera recomendacoes baseadas na predicao'''
        recomendacoes = []
        
        # Recomendacoes baseadas na probabilidade
        if prob_sucesso < 0.3:
            recomendacoes.append("⚠️  Alta probabilidade de fracasso. Considere revisar o escopo do projeto.")
        elif prob_sucesso < 0.5:
            recomendacoes.append("📊 Risco moderado. Implemente medidas de mitigação de riscos.")
        elif prob_sucesso > 0.7:
            recomendacoes.append("✅ Boa probabilidade de sucesso. Mantenha o planejamento atual.")
        
        # Recomendacoes baseadas no ROI
        if roi < 0:
            recomendacoes.append("💰 ROI negativo. Revise o orçamento ou os benefícios esperados.")
        elif roi < 0.2:
            recomendacoes.append("📈 ROI baixo. Procure formas de aumentar os benefícios ou reduzir custos.")
        elif roi > 1:
            recomendacoes.append("🚀 Excelente ROI esperado!")
        
        # Recomendacoes baseadas na complexidade
        if dados_projeto.get('complexity', '').lower() == 'high':
            recomendacoes.append("🔧 Alta complexidade detectada. Considere dividir em fases menores.")
        
        # Recomendacoes baseadas na duracao
        start_date = pd.to_datetime(dados_projeto['start_date'])
        end_date = pd.to_datetime(dados_projeto['end_date'])
        duracao_dias = (end_date - start_date).days
        
        if duracao_dias > 365:
            recomendacoes.append("📅 Projeto longo (>1 ano). Estabeleça marcos intermediários.")
        elif duracao_dias < 30:
            recomendacoes.append("⏱️  Prazo muito curto. Verifique se o escopo é realista.")
        
        return recomendacoes


def exemplo_uso():
    '''Exemplo de como usar o preditor'''
    print("\\n🔮 EXEMPLO DE PREDICAO")
    print("=" * 50)
    
    # Criar preditor
    preditor = PreditorProjetos()
    
    # Exemplo de dados de projeto
    novo_projeto = {
        'project_cost': 1000000,
        'project_benefit': 1500000,
        'start_date': '2024-01-01',
        'end_date': '2024-06-30',
        'project_type': 'INCOME GENERATION',
        'region': 'North',
        'department': 'eCommerce',
        'complexity': 'High',
        'phase': 'Phase 1 - Explore',
        'completion': 0.0,  # Projeto novo
        'year': 2024,
        'month': 1
    }
    
    # Fazer predicao
    resultado = preditor.prever(novo_projeto)
    
    # Mostrar resultados
    print(f"\\n📊 RESULTADO DA PREDICAO:")
    print(f"   Sucesso previsto: {'SIM' if resultado['sucesso'] else 'NAO'}")
    print(f"   Probabilidade de sucesso: {resultado['probabilidade_sucesso']:.1%}")
    print(f"   Confiança na predicao: {resultado['confianca']:.1%}")
    print(f"   ROI esperado: {resultado['roi_esperado']:.1%}")
    
    print(f"\\n💡 RECOMENDACOES:")
    for rec in resultado['recomendacoes']:
        print(f"   {rec}")
    
    return resultado


if __name__ == "__main__":
    # Executar exemplo
    exemplo_uso()