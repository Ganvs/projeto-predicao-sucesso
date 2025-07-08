# Script para fazer predicoes usando o modelo treinado - VERSÃO CORRIGIDA
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class PreditorProjetos:
    """Classe para fazer predicoes de sucesso de projetos - VERSÃO CORRIGIDA"""

    def __init__(self):
        """Inicializa o preditor carregando o modelo"""
        self.modelo = None
        self.scaler = None
        self.label_encoders = None
        self.feature_names = None
        self.threshold = 0.5  # Default, será carregado do arquivo
        self._carregar_modelo()

    def _carregar_modelo(self):
        """Carrega o modelo e componentes salvos"""
        try:
            self.modelo = joblib.load('models/modelo_projetos.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            self.label_encoders = joblib.load('models/label_encoders.pkl')
            self.feature_names = joblib.load('models/feature_names.pkl')

            # ✅ CORREÇÃO: Carregar threshold otimizado
            try:
                self.threshold = joblib.load('models/threshold.pkl')
                print(f"✅ Modelo carregado com threshold otimizado: {self.threshold}")
            except FileNotFoundError:
                print("⚠️  Threshold otimizado não encontrado. Usando 0.5 como padrão.")
                self.threshold = 0.5

        except FileNotFoundError:
            print("❌ Erro: Modelo não encontrado. Execute train.py primeiro!")
            raise

    def preparar_entrada(self, dados_projeto):
        """
        Prepara os dados de entrada para predicao - VERSÃO CORRIGIDA

        Args:
            dados_projeto (dict): Dicionario com os dados do projeto

        Returns:
            pd.DataFrame: DataFrame pronto para predicao
        """
        # Calcular features derivadas
        start_date = pd.to_datetime(dados_projeto['start_date'])
        end_date = pd.to_datetime(dados_projeto['end_date'])
        duracao_dias = (end_date - start_date).days

        # ✅ CORREÇÃO: Novas features mais preditivas
        benefit_cost_ratio = dados_projeto['project_benefit'] / dados_projeto['project_cost']
        custo_por_dia = dados_projeto['project_cost'] / duracao_dias
        beneficio_por_dia = dados_projeto['project_benefit'] / duracao_dias

        # Features indicadoras (baseadas em quantis dos dados de treino)
        # Valores aproximados baseados na análise dos dados
        alto_valor = 1 if dados_projeto['project_benefit'] > 200000 else 0
        projeto_longo = 1 if duracao_dias > 200 else 0

        # Criar DataFrame com todas as features (SEM Completion%)
        features = {
            'Project Cost': dados_projeto['project_cost'],
            'Project Benefit': dados_projeto['project_benefit'],
            'Year': dados_projeto.get('year', datetime.now().year),
            'Month': dados_projeto.get('month', datetime.now().month),
            'Duracao_Dias': duracao_dias,
            'Benefit_Cost_Ratio': benefit_cost_ratio,
            'Custo_Por_Dia': custo_por_dia,
            'Beneficio_Por_Dia': beneficio_por_dia,
            'Alto_Valor': alto_valor,
            'Projeto_Longo': projeto_longo
        }

        # Adicionar features categoricas codificadas
        mapeamento_campos = {
            'Project Type': 'project_type',
            'Region': 'region', 
            'Department': 'department',
            'Complexity': 'complexity',
            'Phase': 'phase'
        }

        for cat_feature, campo_entrada in mapeamento_campos.items():
            if cat_feature in self.label_encoders:
                valor = dados_projeto.get(campo_entrada, 'Unknown')
                try:
                    # Tentar codificar o valor
                    features[cat_feature] = self.label_encoders[cat_feature].transform([valor])[0]
                except ValueError:
                    # Se valor nao conhecido, usar a classe mais comum (primeira)
                    classes = self.label_encoders[cat_feature].classes_
                    features[cat_feature] = self.label_encoders[cat_feature].transform([classes[0]])[0]
                    print(f"⚠️  Valor '{valor}' não conhecido para {cat_feature}. Usando valor padrão: {classes[0]}")

        # Criar DataFrame com ordem correta de features
        df = pd.DataFrame([features])[self.feature_names]

        return df

    def prever(self, dados_projeto):
        """
        Faz a predicao de sucesso do projeto - VERSÃO CORRIGIDA

        Args:
            dados_projeto (dict): Dicionario com os dados do projeto

        Returns:
            dict: Dicionario com predicao e probabilidades
        """
        # Preparar dados
        X = self.preparar_entrada(dados_projeto)

        # Normalizar se necessario (para Logistic Regression)
        if hasattr(self.modelo, 'coef_'):  # E Logistic Regression
            X_scaled = self.scaler.transform(X)
            probabilidades = self.modelo.predict_proba(X_scaled)[0]
        else:  # Random Forest
            probabilidades = self.modelo.predict_proba(X)[0]

        # ✅ CORREÇÃO: Usar threshold otimizado
        predicao = (probabilidades[1] >= self.threshold).astype(int)

        # Calcular metricas adicionais
        benefit_cost_ratio = dados_projeto['project_benefit'] / dados_projeto['project_cost']
        roi = benefit_cost_ratio - 1  # ROI = (Beneficio/Custo) - 1

        resultado = {
            'sucesso': bool(predicao),
            'probabilidade_sucesso': float(probabilidades[1]),
            'probabilidade_fracasso': float(probabilidades[0]),
            'confianca': float(max(probabilidades)),
            'roi_esperado': float(roi),
            'threshold_usado': float(self.threshold),
            'recomendacoes': self._gerar_recomendacoes(dados_projeto, probabilidades[1], roi)
        }

        return resultado

    def _gerar_recomendacoes(self, dados_projeto, prob_sucesso, roi):
        """Gera recomendacoes baseadas na predicao - VERSÃO CORRIGIDA"""
        recomendacoes = []

        # ✅ CORREÇÃO: Recomendações mais precisas baseadas no threshold otimizado
        if prob_sucesso < 0.3:
            recomendacoes.append("🚨 Baixa probabilidade de sucesso. Considere revisar fundamentalmente o projeto.")
        elif prob_sucesso < self.threshold:
            recomendacoes.append("⚠️  Probabilidade de sucesso abaixo do ideal. Implemente medidas de mitigação.")
        elif prob_sucesso < 0.7:
            recomendacoes.append("📊 Probabilidade moderada de sucesso. Monitore de perto os riscos.")
        else:
            recomendacoes.append("✅ Alta probabilidade de sucesso. Mantenha o planejamento atual.")

        # Recomendacoes baseadas no ROI
        if roi < 0:
            recomendacoes.append("💰 ROI negativo. Revise urgentemente o orçamento ou benefícios.")
        elif roi < 0.5:
            recomendacoes.append("📈 ROI baixo (<50%). Procure formas de otimizar custos ou aumentar benefícios.")
        elif roi > 2:
            recomendacoes.append("🚀 Excelente ROI esperado (>200%)!")

        # Recomendacoes baseadas na razão Benefício/Custo
        benefit_cost_ratio = dados_projeto['project_benefit'] / dados_projeto['project_cost']
        if benefit_cost_ratio < 1.5:
            recomendacoes.append("⚖️  Razão benefício/custo baixa. Considere se o projeto vale o investimento.")
        elif benefit_cost_ratio > 5:
            recomendacoes.append("💎 Excelente razão benefício/custo!")

        # Recomendacoes baseadas na complexidade
        complexity = dados_projeto.get('complexity', '').lower()
        if complexity == 'high':
            recomendacoes.append("🔧 Alta complexidade. Considere dividir em fases menores e aumentar o monitoramento.")
        elif complexity == 'low':
            recomendacoes.append("✨ Baixa complexidade. Projeto com boa chance de execução suave.")

        # Recomendacoes baseadas na duracao
        start_date = pd.to_datetime(dados_projeto['start_date'])
        end_date = pd.to_datetime(dados_projeto['end_date'])
        duracao_dias = (end_date - start_date).days

        if duracao_dias > 365:
            recomendacoes.append("📅 Projeto longo (>1 ano). Estabeleça marcos trimestrais e revisões regulares.")
        elif duracao_dias < 30:
            recomendacoes.append("⏱️  Prazo muito curto (<30 dias). Verifique se o escopo é realista.")

        # Recomendacoes baseadas no tipo de projeto
        project_type = dados_projeto.get('project_type', '')
        if project_type == 'INCOME GENERATION':
            recomendacoes.append("💰 Projeto de geração de receita. Monitore métricas de ROI de perto.")
        elif project_type == 'PROCESS IMPROVEMENT':
            recomendacoes.append("⚙️  Projeto de melhoria de processo. Foque em métricas de eficiência.")

        return recomendacoes


def exemplo_uso():
    """Exemplo de como usar o preditor - VERSÃO CORRIGIDA"""
    print("\n🔮 EXEMPLO DE PREDICAO - VERSÃO CORRIGIDA")
    print("=" * 60)

    # Criar preditor
    preditor = PreditorProjetos()

    # Exemplo de dados de projeto FAVORÁVEL
    projeto_favoravel = {
        'project_cost': 50000,
        'project_benefit': 500000,  # ROI de 900%!
        'start_date': '2024-01-01',
        'end_date': '2024-06-30',
        'project_type': 'INCOME GENERATION',
        'region': 'North',
        'department': 'eCommerce',
        'complexity': 'Low',        # Baixa complexidade
        'phase': 'Phase 4 - Implement',
        'year': 2024,
        'month': 1
    }

    # Fazer predicao
    resultado = preditor.prever(projeto_favoravel)

    # Mostrar resultados
    print(f"\n📊 RESULTADO DA PREDICAO (Projeto Favorável):")
    print(f"   Sucesso previsto: {'SIM' if resultado['sucesso'] else 'NÃO'}")
    print(f"   Probabilidade de sucesso: {resultado['probabilidade_sucesso']:.1%}")
    print(f"   Confiança na predicao: {resultado['confianca']:.1%}")
    print(f"   ROI esperado: {resultado['roi_esperado']:.1%}")
    print(f"   Threshold usado: {resultado['threshold_usado']:.2f}")

    print(f"\n💡 RECOMENDACOES:")
    for rec in resultado['recomendacoes']:
        print(f"   {rec}")

    # Exemplo de projeto DESFAVORÁVEL para comparação
    print("\n" + "="*60)

    projeto_desfavoravel = {
        'project_cost': 500000,
        'project_benefit': 400000,  # ROI negativo!
        'start_date': '2024-01-01',
        'end_date': '2025-12-31',   # 2 anos
        'project_type': 'PROCESS IMPROVEMENT',
        'region': 'South',
        'department': 'Admin & BI',
        'complexity': 'High',       # Alta complexidade
        'phase': 'Phase 1 - Explore',
        'year': 2024,
        'month': 1
    }

    resultado2 = preditor.prever(projeto_desfavoravel)

    print(f"\n📊 RESULTADO DA PREDICAO (Projeto Desfavorável):")
    print(f"   Sucesso previsto: {'SIM' if resultado2['sucesso'] else 'NÃO'}")
    print(f"   Probabilidade de sucesso: {resultado2['probabilidade_sucesso']:.1%}")
    print(f"   Confiança na predicao: {resultado2['confianca']:.1%}")
    print(f"   ROI esperado: {resultado2['roi_esperado']:.1%}")

    print(f"\n💡 RECOMENDACOES:")
    for rec in resultado2['recomendacoes']:
        print(f"   {rec}")

    return resultado, resultado2


if __name__ == "__main__":
    # Executar exemplo
    exemplo_uso()
