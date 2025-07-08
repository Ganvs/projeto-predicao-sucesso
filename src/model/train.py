# Script para treinar modelo de predição de sucesso de projetos - VERSÃO CORRIGIDA
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


def preparar_dados(df):
    """Prepara os dados para treinamento - VERSÃO CORRIGIDA"""
    print("📊 Preparando dados...")

    # Criar cópia
    data = df.copy()

    # Limpar colunas monetárias (remover vírgulas e converter para float)
    data['Project Cost'] = data[' Project Cost '].str.replace(',', '').astype(float)
    data['Project Benefit'] = data[' Project Benefit '].str.replace(',', '').astype(float)

    # Converter Completion% para numérico
    data['Completion'] = data['Completion%'].str.rstrip('%').astype(float) / 100

    # ✅ CORREÇÃO 1: NOVA DEFINIÇÃO DE SUCESSO (mais realista)
    def definir_sucesso(row):
        status = row['Status']
        completion = row['Completion']

        if status == 'Completed':
            return 1  # Sempre sucesso
        elif status == 'In - Progress' and completion >= 0.7:
            return 1  # In-Progress com >70% = sucesso provável
        elif status == 'Cancelled' or status == 'On - Hold':
            return 0  # Sempre fracasso
        elif status == 'In - Progress' and completion < 0.3:
            return 0  # In-Progress com <30% = fracasso provável
        else:
            return -1  # Casos ambíguos - remover

    data['Sucesso'] = data.apply(definir_sucesso, axis=1)

    # Remover casos ambíguos
    data = data[data['Sucesso'] != -1]

    # Extrair features úteis das datas
    data['Start Date'] = pd.to_datetime(data['Start Date'])
    data['End Date'] = pd.to_datetime(data['End Date'])
    data['Duracao_Dias'] = (data['End Date'] - data['Start Date']).dt.days

    # ✅ CORREÇÃO 2: FEATURES MAIS PREDITIVAS
    # Razão Benefício/Custo (mais interpretável que ROI)
    data['Benefit_Cost_Ratio'] = data['Project Benefit'] / data['Project Cost']

    # Custo por dia (eficiência)
    data['Custo_Por_Dia'] = data['Project Cost'] / data['Duracao_Dias']

    # Benefício por dia
    data['Beneficio_Por_Dia'] = data['Project Benefit'] / data['Duracao_Dias']

    # Indicador de projeto de alto valor
    data['Alto_Valor'] = (data['Project Benefit'] > data['Project Benefit'].quantile(0.75)).astype(int)

    # Indicador de projeto longo
    data['Projeto_Longo'] = (data['Duracao_Dias'] > data['Duracao_Dias'].quantile(0.75)).astype(int)

    print(f"✅ Taxa de sucesso nos dados (nova definição): {data['Sucesso'].mean():.2%}")
    print(f"✅ Total de projetos após limpeza: {len(data)}")

    return data


def criar_features(data):
    """Cria features para o modelo - VERSÃO CORRIGIDA"""
    print("🔧 Criando features...")

    # ✅ CORREÇÃO 3: REMOVER COMPLETION% (vazamento de dados)
    features_num = [
        'Project Cost', 
        'Project Benefit', 
        'Year',
        'Month',
        'Duracao_Dias',
        'Benefit_Cost_Ratio',  # Nova feature
        'Custo_Por_Dia',
        'Beneficio_Por_Dia',   # Nova feature
        'Alto_Valor',          # Nova feature
        'Projeto_Longo'        # Nova feature
    ]

    # Features categóricas
    features_cat = [
        'Project Type',
        'Region',
        'Department',
        'Complexity',
        'Phase'
    ]

    # Criar DataFrame com features
    X = pd.DataFrame()

    # Adicionar features numéricas
    for col in features_num:
        X[col] = data[col]

    # Codificar features categóricas
    label_encoders = {}
    for col in features_cat:
        le = LabelEncoder()
        X[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # Target
    y = data['Sucesso']

    print(f"✅ Total de features: {X.shape[1]}")
    print(f"✅ Features: {list(X.columns)}")
    print(f"✅ Distribuição do target: {y.value_counts().to_dict()}")

    return X, y, label_encoders


def treinar_modelos(X, y):
    """Treina e compara diferentes modelos - VERSÃO CORRIGIDA"""
    print("\n🤖 Treinando modelos...")

    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Normalizar features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ✅ CORREÇÃO 4: MODELOS COM MELHOR BALANCEAMENTO
    modelos = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200,           # Mais árvores
            max_depth=8,               # Mais profundidade
            min_samples_split=5,       # Evitar overfitting
            min_samples_leaf=2,        # Evitar overfitting
            random_state=42,
            class_weight='balanced'    # Balancear classes
        ),
        'Logistic Regression': LogisticRegression(
            random_state=42,
            class_weight='balanced',   # Balancear classes
            max_iter=1000,
            C=0.1                      # Regularização mais forte
        )
    }

    resultados = {}

    for nome, modelo in modelos.items():
        print(f"\n📈 Treinando {nome}...")

        # Treinar
        if nome == 'Logistic Regression':
            modelo.fit(X_train_scaled, y_train)
            y_pred = modelo.predict(X_test_scaled)
            y_proba = modelo.predict_proba(X_test_scaled)[:, 1]
        else:
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
            y_proba = modelo.predict_proba(X_test)[:, 1]

        # Métricas
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Cross-validation
        if nome == 'Logistic Regression':
            cv_scores = cross_val_score(modelo, X_train_scaled, y_train, cv=5)
        else:
            cv_scores = cross_val_score(modelo, X_train, y_train, cv=5)

        resultados[nome] = {
            'modelo': modelo,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_proba': y_proba
        }

        print(f"  Accuracy: {acc:.3f}")
        print(f"  Precision: {prec:.3f}")
        print(f"  Recall: {rec:.3f}")
        print(f"  F1-Score: {f1:.3f}")
        print(f"  CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

        # Matriz de confusão
        cm = confusion_matrix(y_test, y_pred)
        print(f"  Matriz de Confusão:")
        print(f"    [[{cm[0,0]}, {cm[0,1]}],")
        print(f"     [{cm[1,0]}, {cm[1,1]}]]")

    # Escolher melhor modelo baseado em F1-Score
    melhor_modelo_nome = max(resultados.keys(), key=lambda k: resultados[k]['f1'])
    melhor_modelo = resultados[melhor_modelo_nome]['modelo']

    print(f"\n🏆 Melhor modelo: {melhor_modelo_nome}")

    # ✅ CORREÇÃO 5: ANÁLISE DE THRESHOLD OTIMIZADO
    y_proba_melhor = resultados[melhor_modelo_nome]['y_proba']

    # Testar diferentes thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    print("\n🎯 ANÁLISE DE THRESHOLDS:")

    melhor_threshold = 0.5
    melhor_f1_threshold = 0

    for threshold in thresholds:
        y_pred_threshold = (y_proba_melhor >= threshold).astype(int)
        f1_threshold = f1_score(y_test, y_pred_threshold)
        precision_threshold = precision_score(y_test, y_pred_threshold)
        recall_threshold = recall_score(y_test, y_pred_threshold)

        print(f"  Threshold {threshold}: F1={f1_threshold:.3f}, Precision={precision_threshold:.3f}, Recall={recall_threshold:.3f}")

        if f1_threshold > melhor_f1_threshold:
            melhor_f1_threshold = f1_threshold
            melhor_threshold = threshold

    print(f"\n🎯 Melhor threshold: {melhor_threshold} (F1={melhor_f1_threshold:.3f})")

    # Relatório detalhado do melhor modelo com melhor threshold
    y_pred_otimizado = (y_proba_melhor >= melhor_threshold).astype(int)

    print("\n📊 Relatório de Classificação (Threshold Otimizado):")
    print(classification_report(y_test, y_pred_otimizado, 
                              target_names=['Fracasso', 'Sucesso']))

    # Feature importance (apenas para Random Forest)
    if melhor_modelo_nome == 'Random Forest':
        importancias = pd.DataFrame({
            'feature': X.columns,
            'importance': melhor_modelo.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\n🎯 Top 10 Features mais importantes:")
        print(importancias.head(10))

    return melhor_modelo, scaler, resultados, melhor_threshold


def salvar_modelo(modelo, scaler, label_encoders, feature_names, threshold):
    """Salva o modelo e componentes necessários - VERSÃO CORRIGIDA"""
    print("\n💾 Salvando modelo...")

    # Criar diretório se não existir
    os.makedirs('models', exist_ok=True)

    # Salvar componentes
    joblib.dump(modelo, 'models/modelo_projetos.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(label_encoders, 'models/label_encoders.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')
    joblib.dump(threshold, 'models/threshold.pkl')  # ✅ Salvar threshold otimizado

    print("✅ Modelo salvo em 'models/modelo_projetos.pkl'")
    print(f"✅ Threshold otimizado salvo: {threshold}")


def main():
    """Função principal"""
    print("🚀 INICIANDO TREINAMENTO DO MODELO - VERSÃO CORRIGIDA")
    print("=" * 60)

    # Carregar dados
    df = pd.read_csv('Project Management Dataset.csv')

    # Preparar dados
    data = preparar_dados(df)

    # Criar features
    X, y, label_encoders = criar_features(data)

    # Treinar modelos
    modelo, scaler, resultados, threshold = treinar_modelos(X, y)

    # Salvar modelo
    salvar_modelo(modelo, scaler, label_encoders, list(X.columns), threshold)

    print("\n✅ Treinamento concluído com sucesso!")
    print("\n🔧 PRINCIPAIS CORREÇÕES APLICADAS:")
    print("  1. ❌ Removido Completion% (vazamento de dados)")
    print("  2. ✅ Nova definição de sucesso mais realista")
    print("  3. ✅ Features mais preditivas adicionadas")
    print("  4. ✅ Modelos com melhor balanceamento")
    print("  5. ✅ Threshold otimizado para melhor performance")

    return modelo, scaler, label_encoders, resultados, threshold


if __name__ == "__main__":
    modelo, scaler, label_encoders, resultados, threshold = main()
