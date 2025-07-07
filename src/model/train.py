# Script para treinar modelo de predição de sucesso de projetos
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


def preparar_dados(df):
    '''Prepara os dados para treinamento'''
    print("📊 Preparando dados...")
    
    # Criar cópia
    data = df.copy()
    
    # Limpar colunas monetárias (remover vírgulas e converter para float)
    data['Project Cost'] = data[' Project Cost '].str.replace(',', '').astype(float)
    data['Project Benefit'] = data[' Project Benefit '].str.replace(',', '').astype(float)
    
    # Converter Completion% para numérico
    data['Completion'] = data['Completion%'].str.rstrip('%').astype(float) / 100
    
    # Criar variável alvo binária (1 = Sucesso, 0 = Fracasso)
    # Consideramos "Completed" como sucesso
    data['Sucesso'] = (data['Status'] == 'Completed').astype(int)
    
    # Extrair features úteis das datas
    data['Start Date'] = pd.to_datetime(data['Start Date'])
    data['End Date'] = pd.to_datetime(data['End Date'])
    data['Duracao_Dias'] = (data['End Date'] - data['Start Date']).dt.days
    
    # Criar feature ROI (Return on Investment)
    data['ROI'] = (data['Project Benefit'] - data['Project Cost']) / data['Project Cost']
    
    # Criar feature de eficiência de custo
    data['Custo_Por_Dia'] = data['Project Cost'] / data['Duracao_Dias']
    
    print(f"✅ Taxa de sucesso nos dados: {data['Sucesso'].mean():.2%}")
    
    return data


def criar_features(data):
    '''Cria features para o modelo'''
    print("🔧 Criando features...")
    
    # Features numéricas
    features_num = [
        'Project Cost', 
        'Project Benefit', 
        'Completion',
        'Year',
        'Month',
        'Duracao_Dias',
        'ROI',
        'Custo_Por_Dia'
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
    
    return X, y, label_encoders


def treinar_modelos(X, y):
    '''Treina e compara diferentes modelos'''
    print("\\n🤖 Treinando modelos...")
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalizar features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Modelos para testar
    modelos = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            class_weight='balanced'
        ),
        'Logistic Regression': LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=1000
        )
    }
    
    resultados = {}
    
    for nome, modelo in modelos.items():
        print(f"\\n📈 Treinando {nome}...")
        
        # Treinar
        if nome == 'Logistic Regression':
            modelo.fit(X_train_scaled, y_train)
            y_pred = modelo.predict(X_test_scaled)
        else:
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
        
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
            'cv_std': cv_scores.std()
        }
        
        print(f"  Accuracy: {acc:.3f}")
        print(f"  Precision: {prec:.3f}")
        print(f"  Recall: {rec:.3f}")
        print(f"  F1-Score: {f1:.3f}")
        print(f"  CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    # Escolher melhor modelo baseado em F1-Score
    melhor_modelo_nome = max(resultados.keys(), key=lambda k: resultados[k]['f1'])
    melhor_modelo = resultados[melhor_modelo_nome]['modelo']
    
    print(f"\\n🏆 Melhor modelo: {melhor_modelo_nome}")
    
    # Relatório detalhado do melhor modelo
    if melhor_modelo_nome == 'Logistic Regression':
        y_pred_final = melhor_modelo.predict(X_test_scaled)
    else:
        y_pred_final = melhor_modelo.predict(X_test)
    
    print("\\n📊 Relatório de Classificação:")
    print(classification_report(y_test, y_pred_final, 
                              target_names=['Fracasso', 'Sucesso']))
    
    # Feature importance (apenas para Random Forest)
    if melhor_modelo_nome == 'Random Forest':
        importancias = pd.DataFrame({
            'feature': X.columns,
            'importance': melhor_modelo.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\\n🎯 Top 5 Features mais importantes:")
        print(importancias.head())
    
    return melhor_modelo, scaler, resultados


def salvar_modelo(modelo, scaler, label_encoders, feature_names):
    '''Salva o modelo e componentes necessários'''
    print("\\n💾 Salvando modelo...")
    
    # Criar diretório se não existir
    os.makedirs('models', exist_ok=True)
    
    # Salvar componentes
    joblib.dump(modelo, 'models/modelo_projetos.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(label_encoders, 'models/label_encoders.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')
    
    print("✅ Modelo salvo em 'models/modelo_projetos.pkl'")


def main():
    '''Função principal'''
    print("🚀 INICIANDO TREINAMENTO DO MODELO")
    print("=" * 50)
    
    # Carregar dados
    df = pd.read_csv('data/projetos.csv')
    
    # Preparar dados
    data = preparar_dados(df)
    
    # Criar features
    X, y, label_encoders = criar_features(data)
    
    # Treinar modelos
    modelo, scaler, resultados = treinar_modelos(X, y)
    
    # Salvar modelo
    salvar_modelo(modelo, scaler, label_encoders, list(X.columns))
    
    print("\\n✅ Treinamento concluído com sucesso!")
    
    return modelo, scaler, label_encoders, resultados


if __name__ == "__main__":
    modelo, scaler, label_encoders, resultados = main()