"""
Script para testar o treinamento do modelo
Execute este arquivo para treinar e testar o modelo completo
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Adicionar src ao path
sys.path.append('src')

print("🚀 TESTE DE TREINAMENTO DO MODELO")
print("=" * 60)

# Verificar estrutura de pastas
print("\n📁 Verificando estrutura de pastas...")
pastas_necessarias = ['data', 'src/model', 'models']
for pasta in pastas_necessarias:
    if not os.path.exists(pasta):
        os.makedirs(pasta, exist_ok=True)
        print(f"   ✅ Pasta '{pasta}' criada")
    else:
        print(f"   ✅ Pasta '{pasta}' existe")

# Verificar arquivos de dados
print("\n📊 Verificando arquivos de dados...")
if os.path.exists('data/projetos.csv'):
    print("   ✅ data/projetos.csv encontrado")
else:
    print("   ❌ data/projetos.csv NÃO encontrado!")
    print("   📌 Copie o arquivo projetos.csv para a pasta data/")
    sys.exit(1)

# Importar e executar treinamento
print("\n🤖 Iniciando treinamento...")
try:
    from src.model import train
    modelo, scaler, label_encoders, resultados = train.main()
    print("\n✅ TREINAMENTO CONCLUÍDO COM SUCESSO!")
except Exception as e:
    print(f"\n❌ Erro durante o treinamento: {e}")
    sys.exit(1)

# Testar predição
print("\n🔮 Testando predição com exemplo...")
try:
    from src.model import predict
    
    # Criar um projeto de teste
    projeto_teste = {
        'project_cost': 2000000,
        'project_benefit': 3500000,
        'start_date': '2024-02-01',
        'end_date': '2024-08-30',
        'project_type': 'INCOME GENERATION',
        'region': 'North',
        'department': 'eCommerce',
        'complexity': 'Medium',
        'phase': 'Phase 2 - Develop',
        'completion': 0.25,
        'year': 2024,
        'month': 2
    }
    
    preditor = predict.PreditorProjetos()
    resultado = preditor.prever(projeto_teste)
    
    print("\n📊 RESULTADO DO TESTE:")
    print(f"   Predição: {'SUCESSO' if resultado['sucesso'] else 'FRACASSO'}")
    print(f"   Probabilidade: {resultado['probabilidade_sucesso']:.1%}")
    print(f"   ROI esperado: {resultado['roi_esperado']:.1%}")
    
    print("\n✅ SISTEMA FUNCIONANDO CORRETAMENTE!")
    
except Exception as e:
    print(f"\n❌ Erro durante teste de predição: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("🎉 Todos os testes passaram! O modelo está pronto para uso.")
print("\n📌 Próximos passos:")
print("   1. Execute 'uv run uvicorn src.api.main:app --reload' para iniciar a API")
print("   2. Execute 'uv run streamlit run src/chatbot/app.py' para o chatbot")
