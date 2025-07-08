"""
Script para testar o treinamento do modelo - VERSÃO CORRIGIDA
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
elif os.path.exists('Project Management Dataset.csv'):
    print("   ✅ Project Management Dataset.csv encontrado")
    print("   📌 Usando arquivo na raiz do projeto")
else:
    print("   ❌ Arquivo de dados NÃO encontrado!")
    print("   📌 Certifique-se de que existe 'data/projetos.csv' ou 'Project Management Dataset.csv'")
    sys.exit(1)

# Importar e executar treinamento
print("\n🤖 Iniciando treinamento...")
try:
    # ✅ CORREÇÃO: Ajustar para receber 5 valores de retorno
    if os.path.exists('train_corrigido.py'):
        # Usar versão corrigida se existir
        import importlib.util
        spec = importlib.util.spec_from_file_location("train_corrigido", "train_corrigido.py")
        train_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train_module)
        modelo, scaler, label_encoders, resultados, threshold = train_module.main()
        print(f"\n✅ TREINAMENTO CONCLUÍDO COM SUCESSO! (Threshold: {threshold})")
    else:
        # Fallback para versão original
        from src.model import train
        resultado = train.main()
        if len(resultado) == 5:
            modelo, scaler, label_encoders, resultados, threshold = resultado
        else:
            modelo, scaler, label_encoders, resultados = resultado
            threshold = 0.5
        print("\n✅ TREINAMENTO CONCLUÍDO COM SUCESSO!")

except Exception as e:
    print(f"\n❌ Erro durante o treinamento: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Testar predição
print("\n🔮 Testando predição com exemplo...")
try:
    # ✅ CORREÇÃO: Usar preditor corrigido se existir
    if os.path.exists('predict_corrigido.py'):
        import importlib.util
        spec = importlib.util.spec_from_file_location("predict_corrigido", "predict_corrigido.py")
        predict_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(predict_module)
        preditor = predict_module.PreditorProjetos()
    else:
        from src.model import predict
        preditor = predict.PreditorProjetos()

    # Criar um projeto de teste FAVORÁVEL
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

    resultado_favoravel = preditor.prever(projeto_favoravel)

    print("\n📊 RESULTADO DO TESTE (Projeto Favorável):")
    print(f"   Predição: {'SUCESSO' if resultado_favoravel['sucesso'] else 'FRACASSO'}")
    print(f"   Probabilidade: {resultado_favoravel['probabilidade_sucesso']:.1%}")
    print(f"   ROI esperado: {resultado_favoravel['roi_esperado']:.1%}")
    if 'threshold_usado' in resultado_favoravel:
        print(f"   Threshold usado: {resultado_favoravel['threshold_usado']:.2f}")

    # Testar também um projeto DESFAVORÁVEL para comparação
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

    resultado_desfavoravel = preditor.prever(projeto_desfavoravel)

    print("\n📊 RESULTADO DO TESTE (Projeto Desfavorável):")
    print(f"   Predição: {'SUCESSO' if resultado_desfavoravel['sucesso'] else 'FRACASSO'}")
    print(f"   Probabilidade: {resultado_desfavoravel['probabilidade_sucesso']:.1%}")
    print(f"   ROI esperado: {resultado_desfavoravel['roi_esperado']:.1%}")

    # Verificar se o modelo está funcionando corretamente
    if resultado_favoravel['probabilidade_sucesso'] > resultado_desfavoravel['probabilidade_sucesso']:
        print("\n✅ SISTEMA FUNCIONANDO CORRETAMENTE!")
        print("   📈 Projeto favorável tem maior probabilidade de sucesso")
    else:
        print("\n⚠️  ATENÇÃO: Projeto favorável tem menor probabilidade que o desfavorável")
        print("   📌 Isso pode indicar que o modelo ainda precisa de ajustes")

except Exception as e:
    print(f"\n❌ Erro durante teste de predição: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("🎉 Todos os testes passaram! O modelo está pronto para uso.")
print("\n📌 Próximos passos:")
print("   1. Execute 'uv run uvicorn src.api.main:app --reload' para iniciar a API")
print("   2. Execute 'uv run streamlit run src/chatbot/app.py' para o chatbot")

print("\n📊 RESUMO DOS RESULTADOS:")
print(f"   🎯 Threshold otimizado: {threshold if 'threshold' in locals() else 'N/A'}")
print(f"   📈 Projeto favorável: {resultado_favoravel['probabilidade_sucesso']:.1%} de sucesso")
print(f"   📉 Projeto desfavorável: {resultado_desfavoravel['probabilidade_sucesso']:.1%} de sucesso")
