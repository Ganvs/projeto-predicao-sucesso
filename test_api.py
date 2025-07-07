"""
Script para testar a API
"""
import requests
import json
from datetime import datetime

# URL base da API
BASE_URL = "http://localhost:8000"

def test_status():
    """Testa o status da API"""
    print("🔍 Testando status da API...")
    response = requests.get(f"{BASE_URL}/")
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ API online")
        print(f"   ✅ Modelo carregado: {data['modelo_carregado']}")
        print(f"   ✅ Versão: {data['versao']}")
    else:
        print(f"   ❌ Erro: {response.status_code}")
    return response.status_code == 200


def test_prediction():
    """Testa uma predição"""
    print("\n🔮 Testando predição...")
    
    # Dados de teste
    projeto = {
        "project_cost": 2000000.0,
        "project_benefit": 3500000.0,
        "start_date": "2024-04-01",
        "end_date": "2024-10-30",
        "project_type": "INCOME GENERATION",
        "region": "North",
        "department": "eCommerce",
        "complexity": "High",
        "phase": "Phase 2 - Develop",
        "completion": 0.15
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=projeto,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        resultado = response.json()
        print(f"   ✅ Predição realizada com sucesso!")
        print(f"   📊 Sucesso previsto: {'SIM' if resultado['sucesso'] else 'NÃO'}")
        print(f"   📊 Probabilidade: {resultado['probabilidade_sucesso']:.1%}")
        print(f"   📊 ROI esperado: {resultado['roi_esperado']:.1%}")
        print(f"   💡 Recomendações:")
        for rec in resultado['recomendacoes']:
            print(f"      - {rec}")
    else:
        print(f"   ❌ Erro: {response.status_code}")
        print(f"   ❌ Detalhes: {response.text}")
    
    return response.status_code == 200


def test_endpoints():
    """Testa endpoints auxiliares"""
    print("\n📋 Testando endpoints auxiliares...")
    
    endpoints = [
        "/project-types",
        "/regions", 
        "/departments",
        "/complexities",
        "/phases"
    ]
    
    for endpoint in endpoints:
        response = requests.get(f"{BASE_URL}{endpoint}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ {endpoint}: {len(list(data.values())[0])} opções")
        else:
            print(f"   ❌ {endpoint}: Erro {response.status_code}")


def test_batch():
    """Testa predição em lote"""
    print("\n📦 Testando predição em lote...")
    
    projetos = [
        {
            "project_cost": 1000000.0,
            "project_benefit": 1500000.0,
            "start_date": "2024-01-01",
            "end_date": "2024-06-30",
            "project_type": "PROCESS IMPROVEMENT",
            "region": "South",
            "department": "Warehouse",
            "complexity": "Medium",
            "phase": "Phase 3 - Test",
            "completion": 0.5
        },
        {
            "project_cost": 3000000.0,
            "project_benefit": 5000000.0,
            "start_date": "2024-02-01",
            "end_date": "2025-01-31",
            "project_type": "INCOME GENERATION",
            "region": "East",
            "department": "Sales and Marketing",
            "complexity": "High",
            "phase": "Phase 1 - Explore",
            "completion": 0.0
        }
    ]
    
    response = requests.post(
        f"{BASE_URL}/predict-batch",
        json={"projetos": projetos},
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        resultado = response.json()
        print(f"   ✅ Lote processado!")
        print(f"   ✅ Total: {resultado['total_projetos']} projetos")
        print(f"   ✅ Sucesso: {resultado['processados_com_sucesso']} projetos")
    else:
        print(f"   ❌ Erro: {response.status_code}")


if __name__ == "__main__":
    print("🚀 TESTE DA API DE PREDIÇÃO")
    print("=" * 50)
    print("⚠️  Certifique-se de que a API está rodando!")
    print("    Execute: uv run uvicorn src.api.main:app --reload\n")
    
    try:
        # Executar testes
        if test_status():
            test_prediction()
            test_endpoints()
            test_batch()
            print("\n✅ Todos os testes passaram!")
        else:
            print("\n❌ API não está respondendo. Verifique se está rodando.")
    except requests.exceptions.ConnectionError:
        print("\n❌ Não foi possível conectar à API.")
        print("   Execute primeiro: uv run uvicorn src.api.main:app --reload")
