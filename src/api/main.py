"""
API para servir o modelo de predição de projetos
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List
import sys
import os

# Adicionar o diretório src ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.predict import PreditorProjetos

# Criar instância da aplicação
app = FastAPI(
    title="API de Predição de Sucesso de Projetos",
    description="API para prever o sucesso de projetos usando Machine Learning",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carregar o modelo na inicialização
try:
    preditor = PreditorProjetos()
    print("✅ Modelo carregado com sucesso!")
except Exception as e:
    print(f"❌ Erro ao carregar modelo: {e}")
    preditor = None


# Modelos Pydantic para validação
class ProjetoDados(BaseModel):
    """Modelo de dados de entrada para um projeto"""
    project_cost: float = Field(..., description="Custo do projeto em R$", gt=0)
    project_benefit: float = Field(..., description="Benefício esperado em R$", gt=0)
    start_date: str = Field(..., description="Data de início (YYYY-MM-DD)")
    end_date: str = Field(..., description="Data de término (YYYY-MM-DD)")
    project_type: str = Field(..., description="Tipo do projeto")
    region: str = Field(..., description="Região do projeto")
    department: str = Field(..., description="Departamento responsável")
    complexity: str = Field(..., description="Complexidade (Low/Medium/High)")
    phase: str = Field(..., description="Fase atual do projeto")
    completion: Optional[float] = Field(0.0, description="Percentual de conclusão (0-1)", ge=0, le=1)
    year: Optional[int] = Field(None, description="Ano do projeto")
    month: Optional[int] = Field(None, description="Mês do projeto", ge=1, le=12)
    
    class Config:
        json_schema_extra = {
            "example": {
                "project_cost": 1500000.0,
                "project_benefit": 2500000.0,
                "start_date": "2024-03-01",
                "end_date": "2024-09-30",
                "project_type": "INCOME GENERATION",
                "region": "North",
                "department": "eCommerce",
                "complexity": "High",
                "phase": "Phase 1 - Explore",
                "completion": 0.0
            }
        }


class ResultadoPredicao(BaseModel):
    """Modelo de resposta da predição"""
    sucesso: bool
    probabilidade_sucesso: float
    probabilidade_fracasso: float
    confianca: float
    roi_esperado: float
    recomendacoes: List[str]
    timestamp: str


class StatusResposta(BaseModel):
    """Model de resposta de status"""
    status: str
    modelo_carregado: bool
    versao: str
    timestamp: str


# Endpoints
@app.get("/", response_model=StatusResposta)
async def root():
    """Endpoint raiz - verifica o status da API"""
    return StatusResposta(
        status="online",
        modelo_carregado=preditor is not None,
        versao="1.0.0",
        timestamp=datetime.now().isoformat()
    )


@app.get("/health", response_model=StatusResposta)
async def health_check():
    """Verifica a saúde da API e do modelo"""
    if preditor is None:
        raise HTTPException(status_code=503, detail="Modelo não está carregado")
    
    return StatusResposta(
        status="healthy",
        modelo_carregado=True,
        versao="1.0.0",
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=ResultadoPredicao)
async def predict_project(projeto: ProjetoDados):
    """
    Faz a predição de sucesso para um projeto
    
    Args:
        projeto: Dados do projeto para análise
        
    Returns:
        ResultadoPredicao: Predição com probabilidades e recomendações
    """
    if preditor is None:
        raise HTTPException(status_code=503, detail="Modelo não está disponível")
    
    try:
        # Validar datas
        start = datetime.strptime(projeto.start_date, "%Y-%m-%d")
        end = datetime.strptime(projeto.end_date, "%Y-%m-%d")
        
        if end <= start:
            raise HTTPException(
                status_code=400, 
                detail="Data de término deve ser posterior à data de início"
            )
        
        # Preparar dados para o modelo
        dados_modelo = projeto.model_dump()
        
        # Adicionar ano e mês se não fornecidos
        if dados_modelo['year'] is None:
            dados_modelo['year'] = start.year
        if dados_modelo['month'] is None:
            dados_modelo['month'] = start.month
        
        # Fazer predição
        resultado = preditor.prever(dados_modelo)
        
        # Retornar resultado formatado
        return ResultadoPredicao(
            **resultado,
            timestamp=datetime.now().isoformat()
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Erro nos dados: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")


@app.get("/project-types")
async def get_project_types():
    """Retorna os tipos de projeto válidos"""
    return {
        "project_types": [
            "INCOME GENERATION",
            "PROCESS IMPROVEMENT",
            "WORKING CAPITAL IMPROVEMENT"
        ]
    }


@app.get("/regions")
async def get_regions():
    """Retorna as regiões válidas"""
    return {
        "regions": ["North", "South", "East", "West"]
    }


@app.get("/departments")
async def get_departments():
    """Retorna os departamentos válidos"""
    return {
        "departments": [
            "Admin & BI",
            "eCommerce",
            "Warehouse",
            "Sales and Marketing"
        ]
    }


@app.get("/complexities")
async def get_complexities():
    """Retorna os níveis de complexidade válidos"""
    return {
        "complexities": ["Low", "Medium", "High"]
    }


@app.get("/phases")
async def get_phases():
    """Retorna as fases de projeto válidas"""
    return {
        "phases": [
            "Phase 1 - Explore",
            "Phase 2 - Develop", 
            "Phase 3 - Test",
            "Phase 4 - Implement",
            "Phase 5 - Measure"
        ]
    }


# Exemplo de uso da API em lote
class LoteProjetosRequest(BaseModel):
    """Modelo para requisição em lote"""
    projetos: List[ProjetoDados]


@app.post("/predict-batch")
async def predict_batch(lote: LoteProjetosRequest):
    """Faz predições para múltiplos projetos"""
    if preditor is None:
        raise HTTPException(status_code=503, detail="Modelo não está disponível")
    
    resultados = []
    for i, projeto in enumerate(lote.projetos):
        try:
            dados_modelo = projeto.model_dump()
            resultado = preditor.prever(dados_modelo)
            resultado['projeto_id'] = i
            resultados.append(resultado)
        except Exception as e:
            resultados.append({
                'projeto_id': i,
                'erro': str(e)
            })
    
    return {
        'total_projetos': len(lote.projetos),
        'processados_com_sucesso': len([r for r in resultados if 'erro' not in r]),
        'resultados': resultados
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
