# Sistema de Predição de Sucesso de Projetos

Este projeto utiliza machine learning para prever a probabilidade de sucesso de projetos corporativos, com base em dados históricos e características do projeto. O sistema inclui scripts de treinamento, predição, API e interface web (Streamlit).

## 📦 Estrutura do Projeto

├── data/<br>
│ └── projetos.csv<br>
├── models/<br>
│ └── modelo_projetos.pkl<br>
├── src/<br>
│ ├── model/<br>
│ │ ├── train.py<br>
│ │ └── predict.py<br>
│ ├── api/<br>
│ │ └── main.py<br>
│ └── chatbot/<br>
│ └── app.py<br>
├── test_training.py<br>
├── pyproject.toml<br>
└── README.md<br>

## 🚀 Passo a Passo para Rodar o Projeto

### 1. **Pré-requisitos**

- Python 3.10+ instalado
- [uv](https://github.com/astral-sh/uv) instalado
- Git instalado (opcional, para clonar o repositório)

### 2. **Instale as dependências**

No terminal, execute:

```
uv pip install -r requirements.txt
```

ou, se estiver usando o `pyproject.toml`:

```
pip install -r requirements.txt
uv pip install .
```

### 3. **Prepare os dados**

- Coloque o arquivo `projetos.csv` na pasta `data/`.
- Se quiser usar outro dataset, ajuste o caminho no código.

### 4. **Treine o modelo**

Execute o script de teste e treinamento:

```
uv run python test_training.py
```

O script irá:

- Verificar as pastas e arquivos necessários
- Treinar o modelo
- Salvar o modelo treinado em `models/modelo_projetos.pkl`
- Testar exemplos de predição

### 5. **Inicie a API**

Execute:

```
uv run uvicorn src.api.main:app --reload
```

A API estará disponível em http://localhost:8000/docs (Swagger).

### 6. **Inicie a interface web (Streamlit)**

Execute:

```
uv run streamlit run src/chatbot/app.py
```

Acesse o endereço exibido no terminal para usar a interface gráfica.

---

## 🧪 Testando o Modelo

- O script de teste (`test_training.py`) já executa exemplos de predição.
- Você pode editar os exemplos de projetos no script para testar diferentes cenários.

---

## 🛠️ Principais Arquivos

- `src/model/train.py` — Treinamento do modelo
- `src/model/predict.py` — Predição de sucesso de projetos
- `src/api/main.py` — API FastAPI para integração
- `src/chatbot/app.py` — Interface web (Streamlit)
- `test_training.py` — Teste completo do pipeline
