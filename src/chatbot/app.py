"""
Chatbot para interação com o sistema de predição de projetos
"""
import streamlit as st
import pandas as pd
import requests
from datetime import datetime, date, timedelta
import json
import os

# Configuração da página
st.set_page_config(
    page_title="Assistente de Projetos",
    page_icon="🤖",
    layout="wide"
)

# URL da API
API_URL = "http://localhost:8000"

# Carregar base de usuários
@st.cache_data
def carregar_usuarios():
    """Carrega a base de dados de usuários"""
    try:
        df = pd.read_csv('data/usuarios.csv')
        return df
    except:
        st.error("❌ Erro ao carregar base de usuários")
        return pd.DataFrame()

# Verificar status da API
def verificar_api():
    """Verifica se a API está online"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

# Obter opções válidas da API
@st.cache_data(ttl=3600)
def obter_opcoes():
    """Obtém as opções válidas dos endpoints da API"""
    opcoes = {}
    endpoints = {
        'project_types': 'project-types',
        'regions': 'regions',
        'departments': 'departments',
        'complexities': 'complexities',
        'phases': 'phases'
    }

    for key, endpoint in endpoints.items():
        try:
            response = requests.get(f"{API_URL}/{endpoint}")
            if response.status_code == 200:
                data = response.json()
                opcoes[key] = list(data.values())[0]
            else:
                opcoes[key] = []
        except:
            opcoes[key] = []

    return opcoes

# Inicializar estado da sessão
if 'mensagens' not in st.session_state:
    st.session_state.mensagens = []
    st.session_state.usuario_selecionado = None
    st.session_state.projeto_analisado = {}  # Mudança: renomeado para evitar conflito
    st.session_state.etapa = 'inicio'

# Título e descrição
st.title("🤖 Assistente de Análise de Projetos")
st.markdown("Olá! Sou seu assistente para prever o sucesso de projetos. Vamos começar?")

# Sidebar com informações
with st.sidebar:
    st.header("📊 Informações")

    # Status da API
    api_online = verificar_api()
    if api_online:
        st.success("✅ Sistema Online")
    else:
        st.error("❌ API Offline")
        st.info("Execute: `uv run uvicorn src.api.main:app --reload`")

    # Usuário selecionado
    st.divider()
    st.subheader("👤 Usuário")

    usuarios_df = carregar_usuarios()
    if not usuarios_df.empty:
        usuario_nome = st.selectbox(
            "Selecione seu perfil:",
            options=usuarios_df['Nome'].tolist(),
            index=0 if st.session_state.usuario_selecionado is None else None
        )

        filtro = usuarios_df[usuarios_df['Nome'] == usuario_nome]
        if not filtro.empty:
            usuario_info = filtro.iloc[0]
            st.session_state.usuario_selecionado = usuario_info.to_dict()

            st.info(f"**Cargo:** {usuario_info['Cargo']}")
            st.info(f"**Experiência:** {usuario_info['Experiencia_Anos']} anos")
            st.info(f"**Taxa de Sucesso:** {usuario_info['Taxa_Sucesso']:.0%}")
        else:
            st.warning(f"Usuário '{usuario_nome}' não encontrado.")

    # Botão para limpar conversa
    st.divider()
    if st.button("🔄 Nova Análise", use_container_width=True):
        st.session_state.mensagens = []
        st.session_state.projeto_analisado = {}
        st.session_state.etapa = 'inicio'
        st.rerun()

# Função para adicionar mensagem
def adicionar_mensagem(role, content):
    st.session_state.mensagens.append({"role": role, "content": content})

# Função para coletar dados do projeto
def coletar_dados_projeto():
    """Interface para coletar dados do projeto"""
    opcoes = obter_opcoes()

    with st.form("form_dados_projeto"):  # Mudança: chave única para o form
        st.subheader("📋 Dados do Projeto")

        col1, col2 = st.columns(2)

        with col1:
            nome_projeto = st.text_input("Nome do Projeto")
            custo = st.number_input(
                "Custo do Projeto (R$)", 
                min_value=1000.0,
                value=10000.0,
                step=10000.0
            )
            beneficio = st.number_input(
                "Benefício Esperado (R$)", 
                min_value=1000.0,
                value=150000.0,
                step=10000.0
            )

            tipo = st.selectbox(
                "Tipo do Projeto",
                options=opcoes.get('project_types', [])
            )

            regiao = st.selectbox(
                "Região",
                options=opcoes.get('regions', [])
            )

        with col2:
            departamento = st.selectbox(
                "Departamento",
                options=opcoes.get('departments', [])
            )

            complexidade = st.selectbox(
                "Complexidade",
                options=opcoes.get('complexities', [])
            )

            fase = st.selectbox(
                "Fase Atual",
                options=opcoes.get('phases', [])
            )

            col_data1, col_data2 = st.columns(2)
            with col_data1:
                data_inicio = st.date_input(
                    "Data de Início",
                    value=date.today()
                )
            with col_data2:
                data_fim = st.date_input(
                    "Data de Término",
                    value=date.today() + timedelta(days=180)
                )

            conclusao = st.slider(
                "% de Conclusão",
                min_value=0,
                max_value=100,
                value=0
            )

        submitted = st.form_submit_button("Analisar Projeto", use_container_width=True, type="primary")

        if submitted:
            # Validar datas
            if data_fim <= data_inicio:
                st.error("❌ A data de término deve ser posterior à data de início!")
                return False

            # Salvar dados no session_state com chave diferente
            st.session_state.projeto_analisado = {
                "nome": nome_projeto,
                "project_cost": custo,
                "project_benefit": beneficio,
                "start_date": data_inicio.strftime("%Y-%m-%d"),
                "end_date": data_fim.strftime("%Y-%m-%d"),
                "project_type": tipo,
                "region": regiao,
                "department": departamento,
                "complexity": complexidade,
                "phase": fase,
                "completion": conclusao / 100.0
            }

            return True

    return False

# Função para fazer predição
def fazer_predicao(dados_projeto):
    """Chama a API para fazer a predição"""
    try:
        # Preparar dados para API (remover campo 'nome')
        dados_api = {k: v for k, v in dados_projeto.items() if k != 'nome'}

        response = requests.post(
            f"{API_URL}/predict",
            json=dados_api,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"❌ Erro na API: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"❌ Erro ao conectar com a API: {str(e)}")
        return None

# Função para gerar análise personalizada
def gerar_analise_personalizada(resultado, dados_projeto, usuario):
    """Gera uma análise personalizada combinando resultado e perfil do usuário"""

    analise = f"""
## 📊 Análise do Projeto: {dados_projeto.get('nome', 'Novo Projeto')}

Com base na minha análise e considerando seu perfil como **{usuario['Cargo']}** 
com **{usuario['Experiencia_Anos']} anos de experiência** e uma taxa de sucesso 
histórica de **{usuario['Taxa_Sucesso']:.0%}**, aqui está minha avaliação:

### 🎯 Predição
- **Probabilidade de Sucesso:** {resultado['probabilidade_sucesso']:.1%}
- **Nível de Confiança:** {resultado['confianca']:.1%}
- **ROI Esperado:** {resultado['roi_esperado']:.1%}

### 💡 Recomendações Personalizadas
"""

    # Adicionar recomendações do modelo
    for rec in resultado['recomendacoes']:
        analise += f"\n{rec}\n"

    # Adicionar recomendações baseadas no perfil do usuário
    if usuario['Taxa_Sucesso'] > resultado['probabilidade_sucesso']:
        analise += f"""
\n📌 **Nota especial para você:** Sua taxa de sucesso histórica 
({usuario['Taxa_Sucesso']:.0%}) é superior à probabilidade prevista para este projeto. 
Isso pode indicar que sua experiência pode fazer a diferença!
"""

    # Recomendações por departamento
    if usuario['Departamento'] == dados_projeto['department']:
        analise += f"""
\n✅ **Alinhamento departamental:** O projeto está alinhado com sua área 
({usuario['Departamento']}), o que pode aumentar as chances de sucesso.
"""

    # Recomendações por especialidade
    if usuario['Especialidade'] == dados_projeto['project_type']:
        analise += f"""
\n🎯 **Especialidade matching:** Este projeto está na sua área de especialidade 
({usuario['Especialidade']}), aproveitando sua expertise!
"""

    return analise

# Interface principal do chat
container_chat = st.container()

with container_chat:
    # Mostrar mensagens anteriores
    for msg in st.session_state.mensagens:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Lógica do chatbot
    if st.session_state.etapa == 'inicio' and len(st.session_state.mensagens) == 0:
        adicionar_mensagem(
            "assistant",
            f"""👋 Olá, {st.session_state.usuario_selecionado['Nome'] if st.session_state.usuario_selecionado else 'usuário'}!

Sou seu assistente de análise de projetos. Posso ajudá-lo a prever o sucesso de seus projetos 
usando inteligência artificial.

Para começar, clique no botão abaixo para inserir os dados do seu projeto."""
        )
        st.rerun()

    # Área de entrada de dados
    if st.session_state.etapa == 'inicio':
        if coletar_dados_projeto():
            st.session_state.etapa = 'analisando'
            adicionar_mensagem("user", f"Analisar projeto: {st.session_state.projeto_analisado.get('nome', 'Novo Projeto')}")
            st.rerun()

    # Fazer análise
    elif st.session_state.etapa == 'analisando':
        with st.spinner("🔍 Analisando projeto..."):
            resultado = fazer_predicao(st.session_state.projeto_analisado)

        if resultado:
            analise = gerar_analise_personalizada(
                resultado,
                st.session_state.projeto_analisado,
                st.session_state.usuario_selecionado
            )

            adicionar_mensagem("assistant", analise)
            st.session_state.etapa = 'concluido'
            st.rerun()
        else:
            adicionar_mensagem("assistant", "❌ Não consegui analisar o projeto. Verifique se a API está online.")
            st.session_state.etapa = 'inicio'
            st.rerun()

    # Opções após análise
    elif st.session_state.etapa == 'concluido':
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📊 Analisar Outro Projeto", use_container_width=True):
                adicionar_mensagem("user", "Quero analisar outro projeto")
                adicionar_mensagem("assistant", "Claro! Vamos analisar outro projeto. Por favor, preencha os dados:")
                st.session_state.projeto_analisado = {}
                st.session_state.etapa = 'inicio'
                st.rerun()

        with col2:
            if st.button("📈 Ver Comparação", use_container_width=True):
                adicionar_mensagem("assistant", """
🚧 **Funcionalidade em desenvolvimento!** 

Em breve você poderá comparar múltiplos projetos e ver análises comparativas.
""")
                st.rerun()

        with col3:
            if st.button("💾 Exportar Análise", use_container_width=True):
                adicionar_mensagem("assistant", """
📄 **Exportação disponível em breve!**

Você poderá exportar a análise em PDF ou Excel.
""")
                st.rerun()

# Footer
st.divider()
st.caption("🤖 Assistente de Projetos v1.0 | Powered by Machine Learning")
