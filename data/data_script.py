import pandas as pd

# Carregar dados originais
df_original = pd.read_csv('data/Project Management Dataset.csv')

# Salvar como projetos.csv (simulando que está na pasta data/)
df_original.to_csv('data/projetos.csv', index=False)
print("✅ Arquivo data/projetos.csv criado!")

# Criar base de usuários exemplo
usuarios_data = {
    'Usuario_ID': [1, 2, 3, 4, 5],
    'Nome': ['João Silva', 'Maria Santos', 'Pedro Costa', 'Ana Lima', 'Carlos Souza'],
    'Cargo': [
        'Gerente de Projetos', 
        'Analista de Projetos',
        'Coordenador de TI',
        'Gerente de Operações',
        'Líder Técnico'
    ],
    'Departamento': ['Admin & BI', 'eCommerce', 'Warehouse', 'Sales and Marketing', 'Admin & BI'],
    'Experiencia_Anos': [5, 3, 8, 10, 4],
    'Projetos_Anteriores': [15, 10, 25, 30, 12],
    'Taxa_Sucesso': [0.80, 0.65, 0.90, 0.85, 0.70],
    'Especialidade': [
        'INCOME GENERATION',
        'PROCESS IMPROVEMENT', 
        'WORKING CAPITAL IMPROVEMENT',
        'INCOME GENERATION',
        'PROCESS IMPROVEMENT'
    ]
}

df_usuarios = pd.DataFrame(usuarios_data)
df_usuarios.to_csv('data/usuarios.csv', index=False)
print("✅ Arquivo data/usuarios.csv criado!")

# Mostrar preview dos dados
print("\n📊 Preview dos usuários:")
print(df_usuarios)