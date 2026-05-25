import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração do estilo dos gráficos
sns.set_style("whitegrid")  # Fundo com grid suave
plt.rcParams["figure.figsize"] = (10, 6)  # Tamanho padrão dos gráficos
sns.set_palette("pastel")  # Paleta de cores suaves

# Carregar os dados
df = pd.read_csv("data.csv", sep=";")
df.columns = df.columns.str.strip()  # Remover espaços extras nos nomes das colunas

# Filtrar apenas os alunos que tiveram "Dropout"
dropout_df = df[df["Target"] == "Dropout"]

# 1. Distribuição de Idade dos alunos que tiveram Dropout
plt.figure(figsize=(10, 6))
sns.histplot(dropout_df["Age at enrollment"], bins=15, kde=True, color="skyblue")
plt.title("Distribuição de Idade dos Alunos que Evadiram", fontsize=16)
plt.xlabel("Idade no Momento da Matrícula", fontsize=12)
plt.ylabel("Frequência", fontsize=12)
plt.show()

# 2. Comparação de Nota de Admissão entre Dropout e Graduate
plt.figure(figsize=(10, 6))
sns.boxplot(x="Target", y="Admission grade", data=df, width=0.5)
plt.title("Comparação da Nota de Admissão entre Evadidos e Formados", fontsize=16)
plt.xlabel("Grupo", fontsize=12)
plt.ylabel("Nota de Admissão", fontsize=12)
plt.show()

# 3. Distribuição de Gênero por Grupo (Dropout vs Graduate)
plt.figure(figsize=(10, 6))
sns.countplot(x="Gender", hue="Target", data=df)
plt.title("Distribuição de Gênero por Grupo", fontsize=16)
plt.xlabel("Gênero", fontsize=12)
plt.ylabel("Contagem", fontsize=12)
plt.legend(title="Grupo")
plt.show()

# 4. Situação Financeira (Devedores) por Grupo
plt.figure(figsize=(10, 6))
sns.countplot(x="Debtor", hue="Target", data=df)
plt.title("Situação Financeira (Devedores) por Grupo", fontsize=16)
plt.xlabel("Devedor (1 = Sim, 0 = Não)", fontsize=12)
plt.ylabel("Contagem", fontsize=12)
plt.legend(title="Grupo")
plt.show()

# 5. Bolsistas por Grupo
plt.figure(figsize=(10, 6))
sns.countplot(x="Scholarship holder", hue="Target", data=df)
plt.title("Distribuição de Bolsistas por Grupo", fontsize=16)
plt.xlabel("Bolsista (1 = Sim, 0 = Não)", fontsize=12)
plt.ylabel("Contagem", fontsize=12)
plt.legend(title="Grupo")
plt.show()

# 6. Necessidades Educacionais Especiais por Grupo
plt.figure(figsize=(10, 6))
sns.countplot(x="Educational special needs", hue="Target", data=df)
plt.title("Necessidades Educacionais Especiais por Grupo", fontsize=16)
plt.xlabel("Necessidades Especiais (1 = Sim, 0 = Não)", fontsize=12)
plt.ylabel("Contagem", fontsize=12)
plt.legend(title="Grupo")
plt.show()

# 7. Notas do Primeiro Semestre por Grupo
plt.figure(figsize=(10, 6))
sns.boxplot(x="Target", y="Curricular units 1st sem (grade)", data=df, width=0.5)
plt.title("Notas do Primeiro Semestre por Grupo", fontsize=16)
plt.xlabel("Grupo", fontsize=12)
plt.ylabel("Nota do Primeiro Semestre", fontsize=12)
plt.show()