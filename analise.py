import folium
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from folium import plugins
from folium.plugins import HeatMap

# Ler o conjunto de dados com tipos de dados especificados e desabilitando low_memory
dtype = {"LATITUDE": "float64", "LONGITUDE": "float64", "CIDADE": "str"}
df = pd.read_csv(
    "C:\\Users\\rhamada\\Desktop\\Dados\\BO.csv", dtype=dtype, low_memory=False
)

# Remover linhas com valores ausentes nas coordenadas geográficas
df = df.dropna(subset=["LATITUDE", "LONGITUDE"])

# Etapa 1: Distribuição Geográfica dos Crimes (Mapa de Calor)
crime_map = folium.Map(
    location=[df["LATITUDE"].mean(), df["LONGITUDE"].mean()], zoom_start=10
)
heat_data = [[row["LATITUDE"], row["LONGITUDE"]] for index, row in df.iterrows()]
plugins.HeatMap(heat_data).add_to(crime_map)
crime_map.save("Mapa de Calor.html")

# Gráfico 1: Tipos de Crimes Mais Comuns (Gráfico de Barras)
plt.figure(figsize=(12, 6))
sns.countplot(y="RUBRICA", data=df, order=df["RUBRICA"].value_counts().index[:10])
plt.title("Top 10 Tipos de Crimes Mais Comuns")
plt.xlabel("Número de Ocorrências")
plt.ylabel("Tipo de Crime")
# plt.savefig("crime_types.png")
plt.show()

# Gráfico 2: Status das Ocorrências (Gráfico de Barras)
status_counts = df["FLAG_STATUS"].value_counts()
status_labels = status_counts.index
status_labels = [
    "Concluído" if label == "C" else "Em Trâmite" if label == "T" else label
    for label in status_labels
]
status_values = status_counts.values

# Gráfico de barras para mostrar a distribuição de status
plt.figure(figsize=(10, 6))
plt.bar(status_labels, status_values)
plt.xlabel("Status da Ocorrência")
plt.ylabel("Número de Ocorrências")
plt.title("Distribuição de Status da Ocorrência")
plt.show()

# Gráfico 3: Mostra a contagem de ocorrências para cada uma Delegacia.
# Contagem de ocorrências por delegacia
ocorrencias_por_delegacia = df["DELEGACIA"].value_counts()

# Configurando o tamanho do gráfico
plt.figure(figsize=(12, 6))

# Gráfico de barras
ocorrencias_por_delegacia[:10].plot(kind="bar", color="skyblue")
plt.title("Top 10 Delegacias com Mais Ocorrências de Crimes")
plt.xlabel("Delegacia")
plt.ylabel("Número de Ocorrências")
plt.xticks(rotation=45, ha="right")

# Exibir o gráfico
plt.tight_layout()
plt.show()

# Gráfico 4: Distribuição de Crimes por Cidade
# Filtro das 10 cidades com os maiores índices de ocorrências
top_10_cidades = df["CIDADE"].value_counts().nlargest(10).index
df_top_10 = df[df["CIDADE"].isin(top_10_cidades)]

# Gráfico de barras
plt.figure(figsize=(12, 6))
sns.countplot(data=df_top_10, y="CIDADE", order=top_10_cidades)
plt.title("Top 10 Cidades com Mais Ocorrências de Crimes")
plt.xlabel("Número de Ocorrências")
plt.ylabel("Cidade")
plt.show()

# Gráfico 5: Análise da distribuição dos diferentes tipos de rubricas (crimes) no conjunto de dados.
# Pré-processamento dos rótulos
rubricas_counts = (
    df["RUBRICA"].str.split("(", n=1).str[0]
)  # Remove "(art.)" e tudo depois do primeiro parêntese
rubricas_counts = rubricas_counts.str.replace(
    "A\.I\.-", "", regex=True
)  # Remove "A.I.-"
rubricas_counts = rubricas_counts.str.replace(
    "\(|\)", "", regex=True
)  # Remove parênteses

# Contagem dos tipos de rubricas (crimes)
rubricas_counts = rubricas_counts.value_counts()

# Plotagem do gráfico de barras com tamanho maior
plt.figure(figsize=(16, 8))
rubricas_counts.plot(kind="bar", color="purple")
plt.title("Distribuição dos Tipos de Rubricas (Crimes)")
plt.xlabel("Tipo de Rubrica")
plt.ylabel("Número de Ocorrências")
plt.xticks(rotation=45, ha="right")  # Rotação e alinhamento dos rótulos no eixo x
plt.tight_layout()  # Ajuste automático de layout para evitar cortes
plt.grid(axis="y")

plt.show()

# Gráfico 6: Balanceamento de Classes
# Filtro dos nomes das rubricas, removendo parênteses e "A.I.-"
df["RUBRICA"] = df["RUBRICA"].str.replace(r"\(.*\)", "", regex=True)
df["RUBRICA"] = df["RUBRICA"].str.replace("A.I.-", "", regex=True)

# Gráfico de barras
plt.figure(figsize=(10, 6))
sns.countplot(x="RUBRICA", data=df, order=df["RUBRICA"].value_counts().index)
plt.title("Balanceamento de Classes (Tipos de Crimes)")
plt.xlabel("Tipo de Crime")
plt.ylabel("Contagem")
plt.xticks(rotation=45, ha="right")  # Rotaciona as legendas para melhor visualização
plt.show()

# Gráfico 7: Distribuição de Crimes por Cidade
plt.figure(figsize=(12, 6))
sns.countplot(
    y="RUBRICA", data=df, hue="CIDADE", order=df["RUBRICA"].value_counts().index[:10]
)
plt.title("Distribuição de Crimes por Cidade")
plt.xlabel("Número de Ocorrências")
plt.ylabel("Tipo de Crime")
# plt.savefig("crime_city_distribution.png")
plt.show()

# Gráfico 8: Evolução Temporal de Tipos de Crimes
# Agrupamento dos dados por ano e tipo de crime e conte o número de ocorrências
crime_counts = df.groupby(["ANO_BO", "RUBRICA"]).size().unstack()

# Filtrar tipos de crimes únicos para evitar repetições na legenda
unique_crime_types = df["RUBRICA"].unique()

# Esquema de cores personalizado para os tipos de crimes
colors = sns.color_palette("hsv", len(unique_crime_types))

# Gráficos de linha para cada tipo de crime com cores diferentes
plt.figure(figsize=(12, 6))
for i, crime_type in enumerate(unique_crime_types):
    if crime_type in crime_counts.columns:
        plt.plot(
            crime_counts.index,
            crime_counts[crime_type],
            label=crime_type,
            color=colors[i],
        )

plt.title("Evolução Temporal de Tipos de Crimes")
plt.xlabel("Ano")
plt.ylabel("Número de Ocorrências")
plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1))
plt.grid(True)

plt.show()
