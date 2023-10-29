import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Ignorar avisos do tipo DtypeWarning
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# Ler o conjunto de dados
df = pd.read_csv(
    "C:\\Users\\rhamada\\Desktop\\Dados\\BO.csv",
    dtype={"LATITUDE": "float64", "LONGITUDE": "float64"},
    low_memory=False,
)

# Selecionar as colunas relevantes para o problema
cols = ["ANO_BO", "MES", "LATITUDE", "LONGITUDE", "CIDADE", "RUBRICA"]

# Filtrar o conjunto de dados com as colunas selecionadas
df = df[cols]

# Preencher os valores ausentes com a mediana para as colunas numéricas e com a moda para as colunas categóricas
df["LATITUDE"] = df["LATITUDE"].fillna(df["LATITUDE"].median())
df["LONGITUDE"] = df["LONGITUDE"].fillna(df["LONGITUDE"].median())
df["CIDADE"] = df["CIDADE"].fillna(df["CIDADE"].mode()[0])

# Remover os valores duplicados do conjunto de dados
df = df.drop_duplicates()

# Codificar as colunas categóricas com números inteiros
df["CIDADE"] = df["CIDADE"].astype("category").cat.codes
df["RUBRICA"] = df["RUBRICA"].astype("category").cat.codes

# Separar as variáveis preditoras (X) da variável alvo (y)
X = df.drop("RUBRICA", axis=1)
y = df["RUBRICA"]

# Calcule a contagem de cada classe
class_counts = y.value_counts()
# Obtenha as classes que têm pelo menos 2 instâncias
valid_classes = class_counts[class_counts > 1].index
# Filtrar as instâncias de `X` e `y` que pertencem às classes válidas
X = X[y.isin(valid_classes)]
y = y[y.isin(valid_classes)]

# Dividir o conjunto de dados em treino, validação e teste
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

split = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
for train_index, val_index in split.split(X_train, y_train):
    X_train, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

# Adicionar escalonamento dos dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Definir os algoritmos de classificação a serem testados
classificadores = {
    "Árvore de Decisão": DecisionTreeClassifier(),
    "K Vizinhos Mais Próximos": KNeighborsClassifier(),
    "Regressão Logística": LogisticRegression(max_iter=1000),
    "Rede Neural": MLPClassifier(),
}

# Treinar os modelos de classificação
for nome, classificador in classificadores.items():
    classificador.fit(X_train_scaled, y_train)

# Avaliar o desempenho dos modelos de classificação no conjunto de validação
for nome, classificador in classificadores.items():
    y_pred_val = classificador.predict(X_val_scaled)
    acuracia_val = accuracy_score(y_val, y_pred_val)
    print(f"{nome}: Acurácia no conjunto de validação: {acuracia_val:.2f}")

# Avaliar o desempenho dos modelos de classificação no conjunto de teste
for nome, classificador in classificadores.items():
    y_pred_test = classificador.predict(X_test_scaled)
    acuracia_test = accuracy_score(y_test, y_pred_test)
    print(f"{nome}: Acurácia no conjunto de teste: {acuracia_test:.2f}")

# Calcular as curvas ROC e as áreas sob as curvas
plt.figure(figsize=(10, 6))
for nome, classificador in classificadores.items():
    probabilidade_classe = classificador.predict_proba(X_test_scaled)
    fpr, tpr, _ = roc_curve(y_test, probabilidade_classe[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{nome} (ROC-AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Taxa de Falsos Positivos")
plt.ylabel("Taxa de Verdadeiros Positivos")
plt.title("Curva ROC")
plt.legend(loc="lower right")
plt.show()
