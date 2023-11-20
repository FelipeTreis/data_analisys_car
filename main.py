import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

# Carregando o arquivo CSV como um dataframe Pandas
file_path = '/home/felipe.treis/Downloads/engine_data.csv'
df = pd.read_csv(file_path)

# Análise exploratória dos dados
print("Dados no DataFrame: \n")
print(df.head())  # Visualizando as primeiras linhas para entender a estrutura dos dados

print("\nTipos de dados das colunas: \n")
print(df.dtypes)

# Verificando valores ausentes
missing_values = df.isnull().sum()
print("\nValores ausentes por coluna: \n")
print(missing_values)

# Pré-processamento
imputer = SimpleImputer(strategy='mean')
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)  # Preenchendo os valores ausentes com a média
scaler = StandardScaler()
df_filled[numeric_cols] = scaler.fit_transform(df_filled[numeric_cols])  # Normalizando os dados numéricos

# Preparação dos dados para a tarefa de regressão
X_reg = df_filled.drop('Engine rpm', axis=1)  # Selecionando as features para regressão
y_reg = df_filled['Engine rpm']  # Selecionando a variável alvo para regressão

# Dividindo os dados em conjuntos de treino e teste para a regressão
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Execução da tarefa de regressão (Árvore de Decisão)
model_reg = DecisionTreeRegressor(random_state=42)
model_reg.fit(X_train_reg, y_train_reg)

# Avaliação da regressão
y_pred_reg = model_reg.predict(X_test_reg)
mse_reg = mean_squared_error(y_test_reg, y_pred_reg)
print("\nMétrica de Avaliação da Regressão (MSE): \n")
print(mse_reg)

# Preparação dos dados para a análise de agrupamento
X_cluster = df_filled.drop('Engine rpm', axis=1)  # Selecionando as features para o agrupamento

# Execução da análise de agrupamento (KMeans)
kmeans = KMeans(n_clusters=3, random_state=42)  # Definindo o número de clusters desejados
kmeans.fit(X_cluster)

# Adicionando as labels de cluster aos dados
df_filled['Cluster'] = kmeans.labels_

# Exibindo as informações dos clusters
print("\nInformações dos Clusters:")
print(df_filled['Cluster'].value_counts())
