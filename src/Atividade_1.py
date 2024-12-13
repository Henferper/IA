# %%
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import numerics

df = pd.read_csv("data/housing_sp_city.csv", encoding='ISO-8859-1')
df
# %%
print(df.info())  # Informações gerais sobre o dataset
print(df.describe())  # Estatísticas descritivas para colunas numéricas
print(df.isnull().sum())  # Contagem de valores ausentes por coluna

# Visualizar a distribuição de algumas colunas
df.hist(bins=50, figsize=(15, 10))
plt.tight_layout()
plt.show()
# %%
df['tipo_imovel'] = LabelEncoder().fit_transform(df['tipo_imovel'].fillna("Desconhecido"))
df['bairro'] = LabelEncoder().fit_transform(df['bairro'].fillna("Desconhecido"))

df.head(12)
# %%

numerics = df.select_dtypes(include='number')
numerics = numerics.drop(['cep', 'taxa_condominio', 'iptu_ano'], axis=1)
numerics = numerics.dropna()
# Calculando os quartis
Q1 = numerics.quantile(0.25)
Q3 = numerics.quantile(0.75)
IQR = Q3 - Q1

# Definindo os limites para detectar outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Removendo os outliers para cada coluna
df_sem_outliers = numerics.copy()
for col in numerics.columns:
    df_sem_outliers = df_sem_outliers[
        (df_sem_outliers[col] >= lower_bound[col]) & 
        (df_sem_outliers[col] <= upper_bound[col])
    ]

# Removendo linhas com valores ausentes
df_sem_outliers = df_sem_outliers.dropna()

# Exibindo o número de linhas restantes
print(df_sem_outliers.shape)
# %%
numerics = df.select_dtypes(include='number')
numerics.drop(['cep', 'taxa_condominio', 'iptu_ano'], axis=1)
numerics = numerics.dropna()
numerics.head()
# %%# 
# Calculando os quartis
Q1 = numerics.quantile(0.25)
Q3 = numerics.quantile(0.75)
IQR = Q3 - Q1

# Definindo os limites para detectar outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Removendo os outliers
df_sem_outliers = numerics[(numerics >= lower_bound) & (numerics <= upper_bound)]

df_sem_outliers
# %%
features = ['area_util', 'banheiros', 'suites', 'quartos', 'vagas_garagem']
X = df_sem_outliers[features]
y = df_sem_outliers['preco_aluguel']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Verificar os tamanhos dos datasets de treino e teste
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# %%
columns = ['area_util', 'banheiros', 'suites', 'quartos', 'vagas_garagem']

sns.pairplot(numerics, x_vars=columns, y_vars='preco_aluguel', diag_kind='kde')
# %%
numerics.sort_values('tipo_imovel').plot.scatter('tipo_imovel', 'preco_aluguel')

x = numerics[["tipo_imovel"]]  # features = tipo_imovel
y = numerics[["preco_aluguel"]] # labels = preco_aluguel

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
# %%
# Create a Linear Regression Model
regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)

# Train the model using the trauning sets
regr.fit(x_train, y_train)
# %%
y_predict = regr.predict(x_test)

# Visualizar resultados
plt.scatter(y_test, y_predict, alpha=0.5)
plt.xlabel("Valores Reais")
plt.ylabel("Predições")
plt.title("Predições vs Valores Reais")

plt.scatter(x_test, y_test, color='blue')
plt.plot(x_test, y_predict, color="green")
plt.show()

mse = mean_squared_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)
rmse = mse ** 0.5

print(f'MSE: {mse:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'R2 Score: {r2:.2f}')
# %%
print(f'Mean squared error: {mean_squared_error(y_test, y_predict)}')
print(f'Coefficient of determination: {r2_score(y_test, y_predict)}')