import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Carrega os dados dos clientes que receberam empréstimos
retail_data = pd.read_csv('Retail_Data.csv')

#Carrega os dados dos potenciais clientes
potential_customers = pd.read_csv('Potential_Customers.csv')

#Visualiza os dados
print(retail_data.head())
print(potential_customers.head())  

#Resumo estatístico dos dados
print(retail_data.describe())
print(potential_customers.describe())

#Valida dados ausentes
print(retail_data.isnull().sum())
print(potential_customers.isnull().sum())

#Análise de padrões com distribuição de renda mensal
plt.figure(figsize=(10,6))
sns.histplot(retail_data['renda_mensal'], kde=True, color='blue', label='Clientes Atuais')
sns.histplot(potential_customers['renda_mensal'], kde=True, color='orange', label='Potenciais Clientes')
plt.title('Distribuição de Renda Mensal')
plt.xlabel('Renda Mensal')
plt.ylabel('Frequência')
plt.legend()
plt.show()

#Análise de correlação
plt.figure(figsize=(10,6))
sns.heatmap(retail_data.corr(), annot=True, cmap='coolwarm')
plt.title('Matriz de Correlação - Clientes Atuais')
plt.show()

#identificar padrões interessantes
#Com base nos gráficos e análises, é possível identificar padrões como:
#Renda mensal: Existe uma faixa de renda que é mais comum entre os clientes que receberam empréstimos?
#Tempode relacionamento: Clientes com mais tempo de relacionamento tem maior probabilidade de receber empréstimos?
#Segmentação: Existem segmentos de clientes (estado civil ou educação) que são mais propensos a receber empréstimos?