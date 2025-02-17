import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Carrega os dados dos clientes que receberam empréstimos
retail_data = pd.read_excel('Retail_Data.xlsx')

#Carrega os dados dos potenciais clientes
potential_customers = pd.read_excel('Potential_Customers.xlsx')

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
sns.histplot(retail_data['RENDA'], kde=True, color='blue', label='Clientes Atuais')
sns.histplot(potential_customers['RENDA'], kde=True, color='orange', label='Potenciais Clientes')
plt.title('Distribuição de Renda Mensal')
plt.xlabel('Renda Mensal')
plt.ylabel('Frequência')
plt.legend()
plt.show()

#Análise de correlação
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, annot_kws={"size": 10})
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.title('Matriz de Correlação - Clientes Atuais', fontsize=15)
plt.show()

#identificar padrões interessantes
#Com base nos gráficos e análises, é possível identificar padrões como:
#Renda mensal: Existe uma faixa de renda que é mais comum entre os clientes que receberam empréstimos?
#Tempode relacionamento: Clientes com mais tempo de relacionamento tem maior probabilidade de receber empréstimos?
#Segmentação: Existem segmentos de clientes (estado civil ou educação) que são mais propensos a receber empréstimos?