import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

#Carrega os dados dos clientes que receberam empréstimos
retail_data = pd.read_csv('Retail_Data.csv', on_bad_lines='skip', sep=',')

#Carrega os dados dos potenciais clientes
potential_customers = pd.read_csv('Potential_Customers.csv', on_bad_lines='skip', sep=',')

#Seleciona as features e o target
features = ['anos_com_banco', 'estado_marital', 'nivel_educacao', 'status_emprego', 'renda_mensal', 'balanco_total']
X = retail_data[features]
y = retail_data['recebeu_emprestimo']

#Divide os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Padroniza os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Treina o modelo de Regressão Logística
model = LogisticRegression()
model.fit(X_train, y_train)

#Avalia o modelo
y_pred = model.predict(X_test)
confusion_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(classification_report(y_test, y_pred))

#Visualiza a matriz de confusão
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=['Negativo', 'Possitivo'], yticklabels=['Negativo', 'Postivo'])
plt.xlabel('Predição')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão')
plt.show()

#Faz previsões para os potenciais clientes
potential_customers_scaled = scaler.transform(potential_customers[features])
predictions = model.predict(potential_customers_scaled)

#Adiciona as previsões ao DF de Potenciais clientes
potential_customers['previsao_emprestimo'] = predictions

#Exibe os potenciais clientes que tem maior probabilidade de receber empréstimos
clientes_abordar = potential_customers[potential_customers['previsao_emprestimo'] == 1]
print(clientes_abordar)

#Selecionei as características relevantes e dividi os dados em conjuntos de treino e teste
#Padronização: Dados padronizados para melhorar o desempenho do modelo
#Treinamento: Usei a regressão logística para treinar o modelo
#Avaliação: Avaliei o modelo usando uma matriz de confusão e um relatório de classificação
#Previsão: Fiz previsões para os potenciais clientes e adicionei as previsões ao DataFrame