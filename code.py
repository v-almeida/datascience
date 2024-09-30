# Importação das bibliotecas necessárias
import pandas as pd  # Manipulação de dados
from sklearn.model_selection import train_test_split  # Divisão dos dados em treino/teste
from sklearn.feature_extraction.text import TfidfVectorizer  # Vetorização de texto
from sklearn.linear_model import LogisticRegression  # Modelo de Regressão Logística
from sklearn.metrics import accuracy_score  # Métrica de avaliação de acurácia
import matplotlib.pyplot as plt  # Biblioteca para gráficos
import seaborn as sns  # Gráficos estatísticos
import numpy as np  # Operações com arrays

# 1. Carregar o arquivo SMSSpamCollection que contém dados de mensagens
file_name = './arquivocsv/SMSSpamCollection (1)'  # Localização do arquivo CSV com as mensagens
df = pd.read_csv(file_name, sep='\t', header=None, names=['label', 'text'])  # Leitura do arquivo

# 2. Converter as labels de 'ham' (não spam) e 'spam' para valores binários
# 'ham' será 0 (mensagens normais), 'spam' será 1 (mensagens indesejadas)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 3. Dividir os dados em treino e teste (80% treino, 20% teste)
# X: mensagens (text), y: rótulos (label)
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# 4. Criar um vetorizador TF-IDF, que converte texto em uma representação numérica
# max_features=5000 limita o número de palavras únicas consideradas
tfidf = TfidfVectorizer(max_features=5000)

# 5. Ajustar o vetorizador com os dados de treino e transformar os dados de treino e teste em vetores numéricos
X_train_tfidf = tfidf.fit_transform(X_train)  # Treina o modelo com o conjunto de treino
X_test_tfidf = tfidf.transform(X_test)  # Aplica o modelo no conjunto de teste

# 6. Treinar um modelo de Regressão Logística com os dados de treino
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)  # Ajusta o modelo aos dados

# 7. Prever as probabilidades no conjunto de teste
# predict_proba retorna as probabilidades de cada classe (0 e 1)
y_pred_proba = model.predict_proba(X_test_tfidf)

# Função para gerar gráficos que mostram as probabilidades de classificação do modelo
def plot_binary_classification_probabilities(y_pred_proba, y_test):
    positive_probs = y_pred_proba[:, 1]  # Pegamos as probabilidades da classe positiva (spam)

    # Gráfico 1: Distribuição das probabilidades da classe positiva (população geral)
    plt.hist(positive_probs, bins=10, color='blue', alpha=0.7, edgecolor='black')
    plt.xlabel("Probabilidade da Classe Positiva (Ex.: É spam?) [%]")
    plt.ylabel("Contagem")
    plt.title("Distribuição das Probabilidades da Classe Positiva (População Geral)")
    plt.xticks(ticks=[i/10 for i in range(11)], labels=[f'{i*10}%' for i in range(11)])
    plt.xlim([0, 1])
    plt.show()

    # Gráfico 2: Distribuição das probabilidades para os exemplos que são spam (classe 1)
    positive_indices = (y_test == 1)  # Filtra apenas os exemplos onde y_test é spam
    positive_probs_target = positive_probs[positive_indices]
    plt.hist(positive_probs_target, bins=10, color='red', alpha=0.7, edgecolor='black')
    plt.xlabel("Probabilidade da Classe Positiva (Ex.: É spam?) [%]")
    plt.ylabel("Contagem")
    plt.title("Distribuição das Probabilidades da Classe Positiva (Target Positivo)")
    plt.xticks(ticks=[i/10 for i in range(11)], labels=[f'{i*10}%' for i in range(11)])
    plt.xlim([0, 1])
    plt.show()

    # Gráfico 3: Percentual populacional por faixa de probabilidade
    df_probs = pd.DataFrame({'Probabilidade Positiva': positive_probs, 'Total': [1] * len(positive_probs)})
    bin_edges = [i / 10 for i in range(11)]  # Dividindo em faixas de 10%
    bin_labels = [f"{i*10}% - {(i+1)*10}%" for i in range(10)]
    df_probs['Bin'] = pd.cut(df_probs['Probabilidade Positiva'], bins=bin_edges, labels=bin_labels, include_lowest=True)
    df_binned = df_probs.groupby('Bin').sum().reset_index()
    df_binned['Percentual Populacional'] = (df_binned['Total'] / df_binned['Total'].sum()) * 100

    # Plotar a distribuição percentual
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Bin', y='Percentual Populacional', data=df_binned, color='blue', alpha=0.7, edgecolor='black')
    plt.xlabel("Faixa de Probabilidade")
    plt.ylabel("Percentual Populacional [%]")
    plt.title("Distribuição Percentual da População por Faixa de Probabilidade (Ex.: É spam?)")
    plt.xticks(rotation=45)
    plt.show()

# 8. Chamar a função para gerar os gráficos de probabilidade
plot_binary_classification_probabilities(y_pred_proba, y_test)

# 9. Definir pontos de corte para classificar as mensagens
neg_cutoff = 0.2  # Abaixo de 20% de chance de ser spam, classificamos como não spam (ham)
pos_cutoff = 0.8  # Acima de 80% de chance de ser spam, classificamos como spam

# 10. Classificar as previsões com base nesses pontos de corte
# Se a probabilidade estiver entre 20% e 80%, a mensagem será enviada para "análise manual"
y_pred_custom = np.where(y_pred_proba[:, 1] >= pos_cutoff, 1, np.where(y_pred_proba[:, 1] <= neg_cutoff, 0, 2))

# 11. Avaliar a acurácia excluindo as mensagens classificadas como "análise manual"
manual_review = np.sum(y_pred_custom == 2)  # Contagem das mensagens para análise manual
accuracy_custom = accuracy_score(y_test[y_pred_custom != 2], y_pred_custom[y_pred_custom != 2])  # Acurácia

print(f"Acurácia com análise manual excluída: {accuracy_custom:.2f}")
print(f"Total de exemplos para análise manual: {manual_review} de {len(y_test)} ({manual_review / len(y_test) * 100:.2f}%)")

# Função para justificar a escolha dos pontos de corte
def justify_cutoff_choices(y_pred_proba, neg_cutoff, pos_cutoff):
    positive_probs = y_pred_proba[:, 1]
    total_pop = len(positive_probs)
    neg_pop = np.sum(positive_probs <= neg_cutoff)
    pos_pop = np.sum(positive_probs >= pos_cutoff)
    manual_pop = total_pop - neg_pop - pos_pop

    print(f"População na Classe Negativa (≤ {neg_cutoff*100:.1f}%): {neg_pop} ({neg_pop/total_pop*100:.2f}%)")
    print(f"População na Classe Positiva (≥ {pos_cutoff*100:.1f}%): {pos_pop} ({pos_pop/total_pop*100:.2f}%)")
    print(f"População para Análise Manual: {manual_pop} ({manual_pop/total_pop*100:.2f}%)")

    # Contagem de erros
    false_negatives = np.sum((positive_probs >= pos_cutoff) & (y_test == 0))  # Spam classificado como não spam
    false_positives = np.sum((positive_probs <= neg_cutoff) & (y_test == 1))  # Não spam classificado como spam

    print(f"Falsos Positivos (classificado como ham, mas é spam): {false_positives}")
    print(f"Falsos Negativos (classificado como spam, mas é ham): {false_negatives}")

# 12. Justificar a escolha dos pontos de corte
justify_cutoff_choices(y_pred_proba, neg_cutoff, pos_cutoff)

# 13. Avaliar o modelo original (sem análise manual)
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia do modelo original:", accuracy)
