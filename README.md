<p align="center">
  <img src="Cover Image.jpg" alt="Hotel Sign" width="100%">
</p>


# Projeto de Previsão de Cancelamento de Reservas Hoteleiras

## Índice

1.  [Introdução](#1-introdução)
2.  [Origem dos Dados](#2-origem-dos-dados)
3.  [Estrutura do Projeto](#3-estrutura-do-projeto)
4.  [Processo de Análise e Pré-processamento de Dados](#4-processo-de-análise-e-pré-processamento-de-dados)
    *   [Exploração Inicial](#exploração-inicial)
    *   [Tratamento de Valores Ausentes e Target Leakage](#tratamento-de-valores-ausentes-e-target-leakage)
    *   [Tratamento de Registros Duplicados](#tratamento-de-registros-duplicados)
    *   [Verificação e Conversão de Tipos de Dados](#verificação-e-conversão-de-tipos-de-dados)
    *   [Criação da Feature `total_nights`](#criação-da-feature-total_nights)
    *   [Padronização dos Nomes das Colunas](#padronização-dos-nomes-das-colunas)
    *   [Identificação e Tratamento de Outliers](#identificação-e-tratamento-de-outliers)
    *   [Codificação de Variáveis Categóricas](#codificação-de-variáveis-categóricas)
    *   [Divisão e Escala dos Dados](#divisão-e-escala-dos-dados)
5.  [Treinamento e Avaliação dos Modelos](#5-treinamento-e-avaliação-dos-modelos)
    *   [Regressão Logística](#regressão-logística)
        *   [Avaliação Inicial (Limiar 0.5)](#avaliação-inicial-limiar-05)
        *   [Ajuste do Limiar de Decisão (0.3)](#ajuste-do-limiar-de-decisão-03)
        *   [Reavaliação (Limiar 0.4)](#reavaliação-limiar-04)
    *   [Random Forest Classifier](#random-forest-classifier)
        *   [Treinamento e Avaliação (Limiar 0.5)](#treinamento-e-avaliação-limiar-05)
        *   [Comparação com Regressão Logística](#comparação-com-regressão-logística)
6.  [Conclusão Final](#6-conclusão-final)
7.  [Como Executar o Projeto](#7-como-executar-o-projeto)

---

## 1. Introdução

Este projeto visa desenvolver um modelo de Machine Learning capaz de prever o cancelamento de reservas hoteleiras. A previsão antecipada de cancelamentos permite que hotéis implementem estratégias proativas para minimizar perdas, otimizar a gestão de ocupação e melhorar a receita. O processo envolve desde a exploração inicial dos dados até o treinamento e avaliação de diferentes modelos de classificação.

## 2. Origem dos Dados

Os dados utilizados neste projeto são provenientes do dataset **"Hotel Booking Demand"**, disponível no Kaggle Hub, fornecido por jessemostipak. O dataset contém uma variedade de informações sobre reservas de hotéis, incluindo detalhes do hóspede, datas de chegada, tipo de hotel, preço e status de cancelamento.

## 3. Estrutura do Projeto

Este projeto é implementado em um único notebook Jupyter (Google Colab), que segue um fluxo sequencial através das etapas de:

*   Carregamento de dados
*   Pré-processamento e limpeza de dados
*   Engenharia de Features
*   Treinamento de modelos
*   Avaliação de desempenho

Todos os passos são executados e documentados dentro deste ambiente único.

## 4. Processo de Análise e Pré-processamento de Dados

Uma etapa crucial para a construção de um modelo robusto é a preparação dos dados.

### Exploração Inicial

*   **`df.info()`** e **`df.describe()`** foram utilizados para uma compreensão inicial da estrutura, tipos de dados e estatísticas descritivas do dataset.
*   Identificamos colunas com valores ausentes: `children`, `country`, `agent`, e `company`.

### Tratamento de Valores Ausentes e Target Leakage

*   **`company`** (94.31%) e **`agent`** (13.69%) foram removidas devido ao alto volume de dados ausentes.
*   **`children`** (0.003%) foi preenchida com `0`.
*   **`country`** (0.41%) foi preenchida com `'Unknown'`.
*   As colunas **`reservation_status`** e **`reservation_status_date`** foram removidas para evitar *target leakage*, pois contêm informações que diretamente revelam o status de cancelamento e poderiam artificialmente inflar o desempenho do modelo.

### Tratamento de Registros Duplicados

*   Foram identificados **32.280 registros duplicados**.
*   Uma análise mostrou que os registros duplicados distorciam significativamente a taxa de cancelamento (58.64% de cancelamentos em duplicatas vs. 27.28% em registros únicos).
*   Todos os registros duplicados foram removidos, resultando em **87.110 registros únicos**, garantindo uma representação mais fiel da realidade e evitando viés no modelo.

### Verificação e Conversão de Tipos de Dados

*   10 colunas do tipo `object` (categóricas) foram convertidas para o tipo `category` para otimização de memória e performance.
*   A coluna `children` foi convertida de `float` para `int`.
*   Houve uma redução significativa no uso de memória do DataFrame, de 19.3+ MB para **13.5 MB**.

### Criação da Feature `total_nights`

*   Uma nova feature, **`total_nights`**, foi criada pela soma de `stays_in_weekend_nights` e `stays_in_week_nights`. Esta feature direta fornece ao modelo a duração total da estadia, que é um fator crucial no comportamento de reserva e cancelamento.

### Padronização dos Nomes das Colunas

*   Todos os nomes das colunas foram convertidos para o formato `snake_case` (minúsculas com sublinhados) para melhorar a consistência e a legibilidade, facilitando a manipulação e a manutenção do código.

### Identificação e Tratamento de Outliers

*   As estatísticas descritivas e boxplots revelaram outliers em `adr`, `adults`, `babies`, e `total_nights`.
*   Registros com 0 hóspedes (`adults` + `children` + `babies` == 0) foram removidos (166 registros).
*   Registros com `adr` negativo foram removidos (1 registro).
*   Valores muito altos de `adr` foram limitados ao percentil 99.5 (capping em 285.00) para mitigar seu impacto sem remover muitos dados.
*   Registros com números implausíveis de `adults` (>4) e `babies` (>2) foram removidos (18 registros).
*   Após todo o tratamento, o DataFrame final possui **86.925 registros**.

### Codificação de Variáveis Categóricas

*   As colunas categóricas restantes foram transformadas em formato numérico utilizando **One-Hot Encoding** (`pd.get_dummies`), criando um DataFrame `df_encoded` com **248 colunas**.

### Divisão e Escala dos Dados

*   Os dados foram divididos em features (**X**) e target (**y** - `is_canceled`).
*   Em seguida, foram divididos em conjuntos de treinamento (80%) e teste (20%), com estratificação (`stratify=y`) para garantir que a distribuição da classe `is_canceled` fosse preservada em ambos os conjuntos.
*   As features numéricas foram escaladas utilizando **`StandardScaler`**, garantindo que todas as features contribuam igualmente para o modelo e melhorando o desempenho de algoritmos sensíveis à escala.

## 5. Treinamento e Avaliação dos Modelos

Dois modelos de classificação foram explorados para prever o cancelamento de reservas.

### Regressão Logística

O modelo de Regressão Logística foi utilizado como um *baseline* devido à sua simplicidade e interpretabilidade.

#### Avaliação Inicial (Limiar 0.5)

*   **Acurácia:** 79.91%
*   **Precisão:** 68.79%
*   **Recall:** 48.34%
*   **F1-Score:** 56.78%
*   **AUC:** 0.85
*   **Conclusão:** Bom desempenho geral, mas recall baixo, indicando que o modelo estava perdendo muitos cancelamentos reais.

#### Ajuste do Limiar de Decisão (0.3)

Para tentar melhorar o recall, o limiar de decisão foi ajustado para 0.3.

*   **Acurácia:** 76.42% (queda)
*   **Precisão:** 54.98% (queda significativa)
*   **Recall:** 75.16% (aumento significativo)
*   **F1-Score:** 63.50% (aumento)
*   **Conclusão:** O recall melhorou drasticamente, mas com um custo elevado na precisão (mais falsos positivos). O F1-Score, que equilibra ambas as métricas, teve um aumento.

#### Reavaliação (Limiar 0.4)

Um limiar intermediário de 0.4 foi testado para buscar um melhor equilíbrio.

*   **Acurácia:** 79.72%
*   **Precisão:** 63.05%
*   **Recall:** 62.16%
*   **F1-Score:** 62.60%
*   **Conclusão:** Este limiar apresentou um bom equilíbrio entre precisão e recall, sendo um ponto intermediário que oferece um recall razoável sem o sacrifício excessivo da precisão observado no limiar de 0.3.

### Random Forest Classifier

Para tentar superar os resultados da Regressão Logística, foi implementado um modelo Random Forest, conhecido por sua capacidade de lidar com relações não lineares e complexidade nos dados.

#### Treinamento e Avaliação (Limiar 0.5)

*   **Acurácia:** 84.48%
*   **Precisão:** 77.51%
*   **Recall:** 60.79%
*   **F1-Score:** 68.14%
*   **AUC:** 0.90

#### Comparação com Regressão Logística (Limiar Padrão 0.5)

| Métrica    | Regressão Logística (0.5) | Random Forest (0.5) |
| :--------- | :------------------------ | :------------------ |
| **Acurácia** | 79.91%                    | **84.48%**          |
| **Precisão** | 68.79%                    | **77.51%**          |
| **Recall**   | 48.34%                    | **60.79%**          |
| **F1-Score** | 56.78%                    | **68.14%**          |
| **AUC**      | 0.85                      | **0.90**            |

## 6. Conclusão Final

O modelo **Random Forest demonstrou um desempenho superior** em todas as métricas de avaliação quando comparado à Regressão Logística. Ele conseguiu uma maior acurácia geral, uma melhor capacidade de identificar cancelamentos reais (recall) e foi mais preciso em suas previsões de cancelamento, resultando em menos falsos positivos. A área sob a curva ROC (AUC) de 0.90 confirma sua excelente capacidade discriminatória.

Este modelo Random Forest, com seu desempenho robusto, é a escolha recomendada para prever cancelamentos de reservas, permitindo que a empresa implemente estratégias proativas para mitigar perdas e otimizar a gestão de reservas. Este projeto reforça a importância de um pré-processamento cuidadoso e da experimentação com diferentes modelos para alcançar os melhores resultados.

## 7. Como Executar o Projeto

Para executar este projeto e replicar os resultados, siga os passos abaixo:

1.  **Clonar o Repositório**: Faça um clone deste repositório para sua máquina local usando o Git:
    ```bash
    git clone https://github.com/EricoCoutoJr/Projeto-de-Previs-o-de-Cancelamento-de-Reservas-Hoteleiras.git
    cd Projeto-de-Previs-o-de-Cancelamento-de-Reservas-Hoteleiras
    ```

2.  **Configurar o Ambiente**: Recomenda-se criar um ambiente virtual para gerenciar as dependências do projeto:
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows, use `venv\Scripts\activate`
    ```

3.  **Instalar Dependências**: Instale todas as bibliotecas necessárias listadas no `requirements.txt` (ou manualmente, se preferir): 
    ```bash
    pip install -r requirements.txt
    # Caso não exista um requirements.txt, as dependências são: 
    # pip install kagglehub pandas numpy matplotlib seaborn scikit-learn
    ```

4.  **Autenticação Kaggle**: O dataset é baixado via `kagglehub`. Para que isso funcione, você precisará configurar suas credenciais do Kaggle:
    *   Vá para sua conta no Kaggle (`kaggle.com/me/account`).
    *   Na seção "API", clique em "Create New API Token" para baixar o arquivo `kaggle.json`.
    *   Mova este arquivo `kaggle.json` para a pasta `~/.kaggle/` (Linux/macOS) ou `C:\Users\<Your Username>\.kaggle\` (Windows).
    *   Certifique-se de que o arquivo `kaggle.json` tenha as permissões corretas (leitura/escrita apenas para o proprietário). Exemplo no Linux/macOS: `chmod 600 ~/.kaggle/kaggle.json`.

5.  **Abrir o Notebook**: Abra o arquivo `.ipynb` do projeto em seu ambiente Jupyter preferido (Jupyter Lab, Jupyter Notebook, VS Code, Google Colab).

6.  **Executar as Células**: Execute as células do notebook sequencialmente, de cima para baixo. Cada célula contém um passo do processo de análise, pré-processamento, treinamento ou avaliação de modelos.

7.  **Analisar Saídas**: Observe as saídas de cada célula, que incluem estatísticas, gráficos e métricas de desempenho. As células de texto explicam o contexto e os resultados de cada etapa.

Ao seguir estas instruções, você poderá explorar e entender todo o fluxo do projeto, desde a importação dos dados brutos até as conclusões sobre o desempenho dos modelos.
