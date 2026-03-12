# Perfuracao de Pocos

Projeto de machine learning supervisionado para recomendar a melhor regiao para abertura de novos pocos de petroleo.

O notebook original deste repositorio continua sendo a base analitica do estudo. A diferenca agora e que o projeto foi reorganizado em um formato mais didatico, com funcoes reutilizaveis, script principal e documentacao orientada a negocio.

---

## Visao Geral

A empresa ficticia `OilyGiant` precisa decidir em qual das tres regioes deve investir no desenvolvimento de 200 novos pocos.

O desafio nao e apenas prever o volume de reservas de cada poco. A decisao final depende de tres camadas:

- qualidade preditiva do modelo
- lucro potencial ao selecionar os 200 melhores pocos
- risco de prejuizo ao simular a operacao com bootstrapping

Por restricao do problema, o modelo utilizado deve ser **Regressao Linear**.

---

## Objetivo de Negocio

Queremos responder uma pergunta simples:

**Qual regiao oferece o melhor equilibrio entre retorno esperado e risco de perda?**

As regras do estudo sao:

- 500 pontos sao avaliados por regiao
- apenas os 200 melhores entram no plano de desenvolvimento
- o investimento total e de `USD 100.000.000`
- cada unidade de `product` representa milhares de barris
- cada unidade gera `USD 4.500` em receita
- so podemos recomendar regioes com risco de prejuizo abaixo de `2.5%`

---

## Motivacao das Ferramentas

O projeto foi estruturado para deixar claro o papel de cada ferramenta usada no estudo:

- `pandas`: leitura, limpeza e organizacao dos dados tabulares
- `numpy`: calculos numericos e simulacao do bootstrap
- `scikit-learn`: divisao treino-validacao e treinamento da regressao linear
- `matplotlib`: mantido como dependencia para expandir o notebook com graficos, se desejado

Escolhas metodologicas:

- `LinearRegression`: exigencia explicita do enunciado
- `train_test_split`: separa treino e validacao sem misturar avaliacao com treinamento
- `RMSE`: mede o erro medio das previsoes em uma escala interpretavel para o problema
- `baseline pela media`: mostra se o modelo realmente agrega valor frente a uma previsao ingenua
- `top 200 por previsao`: traduz o modelo para uma decisao operacional concreta
- `bootstrapping`: aproxima a incerteza do processo real e quantifica risco

---

## Principais Achados do Notebook

Os resultados abaixo foram extraidos do notebook base:

| Regiao | Media prevista | Media real |    RMSE | Baseline RMSE |
| ------ | -------------: | ---------: | ------: | ------------: |
| 0      |        92.5926 |    92.0786 | 37.5794 |       44.2896 |
| 1      |        68.7285 |    68.7231 |  0.8931 |       46.0214 |
| 2      |        94.9650 |    94.8842 | 40.0297 |       44.9023 |

Leituras importantes:

- A `regiao 1` tem poder preditivo excepcionalmente alto, com erro muito baixo.
- As `regioes 0 e 2` apresentam medias de reserva maiores, o que inicialmente parece mais atrativo.
- No recorte deterministico dos 200 melhores pocos, o notebook indica vantagem inicial da `regiao 0`.
- Entretanto, quando o estudo incorpora incerteza com bootstrap, a `regiao 1` passa a ser a recomendacao final, com valores mais reais e incerteza calculada.

---

## Estrutura do Projeto

```bash
.
|-- data/
|   |-- geo_data_0.csv
|   |-- geo_data_1.csv
|   `-- geo_data_2.csv
|-- notebook/
|   `-- drill_location.ipynb
|-- notes/
|   `-- diretriz.txt
|-- reports/
|   |-- bootstrap_report.csv
|   `-- model_report.csv
|-- src/
|   |-- __init__.py
|   |-- business.py
|   |-- data.py
|   `-- modeling.py
|-- main_code.py
|-- requirements.txt
|-- train.py
`-- README.md
```

---

## Fluxo do Projeto

### 1. Preparacao dos dados

- leitura dos tres arquivos CSV
- remocao de duplicatas
- descarte da coluna `id`, que nao agrega valor preditivo
- separacao entre variaveis explicativas e alvo `product`

### 2. Treinamento do modelo

- divisao `75/25` entre treino e validacao
- treino de uma regressao linear por regiao
- comparacao do RMSE do modelo com um baseline pela media

### 3. Avaliacao economica

- calculo do volume minimo medio por poco para nao haver prejuizo
- selecao dos 200 melhores pocos usando as previsoes do modelo
- conversao do volume em lucro esperado

### 4. Analise de risco

- 1000 simulacoes bootstrap
- amostras de 500 pocos com reposicao
- selecao dos 200 melhores em cada simulacao
- calculo de lucro medio, intervalo de confianca e risco de perda

---

## Como Executar

```bash
pip install -r requirements.txt
python train.py
```

Ou, se preferir manter o nome antigo:

```bash
python main_code.py
```

Ao executar, o script deve gerar:

- `reports/model_report.csv`
- `reports/bootstrap_report.csv`

---

## O que foi melhorado em relacao ao estado anterior

- o notebook deixou de ser a unica forma de entender o projeto
- a logica foi separada em modulos com responsabilidade clara
- o ponto de entrada do pipeline ficou explicito
- a documentacao agora explica nao so o que foi feito, mas por que foi feito
