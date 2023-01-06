---
layout: post
title: Covariate Shift&#58; Formulando o problema
featured-img: covariate_0_formulando
category: [🇧🇷, dataset shift]
mathjax: true
summary: Apresentando o cenário do dataset shift com um caso ilustrativo.
---

<p><div align="justify">Um dos objetivos do aprendizado supervisionado é tentar reconhecer padrões entre variáveis explicativas e uma variável alvo. Matematicamente, temos um vetor aleatório $V = (X_1, X_2, \cdots, X_n, Y)$ e supomos que existe uma relação entre as variáveis explicativas $X_i$ e a variável alvo $Y$ do tipo</div></p>

$$
\begin{equation*}
Y \sim f(X_1, X_2,\cdots, X_n) + \varepsilon,
\end{equation*}
$$

<p><div align="justify">onde $f:\mathbb{R}^n\to \mathbb{R}$ é uma função qualquer e $\varepsilon$ é uma variável aleatória com média $0$ chamada de ruído (que possivelmente pode mudar dependendo dos valores de $X_i$ também). O objetivo do paradigma de aprendizado supervisionado é estimar a função $f$ com observações anteriores (uma amostra do vetor aleatório $V$).</div></p>

<p><div align="justify">Em geral, quando fazemos validações <i>hold-out</i> e <i>k-fold</i>, esperamos que o desempenho da nossa função estimada continue sendo o mesmo que no conjunto de validação quando o modelo se deparar com dados novos. O problema de aprendizado de máquina em <b>ambientes não-estacionários</b> traz novos desafios ao nosso trabalho:  e se houver um <i><b>Dataset Shift</b></i>, isto é, e se a distribuição do vetor aleatório $V$ for diferente nos dados que ainda não conhecemos? É razoável esperar que o modelo mantenha o desempenho obtido na validação?</div></p>

<p><div align="justify">Dentro desse contexto há pelo menos duas variações clássicas. A primeira é o <b><i>Concept Shift</i></b>, que ocorre quando a função $f$ que relaciona as variáveis $X_i$ e $Y$ muda. Um problema aparentemente mais sutil, mas igualmente apavorante é o caso em que a relação entre as variáveis explicativas e a variável alvo é conservada, mas a distribuição das variáveis $X_i$ nos novos exemplos é diferente da distribuição das variáveis $X_i$ nos dados de treinamento. Esse é o <i><b>Covariate Shift</b></i>, situação que vamos aprender a identificar nesta série de posts e dar uma possível abordagem de solução.</div></p>

<p><div align="justify">Antes, para esclarecer as ideias com uma situação prática, vamos construir artificialmente um cenário que apresenta <i>Covariate Shift</i> e analisar o problema que surge se isso não é identificado e tratado de maneira adequada.</div></p>

# Exemplo de dataset shift entre dados de treino e dados de produção

<p><div align="justify">Sejam $X$ uma variável aleatória tal que $X\sim \mathcal{N}(0,1)$, $f:\mathbb{R}\to\mathbb{R}$  uma função da forma $f(t) = \cos(2\pi t)$ e $\varepsilon$  o ruído modelado como $\varepsilon \sim \mathcal{N}(0,0.25)$. Construiremos um conjunto de dados gerados por esse experimento aleatório.</div></p>

```python
def f(X):
    return np.cos(2*np.pi*X) 

def f_ruido(X):
    '''
    Retorna uma amostra da v.a. Y, fixada uma amostra do v.a. X,
    entregue como parâmetro.
    '''
    return f(X) + np.random.normal(0, 0.5, size = X.shape[0])
    
def sample(n, mean = 0):
    '''
    Retorna uma amostra do vetor aleatório (X,Y).
    O parâmetro n é o tamanho da amostra desejada, e o
    valor mean é a média da normal, distribuição de X.
    '''
    X = np.random.normal(mean, 1, size=n)
    Y = f_ruido(X)
    return X.reshape(-1, 1), Y.reshape(-1, 1)
```

<p><div align="justify">Neste exemplo, faremos este experimento $100$ vezes, criando nossos dados com a média de $X$ em $0$ como comentado anteriormente.</div></p>

```python
X_past, Y_past = sample(100)
```

<p><div align="justify">Apesar do ruído ser da ordem de grandeza $f$, é possível visualizar o padrão da função que guia a geração dos dados, como observamos na Figura 1. Estamos interessados em fazer previsões: dadas novas observações de $x$, queremos estimar os respectivos valores para $y$.</div></p>

<p><center><img src="{{ site.baseurl }}/assets/img/covariate_0_formulando_post/imagem1.png"></center>
<center><b>Figura 1</b>: Os pontos azuis são nossas observações e em preto temos a curva que gerou os dados.</center></p>

<p><div align="justify">Usaremos um modelo simples para fazer a regressão, a Árvore de Decisão. Com um <i>GridSearch</i>, escolhemos o melhor valor para o mínimo de exemplos por folha (parâmetro de regularização, evitando <i>overfit</i>). Olhando, ainda, a validação cruzada para um <i>k-fold</i> com 5 pastas, estimamos o valor de $R^2$ se utilizássemos a árvore em dados nunca vistos.</div></p>

```python
dtr = DecisionTreeRegressor()
param = {'min_samples_leaf': np.arange(1,10,1)}
grid_search = GridSearchCV(dtr, param, cv = 5, scoring= 'r2', return_train_score=True)
grid_search.fit(X_past, Y_past)
```

<p><div align="justify">Obtemos um $R^2=0.486$, demonstrando há um aprendizado dos padrões dos dados (apesar de ser um modelo simples e termos poucos exemplos). Treinando novamente a árvore em todos os exemplos, podemos ver na Figura 2 que ele se aproxima da função original $f$.</div></p>

```python
dtr = DecisionTreeRegressor(min_samples_leaf = grid_search.best_params_['min_samples_leaf'])
dtr.fit(X_past,Y_past)
```

<p><center><img src="{{ site.baseurl }}/assets/img/covariate_0_formulando_post/imagem2.png"></center>
<center><b>Figura 2</b>: Adicionamos, em vermelho, a função estimada pelo modelo que tenta se aproximar da geradora, em preto.</center></p>

<p><div align="justify">Graficamente, vemos que o modelo faz um trabalho razoável ao redor do $0$, e perde a qualidade nas bordas, onde há menos exemplos de treinamento.</div></p>

<p><div align="justify">Imaginemos agora que o cenário mudou: apesar da relação entre $X$ e $Y$ continuar sendo a mesma, por algum motivo, a variável $X$ não é tem mais distribuição dada por $X\sim \mathcal{N}(0,1)$. Nesta variação, ela é dada por $X\sim \mathcal{N}(2,1)$, ou seja, temos uma translação da distribuição.</div></p>

```python
X_new, Y_new = sample(100, mean = 2)
```

<p><div align="justify">Os dados agora estão distribuídos mais a direita, como podemos visualizar na Figura 3.</div></p>

<p><center><img src="{{ site.baseurl }}/assets/img/covariate_0_formulando_post/imagem3.png"></center>
<center><b>Figura 3</b>: Histograma comparando a distribuição das duas amostras que temos. Em azul a feita quando X tinha média 0 e em laranja a nova, com média 2.</center></p>

<p><div align="justify">É razoável esperar que o desempenho do nosso modelo continue o mesmo? Podemos ver na Figura 4 que não.</div></p>

<p><center><img src="{{ site.baseurl }}/assets/img/covariate_0_formulando_post/imagem4.png"></center>
<center><b>Figura 4</b>: Agora, para os novos dados em laranja, extendemos o domínio do nosso modelo e vemos que ele não faz um bom trabalho tentando aproximar a curva geradora.</center></p>

<p><div align="justify">Como esperado, a qualidade do modelo cai para um $R^2$ de $-0.283$ nos dados novos. Isto lembrando que a relação entre $X$ e $Y$ não mudou, apenas a distribuição de $X$.</div></p>

# Identificando <i>Covariate Shfit</i>

<p><div align="justify">Dada a motivação inicial, temos um desafio resumido no seguinte enunciado:</div></p>

<p><div align="justify">Seja $X$ e $Z$ variáveis (ou vetores) aleatórias. Suponha que amostre $X$ de forma independente $N\in\mathbb{N}^*$ vezes e também $Z$ seja amostrada também de forma independente $M\in \mathbb{N}^*$ vezes ficando com as amostras $\{x_1, x_2, \cdots, x_N \} $ e $\{z_1, z_2, \cdots, z_M \} $. Como saber se $X\sim Z$ olhando apenas para as duas amostras? No contexto específico do <i>Covariate Shift</i>, vamos estar comparando as amostras das covariáveis no treino e depois em produção.</div></p>

<p><div align="justify">No geral, o monitoramento da distribuição das covariáveis precisa ser simples. Métodos básicos são escolhidos no lugar de técnicas complexas privilegiando eficiência computacional. Além disso, a análise costuma ser feita olhando covariável por covariável, identificando mudanças nessas distribuições marginais. Entre os métodos clássicos univariados, se destacam:</div></p>

- <p><div align="justify">Comparação de estatísticas: médias, variâncias e alguns quantis amostrais;</div></p>

- <p><div align="justify">Comparação de frequências para distribuições discretas e dados categóricos;</div></p>

- <p><div align="justify">Teste de Kolmogorov-Smirnov;</div></p>

- <p><div align="justify">Divergência de Kullback-leibler.</div></p>

<p><div align="justify">Muitas vezes esse monitoramente é feito aliado à análise da distribuição da saída do modelo. Se antes o nosso modelo indicava que $10%$ dos dados eram de uma classe e agora ele diz que $20%$ são daquela classe, temos um bom indicativo de que a distribuição de entrada mudou.</div></p>

<p><div align="justify">Nesta série de postagens pretendo apresentar alguns métodos um pouco mais alternativos para identificar o <i>Covariate Shift</i>. Em seguida, vamos entender porque ele ocorre sob a luz da minimização do risco empírico de Vapnik. Derivaremos uma maneira muito elegante de tratá-lo a partir de uma técnica que utilizamos para identificação do <i>dataset shift</i>.</div></p>

