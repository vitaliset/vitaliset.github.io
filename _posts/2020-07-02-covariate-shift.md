---
layout: post
title: Covariate Shift 0: Formulando o problema
mathjax: true
summary: Nest post apresento um problema clássico de dataset shift com um exemplo visual
---

# Covariate Shift 0: Formulando o problema

Um dos objetivos do aprendizado supervisionado é tentar reconhecer padrões entre variáveis explicativas e uma variável alvo. Matematicamente, temos um vetor aleatório $V = (X_1, X_2, \cdots, X_n, Y)$ e supomos que existe uma relação entre as variáveis explicativas $X_i$ e a variável alvo $Y$ do tipo
$$
Y \sim f(X_1, X_2,\cdots, X_n) + \varepsilon,
$$
onde $f:\mathbb{R}^n\to \mathbb{R}$ é uma função qualquer e $\varepsilon$ é uma variável aleatória com média $0$ chamada de ruído. O objetivo do paradigma de aprendizado supervisionado é estimar a função $f$ com observações anteriores (uma amostra do vetor aleatório $V$).

Em geral, quando fazemos validações hold-out e k-fold, esperamos que o desempenho da nossa função estimada continue sendo o mesmo que no conjunto de validação quando o modelo se deparar com dados novos. O problema de aprendizado de máquina em **ambientes não-estacionários** traz novos desafios ao nosso trabalho:  e se a distribuição do vetor aleatório $V$ variar?

É razoável esperar que o nosso modelo mantenha o desempenho da validação quando a função $f$ que relaciona as variáveis $X_i$ e $Y$ muda? Esse cenário é conhecimento como **Concept Shift** e é uma possível adversidade para muitas aplicações de aprendizado de máquina.

Um problema mais sutil, mas igualmente apavorante é o caso em que a relação entre as variáveis explicarivas e a variável alvo é conservado, mas a distribuição das variáveis $X_i$ nos novos exemplos é diferente da distribuição das variáveis $X_i$ nos dados de treinamento. Esse é o **Covariate Shift**, situação que vamos aprender a identificar nesta série de posts e dar uma possível abordagem de solução.

Antes, para esclarecer as ideias com uma situação prática, vamos construir articifialmente um Covariate Shift e identificar os problemas que acontecem se isso não é tratado de maneira séria.

Sejam $X$ uma variável aleatória tal que $X\sim \mathcal{N}(0,1)$, $f:\mathbb{R}\to\mathbb{R}$  uma função da forma $f(t) = \cos(2\pi t)$ e o ruído $\varepsilon$ modelado como $\varepsilon \sim \mathcal{N}(0,0.5)$. Construímos um conjunto de dados são gerados por esse experimento aleatório.

```python
def f(x):
    return np.cos(2*np.pi*x) 

def f_ruido(x):
    return f(x) + np.random.normal(0, 0.5)
    
def sample(n):
    '''
    Retorna uma amostra do vetor aleatório (X,Y).
    A variável n é o tamanho da amostra desejada.
    '''
    x = np.random.normal(0, 1, size=n)
    y = f_ruido(x)
    
    return x, y
```

Fazemos este experimento $100$ vezes, criando nossos dados. Apesar do ruído ser da ordem de grandeza de $f$, há ainda uma memória da função que os origino como podemos observar na Figura 1.

![Figura 1]({{ "assets/img/sonho1.png" | absolute_url }})

Fitamos uma árvore de regressão tentando utilizar os valores de $x$ para prever os valores de $y$. Fazendo um GridSearch básico no número de exemplos por folha chegamos em uma árvore com $R^2$ de $0.7970$ nos dados de treino.



Mas agora, **apesar da relação entre $X$ e $Y$ continuar sendo a mesma**, por algum motivo, a variável $X$ não é tem mais distribuição dada por $X\sim \mathcal{N}(0,1)$. Nesta variação, ela é dada por $X\sim \mathcal{N}(2,0)$.


É razoável esperar que o desempenho do nosso modelo continue o mesmo?

Fitamos uma árvore de regressão tentando utilizar os valores de $x$ para prever os valores de $y$. Fazendo um GridSearch básico no número de exemplos por folha chegamos em uma árvore com $R^2$ de $0.7970$ nos dados de treino.





#### Referências

[MathExchange](https://datascience.stackexchange.com/questions/28331/different-test-set-and-training-set-distribution)

(https://www.analyticsvidhya.com/blog/2017/07/covariate-shift-the-hidden-problem-of-real-world-data-science/)

(https://maxhalford.github.io/blog/subsampling-1/)

(https://blog.bigml.com/2014/01/03/simple-machine-learning-to-detect-covariate-shift/)

(https://mitpress.mit.edu/books/machine-learning-non-stationary-environments)



(https://mitpress.mit.edu/books/dataset-shift-machine-learning)