---
layout: page
title: Uma sugestão de guia de estudos para mentoria em Ciência de Dados
permalink: /tutoring-guideline/
mathjax: true
---

Essa página sugere um roteiro de estudos em Ciência de Dados com foco em Aprendizado de Máquina. Ela abrange vários assuntos básicos e sugere continuações de estudo ou focos que o mentorado pode começar a se especializar no final.

Dependendo do nível técnico do mentorado no início da mentoria, várias etapas podem ser puladas. De forma geral, as atividades roterizadas nesse guia supõe um conhecimento, a nível de ciclo básico de graduação em curso de exatas, que detalho com mais cuidado a seguir, dando ainda alternativas de cursos para quando esse pré-requisito não se aplica.

## Pré-requisitos esperados do mentorado

- **Lógica de programação**: Não é esperado que o mentorado já tenha utilizado Python, mas aguarda-se que já tenha experiência prévia com programação em outra linguagem e esteja confortável com as estruturas de repetição, condições e operadores lógicos, além de algumas estruturas de dados básicas como listas, vetores e strings. Se esse não for o caso do mentorado, o curso de [introdução à programação do Kaggle](https://www.kaggle.com/learn/intro-to-programming) pode ser um bom começo. Se ainda assim, for necessário gastar mais tempo vendo esses tópicos, aulas selecionadas da [playlist de Python do Curso em Vídeo](https://youtube.com/playlist?list=PLvE-ZAFRgX8hnECDn1v9HNTI71veL3oW0) podem ser uma boa referência.

- **Probabilidade**: A linguagem de Aprendizado de Máquina é intrinsicamente probabilística. É necessário que o mentorado esteja confortável com os conceitos básicos de probabilidade (no mínimo em um [nível introdutório do assunto](https://gradmat.ufabc.edu.br/disciplinas/ipe/)). Isso significa, conhecer a definição (ingênua) de espaço de probabilidade, entender sobre probabilidade condicional e independência, variáveis aleatórias discretas e contínuas, conhecer as principais distribuições de variáveis aleatórias e saber calcular esperanças, variâncias, correlações etc. Caso esse assunto seja intimidador ainda, é sugerido uma revisão detalhada dos tópicos seguindo, por exemplo, o curso [Introdução à Probabilidade e Estatístíca (IPE) do professor Rafael Grisi da UFABC](https://www.youtube.com/channel/UCKGYUbUHWSDr8pFaDd7JWCg/videos).

- **Matemática de forma geral**: Além de probabilidade, outros tópicos de matemática, como noções de otimização, cálculo diferencial, operações matriciais e estatística são importantes para um entendimento menos superficial dos algoritmos de Aprendizado de Máquina. Se o mentorado tem uma base matemática fraca, em algum desses assuntos (ou em todos), a especialização [Mathematics for Machine Learning and Data Science Specialization](https://www.coursera.org/specializations/mathematics-for-machine-learning-and-data-science#courses) pode ser uma grande amiga para visitar aulas específicas ou se debruçar por algumas semanas com as bases principais antes da Ciência de Dados.

## Ensinamentos do HDFS: redundância é importante

Durante esse guia de materiais de mentoria, por selecionar partes específicas de cursos diferentes, vários assuntos podem ser vistos em repetição com uma abordagem ligeiramente diferente. Isso é intencional. Acredito que para formentar essa base da forma mais robusta possível, é importante que os assuntos realmente estejam absorvidos e vê-los algumas vezes reforça isso.

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line2.png"></center></p>

## 0) Como usar esse guia de atividades de tutoria?

Logo abaixo do título da atividade, temos uma breve motivação. Ela serve para o tutor saber porque aquela atividade faz sentido naquele momento (pode ser compartilhado com o mentorado se achar que é importante explicar a tarefa).

### 0.1) Requisitos sugeridos

Os pré-requisitos são para saber as dependências sugerida entre as atividades caso queira seguir outra ordem ou pular etapas.

### 0.2) Descrição

A descrição é o texto sugerido para passar para o aluno na ferramenta utilizada para registrar as atividades. Eu sugiro usar, como ferramenta, o [Trello](https://trello.com/): ele tem uma estrutura estilo Kanban que você pode organizar os cards em colunas (algo como: "to do", "in progress" e "done"). Essa arquitetura é bem parecida com a forma que as ferramentas de organização de tarefas são utilizadas em várias equipes de dados que usam o Scrum (ou alguma variação dele).

Aqui é onde o link para o material de estudo deve ser disponibilizado. De forma geral, as recomendações disponibilizam o conteúdo de forma gratuíta e os materiais estão em inglês majoritariamente, com alguns poucos tópicos em português. Se eventualmente algum link esteja quebrado, por favor, entre em contato comigo para verificação e correção.

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line2.png"></center></p>

# Atividades recorrentes

As atividades recorrentes são aquelas que não são feitas pontualmente e em sequência são finalizadas. São atividades que, a partir do momento que você tem os requisitos para realizá-las, você deveria fazê-la de forma recorrente durante as semanas da tutoria. 

## A) HackerRank

O HackerRank tem várias playlists interessantes que te ensinam com exercícios sobre particularidades do Python. Você conhece estrutura de dados não triviais enquanto revisita as que você acha que já domina aprendendo formas novas de trabalhar com elas. Variações dessa ferramentas podem ser utilizadas em processos seletivos (principalmente para posições mais próximas de Engenharia de Software, como é o caso de um Machine Learning Engineer).

### A.1) Requisitos sugeridos

Atividade 2.

### A.2) Descrição

Fazer pelo menos 1 hora de atividades propostas nas playlists de Python do [HackerRank](https://www.hackerrank.com/domains/python) toda semana. Rapidinho você já vai ter visto a maioria dos tipos relevantes e eles vão te ajudar a deixar seu código mais _pythonic_.

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## B) Stratascratch

Pandas é a principal biblioteca de manipulação de dados do Python. Aliado com o SQL, vai ser a principal maneira de tratar dados no dia a dia de ciência de dados. Você só vai dominar a sintaxe do pandas se utilizá-lo de forma recorrente. A ideia aqui é ter alguns exercícios mais ou menos clássicos para estar se familiarizando com as manipulações principais.

### B.1) Requisitos sugeridos

Atividade 12.

### B.2) Descrição

Fazer pelo menos 0.5 horas de atividades propostas utilizando Pandas do [Stratascratch]([stratascratch.com](http://stratascratch.com/)) toda semana. Você consegue resolver em SQL também e pode ser útil para treinar essa ferramenta eventualmente.

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line2.png"></center></p>

# Atividades "Sequenciais"

Atividades sequenciais idealizadas nesse roteiro.

## 1) Instalar Python (pelo Anaconda)

A ideia é deixar o ambiente pronto para utilizar Jupyter Notebooks. São ótimas ferramentas para exploração e, na prática, é o ambiente mais adotado no dia-a-dia do cientista de dados para prototipação de código. Durante os encontros é sempre útil mostrar os atalhos mais comuns e boas práticas (deixar o notebook preparado para um "restart and run all" obtendo resultados reproduzíveis com `random_state`s fixados), mas falar todos agora só vai confundir o mentorando. Eventualmente, no final, podemos mostrar mais coisas de IDE, como VSCode etc, mas não acho útil no início.

### 1.1) Requisitos sugeridos

Não há.

### 1.2) Descrição

Seguir os passos descritos aqui:  [Instalando Python para Aprendizado de Máquina - LAMFO](https://lamfo-unb.github.io/2017/06/10/Instalando-Python/).
Não precisa instalar TensorFlow/Git por enquanto.

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 2) Nivelamento Python (Kaggle Learn)

Python é **A LINGUAGEM** para Ciência de Dados: existem muitas bibliotecas de qualidade prontas e a maioria das empresas adota como padrão na esteira de produtização. O curso do Kaggle Learn é ótimo porque foca nas partes que serão mais úteis pensando nas bibliotecas principais de Aprendizado de Máquina. É um curso focado em quem está migrando de linguagem, ou seja, já assume conhecimento básico de lógica de programação.

### 2.1) Requisitos sugeridos

Atividade 1. Já assume conhecimento de programação básico em alguma linguagem.

### 2.2) Descrição 

[Learn Python Tutorials](https://www.kaggle.com/learn/python). Tempo estimado de 5 horas.

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 3) Aula introdutória do curso (básico) de ML do Andrew

Aqui, o didático Andrew apresenta os tipos de aprendizado e dá exemplos legais. É um bom ponto pra pedir pro mentorado dar exemplos de problemas que dá pra aplicar e também motivar algumas aplicações não triviais.

### 3.1) Requisitos sugeridos

Não há.

### 3.2) Descrição

- [Supervised Machine Learning: Regression and Classification by Andrew Ng](https://www.youtube.com/playlist?list=PLkDaE6sCZn6FNC6YRfRQc_FbeQrF8BwGI)
- #3 - What is machine learning? (5min)
- #4 - Supervised learning part 1 (6min)
- #5 - Supervised learning part 2 (7min)
- #6 - Unsupervised learning part 1 (8min)
- #7 - Unsupervised learning part 2 (3min)

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 4) Aula de Regressão Linear do curso (básico) de ML do Andrew

Regressão Linear é um dos algoritmos mais simples de Aprendizado de Máquina. É o terreno perfeito para apresentar as principais ideias do processo de aprendizado supervisionado, enquanto introduz nomeclatura e intuição. É fundamental que essa atividade seja feita com muita atenção e cuidado.

O curso do Andrew tem alguns notebooks auxiliares que podem ser legais de explorar dependendo do perfil do mentorado. Eu, particularmente, não acho essencial dado que as aulas já são bem visuais e já explicam os conceitos apresentados. De qualquer forma, você consegue baixar o zip nesse [repositório](https://github.com/greyhatguy007/Machine-Learning-Specialization-Coursera) (se eventualmente esse repositório ficar indisponível, não deve ser díficil achar outro repositório com esses arquivos [procurando no google](https://www.google.com/search?q=supervised+machine+learning%3A+regression+and+classification+notebooks+github&sxsrf=AJOqlzUQ11tr1y9XmW0QVpXNVUjS_8bIMg%3A1676862919336&ei=x-XyY6aQFOy81sQP1tGg4Ak&oq=Supervised+Machine+Learning%3A+Regression+and+Classification+notebooks&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQAxgAMgcIIxCwAxAnMgoIABBHENYEELADMgoIABBHENYEELADMgoIABBHENYEELADMgoIABBHENYEELADMgoIABBHENYEELADMgoIABBHENYEELADMgoIABBHENYEELADMgoIABBHENYEELADSgQIQRgAUABYAGDdC2gBcAF4AIABAIgBAJIBAJgBAMgBCcABAQ&sclient=gws-wiz-serp)).

### 4.1) Requisitos sugeridos

Atividade 3.

### 4.2) Descrição

- [Supervised Machine Learning: Regression and Classification by Andrew Ng](https://www.youtube.com/playlist?list=PLkDaE6sCZn6FNC6YRfRQc_FbeQrF8BwGI)
- #9 - Linear regression model part 1 (10min)
- #10 - Linear regression model part 2 (6min)
- #11 - Cost function formula (9min)
- #12 - Cost function intuition (15min)
- #13 - Visualizing the cost function (8min)
- #14 - Visualization examples (6min)
- #15 - Gradient descent (8min)
- #16 - Implementing gradient descent (9min)
- #17 - Gradient descent intuition (7min)
- #18 - Learning rate (9min)
- #19 - Gradient descent for linear regression (6min)
- #20 - Running gradient descent (5min)

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 5) Introdução ao Numpy

Numpy é a principal ferramenta para mexer em vetores e matrizes no Python. Todas as outras principais bibliotecas de Aprendizado de Máquina são construídas em cima dela. É primordial entender o básico, em um primeiro momento, e, eventualmente, masterizar.

### 5.1) Requisitos sugeridos

Atividade 2.

### 5.2) Descrição

- Numpy in 5 min: [Learn NUMPY in 5 minutes - BEST Python Library!](https://youtu.be/xECXZ3tyONo) (20 min)

- Se precisar ver mais, esse vídeo aqui também é interessante: [Complete Python NumPy Tutorial (Creating Arrays, Indexing, Math, Statistics, Reshaping)](https://youtu.be/GB9ByFAIAH4) (1 hora)

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 6) Implementing from ground up: Regressão Linear Simples

Essa atividade é uma forma de colocar em prática tudo que vimos até agora. 

### 6.1) Requisitos sugeridos

Atividades 2, 3, 4 e 5.

### 6.3) Descrição

- Utilizando Python (principalmente o `numpy`), você deve construir uma função que recebe seu conjunto de dados `X_train`, `y_train` e retorna os pesos de uma regressão linear simples (ou seja, `X_train` é unidimensional), utilizando gradiente descendente para fazer esse cálculo, como visto no curso do Andrew Ng. Pode criar o seu conjunto `X_train`, `y_train` como quiser, mas seu código deve ser robusto o suficiente para poder trocar os valores e continuar rodando da forma correta.
- Defina critérios de parada que você achar apropriado para o gradiente descendente.
- Em seguida, com os pesos calculados, você deverá fazer uma função que prevê os `y` para um conjunto `X` qualquer.
- Pode ser interessante utilizar algumas bibliotecas gráficas para visualizar o que você está fazendo. A mais famosa, e que eu mais gosto é o matplotlib. Nessa [playlist do Corey Schafer](https://youtube.com/playlist?list=PL-osiE80TeTvipOqomVEeZ1HRrcEvtZB_) temos demonstrações de vários gráficos úteis.

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 7) Curso de básico de ML do Kaggle Learn

Tirando problemas específicos, o dia a dia do cientista de dados não é criando os modelos do zero. O scikit-learn é uma das mais robustas bibliotecas com dezenas de modelos já construídos seguindo os melhores padrões de desenvolvimento de software. Tem uma comunidade Open Source incrível que dá suporte e guia os desenvolvimentos dela. Esse curso do Kaggle é um primeiro contato com o scikit-learn. É importante para conhecer o padrão de fit/predict que é o estabelecido, de forma geral, em Aprendizado de Máquina.

### 7.1) Requisitos sugeridos

Atividades 2, 3 e 4.

### 7.2) Descrição

[Kaggle Learn - Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning). Tempo estimado de 3 horas.

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 8) Introdução à Programação Orientada à Objetos (POO)

A orientação orientada à objetos é o paradigma de programação principal do Python. A forma de abstração  que ele nos oferece é muito poderosa e permite construir códigos complexos de uma forma estruturada e reaproveitável, com manutenção facilitada. A ideia dessa atividade não é ficar um mestre em POO, mas conhecer por cima a ideia para saber que existe e entender que o scikit-learn e outras bibliotecas do Python utilizam ela. No futuro, esse tópico pode ser revisto, entendendo agora a utilização de heranças e boas práticas (como os princípios SOLID).  

### 8.1) Requisitos sugeridos

Atividade 7.

### 8.2) Descrição

- [Python OOP Tutorial 1: Classes and Instances](https://youtu.be/ZDa-Z5JzLYM) (15 min)
- Extra: uma discussão sobre diferentes formas de se programar (paradigmas): [1.1 - Programação Funcional em Haskell: Paradigmas de Programação](https://youtu.be/CzGSaXbPFRA) (27 min) - O Python tem várias coisas bem úteis de programação funcional então é legal conhecer por cima as ideias também.

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 9) Aula de Regressão Linear Multivariada (e polinomial) do curso (básico) de ML do Andrew

Na vida real vamos utilizar dezenas, centenas, milhares de variáveis para fazer nossas previsões. Mesmo ainda sendo um algoritmo pouco complexo, generalizar o caso da regressão linear simples dá algumas pitadas de onde queremos chegar eventualmente. Além disso, nessa aula o Andrew explica a ideia de vetorização de código (stop using for loops!).

### 9.1) Requisitos sugeridos

Atividade 4.

### 9.2) Descrição

- [Supervised Machine Learning: Regression and Classification by Andrew Ng](https://www.youtube.com/playlist?list=PLkDaE6sCZn6FNC6YRfRQc_FbeQrF8BwGI)
- #21 - Multiple features (9min)
- #22 - Vectorization part 1 (6min)
- #23 - Vectorization part 2 (6min)
- #24 - Gradient descent for multiple linear regression (7min)
- #25 - Feature scaling part 1 (6min)
- #26 - Feature scaling part 2 (7min)
- #27 - Checking gradient descent for convergence (5min)
- #28 - Choosing the learning rate (6min)
- #29 - Feature engineering (3min)
- #30 - Polynomial regression (5min)

Em particular, nessa atividade, pode ser necessário revisar operações matriciais. Um material rápido e direto ao ponto para isso pode ser algumas das aulas da versão anterior desse curso:

- [Machine Learning — Andrew Ng, Stanford University [FULL COURSE]](https://www.youtube.com/playlist?list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN)
- Lecture 3.1 — Linear Algebra Review : Matrices And Vectors - 8:46
- Lecture 3.2 — Linear Algebra Review : Addition And Scalar Multiplication - 6:54
- Lecture 3.3 — Linear Algebra Review : Matrix Vector Multiplication - 13:40
- Lecture 3.4 — Linear Algebra Review : Matrix-Matrix Multiplication - 11:10
- Lecture 3.5 — Linear Algebra Review : Matrix Multiplication Properties - 9:03
- Lecture 3.6 — Linear Algebra Review : Inverse And Transpose - 11:14

Se quiser ver mais sobre o que o Andrew chama de "normal equation":
- [Machine Learning — Andrew Ng, Stanford University [FULL COURSE]](https://www.youtube.com/playlist?list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN)
- Lecture 4.6 — Linear Regression With Multiple Variables : Normal Equation - 16:18
- Lecture 4.7 — Linear Regression With Multiple Variables : Normal Equation Non Invertibility - 5:59

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 10) Implementing from ground up: Regressão Polinomial + POO

A estrutura orientada a objeto do scikit-learn precisa ser sua amiga. A ideia dessa atividade é tentar abrir um pouco a caixa preta dos estimadores do scikit-learn implementando do zero o caso particular de regressão multivariada (quando as dimensões extras são potências da primeira componente, como o Andrew explica em um dos vídeos da atividade 9).
A ideia aqui é ficar um pouco mais familiar com a forma como o scikit-learn funciona, treinando POO.

### 10.1) Requisitos sugeridos

Atividades 8 e 9.

### 10.2) Descrição

A ideia dessa atividade é estruturar de forma mais elegante o que você fez na atividade 8, colocando dentro de uma classe no formato dos estimadores vistos no curso do Kaggle sobre scikit-learn. Idealmente, boa parte do código anterior, será reaproveitado nessa atividade.

- Você deve construir uma classe chamada `PolynomialRegression` que recebe um parâmetro na sua inicialização chamado `degree`.
- Essa classe precisa ter dois métodos, o `fit` e o `predict`. O método `fit` recebe duas entradas: o `X` e o `y`. `X` é tal que `X.shape = (n_samples, 1)` e `y` é tal que `y.shape = (n_samples,)`, `n_samples` sendo o número de amostras. Repare que `X = np.array([[1, 2, 3]]).T` e `y = np.array([1, 2, 3])` satisfazem essas restrições (apenas um exemplo, use outros valores quaisquer). O método `predict` recebe apenas uma entrada: o `X`, com as mesmas retrições de `.shape` descritas anteriormente.
- A função fit calcula polinômios de `X` de grau até `degree` (sugestão: utilize um [list comprehension](https://www.w3schools.com/python/python_lists_comprehension.asp) com um [`np.hstack`](https://numpy.org/doc/stable/reference/generated/numpy.hstack.html)) e em seguida aplica um gradiente descendente para encontrar os coeficientes dessa regressão linear multivariada. Isso deve ser uma generalização da função anterior que você construiu para a regressão linear simples.
- Os  coeficientes aprendidos durante o `fit` devem ser armazenados em atributos da sua classe com um underline ao final do nome. Essa é a estrutura adotada pelo scikit-learn para guardar informações que foram aprendidas durante o treinamento.
- Por fim, a função `predict`, que recebe `X` deve fazer a mesma transformação polinomial nesse novo X e fazer as devidas multiplicações da regressão múltipla (algo como `X_poly*w + b`) para obter a previsão que é o que será retornado pela função.

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 11) Validação de modelos de regressão

Em relação à atividade prática, é de nosso interesse, por exemplo, saber qual o melhor valor de `degree` que devemos utilizar no nosso modelo. Para medir a noção de melhor ou pior, assim como no caso da regressão linear, precisamos definir uma métrica de avaliação. Aqui iremos ver algumas outras além da "mean squared error" além de aplicar essa ideia na atividade anterior.

### 11.1) Requisitos sugeridos

Atividade 10.

### 11.2) Descrição

Métricas de Regressão
- [Regression Metrics Review I - How to Win a Data Science Competition Learn from Top Kagglers](https://www.bilibili.com/video/BV117411y7Fa?p=30)
- [Regression Metrics Review II - How to Win a Data Science Competition Learn from Top Kagglers](https://www.bilibili.com/video/BV117411y7Fa?p=31)
Obs: Os vídeos dessa seção são de uma plataforma de compartilhamento de vídeos asiática pois os originais, que estavam no Coursera, estão indisponíveis em decorrência das sansões aplicadas a Rússia pela guerra na Ucrância. Em particular, no Coursera os cursos originados de universidades russas foram indiponibilizados.

No curso introdutório de Aprendizado de Máquina do Kaggle Learn que você fez, ideias iniciais de validação de modelo com um conjunto hold out foram apresentadas. Um exercício interessante é aplicar essas mesma ideia na atividade anterior separando seu conjunto em uma parte para treino e outra para teste.
- [Training and testing - Machine Learning for Trading](https://youtu.be/P2NqrFp8usY)
- Escolha algumas das [métricas discutidas aqui que estejam disponíveis no scikit-learn](https://scikit-learn.org/stable/modules/classes.html#regression-metrics) e veja como ela muda (tanto no conjunto de treino quanto no conjunto de teste) variando o valor de `degree` da sua implementação anterior.
- [Fundamentos de Aprendizagem de Máquina: Viés e Variância - StatQuest](https://youtu.be/EuBBz3bI-aA)
- Veremos depois com mais detalhe discussões sobre viés/variância e underfitting/overfitting, mas tente pensar o que acontece com o modelo polinomial quando mudamos o valor de `degree`. Para quais valores de `degree` temos underfitting e para quais valores temos `overfitting`?

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 12) Introdução ao Pandas

Pandas é a biblioteca de manipulações mais utilizada para estruturar seus dados em Python. Aliada ao Spark e ao SQL, você terá um stack muito robusto para as diferentes tarefas e cenários de manipulação de dados. O Pandas é, talvez, a mais importante na prática e masterizar ela ajudará a aprender as outras com mais facilidade.

### 12.1) Requisitos sugeridos

Atividade 5.

### 12.2) Descrição

Algumas referências. Se estiver muito redundante, pode pular alguma(s) das sugestões.
- [Kaggle Learn - Intro to Pandas](https://www.kaggle.com/learn/pandas). Tempo estimado de 4 horas.
- [Pandas for Data Science in 20 Minutes : Python Crash Course](https://www.youtube.com/watch?v=tRKeLrwfUgU) (23 min) 
- [Complete Python Pandas Data Science Tutorial! (Reading CSV/Excel files, Sorting, Filtering, Groupby)](https://youtu.be/vmEHCJofslg)  (~1 hora) 
- [Python pandas — An Opinionated Guide](https://youtube.com/playlist?list=PLgJhDSE2ZLxaENZWWF_VOUa5886KiUd15) (~2 horas)

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 13) Aula de Regressão Logística do curso (básico) de ML do Andrew

A regressão logística é uma generalização natural da regressão linear para o caso de classificação binária em que, por construção, é esperado que o output do seu modelo seja um valor entre 0 e 1 com interpretação de probabilidade de uma das classes. Nessa aula do Andrew, alguns assuntos a mais são abordados como underfitting/overfitting e regularização.

### 13.1) Requisitos sugeridos

Atividade 11.

### 13.2) Motivação

- [Supervised Machine Learning: Regression and Classification by Andrew Ng](https://www.youtube.com/playlist?list=PLkDaE6sCZn6FNC6YRfRQc_FbeQrF8BwGI)
- #31 - Classification Motivations (9min)
- #32 - Logistic regression (9min)
- #33 - Decision boundary (10min)
- #34 - Cost function for logistic regression (11min)
- #35 - Simplified Cost Function for Logistic Regression (5min)
- #36 - Gradient Descent Implementation (6min)
- #37 - The problem of overfitting (11min)
- #38 - Addressing overfitting (8min)
- #39 - Cost function with regularization (9min)
- #40 - Regularized linear regression (8min)
- #41 - Regularized logistic regression (5min)

Vale dar uma olhada rápida na ideia de generalização para o caso multiclasse:
- [Machine Learning — Andrew Ng, Stanford University [FULL COURSE]](https://www.youtube.com/playlist?list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN)
- Lecture 6.7 — Logistic Regression : MultiClass Classification OneVsAll - 6:16

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 14) Métricas de classificação

Assim como no caso de regressão, existem maneiras de avaliar a qualidade do nosso modelo no problema de classificação.

### 14.1) Requisitos sugeridos

Atividade 13.

### 14.2) Descrição

- [The 5 Classification Evaluation metrics every Data Scientist must know](https://towardsdatascience.com/the-5-classification-evaluation-metrics-you-must-know-aa97784ff226)
- [Classification Metrics Review - How to Win a Data Science Competition Learn from Top Kagglers](https://www.bilibili.com/video/BV117411y7Fa?p=32)
Obs: Os vídeos dessa seção são de uma plataforma de compartilhamento de vídeos asiática pois os originais, que estavam no Coursera, estão indisponíveis em decorrência das sansões aplicadas a Rússia pela guerra na Ucrância. Em particular, no Coursera os cursos originados de universidades russas foram indiponibilizados.

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 15) Árvores de Decisão e Regressão

Algoritmos baseados em árvores são os mais utilizados, de maneira geral, para dados tabulares. Entender bem o caso base ajudará a compreender as maneiras mais robustas (quando fazemos cômites).

### 15.1) Requisitos sugeridos

Atividade 14.

### 15.2) Descrição

- [CART - Classification And Regression Trees - StatQuest](https://youtube.com/playlist?list=PLblh5JKOoLUKAtDViTvRGFpphEc24M-QH)
- Decision and Classification Trees, Clearly Explained!!! (18min)
- Decision Trees, Part 2 - Feature Selection and Missing Data (5min)
- Regression Trees, Clearly Explained!!! (22min)
- How to Prune Regression Trees, Clearly Explained!!! (16min)

- [Decision Trees na Prática (Scikit-learn / Python) - Mario Filho](https://youtu.be/BDqejVlCfvc) (30min)

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 16) Random Forest

pass

### 16.1) Requisitos sugeridos

Atividade 15.

### 16.2) Descrição

- [Bootstrapping Main Ideas!!! - StatQuest](https://youtu.be/Xz0x-8-cgaQ) (9min)
- [StatQuest: Random Forests Parte 1 - Construindo, Usando e Avaliando - StatQuest](https://youtu.be/J4Wdy0Wc_xQ) (10min)
- [Random Forest na Prática (Scikit-learn / Python) - Mario Filho](https://youtu.be/RtA1rjhuavs) (25min)

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 17) Brincando com o Kaggle

pass

### 17.1) Requisitos sugeridos

Atividades 12 e 16.

### 17.2) Descrição

Escolher um dataset do Kaggle e limpar os dados + aplicar um modelo de ML, avaliando os resultados (qual métrica utilizar pensando no problema que estou preocupado em resolver?)

- [Find Open Datasets and Machine Learning Projects | Kaggle](https://www.kaggle.com/datasets)
- Sugestão: [Heart Attack Analysis & Prediction Dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset)

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line2.png"></center></p>

## T0 D0:

KNN + post de distâncias

- Semana 3 e Semana 4 do curso do Andrew

- Intermediate Machine Learning - Kaggle Learn: https://www.kaggle.com/learn/intermediate-machine-learning + https://www.kaggle.com/code/alexisbcook/parsing-dates
- Feature Engineering - Kaggle Learn: https://www.kaggle.com/learn/feature-engineering
- Otimização de hiperparâmetros: https://youtu.be/ttE0F7fghfk

- GitHub: https://learn.microsoft.com/pt-BR/training/modules/introduction-to-github/

- Intro to AI Ethics - Kaggle Learn: https://www.kaggle.com/learn/intro-to-ai-ethics
- Machine Learning Explainability - Kaggle Learn: https://www.kaggle.com/learn/machine-learning-explainability

- Semana 1 Unsupervised learning: https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning?specialization=machine-learning-introduction#syllabus

## Caminhos possíveis e/ou estudo individual

- Deep Learning
- Intro to Deep Learning - Kaggle Learn: https://www.kaggle.com/learn/intro-to-deep-learning
- Semana 1, 2 do curso do Andrew
- Avançado: Especialização em DeepLearning/ cursos do deeplearning.ai (imagem/texto/som etc)

- eXplainable AI
- Curso da UFABC Hal: https://www.youtube.com/@ufabchal/playlists?view=1&sort=dd&shelf_id=0

- Sistemas de recomendação
- Semana 2 https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning?specialization=machine-learning-introduction#syllabus

- Online Learning

- Robustez de modelo
- Leakeage de forma geral: https://youtu.be/qPYqeD2OUl4
- Out of distribution generalization LG: https://youtu.be/Gq20DI9punw
- Adaptação de domínio
- http://ciml.info/dl/v0_99/ciml-v0_99-ch08.pdf
- Meus posts

- Análise de Sobrevivência
- MeetUp Jessica: https://youtu.be/WZNmlT-arF0
- Pablo

- Calibração de Modelos
- Ale

- aprendizado semisupervisionado
- will

- Aprendizado por Reforço
- Multi Armed Bandits: https://gdmarmerola.github.io/ts-for-bernoulli-bandit/
- Semana 3 https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning?specialization=machine-learning-introduction#syllabus

- Inferencia Causal
- Livro Facure: https://matheusfacure.github.io/python-causality-handbook/landing-page.html

- Feature Selection
- https://vitaliset.github.io/boruta/

- SQL - Kaggle Learn: https://www.kaggle.com/learn/intro-to-sql
- SQL 2 - Kaggle Learn: https://www.kaggle.com/learn/advanced-sql

- Fairness -> Mais?
- Ramon: https://youtu.be/iBfpL0aA7w4
- Palestra geral Tatiana: https://youtu.be/LWt4LZmpasc

- Algumas discussões sobre gestão
- Moneda: https://datascienceleadership.com/
- Edu: https://youtu.be/0ELffU6j_Tk

- Monitoramento de Modelos
- MeetUp Nubank - https://youtu.be/iAiY9L47eak

- MLOps
- Curso Básico do Andrew

- Inferencia de Rejeitados
- MeetUp Jessica: https://youtu.be/CJeMJfUYwkM

Colocar o texto: https://vitaliset.github.io/k-fold/