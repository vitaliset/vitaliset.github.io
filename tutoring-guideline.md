---
layout: page
title: Uma sugestão de guia de estudos para mentoria em Ciência de Dados
permalink: /tutoring-guideline/
mathjax: true
---

Essa página sugere um roteiro de estudos em Ciência de Dados com foco em Aprendizado de Máquina. Ela abrange vários assuntos básicos e sugere continuações de estudo ou focos que o mentorado pode começar a se especializar no final.

Dependendo do nível técnico do mentorado no início da mentoria, várias etapas podem ser puladas. De forma geral, as atividades roterizadas nesse guia supõe um conhecimento, a nível de ciclo básico de graduação em curso de exatas, que detalho com mais cuidado a seguir, dando ainda alternativas de cursos para quando esse pré-requisito não se aplica.

## Pré-requisitos esperados do mentorado

Numa conversa com o mentorado, é importante sentir se ele tem os pré requisitos para seguir essa sugestão de tópicos ou não. Caso não seja o caso, várias coisas precisam ser adaptadas, principalmente no início e já dou algumas sugestões nessa seção.

- **Lógica de programação**: Não é esperado que o mentorado já tenha utilizado Python, mas aguarda-se que já tenha experiência prévia com programação em outra linguagem e esteja confortável com laços de repetição, condições e operadores lógicos, além de algumas estruturas de dados básicas como listas, vetores e strings. Se esse não for o caso do mentorado, o curso de [introdução à programação do Kaggle](https://www.kaggle.com/learn/intro-to-programming) pode ser um bom começo. Se ainda assim, for necessário gastar mais tempo vendo esses tópicos, aulas selecionadas da [playlist de Python do Curso em Vídeo](https://youtube.com/playlist?list=PLvE-ZAFRgX8hnECDn1v9HNTI71veL3oW0) podem ser uma boa referência.

- **Estatística descritiva**: Visualizações clássicas como gráficos de dispersão, gráficos de barra/coluna, histogramas, box plots, gráficos de pizza, além de medidas resumo de posição e dispersão, como média, mediana, percentis, desvio padrão etc, precisam ser assuntos que o mentorado já tenha alguma familiaridade. Dependendo do nível do mentorado, uma simples revisão superficial do assunto ([como essa aula de análise descritiva da Univesp](https://youtu.be/42ArF0YCWm8)) provavelmente já é suficiente. Em outros casos, será necessário um pouco mais de atenção e daí o foco das primeiras semanas pode mudar totalmente para ensinar Python a partir de exemplos de visualização de dados (por exemplo com o [Kaggle Learn de Data Visualization com seaborn](https://www.kaggle.com/learn/data-visualization)), ao invés de já começar com Aprendizado de Máquina.

- **Probabilidade**: A linguagem de Aprendizado de Máquina é intrinsicamente probabilística. É recomendado que o mentorado esteja confortável com os conceitos básicos de probabilidade (no mínimo em um [nível introdutório do assunto](https://gradmat.ufabc.edu.br/disciplinas/ipe/)). Isso significa, conhecer a definição (ingênua) de espaço de probabilidade, entender sobre probabilidade condicional e independência, variáveis aleatórias discretas e contínuas, conhecer as principais distribuições de variáveis aleatórias e saber calcular esperanças, variâncias, correlações etc. Obviamente não precisa conhecer tudo nos mínimos detalhes, mas se o assunto é intimidador ainda, é sugerido uma revisão dos tópicos seguindo, por exemplo, o curso [Introdução à Probabilidade e Estatístíca (IPE) do professor Rafael Grisi da UFABC](https://www.youtube.com/channel/UCKGYUbUHWSDr8pFaDd7JWCg/videos).

- **Matemática de forma geral**: Além de estatística descritiva e probabilidade básica, outros tópicos de matemática, como noções de otimização, cálculo diferencial, operações matriciais e estatística inferencial são importantes para um entendimento menos superficial dos algoritmos de Aprendizado de Máquina. Se o mentorado tem uma base matemática fraca, em algum desses assuntos (ou em todos), a especialização [Mathematics for Machine Learning and Data Science Specialization](https://www.coursera.org/specializations/mathematics-for-machine-learning-and-data-science#courses) pode ser uma grande amiga para visitar aulas específicas ou se debruçar por algumas semanas com as bases principais da Ciência de Dados.

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

# Cronograma 

<p><center><img src="{{ site.baseurl }}/assets/img/ordem_atividades.png"></center></p>

Terminada essa etapa de atividades, a sugestão é que seja feito algum desafio do Kaggle com cuidado no tempo restante da tutoria (pensando numa tutoria de aproximadamente 6 meses). Nesse momento, é importante entender o problema de forma detalhada e pode ser útil trabalhar em "releases", em que a cada x semanas se tenha uma versão do modelo um pouco mais estruturada e com testes novos.

Obviamente essas atividades não tão escritas em pedra e podem ser modificadas dependendo do viés do tutorado. Por exemplo, um tutorado mais próximo da área de saúde pode ter interesse em redes neurais e portanto faz sentido algumas atividades serem substituídas, principalmente das semanas finais.

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

Fazer pelo menos 30 minutos de atividades propostas utilizando Pandas do [Stratascratch]([stratascratch.com](http://stratascratch.com/)) toda semana. Você consegue resolver em SQL também e pode ser útil para treinar essa ferramenta eventualmente.

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line2.png"></center></p>

# Atividades Sequenciais

Sequência idealizada de atividades para o ciclo de estudos da mentoria.

## 1) Instalar Python (pelo Anaconda)

A ideia é deixar o ambiente pronto para utilizar Jupyter Notebooks. São ótimas ferramentas para exploração e, na prática, é o ambiente mais adotado no dia-a-dia do cientista de dados para prototipação de código. Durante os encontros é sempre útil mostrar os atalhos mais comuns e boas práticas (deixar o notebook preparado para um "restart and run all" obtendo resultados reproduzíveis com `random_state`s fixados), mas falar todos agora só vai confundir o mentorando. Eventualmente, no final, podemos mostrar mais coisas de IDE, como VSCode etc, mas não acho útil no início.

### 1.1) Requisitos sugeridos

Não há.

### 1.2) Descrição

- [Instalando Python para Aprendizado de Máquina - LAMFO](https://lamfo-unb.github.io/2017/06/10/Instalando-Python/). Não precisa instalar o TensorFlow, nem o Git por enquanto.

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 2) Aula introdutória da especialização (básica) de ML do Andrew

Aqui, o didático Andrew apresenta os tipos de aprendizado e dá exemplos legais. É um bom ponto pra pedir pro mentorado dar exemplos de problemas que dá pra aplicar e também motivar algumas aplicações não triviais.

### 2.1) Requisitos sugeridos

Não há.

### 2.2) Descrição

- [Supervised Machine Learning: Regression and Classification by Andrew Ng](https://www.youtube.com/playlist?list=PLkDaE6sCZn6FNC6YRfRQc_FbeQrF8BwGI)
    - #3 - What is machine learning? (5min)
    - #4 - Supervised learning part 1 (6min)
    - #5 - Supervised learning part 2 (7min)
    - #6 - Unsupervised learning part 1 (8min)
    - #7 - Unsupervised learning part 2 (3min)

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 3) Nivelamento Python (Kaggle Learn)

Python é **A LINGUAGEM** para Ciência de Dados: existem muitas bibliotecas de qualidade prontas e a maioria das empresas adota como padrão na esteira de produtização. O curso do Kaggle Learn é ótimo porque foca nas partes que serão mais úteis pensando nas bibliotecas principais de Aprendizado de Máquina. É um curso focado em quem está migrando de linguagem, ou seja, já assume conhecimento básico de lógica de programação.

### 3.1) Requisitos sugeridos

Atividade 1. Já assume conhecimento de programação básico em alguma linguagem.

### 3.2) Descrição 

- [Kaggle Learn - Python Tutorials](https://www.kaggle.com/learn/python) (5h).

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 4) Aula de Regressão Linear da especialização (básica) de ML do Andrew

Regressão Linear é um dos algoritmos mais simples de Aprendizado de Máquina. É o terreno perfeito para apresentar as principais ideias do processo de aprendizado supervisionado, enquanto introduz nomeclatura e intuição. É fundamental que essa atividade seja feita com muita atenção e cuidado.

O curso do Andrew tem alguns notebooks auxiliares que podem ser legais de explorar dependendo do perfil do mentorado. Eu, particularmente, não acho essencial dado que as aulas já são bem visuais e já explicam os conceitos apresentados. De qualquer forma, você consegue baixar o zip nesse [repositório](https://github.com/greyhatguy007/Machine-Learning-Specialization-Coursera) (se eventualmente esse repositório ficar indisponível, não deve ser díficil achar outro repositório com esses arquivos [procurando no google](https://www.google.com/search?q=supervised+machine+learning%3A+regression+and+classification+notebooks+github&sxsrf=AJOqlzUQ11tr1y9XmW0QVpXNVUjS_8bIMg%3A1676862919336&ei=x-XyY6aQFOy81sQP1tGg4Ak&oq=Supervised+Machine+Learning%3A+Regression+and+Classification+notebooks&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQAxgAMgcIIxCwAxAnMgoIABBHENYEELADMgoIABBHENYEELADMgoIABBHENYEELADMgoIABBHENYEELADMgoIABBHENYEELADMgoIABBHENYEELADMgoIABBHENYEELADMgoIABBHENYEELADSgQIQRgAUABYAGDdC2gBcAF4AIABAIgBAJIBAJgBAMgBCcABAQ&sclient=gws-wiz-serp)).

### 4.1) Requisitos sugeridos

Atividade 2.

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

Atividade 3.

### 5.2) Descrição

- Numpy in 5 min: [Learn NUMPY in 5 minutes - BEST Python Library!](https://youtu.be/xECXZ3tyONo) (20 min)

- Se precisar ver mais, esse vídeo aqui também é interessante: [Complete Python NumPy Tutorial (Creating Arrays, Indexing, Math, Statistics, Reshaping)](https://youtu.be/GB9ByFAIAH4) (1 hora)

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 6) Implementing from ground up: Regressão Linear Simples

Essa atividade é uma forma de colocar em prática tudo que vimos até agora. 

### 6.1) Requisitos sugeridos

Atividades 4 e 5.

### 6.3) Descrição

- Utilizando Python (principalmente o `numpy`), você deve construir uma função que recebe seu conjunto de dados `X_train`, `y_train` e retorna os pesos de uma regressão linear simples (ou seja, `X_train` é unidimensional), utilizando gradiente descendente para fazer esse cálculo, como visto no curso do Andrew Ng. Pode criar o seu conjunto `X_train`, `y_train` como quiser, mas seu código deve ser robusto o suficiente para poder trocar os valores e continuar rodando da forma correta.
- Defina critérios de parada que você achar apropriado para o gradiente descendente.
- Em seguida, com os pesos calculados, você deverá fazer uma função que prevê os `y` para um conjunto `X` qualquer.
- Pode ser interessante utilizar algumas bibliotecas gráficas para visualizar o que você está fazendo. A mais famosa, e que eu mais gosto é o matplotlib. Nessa [playlist do Corey Schafer](https://youtube.com/playlist?list=PL-osiE80TeTvipOqomVEeZ1HRrcEvtZB_) temos demonstrações de vários gráficos úteis.

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 7) Curso de básico de ML do Kaggle Learn

Tirando problemas específicos, o dia a dia do cientista de dados não é criando os modelos do zero. O scikit-learn é uma das mais robustas bibliotecas com dezenas de modelos já construídos seguindo os melhores padrões de desenvolvimento de software. Tem uma comunidade Open Source incrível que dá suporte e guia os desenvolvimentos dela. Esse curso do Kaggle é um primeiro contato com o scikit-learn. É importante para conhecer o padrão de fit/predict que é o estabelecido, de forma geral, em Aprendizado de Máquina.

### 7.1) Requisitos sugeridos

Atividades 4 e 5.

### 7.2) Descrição

- [Kaggle Learn - Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning) (3h).

- Extra: pode ser util dar uma lida por cima da página de [getting started do scikit-learn](https://scikit-learn.org/stable/getting_started.html).

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 8) Introdução à Programação Orientada a Objetos (POO)

A orientação orientada à objetos é o paradigma de programação principal do Python. A forma de abstração  que ele nos oferece é muito poderosa e permite construir códigos complexos de uma forma estruturada e reaproveitável, com manutenção facilitada. A ideia dessa atividade não é ficar um mestre em POO, mas conhecer por cima a ideia para saber que existe e entender que o scikit-learn e outras bibliotecas do Python utilizam ela. No futuro (provavelmente não no tempo dessa mentoria), esse tópico pode ser revisto, entendendo agora a utilização de heranças e boas práticas (como os princípios SOLID e design patterns).  

### 8.1) Requisitos sugeridos

Atividade 7.

### 8.2) Descrição

- [Python OOP Tutorial 1: Classes and Instances](https://youtu.be/ZDa-Z5JzLYM) (15 min)
- Tente criar algum cenário simples em Python em que você usa classes. Por exemplo, crie uma classe abstrata que representa a entidade "cachorro" e tem dois atributos: "nome" e "raça". O cachorro precisa ainda ter um método chamado "pedido_para_sentar" que recebe uma string e se essa string é o nome do cachorro então ele printa que o cachorro sentou.

- Extra: uma discussão sobre diferentes formas de se programar (paradigmas): [1.1 - Programação Funcional em Haskell: Paradigmas de Programação](https://youtu.be/CzGSaXbPFRA) (27 min) - O Python tem várias coisas bem úteis de programação funcional então é legal conhecer por cima as ideias também.

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 9) Aula de Regressão Linear Multivariada (e polinomial) da especialização (básica) de ML do Andrew

Na vida real vamos utilizar dezenas, centenas, milhares de variáveis para fazer nossas previsões não apenas uma como na regressão linear simples. Mesmo ainda sendo um algoritmo pouco complexo, generalizar o caso da regressão linear simples dá algumas pitadas de onde queremos chegar eventualmente. Além disso, nessa aula o Andrew explica a ideia de vetorização de código (stop using `for` loops!).

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

Se quiser ver mais sobre o que o Andrew chama de "normal equation" - que nada mais é que a solução análitica dos pesos da regressão linear (em contrates com o metódo numérico iterativo aproximado dado pelo gradiente descendente):
- [Machine Learning — Andrew Ng, Stanford University [FULL COURSE]](https://www.youtube.com/playlist?list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN)
    - Lecture 4.6 — Linear Regression With Multiple Variables : Normal Equation - 16:18
    - Lecture 4.7 — Linear Regression With Multiple Variables : Normal Equation Non Invertibility - 5:59

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 10) Implementing from ground up: Regressão Polinomial + POO

A estrutura orientada a objeto do scikit-learn precisa ser sua amiga. A ideia dessa atividade é tentar abrir um pouco a caixa preta dos estimadores do scikit-learn implementando do zero o caso particular de regressão multivariada (quando as dimensões extras são potências da primeira componente, como o Andrew explica em um dos vídeos da atividade 9).
A ideia aqui é ficar um pouco mais familiar com a forma como o scikit-learn funciona, treinando POO.

### 10.1) Requisitos sugeridos

Atividades 4, 8 e 9.

### 10.2) Descrição

A ideia dessa atividade é estruturar de forma mais elegante o que você fez na atividade 8, colocando dentro de uma classe no formato dos estimadores vistos no curso do Kaggle sobre scikit-learn. Idealmente, boa parte do código anterior, será reaproveitado nessa atividade.

- Você deve construir uma classe chamada `PolynomialRegression` que recebe um parâmetro na sua inicialização chamado `degree`.
- Essa classe precisa ter dois métodos, o `fit` e o `predict`. O método `fit` recebe duas entradas: o `X` e o `y`. `X` é tal que `X.shape = (n_samples, 1)` e `y` é tal que `y.shape = (n_samples,)`, `n_samples` sendo o número de amostras. Repare que `X = np.array([[1, 2, 3]]).T` e `y = np.array([1, 2, 3])` satisfazem essas restrições (apenas um exemplo, use outros valores quaisquer). O método `predict` recebe apenas uma entrada: o `X`, com as mesmas retrições de `.shape` descritas anteriormente.
- A função fit calcula polinômios de `X` de grau até `degree` (sugestão: utilize um [list comprehension](https://www.w3schools.com/python/python_lists_comprehension.asp) com um [`np.hstack`](https://numpy.org/doc/stable/reference/generated/numpy.hstack.html)) e em seguida aplica um gradiente descendente para encontrar os coeficientes dessa regressão linear multivariada. Isso deve ser uma generalização da função anterior que você construiu para a regressão linear simples.
- Os  coeficientes aprendidos durante o `fit` devem ser armazenados em atributos da sua classe com um underline ao final do nome. Essa é a estrutura adotada pelo scikit-learn para guardar informações que foram aprendidas durante o treinamento.
- Por fim, a função `predict`, que recebe `X` deve fazer a mesma transformação polinomial nesse novo X e fazer as devidas multiplicações da regressão múltipla (algo como `X_poly*w + b`) para obter a previsão que é o que será retornado pela função.

- Extra: pode ser interessante dar uma olhada em como o [scikit-learn sugere a implementação de modelos](https://scikit-learn.org/dev/developers/develop.html). Não precisa se preocupar com o que ele chama de `BaseEstimator` e mixins.

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 11) Validação de modelos de regressão

Em relação à atividade prática, é de nosso interesse, por exemplo, saber qual o melhor valor de `degree` que devemos utilizar no nosso modelo. Para medir a noção de melhor ou pior, assim como no caso da regressão linear, precisamos definir uma métrica de avaliação. Aqui iremos ver algumas outras além da "mean squared error" além de aplicar essa ideia na atividade anterior.

### 11.1) Requisitos sugeridos

Atividade 10.

### 11.2) Descrição

Métricas de Regressão
- [Regression Metrics Review I - How to Win a Data Science Competition Learn from Top Kagglers](https://www.bilibili.com/video/BV117411y7Fa?p=30) (15min)
- [Regression Metrics Review II - How to Win a Data Science Competition Learn from Top Kagglers](https://www.bilibili.com/video/BV117411y7Fa?p=31) (9min)

Obs: Os vídeos dessa seção são de uma plataforma de compartilhamento de vídeos asiática pois os originais, que estavam no Coursera, estão indisponíveis em decorrência das sansões aplicadas a Rússia pela guerra na Ucrância. Em particular, no Coursera os cursos originados de universidades russas foram indiponibilizados.

No curso introdutório de Aprendizado de Máquina do Kaggle Learn que você fez, ideias iniciais de validação de modelo com um conjunto hold out foram apresentadas. Um exercício interessante é aplicar essas mesma ideia na atividade anterior (10) separando seu conjunto em uma parte para treino e outra para teste.
- [Training and testing - Machine Learning for Trading](https://youtu.be/P2NqrFp8usY) (3min)
- Escolha algumas das [métricas discutidas que estejam disponíveis no scikit-learn](https://scikit-learn.org/stable/modules/classes.html#regression-metrics) e veja como ela se comporta (tanto no conjunto de treino quanto no conjunto de teste) variando o valor de `degree` da sua implementação.
- [Fundamentos de Aprendizagem de Máquina: Viés e Variância - StatQuest](https://youtu.be/EuBBz3bI-aA) (7min)
- Veremos depois com mais detalhe discussões sobre viés/variância e underfitting/overfitting, mas tente pensar o que acontece com o modelo polinomial quando mudamos o valor de `degree`. Para quais valores de `degree` temos underfitting e para quais valores temos `overfitting`?

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 12) Introdução ao Pandas

Pandas é a biblioteca de manipulações mais utilizada para estruturar seus dados em Python. Aliada ao Spark e ao SQL, você terá um stack muito robusto para as diferentes tarefas e cenários de manipulação de dados. O Pandas é, talvez, a mais importante na prática e masterizar ela ajudará a aprender as outras com mais facilidade.

### 12.1) Requisitos sugeridos

Atividade 5.

### 12.2) Descrição

Algumas referências. Se estiver muito redundante, pode pular alguma(s) das sugestões.
- [Kaggle Learn - Intro to Pandas](https://www.kaggle.com/learn/pandas) (4h).
- [Pandas for Data Science in 20 Minutes : Python Crash Course](https://www.youtube.com/watch?v=tRKeLrwfUgU) (23 min) 
- [Complete Python Pandas Data Science Tutorial! (Reading CSV/Excel files, Sorting, Filtering, Groupby)](https://youtu.be/vmEHCJofslg)  (~1 hora) 
- [Python pandas — An Opinionated Guide](https://youtube.com/playlist?list=PLgJhDSE2ZLxaENZWWF_VOUa5886KiUd15) (~2 horas)

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 13) Aula de Regressão Logística da especialização (básica) de ML do Andrew

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
    - Lecture 6.7 — Logistic Regression : MultiClass Classification OneVsAll (6min)

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 14) Métricas de classificação

Assim como no caso de regressão, existem maneiras de avaliar a qualidade do nosso modelo no problema de classificação.

### 14.1) Requisitos sugeridos

Atividade 13.

### 14.2) Descrição

- [The 5 Classification Evaluation metrics every Data Scientist must know](https://towardsdatascience.com/the-5-classification-evaluation-metrics-you-must-know-aa97784ff226)
- [Classification Metrics Review - How to Win a Data Science Competition Learn from Top Kagglers](https://www.bilibili.com/video/BV117411y7Fa?p=32)

Obs: Os vídeos dessa seção são de uma plataforma de compartilhamento de vídeos asiática pois os originais, que estavam no Coursera, estão indisponíveis em decorrência das sansões aplicadas a Rússia pela guerra na Ucrância. Em particular, no Coursera os cursos originados de universidades russas foram indiponibilizados.

- Uma sugestão um pouco mais rigorosa é o [capítulo sobre métricas de classificação do livro do DataLab](https://pibieta.github.io/imbalanced_learning/notebooks/Metrics%201%20-%20Intro%20%26%20ROC%20AUC.html#)

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 15) Árvores de Decisão e Regressão

Algoritmos baseados em árvores são os mais utilizados, de maneira geral, para dados tabulares. Entender bem o caso base ajudará a compreender as maneiras mais robustas de utilizá-las (quando fazemos cômites).

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

TODO

### 16.1) Requisitos sugeridos

Atividade 15.

### 16.2) Descrição

- [Bootstrapping Main Ideas!!! - StatQuest](https://youtu.be/Xz0x-8-cgaQ) (9min)
- [StatQuest: Random Forests Parte 1 - Construindo, Usando e Avaliando - StatQuest](https://youtu.be/J4Wdy0Wc_xQ) (10min)
- [Random Forest na Prática (Scikit-learn / Python) - Mario Filho](https://youtu.be/RtA1rjhuavs) (25min)
- [Out-of-Bag Error (OOB Error) - No Learning](https://youtu.be/FUsCBp_4UwE)
- [Random Forest - Feature Importance - I2ML](https://youtu.be/cw4qG9ePZ9Y) (8min)
- [Permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html) (Obs: Não só aplicável às Random Forests)
- [Random Forest - Machine Learning Interview Q&A for Data Scientists - Data Science Interviews - Emma Ding](https://youtu.be/vc88wyUz5jw)

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 17) Análise Exploratória de Dados

Na prática, antes da modelagem, é muito importante entender quais variáveis o cientista tem disponíveis para criação do modelo. Nem sempre o problema está bem empacotado e a definição do que você quer modelar (e quais métricas otimizar) pode vir de um bom entendimento dos dados disponíveis conjuntamente com alinhamentos com a área de negócios interessada.

- Pensar em **hipóteses de negócio que você gostaria de validar**, são uma ótima maneira de fazer uma análise exploratória. Será que você já tem alguma intuição sobre o problema que pode, inclusive, te ajudar depois a modelar o problema de uma forma diferente?
    - Num problema de inadimplência de crédito (previsão se uma pessoa vai pagar ou não uma dívida), podemos, por exemplo, ter a intuição de que, para pessoas de baixa renda, a presença ou não de uma dívida não quitada anterior pode ser crucial para a pessoa pagar o próximo empréstimo de interesse. Enquanto para pessoas de alta renda, isso é menos importante. Esse é um tipo de pergunta que você pode diretamente testar e essa hipótese pode se tornar uma regra de negócio que vira um benchmark que você gostaria de bater depois com seu modelo. Será que essa regra sozinha já tem uma performance boa o suficiente que nem justifica a criação de um modelo?
    - Ou ainda, a sua hipótese pode te ajudar a entender onde está o maior foco de interesse da sua modelagem. Por exemplo, se você tem o valor da dívida, pode agrupar seus dados para identificar quais os grupos que "se mal identificados pelo modelo" podem trazer maior prejuízo. Imagine que, apesar de pessoas de alta renda serem minoria da sua base de desenvolvimento (10%, por exemplo), estão associadas com 90% do valor total em empréstimo. O seu modelo identificar bem os maus pagadores nesse grupo pode ser muito mais importante que identificar maus pagadores de forma geral na população. Isso poderia te guiar na etapa de modelagem e avalição a segmentar suas métricas, usar `sample_weights` para valorizar grupos de interesse e até quebrar seu modelo por renda, por exemplo.

- Estudar, com alguns gráficos e estatísticas, sua amostra de desenvolvimento te dá insights sobre **quais são as variáveis que estão relacionadas com o problema e alguns possíveis feature engineering** que você pode criar, além de **identificar problemas que você precisará resolver** para ter uma modelagem adequada.
    - É muito comum tentar filtrar algumas variáveis nessa etapa baseado em alguma medida de correlação. Apesar de eu achar que isso é feito melhor mais para frente no pipeline de desenvolvimento, pode ser necessário fazer isso durante a exploração se o problema tiver dezenas de colunas e o tempo for curto, nor obrigando a focar apenas nas de maior interesse. Rodar correlações simples ou algoritimos que metrificam a importância das variáveis (como um algoritmo baseado em árvores) podem ser super úteis aqui, mas devem ser usados com cuidado.
    - Saber quais variáveis você tem, alinhado com entendimento de negócios pode te fazer pensar em variáveis (muitas vezes feitas apenas com operações matemáticas triviais como a soma) que simplificam a vida do modelo (principalmente árvores que só conseguem fazer cortes paralelos aos eixos). Por exemplo, uma variável super importante em crédito é o "comprometimento de renda", ou seja, qual a porcentagem do salário que precisaria ficar "reservado" para pagar a parcela de um financiamento ativo (`valor_parcela/valor_salario`). Criar ela nessa etapa e fazer alguns gráficos para avaliá-la pode ser considerada uma atividade exploratória útil.
    - É muito comum ter bases com problemas de valores vazios ou, pior, mal preenchidos. Se você sabe que determinada variável só pode estar em um certo range e você encontra valores fora desse range, você precisa entender porque isso aconteceu. Analisar criticamente os valores presentes pode te ajudar a traçar uma estratégia para tratá-los ou possivelmente dropá-los (com o cuidado que dropar linhas no teste só deve ser feita se de fato aquilo não for visto na vida real, sem trapaças). (_Num problema de Kaggle/portifólio raramente você tem essa resposta, mas em uma empresa, esse pode ser o momento em que você conversa com a engenharia e entende se, na utilização do modelo, o dado vai estar de qual forma etc._)

- Além disso, esse entendimento te guia a escolher modelos que são mais apropriados para o tipo de dados que você têm.
    - Por exemplo, escolhendo modelos que nativamente lidem bem com variáveis categóricas (como o CatBoost).
    - Pensando em valores faltantes de dados, dependendo do caso que você está ([MCAR, MAR ou MNAR](https://youtu.be/YpqUbirqFxQ)), você vai definir possíveis estratégias de imputação nesse momento (que podem ser testadas, depois, na sua otimização de hiperparâmetros) ou, mesmo, se quer utilizar um modelo que lida nativamente com esse tipo de dados faltante (como o lightGBM).

- Por fim, você pode explorar tendências temporais nos dados. Aprendizado de Máquina trabalha com o pressuposto de que os dados são estáveis ao longo do tempo, o que não necessariamente é uma realidade. Em muitas aplicações é importante quebrar seus dados respeitando a lógica temporal e essa parte da análise exploratória pode te contar se isso é importante no problema em questão ou não. 

Opinião pessoal:

- Pensando principalmente em aprendizado supervisionado, os pontos anteriores (validação de hipóteses de negócio, estudos sobre variáveis relevantes, criação de variáveis novas, identificação de problemas e questões relacionadas) são os principais resultados de um EDA e devem ser o foco da sua exploração. Fazer gráficos "por fazer", que não trazem informação relevante, devem ser evitados a todo custo uma vez que deixam sua análise prolixa e sem o foco de interesse, que é o modelo final.

- Facilmente você pode se pegar saindo de algum desses objetivos anteriores, mas vale sempre se questionar: "porque estou fazendo esse gráfico ou calculando essa estatística?" Se o motivo está claro para você, então provavelmente fazer tal gráfico/cálculo faz sentido.

- Posso estar sendo um pouco duro com as análises exploratórias de forma geral, mas é muito comum, principalmente em DSs no início de carreira (leia-se cases de entrevistas e projetos de portifólio), notebooks enormes com inúmeros gráficos e `.head()` de tabelas que deixam o código díficil de navegar e que nada acrescentam no processo de modelagem ou na história contada. Em 99% das vezes parecem completamente desconectados do problema de interesse e se, excluídos do notebook, nada mudariam no outcome.

- Em outros casos, em que não há necessariamente um modelo envolvido, se extender na análise exploratória pode fazer sentido, mas raramente eu fugiria de alguma das motivações anteriores.

### 17.1) Requisitos sugeridos

Atividades 12.

### 17.2) Descrição

TODO: Achar kerneis legais.

- [Exemplo de notebook que faz uma análise exploratória focada procurar erros nos dados](https://github.com/vitaliset/projetos-de-estudo/blob/main/New%20York%20City%20Taxi%20Fare%20Prediction/1_procurando_erros.ipynb).

Extras:
- Iremos discutir aspectos de validação out-of-time no futuro, mas esse notebook discuti uma análise exploratória para estudo de estabilidade temporal com uma [técnica interessante](https://vitaliset.github.io/covariate-shift-2-classificador-binario/):
    - [Exemplo de notebook que faz uma análise exploratória focada em problemas de drift temporal dos dados](https://github.com/vitaliset/projetos-de-estudo/blob/main/New%20York%20City%20Taxi%20Fare%20Prediction/2_dinamica_temporal.ipynb).

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 18) Kaggle Challenge

TODO

### 18.1) Requisitos sugeridos

Atividades 16 e 17.

### 18.2) Descrição

Escolher um dataset do Kaggle e limpar os dados + aplicar um modelo de ML, avaliando os resultados (qual métrica utilizar pensando no problema que estou preocupado em resolver?)

- [Find Open Datasets and Machine Learning Projects - Kaggle](https://www.kaggle.com/datasets)
- Sugestão: [Heart Attack Analysis & Prediction Dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset)

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 19) Aula sobre dicas práticas da especialização (básica) de ML do Andrew

TODO: Discussão em alto nível sobre validação, um pouco do ciclo de vida do desenvolvimento de um modelo de Aprendizado de Máquina Supervisionado, MLOps e Ética.

### 19.1) Requisitos sugeridos

Atividade 18.

### 19.2) Descrição

Infelizmente, a partir do segundo curso do Andrew, é necessário se inscrever pelo Coursera porque os vídeos não estão disponíveis no YouTube. Não se preocupe, o conteúdo continua gratuíto se você se inscrever como **ouvinte**.

- [Advanced Learning Algorithms by Andrew Ng](https://www.coursera.org/learn/advanced-learning-algorithms?specialization=machine-learning-introduction)
- Semana 3: Advice for applying machine learning
    - Deciding what to try next (3min)
    - Evaluating a model (10min)
    - Model selection and training/cross validation/test sets (13min) - Obs1*
    - Diagnosing bias and variance (11min)
    - Regularization and bias/variance (10min)
    - Establishing a baseline level of performance (9min) - Obs2*
    - Learning curves (11min)
    - Deciding what to try next revisited (8min)
    - Bias/variance and neural networks (10min) - Pode pular ou assistir apenas "por cima" por ser específico de Redes Neurais
    - Iterative loop of ML development (7min)
    - Error analysis (8min)
    - Adding data (14min)
    - Transfer learning: using data from a different task (11min) - Pode pular ou assistir apenas "por cima" por ser específico de Redes Neurais
    - Full cycle of a machine learning project (8min)
    - Fairness, bias, and ethics (9min)
    - Error metrics for skewed datasets (11min)
    - Trading off precision and recall (11min)

- Obs1*: O Andrew usa o termo "cross validation" de uma forma diferente do tradicional, que costuma ser usada quando estamos falando do k-fold ou alguma variação dele (como o [`sklearn.model_selection.StratifiedKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html) ou o [`sklearn.model_selection.TimeSeriesSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html#sklearn.model_selection.TimeSeriesSplit)). O conjunto que ele se refere é comumente chamado de conjunto de validação (enquanto o último conjunto é chamado de conjunto de teste).
    - [Fundamentos de aprendizado de máquina: Validação Cruzada - StatQuest](https://youtu.be/fSytzGwwBVw)
- Obs2*: O seu baseline normalmente é um algoritmo mais simples (ou o modelo que já existe) numa aplicação real.

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 20) Aula de árvores de decisão da especialização (básica) de ML do Andrew

TODO

### 20.1) Requisitos sugeridos

Atividade 19.

### 20.2) Descrição

Infelizmente, a partir do segundo curso do Andrew, é necessário se inscrever pelo Coursera porque os vídeos não estão disponíveis no YouTube. Não se preocupe, o conteúdo continua gratuíto se você se inscrever como **ouvinte**.

- [Advanced Learning Algorithms by Andrew Ng](https://www.coursera.org/learn/advanced-learning-algorithms?specialization=machine-learning-introduction)
- Semana 4: Decision trees
    - Decision tree model (7min)
    - Learning Process (11min)
    - Measuring purity (7min)
    - Choosing a split: Information Gain (11min)
    - Putting it together (9min)
    - Using one-hot encoding of categorical features (5min)
    - Continuous valued features (6min)
    - Regression Trees (optional) (9min)
    - Using multiple decision trees (3min)
    - Sampling with replacement (3min)
    - Random forest algorithm (6min)
    - XGBoost (6min)
    - When to use decision trees (6min)

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 21) Curso de intermediário de ML do Kaggle Learn

TODO

### 21.1) Requisitos sugeridos

Atividade 20.

### 21.2) Descrição

- [Kaggle Learn - Intermediate Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning) (4h).
- [Validação out of time - Experian (Serasa)](https://www.experian.com/blogs/insights/2018/06/understanding-validation-samples-within-model-development/).
- [Diagrama de validação out-of-space e out-of-time - Documentação do fklearn - Nubank](https://fklearn.readthedocs.io/en/latest/examples/fklearn_overview.html?highlight=out-of-ID#Spliting-the-dataset-into-train-and-holdout).

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 22) Otimização de hiperparâmetros

TODO

### 22.1) Requisitos sugeridos

Atividade 21.

### 22.2) Descrição

- [Hyperparameter optimization - Wikipedia](https://en.wikipedia.org/wiki/Hyperparameter_optimization)
- [Hyperparameter Optimization - The Math of Intelligence #7 - Siraj Raval](https://youtu.be/ttE0F7fghfk) (10min). Ele é meio impreciso em alguns poucos momentos não muito relevantes e entra em alguns tópicos que não são essenciais, mas é interessante para ver a ideia geral.
- [Nunca Mais Use Grid Search Para Ajustar Hiperparâmetros - Mario Filho](https://youtu.be/WhnkeasZNHI) (32min)

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 23) Curso de feature engineering do Kaggle Learn

TODO

### 23.1) Requisitos sugeridos

Atividade 21.

### 23.2) Descrição

- [Kaggle Learn - Feature Engineering](https://www.kaggle.com/learn/feature-engineering) (5h).
- [Aula avulsa do curso de Data Cleaning sobre dataframes com colunas do tipo data](https://www.kaggle.com/code/alexisbcook/parsing-dates) (1h).
- [One-Hot, Label, Target e K-Fold Target Encoding, claramente explicados!!! - StatQuest](https://youtu.be/589nCGeWG1w)
<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 24) Curso de explicabilidade do Kaggle Learn

TODO

### 24.1) Requisitos sugeridos

Atividade 21.

### 24.2) Descrição

- [Kaggle Learn - Machine Learning Explainability](https://www.kaggle.com/learn/machine-learning-explainability) (4h).
- [SHAP Values Explained Exactly How You Wished Someone Explained to You - Samuele Mazzanti](https://towardsdatascience.com/shap-explained-the-way-i-wish-someone-explained-it-to-me-ab81cc69ef30)

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 25) Curso de ética do Kaggle Learn

TODO

### 25.1) Requisitos sugeridos

Atividade 21.

### 25.2) Descrição

- [Kaggle Learn - Intro to AI Ethics](https://www.kaggle.com/learn/intro-to-ai-ethics) (4h).

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 26) Aula aprendizado não supervisionado da especialização (básica) de ML do Andrew

TODO

### 26.1) Requisitos sugeridos

Atividade 21.

### 26.2) Descrição

Infelizmente, a partir do segundo curso do Andrew, é necessário se inscrever pelo Coursera porque os vídeos não estão disponíveis no YouTube. Não se preocupe, o conteúdo continua gratuíto se você se inscrever como **ouvinte**.

- [Unsupervised Learning, Recommenders, Reinforcement Learning](https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning)
- Semana 1: Unsupervised learning
    - What is clustering? (4min)
    - K-means intuition (6min)
    - K-means algorithm (9min)
    - Optimization objective (11min)
    - Initializing K-means (8min)
    - Choosing the number of clusters (7min)
    - Finding unusual events (11min)
    - Gaussian (normal) distribution (10min)
    - Anomaly detection algorithm (11min)
    - Developing and evaluating an anomaly detection system (11min)
    - Anomaly detection vs. supervised learning (8min)
    - Choosing what features to use (14min)

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 27) Algoritmos baseados em distância

TODO

### 27.1) Requisitos sugeridos

Atividade 26.

### 27.2) Descrição

- [K-nearest neighbors, Clearly Explained - StatQuest](https://youtu.be/HVXime0nQeI) (5min)
- [K-means clustering = StatQuest](https://youtu.be/4b5d3muPQmA) (9min)
- [Chapter 3 - GEOMETRY AND NEAREST NEIGHBORS - A Course in Machine Learning by Hal Daumé III](http://ciml.info/)
- [Generalizando distância - Carlo Lemos](https://vitaliset.github.io/distancia/)

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 28) Introdução ao GitHub

TODO

### 28.1) Requisitos sugeridos

Não há.

### 28.2) Descrição

- [GitHub - Guia Completo do Iniciante - Felipe - Dev Samurai](https://youtu.be/UbJLOn1PAKw) (22min)
- [Introdução ao GitHub - Treinamento Microsoft](https://learn.microsoft.com/pt-BR/training/modules/introduction-to-github/)
- [Learn Git Branching](https://learngitbranching.js.org/?locale=pt_BR)

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line2.png"></center></p>

# Discussões extras e alguns tópicos de estudo individual pós tutoria

Essa é uma lista (não exaustiva) de temas interessantes que podem ser usados para se aprofundar (para mentorados que já estão mais avançados) após o ciclo idealizado. Aqui a escolha deve ser do mentorado sobre os assuntos pensando também no tempo restante e na relevância dos assuntos.

## SQL
- [Kaggle Learn - Intro to SQL](https://www.kaggle.com/learn/intro-to-sql)
- [Kaggle Learn - Advanced SQL](https://www.kaggle.com/learn/advanced-sql)

## Algoritmos Clássicos de ML que não foram vistos anteriormente, mas são importantes
- Supervisionado
    - [StatQuest: Linear Discriminant Analysis (LDA) clearly explained](https://youtu.be/azXCzI57Yfc)
    - [Naive Bayes, Clearly Explained!!!](https://youtu.be/O2L2Uv9pdDA)
    - [Gaussian Naive Bayes, Clearly Explained!!!](https://youtu.be/H3EjCKtlVog)
    - [Support Vector Machines, Clearly Explained!!!](https://youtu.be/efR1C6CvhmE)
    - [Support Vector Machines Part 2: The Polynomial Kernel (Part 2 of 3)](https://youtu.be/Toet3EiSFcM)
    - [Support Vector Machines Part 3: The Radial (RBF) Kernel (Part 3 of 3)](https://youtu.be/Qc5IyLW_hns)
- Redução de Dimensionalidade
    - [StatQuest: Ideias principais de PCA em somente 5 minutos!!!](https://youtu.be/HMOI_lkzW08)
    - [StatQuest: Principal Component Analysis (PCA), Step-by-Step](https://youtu.be/FgakZw6K1QQ)
    - [StatQuest: MDS and PCoA](https://youtu.be/GEn-_dAyYME)
    - [StatQuest: t-SNE, Clearly Explained](https://youtu.be/NEaUSP4YerM)
- Clustering
    - [StatQuest: Hierarchical Clustering](https://youtu.be/7xHsRkOdVwo)
    - [Clustering with DBSCAN, Clearly Explained!!!](https://youtu.be/RDZUdRSDOok)

## eXplainable AI
- [You are underutilizing SHAP values: understanding populations and events - Estevão Uyrá](https://towardsdatascience.com/you-are-underutilizing-shap-values-understanding-populations-and-events-7f4a45202d5)
- [xAI - Fabrício Olivetti - UFABC](https://www.youtube.com/@ufabchal/playlists?view=1&sort=dd&shelf_id=0)
- [The Science Behind InterpretML: Explainable Boosting Machine](https://youtu.be/MREiHgHgl0k)

## Cursos de Aprendizado de Máquina com um pouco mais de rigor
- [Introduction to Statistical Learning](https://youtube.com/playlist?list=PLOg0ngHtcqbPTlZzRHA2ocQZqB1D_qZ5V)
- [Learning from Data - Yaser Abu Mostafa - Caltech](https://youtube.com/playlist?list=PLnIDYuXHkit4LcWjDe0EwlE57WiGlBs08)

## Boosting Trees
- [Bagging vs Boosting - Ensemble Learning In Machine Learning Explained - WhyML](https://youtu.be/tjy0yL1rRRU)
- [AdaBoost, Clearly Explained](https://youtu.be/LsK-xG1cLYA)
- [Gradient Boost playlist - StatQuest](https://youtube.com/playlist?list=PLblh5JKOoLUJjeXUvUE0maghNuY2_5fY6)
- [XGBoost playlist - StatQuest](https://youtube.com/playlist?list=PLblh5JKOoLULU0irPgs1SnKO6wqVjKUsQ)
- [XGBoost vs LightGBM: How Are They Different - neptune.ai/blog](https://neptune.ai/blog/xgboost-vs-lightgbm)
- [Ensemble: Boosting, Bagging, and Stacking - Machine Learning Interview Q&A for Data Scientists - Emma Ding](https://youtu.be/sN5ZcJLDMaE)
- [Gradient Boosting (GBM) and XGBoost - Machine Learning Interview Q&A for Data Scientists - Emma Ding](https://youtu.be/yw-E__nDkKU)

## MLOps
- [Curso Machine Learning Engineering for Production (MLOps) by Andrew Ng](https://youtube.com/playlist?list=PLkDaE6sCZn6GMoA0wbpJLi3t34Gd8l0aK)
- Monitoramento de Modelos
    - [ML Drift: Identifying Issues Before You Have a Problem - Fiddler AI](https://youtu.be/uOG685WFO00)
    - [Monitoramento de modelos: por quê monitorar, boas práticas e aprendizados - Nubank ML Meetup](https://youtu.be/iAiY9L47eak)

## Fairness
- [A Justiça da Sociedade Algorítmica - Ramon Vilarino](https://youtu.be/iBfpL0aA7w4)
- [Fairness em Machine Learning - Nubank ML Meetup - Tatyana Zabanova](https://youtu.be/LWt4LZmpasc)
- [Dealing with bias and fairness in AI systems - Pedro Saleiro](https://youtu.be/yqLzzoBYDRM)

## Redes Neurais (Deep Learning)
- [Kaggle Learn - Intro to Deep Learning](https://www.kaggle.com/learn/intro-to-deep-learning)
- [Advanced Learning Algorithms by Andrew Ng](https://www.coursera.org/learn/advanced-learning-algorithms?specialization=machine-learning-introduction)
    - Semana 1: Neural Networks
    - Semana 2: Neural network training
- [Deep Learning Specialization by Andrew Ng](https://www.coursera.org/specializations/deep-learning)
    - [Course 1 - Neural Networks and Deep Learning](https://youtube.com/playlist?list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0)
    - [Course 2 - Improving Deep Neural Networks: Hyperparameter Tuning, , Regularization and Optimization](https://youtube.com/playlist?list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc)
    - [Course 3 -Structuring Machine Learning Projects](https://youtube.com/playlist?list=PLkDaE6sCZn6E7jZ9sN_xHwSHOdjUxUW_b)
    - [Course 4 - Convolutional Neural Networks](https://youtube.com/playlist?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF)
    - [Course 5 - Sequence Models](https://youtube.com/playlist?list=PLkDaE6sCZn6F6wUI9tvS_Gw1vaFAx6rd6)
- [Transformers with Hugging Face](https://huggingface.co/course/chapter1/1)
- [Tranformers doc - Hugging Face](https://huggingface.co/docs/transformers/index)

## Shallow NLP
- [Python Tutorial: re Module - How to Write and Match Regular Expressions (Regex) - Corey Schafer](https://youtu.be/K8L6KVGG-7o)
- [Modelo Bag of Words - IA Expert Academy](https://youtu.be/13R_-qLXvzA)
- [TF-IDF (Term Frequency - Inverse Document Frequency) - IA Expert Academy](https://youtu.be/RVx_QYZPGaU)
- [Remoção de stop words com Python e NLTK - IA Expert Academy](https://youtu.be/og-t1HLey7I)
- [Stemming com NLTK e Python - IA Expert Academy](https://youtu.be/kTlYIuveYJE)
- [Word2Vec - Skipgram and CBOW - The Semicolon](https://youtu.be/UqRCEmrv1gQ)

## Sistemas de Recomendação (RecSys)
- [Unsupervised Learning, Recommenders, Reinforcement Learning by Andrew Ng](https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning?specialization=machine-learning-introduction#syllabus)
    - Semana 2: Recommender systems

## Aprendizado por Reforço
- [Multi-Armed Bandit - Matheus Facure](https://matheusfacure.github.io/2017/03/04/bernoulli-bandits-thompson/)
- [Multi-Armed Bandit - Guilherme Marmerola](https://gdmarmerola.github.io/ts-for-bernoulli-bandit/)
- [Introdução ao Aprendizado por Reforço - Itau Data Science Meetup - Caique R. Almeida](https://youtu.be/zerTbKzNq1o)
- [Unsupervised Learning, Recommenders, Reinforcement Learning by Andrew Ng](https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning?specialization=machine-learning-introduction#syllabus)
    - Semana 3: Reinforcement learning
- [Deep Reinforcement Learning Course - Hugging Face](https://huggingface.co/deep-rl-course/unit0/introduction)

## Séries temporais com Aprendizado de Máquina
- [Multiple Time Series Forecasting With Scikit-Learn - Mario Filho](https://youtu.be/RRd2wzMRpOc)

## Robustez de Modelo
- [Como leakage de dados acaba com backtesting - Nubank ML Meetup - Tatyana Zabanova](https://youtu.be/qPYqeD2OUl4)
- [Generalização de domínio, invariância e a Floresta Temporalmente Robusta - Nubank ML Meetup - Luis Moneda](https://youtu.be/Gq20DI9punw)
- [Train/test mismatch e adaptação de domínio - A Course in Machine Learning by Hal Daumé III](http://ciml.info/dl/v0_99/ciml-v0_99-ch08.pdf)

## Inferência Causal
- [Causal Inference for The Brave and True - Matheus Facure](https://matheusfacure.github.io/python-causality-handbook/landing-page.html)

## Feature Selection
- [Feature Selection - Machine Learning Interview Q&A for Data Scientists - Data Science Interviews - Emma Ding](https://youtu.be/7tW29jBceRw)
- [Feature Selection Usando Scikit-Learn - Mario Filho](https://www.youtube.com/live/Bcn5e7LYMhg?feature=share)
- [Como Remover Variáveis Irrelevantes de um Modelo de Machine Learning - Mario Filho](https://www.youtube.com/live/6-mKATDSQmk?feature=share)
- [Seleção de variáveis: uma utilização crítica do Boruta - Carlo Lemos](https://vitaliset.github.io/boruta/)

## Python
- [Intermediate Python in 6 hours - freeCodeCamp.org](https://youtu.be/HGOBQPFzWKo)
- [Python Programming Beginner Tutorials - Corey Schafer](https://youtube.com/playlist?list=PL-osiE80TeTskrapNbzXhwoFUiLCjGgY7)
- [Python Tutorial: Iterators and Iterables - What Are They and How Do They Work? - Corey Schafer](https://youtu.be/jTYiNjvnHZY)
- [Python Tutorial: Itertools Module - Iterator Functions for Efficient Looping - Corey Schafer](https://youtu.be/Qu3dThVy6KQ)
- [8 Design Patterns EVERY Developer Should Know](https://youtu.be/tAuRQs_d9F8)
- [Software Design in Python - ArjanCodes](https://youtube.com/playlist?list=PLC0nd42SBTaNuP4iB4L6SJlMaHE71FG6N)
- Write better Python code - ArjanCodes
    - [Part 1: Cohesion and coupling](https://youtu.be/eiDyK_ofPPM)
    - [Part 2: Dependency inversion](https://youtu.be/Kv5jhbSkqLE)
    - [Part 3: The strategy pattern](https://youtu.be/WQ8bNdxREHU)
    - [Part 4: The observer pattern](https://youtu.be/oNalXg67XEE)
    - [Part 5: Unit testing and code coverage](https://youtu.be/jmP3fp_BhmE)
    - [Part 6: Template method and bridge](https://youtu.be/t0mCrXHsLbI)
    - [Part 7a: Exception handling](https://youtu.be/ZsvftkbbrR0)
    - [Part 7b: Monadic error handling](https://youtu.be/J-HWmoTKhC8)
    - [Part 8: Software architecture](https://youtu.be/ihtIcGkTFBU)
    - [Part 9: SOLID principles](https://youtu.be/pTB30aXS77U)
    - [Part 10: Object creation patterns](https://youtu.be/Rm4JP7JfsKY)
- Programação Funcional
    - [Introdução aos paradigmas da Programação - Itaú Data Science Meetup - Clarissa David](https://youtu.be/XDJ28JZws_I)
    - [returns](https://returns.readthedocs.io/en/latest/index.html)
    - [functools](https://docs.python.org/3/library/functools.htmll)
    - [toolz](https://toolz.readthedocs.io/en/latest/)
- Flask
    - [Flask Tutorials - Corey Schafer](https://youtube.com/playlist?list=PL-osiE80TeTs4UjLw5MM6OjgkjFeUxCYH)
- Streamlit
    - [Streamlit: Criando aplicações web - Eduardo Mendes](https://www.youtube.com/live/Ie5ef_R_k6I?feature=share)
- PySpark
    - [PySpark Tutorial - freeCodeCamp.org](https://youtu.be/_C8kWso4ne4)

## Conformal Prediction
- [Distribution-Free Uncertainty Quantification](https://youtube.com/playlist?list=PLBa0oe-LYIHa68NOJbMxDTMMjT8Is4WkI)
- [“MAPIE” Explained Exactly How You Wished Someone Explained to You - Samuele Mazzanti](https://towardsdatascience.com/mapie-explained-exactly-how-you-wished-someone-explained-to-you-78fb8ce81ff3)

## Análise de Sobrevivência
- [Análise de sobrevivência - Nubank ML Meetup - Jessica De Sousa](https://youtu.be/WZNmlT-arF0)
- [Introduction to Survival Analysis - Pablo Ibieta](https://medium.com/datalab-log/survival-analysis-in-python-5aa9af04318c)

## Calibração de Probabilidade
- [Você deve calibrar seu modelo desbalanceado (ou não?): um tutorial completo - Alessandro Gagliardi](https://medium.com/datalab-log/você-deve-calibrar-seu-modelo-desbalanceado-ou-não-3f111160653a)

## Aprendizado Semissupervisionado
- [Uma Introdução ao Aprendizado Semissupervisionado (SSL) - Willian Dihanster](https://medium.com/datalab-log/uma-introdução-ao-aprendizado-semissupervisionado-ssl-9f2354314796)

## Inferência de Rejeitados
- [Counterfactual evaluation and Reject Inference - Nubank ML Meetup - Jessica De Sousa](https://youtu.be/CJeMJfUYwkM)

## Problem Solving
- [Resolução de Problemas - Nubank ML Meetup - Victor Goraieb](https://youtu.be/BB_w4e-NBek)

## Online Learning
- [Machine Learning Types - Batch Learning VS Online Learning - Rocketing Data Science](https://youtu.be/HTGCgErym1E)
- [Online Learning - Andrew Ng](https://youtu.be/dnCzy_XKGbA)
- [Awesome Online Machine Learning Repo - Max Halford](https://github.com/online-ml/awesome-online-machine-learning)
- [Read the docs of RiverML](https://riverml.xyz/)

## Quantum Machine Learning
- [Quantum Machine Learning e o classificador variacional quântico - Itau Data Science Meetup - André Juan](https://youtu.be/2XOfnq4niwQ)

## Análise de Algoritmos
- [Análise de algoritmos - Carla Quem Disse - UFABC](https://youtube.com/playlist?list=PLncEdvQ20-mgGanwuFczm-4IwIdIcIiha)

## Open Source
- Contribuindo com o scikit-learn
    - [Scikit-learn sprint instructions - Andreas Mueller](https://youtu.be/5OL8XoMMOfA)
    - [Sprint Instructions for scikit-learn Vol 2 - Andreas Mueller](https://youtu.be/p_2Uw2BxdhA)
    - [Contributing to scikit-learn: An Example Pull Request - Reshama Shaikh](https://youtu.be/PU1WyDPGePI)
    - [3 Components of Reviewing a Pull Request (scikit-learn) - Thomas Fan](https://youtu.be/dyxS9KKCNzA)
- [How to Contribute to Open Source for Data Scientists - Data Umbrella](https://youtube.com/playlist?list=PLBKcU7Ik-ir9Ol1MOQ5LVAseoH4IwqsFE)

## Discussões sobre gestão em DS
- [Data Science Leadership - Luis Moneda](https://datascienceleadership.com/)
- [Gestão de Cientistas de Dados - Uma Abordagem Heurística não Holística - Nubank ML Meetup - Eduardo Hruschka](https://youtu.be/0ELffU6j_Tk)
- [Painel sobre Liderança em times de Ciência de Dados - Nubank ML Meetup](https://youtu.be/FyUWiIh4yLE)

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line2.png"></center></p>

As linhas horizontais desse site foram feitos no [Silk](http://weavesilk.com/).
