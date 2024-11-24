---
layout: page
title: Roteiro para Mentores: Orientando iniciantes em carreiras de Ciência de Dados
permalink: /tutoring-guideline/
mathjax: true
---

_This page is a resource for mentors, offering a structured curriculum to guide beginners through their data science journey. It provides mentors with a clear roadmap and curated materials to effectively support their mentees in building foundational skills and advancing in the field._

_If you're curious, consider using your browser to translate this page from Portuguese to English and explore the content!_

___

Essa página sugere um roteiro de estudos em Ciência de Dados com foco em Aprendizado de Máquina. Ela abrange vários assuntos básicos e sugere continuações de estudo ou áreas de especialização que o mentorado pode escolher ao final.

Dependendo do nível técnico do mentorado no início da mentoria, várias etapas podem ser puladas. De forma geral, as atividades roteirizadas neste guia supõem um conhecimento equivalente ao ciclo básico de graduação em cursos da área de exatas, o qual detalharei com mais cuidado a seguir, dando ainda alternativas de cursos para quando esse pré-requisito não se aplica.

$\oint$ _Os materiais apresentados aqui foram selecionados com grande intersecção com os recursos que utilizei durante minha preparação para ingressar no mercado de ciência de dados. Posteriormente, muitos desses materiais foram empregados em mentorias que conduzi com profissionais em início de carreira._

## Pré-requisitos esperados do mentorado

Em uma conversa com o mentorado, é importante perceber se ele possui os pré-requisitos para seguir a sugestão de tópicos ou não. Caso contrário, diversas adaptações precisam ser feitas, principalmente no início, e já forneço algumas sugestões nesta seção.

- **Lógica de programação**: Não é esperado que o mentorado já tenha experiência com Python, mas é desejável que tenha familiaridade com outra linguagem de programação e esteja confortável com laços de repetição, condições e operadores lógicos, além de estruturas de dados básicas como listas, dicionários e tipos básicos como strings. Caso o mentorado não possua esse conhecimento, o curso de [introdução à programação do Kaggle](https://www.kaggle.com/learn/intro-to-programming) pode ser um bom ponto de partida. Se for necessário dedicar ainda mais tempo a esses tópicos, aulas selecionadas da [playlist de Python do Curso em Vídeo](https://youtube.com/playlist?list=PLvE-ZAFRgX8hnECDn1v9HNTI71veL3oW0) podem ser uma referência útil.

- **Estatística descritiva**: É esperado que o mentorado tenha familiaridade com visualizações clássicas, como gráficos de dispersão, gráficos de barras/colunas, histogramas, box plots, gráficos de pizza, além de medidas-resumo de posição e dispersão, como média, mediana, percentis, desvio padrão, entre outros. Dependendo do nível do mentorado, uma simples revisão superficial do assunto ([como nesta aula de análise descritiva da Univesp](https://youtu.be/42ArF0YCWm8)) pode ser suficiente. Em outros casos, pode ser necessário maior atenção, com o foco inicial das semanas dedicado ao ensino de Python por meio de exemplos de visualização de dados (por exemplo, utilizando o [Kaggle Learn de Data Visualization com seaborn](https://www.kaggle.com/learn/data-visualization)), em vez de começar diretamente com Aprendizado de Máquina.

- **Probabilidade**: A linguagem do Aprendizado de Máquina é intrinsecamente probabilística. Recomenda-se que o mentorado esteja confortável com conceitos básicos de probabilidade (pelo menos em um [nível introdutório](https://gradmat.ufabc.edu.br/disciplinas/ipe/)). Isso inclui compreender a definição (ingênua) de espaço de probabilidade, probabilidade condicional e independência, variáveis aleatórias discretas e contínuas, as principais distribuições de variáveis aleatórias, bem como saber calcular esperanças, variâncias, correlações, entre outros. Não é necessário dominar todos os detalhes, mas, caso o tema seja intimidador, é recomendada uma revisão desses tópicos, como no curso [Introdução à Probabilidade e Estatística (IPE) do professor Rafael Grisi da UFABC](https://www.youtube.com/channel/UCKGYUbUHWSDr8pFaDd7JWCg/videos).

- **Matemática de forma geral**: Além de estatística descritiva e probabilidade básica, outros tópicos matemáticos, como noções de otimização, cálculo diferencial, operações matriciais e estatística inferencial, são fundamentais para um entendimento mais aprofundado dos algoritmos de Aprendizado de Máquina. Se o mentorado possui uma base matemática frágil em algum desses tópicos (ou em todos), a especialização [Mathematics for Machine Learning and Data Science Specialization](https://www.coursera.org/specializations/mathematics-for-machine-learning-and-data-science#courses) pode ser uma excelente aliada, permitindo o estudo de aulas específicas ou uma imersão de algumas semanas nos fundamentos essenciais da Ciência de Dados.

## Ensinamentos do HDFS: redundância é importante

Durante este guia de materiais de mentoria, ao selecionar partes específicas de cursos diferentes, vários assuntos podem ser abordados repetidamente, mas com uma abordagem ligeiramente diferente. Isso é intencional. Acredito que, para fortalecer essa base da forma mais robusta possível, é importante que os assuntos sejam realmente absorvidos, e revê-los algumas vezes ajuda a reforçar esse aprendizado. :)

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line2.png"></center></p>

## 0) Como usar esse guia de atividades de tutoria?

Logo abaixo do título da atividade, há uma breve motivação. Ela serve para que o tutor entenda por que aquela atividade faz sentido naquele momento (e pode ser compartilhada com o mentorado, caso o tutor julgue importante explicar a tarefa).

### 0.1) Requisitos sugeridos

Os pré-requisitos servem para indicar as dependências sugeridas entre as atividades, caso se deseje seguir uma ordem diferente ou pular etapas.

### 0.2) Descrição

A descrição é o texto sugerido para ser passado ao aluno na ferramenta utilizada para registrar as atividades. Recomendo o uso do [Trello](https://trello.com/), uma ferramenta com estrutura no estilo Kanban, onde é possível organizar os cards em colunas (como "To Do", "In Progress" e "Done"). Essa estrutura é bastante semelhante à forma como ferramentas de organização de tarefas são utilizadas em muitas equipes de dados que empregam o Agile (ou alguma variação dele).

É aqui que o link para o material de estudo deve ser disponibilizado. De forma geral, as recomendações oferecem o conteúdo de forma gratuita, e os materiais estão majoritariamente em inglês, com alguns tópicos em português. Caso algum link esteja quebrado, por favor, entre em contato comigo para que eu possa verificar e corrigir. :)

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line2.png"></center></p>

# Cronograma 

<p><center><img src="{{ site.baseurl }}/assets/img/ordem_atividades.png"></center></p>

Concluída essa etapa de atividades, a sugestão é realizar algum desafio no Kaggle, levando em consideração o tempo restante da tutoria (considerando uma duração aproximada de 6 meses). Nesse momento, é importante compreender o problema de forma detalhada, e pode ser útil trabalhar em "releases", em que, a cada x semanas, uma versão mais estruturada do modelo seja desenvolvida, incluindo novos testes.

Obviamente, essas atividades não estão gravadas em pedra e podem ser ajustadas conforme o interesse do tutorado. Por exemplo, um tutorado mais próximo da área de saúde pode ter interesse em redes neurais para imagens, por exemplo, o que tornaria relevante a substituição de algumas atividades, especialmente nas semanas finais.

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line2.png"></center></p>

# Atividades recorrentes

As atividades recorrentes não são realizadas pontualmente nem finalizadas em sequência. São atividades que, uma vez atendidos os requisitos para sua execução, devem ser realizadas de forma contínua ao longo das semanas da tutoria.

### A) HackerRank

O HackerRank oferece diversas playlists interessantes que ensinam, por meio de exercícios, particularidades do Python. Você terá a oportunidade de conhecer estruturas de dados não triviais enquanto revisita as que já domina, aprendendo novas maneiras de trabalhar com elas. Ferramentas como essa são frequentemente utilizadas em processos seletivos, principalmente para posições mais ligadas à Engenharia de Software, como a de Machine Learning Engineer.

### A.1) Requisitos Sugeridos

Atividade 2.

### A.2) Descrição

Dedicar pelo menos 1 hora por semana às atividades propostas nas playlists de Python do [HackerRank](https://www.hackerrank.com/domains/python). Com pouco tempo de prática, você terá explorado a maioria dos tipos de estruturas relevantes, o que ajudará a tornar seu código mais _pythonic_.

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## B) Stratascratch

Pandas é a principal biblioteca de manipulação de dados do Python. Aliada ao SQL, será a principal ferramenta para tratar dados no dia a dia da ciência de dados. Você só dominará a sintaxe do Pandas se utilizá-lo de forma recorrente. A ideia aqui é apresentar alguns exercícios, mais ou menos clássicos, para se familiarizar com as manipulações principais.

### B.1) Requisitos sugeridos

Atividade 12.

### B.2) Descrição

Realizar pelo menos 30 minutos de atividades propostas utilizando o Pandas do [Stratascratch](http://stratascratch.com) toda semana. Você também pode resolvê-las em SQL, o que pode ser útil para treinar essa ferramenta eventualmente.

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line2.png"></center></p>

# Atividades Sequenciais

Sequência idealizada de atividades para o ciclo de estudos da mentoria.

## 1) Instalar Python (pelo Anaconda)

A ideia é preparar o ambiente para utilizar Jupyter Notebooks. Essas são ferramentas excelentes para exploração e, na prática, representam o ambiente mais adotado no dia a dia do cientista de dados para prototipação de código. Durante os encontros, é sempre útil apresentar os atalhos mais comuns e boas práticas, como deixar o notebook preparado para um "restart and run all" com resultados reproduzíveis (utilizando random states fixados). No entanto, abordar tudo isso agora pode acabar confundindo o mentorando. Eventualmente, ao final, podemos introduzir outras ferramentas, como IDEs, incluindo o VSCode, mas acredito que isso não seja útil no início.

### 1.1) Requisitos sugeridos

Não há.

### 1.2) Descrição

- [Instalando Python para Aprendizado de Máquina - LAMFO](https://lamfo-unb.github.io/2017/06/10/Instalando-Python/). Não é necessário instalar o TensorFlow nem o Git por enquanto.

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 2) Aula introdutória da especialização (básica) de ML do Andrew

Aqui, o didático Andrew apresenta os tipos de aprendizado e oferece exemplos interessantes. É uma ótima oportunidade para pedir ao mentorando que sugira exemplos de problemas onde esses conceitos podem ser aplicados, além de motivar discussões sobre aplicações menos triviais.

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

Python é **A LINGUAGEM** para Ciência de Dados: ela conta com muitas bibliotecas de alta qualidade já disponíveis e é amplamente adotada como padrão na esteira de produtização pelas empresas. O curso do Kaggle Learn é excelente porque foca nas partes mais úteis, especialmente considerando as principais bibliotecas de Aprendizado de Máquina. É um curso direcionado para quem está migrando de outra linguagem, ou seja, já parte do pressuposto de que o aluno tem conhecimento básico de lógica de programação.

### 3.1) Requisitos sugeridos

Atividade 1. Já assume conhecimento de programação básico em alguma linguagem.

### 3.2) Descrição 

- [Kaggle Learn - Python Tutorials](https://www.kaggle.com/learn/python) (5h).

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 4) Aula de Regressão Linear da especialização (básica) de ML do Andrew

A Regressão Linear é um dos algoritmos mais simples de Aprendizado de Máquina, sendo o terreno ideal para introduzir as principais ideias do aprendizado supervisionado, além de apresentar nomenclaturas e desenvolver a intuição. É essencial que essa atividade seja realizada com muita atenção e cuidado.

O curso do Andrew Ng inclui alguns notebooks auxiliares que podem ser interessantes de explorar, dependendo do perfil do mentorado. No entanto, eu, pessoalmente, não os considero indispensáveis, pois as aulas são bastante visuais e já explicam bem os conceitos apresentados.  Se for necessário, você pode baixar os notebooks em um repositório como este: [Machine-Learning-Specialization-Coursera](https://github.com/greyhatguy007/Machine-Learning-Specialization-Coursera). Caso esse repositório fique indisponível, será relativamente fácil encontrar outros semelhantes [pesquisando no Google](https://www.google.com/search?q=supervised+machine+learning%3A+regression+and+classification+notebooks+github&sxsrf=AJOqlzUQ11tr1y9XmW0QVpXNVUjS_8bIMg%3A1676862919336&ei=x-XyY6aQFOy81sQP1tGg4Ak&oq=Supervised+Machine+Learning%3A+Regression+and+Classification+notebooks&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQAxgAMgcIIxCwAxAnMgoIABBHENYEELADMgoIABBHENYEELADMgoIABBHENYEELADMgoIABBHENYEELADMgoIABBHENYEELADMgoIABBHENYEELADMgoIABBHENYEELADMgoIABBHENYEELADSgQIQRgAUABYAGDdC2gBcAF4AIABAIgBAJIBAJgBAMgBCcABAQ&sclient=gws-wiz-serp).

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

## 5) Introdução ao NumPy

NumPy é a principal ferramenta para manipulação de vetores e matrizes no Python, servindo como base para praticamente todas as outras bibliotecas importantes de Aprendizado de Máquina. Inicialmente, é fundamental compreender o básico dessa biblioteca e, com o tempo, buscar uma compreensão mais aprofundada para dominá-la.

### 5.1) Requisitos sugeridos

Atividade 3.

### 5.2) Descrição

- Numpy in 5 min: [Learn NUMPY in 5 minutes - BEST Python Library!](https://youtu.be/xECXZ3tyONo) (20 min)

- Se precisar ver mais, esse vídeo aqui também é interessante: [Complete Python NumPy Tutorial (Creating Arrays, Indexing, Math, Statistics, Reshaping)](https://youtu.be/GB9ByFAIAH4) (1 hora)

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 6) Implementing from ground up: Regressão Linear Simples

Essa atividade serve como uma oportunidade para praticar e consolidar tudo o que foi aprendido até agora.

### 6.1) Requisitos sugeridos

Atividades 4 e 5.

### 6.3) Descrição

- Utilizando Python (principalmente o NumPy), você deve construir uma função que receba seu conjunto de dados `X_train` e `y_train` (NumPy arrays) e retorne os pesos de uma regressão linear simples (ou seja, `X_train` é unidimensional), utilizando o gradiente descendente para realizar o cálculo, conforme abordado no curso do Andrew Ng. Você pode criar os conjuntos `X_train` e `y_train` como preferir, mas o código deve ser robusto o suficiente para permitir a troca dos valores sem comprometer o funcionamento.
- Defina critérios de parada que considerar apropriados para o gradiente descendente.
- Em seguida, com os pesos calculados, construa uma função que preveja os valores de `y` para um conjunto `X` qualquer.
- Pode ser interessante utilizar bibliotecas gráficas para visualizar o que está sendo feito. A mais famosa é o matplotlib. Nesta [playlist do Corey Schafer](https://youtube.com/playlist?list=PL-osiE80TeTvipOqomVEeZ1HRrcEvtZB_) há demonstrações de vários tipos de gráficos úteis.

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 7) Curso de básico de ML do Kaggle Learn

Exceto por problemas específicos, o dia a dia do cientista de dados não envolve criar modelos do zero. O scikit-learn é uma das bibliotecas mais robustas e amplamente utilizadas, com dezenas de modelos pré-construídos que seguem os melhores padrões de desenvolvimento de software. Além disso, conta com uma comunidade Open Source incrível que fornece suporte e orienta a evolução da biblioteca.

Este curso do Kaggle serve como uma introdução ao scikit-learn, sendo uma excelente oportunidade para aprender o padrão fit/predict, amplamente estabelecido e utilizado no campo de Aprendizado de Máquina.

### 7.1) Requisitos sugeridos

Atividades 4 e 5.

### 7.2) Descrição

- [Kaggle Learn - Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning) (3h).

- Extra: pode ser útil dar uma lida rápida na página de ["getting started" do scikit-learn](https://scikit-learn.org/stable/getting_started.html).

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 8) Introdução à Programação Orientada a Objetos (POO)

A orientação a objetos é o paradigma de programação principal do Python. A forma de abstração que ele nos oferece é muito poderosa, permitindo a construção de códigos complexos de maneira estruturada e reaproveitável, além de facilitar a manutenção. O objetivo desta atividade não é se tornar um mestre em POO, mas ter uma visão geral para saber que ela existe e compreender como o scikit-learn e outras bibliotecas do Python a utilizam. No futuro (provavelmente fora do escopo desta mentoria básica inicial), esse tópico pode ser revisitado, aprofundando-se no uso de heranças e boas práticas, como os princípios SOLID e os design patterns.

### 8.1) Requisitos sugeridos

Atividade 7.

### 8.2) Descrição

- [Python OOP Tutorial 1: Classes and Instances](https://youtu.be/ZDa-Z5JzLYM) (15 min)
- Tente criar um cenário simples em Python no qual você utiliza classes. Por exemplo, crie uma classe abstrata que represente a entidade "Cachorro" e que tenha dois atributos: "nome" e "raça". O cachorro também deve ter um método chamado "pedido_para_sentar", que recebe uma string. Se essa string for igual ao nome do cachorro, o método deve imprimir que o cachorro sentou.

- Extra: uma discussão sobre diferentes formas de se programar (paradigmas): [1.1 - Programação Funcional em Haskell: Paradigmas de Programação](https://youtu.be/CzGSaXbPFRA) (27 min) - O Python tem várias coisas bem úteis de programação funcional então é legal conhecer por cima as ideias também.

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 9) Aula de Regressão Linear Multivariada (e polinomial) da especialização (básica) de ML do Andrew

Na vida real, utilizamos dezenas, centenas ou até milhares de variáveis para realizar nossas previsões, e não apenas uma, como na regressão linear simples. Embora ainda seja um algoritmo relativamente simples, generalizar o caso da regressão linear simples já nos dá uma ideia de onde queremos chegar eventualmente. Além disso, nesta aula, o Andrew explica o conceito de vetorização de código (evite usar loops `for`!).

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

Se quiser saber mais sobre o que Andrew chama de "normal equation", que nada mais é do que a solução analítica dos pesos na regressão linear (em contraste com o método numérico iterativo aproximado fornecido pelo gradiente descendente):
- [Machine Learning — Andrew Ng, Stanford University [FULL COURSE]](https://www.youtube.com/playlist?list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN)
    - Lecture 4.6 — Linear Regression With Multiple Variables : Normal Equation - 16:18
    - Lecture 4.7 — Linear Regression With Multiple Variables : Normal Equation Non Invertibility - 5:59

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 10) Implementing from ground up: Regressão Polinomial + POO

A estrutura orientada a objetos do scikit-learn precisa se tornar sua aliada. O objetivo desta atividade é explorar um pouco a "caixa preta" dos estimadores do scikit-learn, implementando do zero o caso particular de regressão multivariada (quando as dimensões extras correspondem a potências da primeira componente, como Andrew explica em um dos vídeos da atividade 9). 

A ideia aqui é se familiarizar um pouco mais com a forma como o scikit-learn funciona, praticando POO.

### 10.1) Requisitos sugeridos

Atividades 4, 8 e 9.

### 10.2) Descrição

O objetivo desta atividade é estruturar, de forma mais elegante, o que você desenvolveu na atividade 8, encapsulando o código em uma classe no formato dos estimadores apresentados no curso do Kaggle sobre scikit-learn. Idealmente, boa parte do código anterior será reaproveitada nesta atividade.

- Você deve criar uma classe chamada `PolynomialRegression`, que recebe um parâmetro em sua inicialização chamado `degree`.
- Essa classe precisa conter dois métodos: `fit` e `predict`. O método `fit` deve receber duas entradas: `X` e `y`. O `X` deve atender à forma `X.shape = (n_samples, 1)` e o `y` deve atender à forma `y.shape = (n_samples,)`, onde `n_samples` é o número de amostras. Por exemplo, `X = np.array([[1, 2, 3]]).T` e `y = np.array([1, 2, 3])` satisfazem essas condições (apenas como exemplo; use outros valores quaisquer). O método `predict` deve receber apenas uma entrada, `X`, com as mesmas restrições de `.shape` descritas anteriormente.
- O método `fit` deve calcular os polinômios de `X` até o grau definido em `degree` (sugestão: utilize um [list comprehension](https://www.w3schools.com/python/python_lists_comprehension.asp) junto com [`np.hstack`](https://numpy.org/doc/stable/reference/generated/numpy.hstack.html)) e, em seguida, aplicar o gradiente descendente para encontrar os coeficientes dessa regressão linear multivariada. Isso deve ser uma generalização da função que você desenvolveu anteriormente para a regressão linear simples.
- Os coeficientes aprendidos durante o `fit` devem ser armazenados em atributos da classe com um **sufixo de underline** no nome. Essa é a convenção adotada pelo scikit-learn para guardar informações aprendidas durante o treinamento.
- Finalmente, o método `predict`, que recebe o `X`, deve aplicar a mesma transformação polinomial nesse novo `X` e realizar as multiplicações da regressão múltipla (algo como `X_poly * w + b`) para obter as previsões, que devem ser retornadas pela função.

- **Extra**: Pode ser interessante consultar como o [scikit-learn sugere a implementação de modelos](https://scikit-learn.org/dev/developers/develop.html). Não se preocupe com o que eles chamam de `BaseEstimator` e mixins, que é um assunto mais avançado de orientação à objetos.

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 11) Validação de modelos de regressão

Em relação à atividade prática, é de nosso interesse determinar, por exemplo, qual o melhor valor de `degree` a ser utilizado em um modelo. Para avaliar o desempenho do modelo e definir o que significa ser "melhor" ou "pior", assim como no caso da regressão linear, precisamos estabelecer uma métrica de avaliação. Nesta atividade, exploraremos algumas métricas além da "mean squared error" e aplicaremos essa ideia na atividade anterior. 

### 11.1) Requisitos sugeridos

Atividade 10.

### 11.2) Descrição

Métricas de Regressão
- [Regression Metrics Review I - How to Win a Data Science Competition Learn from Top Kagglers](https://www.bilibili.com/video/BV117411y7Fa?p=30) (15min)
- [Regression Metrics Review II - How to Win a Data Science Competition Learn from Top Kagglers](https://www.bilibili.com/video/BV117411y7Fa?p=31) (9min)

Obs: Os vídeos desta seção estão hospedados em uma plataforma de compartilhamento de vídeos asiática, pois os originais, que estavam no Coursera, tornaram-se indisponíveis devido às sanções aplicadas à Rússia pela guerra na Ucrânia. No Coursera, os cursos originados de universidades russas foram desativados.

No curso introdutório de Aprendizado de Máquina do Kaggle Learn, que você realizou, foram apresentadas ideias iniciais sobre validação de modelo utilizando um conjunto de validação (hold-out). Um exercício interessante é aplicar essa mesma ideia à atividade anterior (10), separando o conjunto de dados em uma parte para treino e outra para teste.

- [Training and testing - Machine Learning for Trading](https://youtu.be/P2NqrFp8usY) (3 min)  
- Escolha algumas das [métricas discutidas disponíveis no scikit-learn](https://scikit-learn.org/stable/modules/classes.html#regression-metrics) e analise como elas se comportam (tanto no conjunto de treino quanto no de teste) ao variar o valor de `degree` na sua implementação.  
- [Fundamentos de Aprendizagem de Máquina: Viés e Variância - StatQuest](https://youtu.be/EuBBz3bI-aA) (7 min)  

Mais adiante, discutiremos com mais profundidade os conceitos de viés/variância e underfitting/overfitting. Por ora, tente refletir sobre o que acontece com o modelo polinomial ao alterar o valor de `degree`. Para quais valores de `degree` ocorre underfitting? E para quais valores ocorre overfitting?  

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 12) Introdução ao Pandas

O Pandas é a biblioteca de manipulação de dados mais amplamente utilizada para estruturar seus dados em Python. Aliada ao Spark e ao SQL, ela compõe um stack extremamente robusto para diferentes tarefas e cenários de manipulação de dados. O Pandas é, talvez, a mais natural para aprender quando se está estudando Python, e dominá-la facilitará o aprendizado das outras ferramentas. 

### 12.1) Requisitos sugeridos

Atividade 5.

### 12.2) Descrição

Algumas referências. Caso ache que estão muito redundantes, sinta-se à vontade para pular algumas delas.
- [Kaggle Learn - Intro to Pandas](https://www.kaggle.com/learn/pandas) (4h).
- [Pandas for Data Science in 20 Minutes : Python Crash Course](https://www.youtube.com/watch?v=tRKeLrwfUgU) (23 min) 
- [Complete Python Pandas Data Science Tutorial! (Reading CSV/Excel files, Sorting, Filtering, Groupby)](https://youtu.be/vmEHCJofslg)  (~1 hora) 
- [Python pandas — An Opinionated Guide](https://youtube.com/playlist?list=PLgJhDSE2ZLxaENZWWF_VOUa5886KiUd15) (~2 horas)

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 13) Aula de Regressão Logística da especialização (básica) de ML do Andrew

A regressão logística é uma generalização natural da regressão linear para o caso de classificação binária, em que, por construção, espera-se que o output do modelo seja um valor entre 0 e 1, com interpretação como a probabilidade de uma das classes. Nesta aula do Andrew, são abordados alguns tópicos adicionais, como underfitting, overfitting e regularização.

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

Assim como no caso da regressão, existem diversas maneiras de avaliar a qualidade de um modelo em problemas de classificação.

### 14.1) Requisitos sugeridos

Atividade 13.

### 14.2) Descrição

- [The 5 Classification Evaluation metrics every Data Scientist must know](https://towardsdatascience.com/the-5-classification-evaluation-metrics-you-must-know-aa97784ff226)
- [Classification Metrics Review - How to Win a Data Science Competition Learn from Top Kagglers](https://www.bilibili.com/video/BV117411y7Fa?p=32)

Obs: Os vídeos desta seção estão hospedados em uma plataforma de compartilhamento de vídeos asiática, pois os originais, que estavam no Coursera, tornaram-se indisponíveis devido às sanções aplicadas à Rússia pela guerra na Ucrânia. No Coursera, os cursos originados de universidades russas foram desativados.

- Uma sugestão mais rigorosa é o [capítulo sobre métricas de classificação do livro do DataLab](https://pibieta.github.io/imbalanced_learning/notebooks/Metrics%201%20-%20Intro%20%26%20ROC%20AUC.html#)

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 15) Árvores de Decisão e Regressão

Algoritmos baseados em árvores estão entre os mais utilizados, de maneira geral, para trabalhar com dados tabulares. Compreender bem o caso básico é essencial para entender as formas mais robustas de utilizá-los, especialmente quando empregamos comitês.

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

As Random Forests são um poderoso exemplo de comitês baseados em árvores de decisão, projetados para aumentar a precisão e a robustez dos modelos. Compreender suas ideias centrais, como bootstrapping, o erro fora da amostra (OOB) e a importância das variáveis, é essencial para explorar seu potencial em dados tabulares.

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

Na prática, antes da modelagem, é muito importante entender quais variáveis o cientista tem disponíveis para a criação do modelo. Nem sempre o problema está bem definido, e a delimitação do que se deseja modelar (e quais métricas otimizar) pode surgir de um bom entendimento dos dados disponíveis, em conjunto com alinhamentos com a área de negócios interessada.

- Pensar em **hipóteses de negócio que você gostaria de validar** é uma ótima maneira de realizar uma análise exploratória. Será que você já tem alguma intuição sobre o problema que possa, inclusive, ajudá-lo a modelar o problema de uma forma diferente no futuro?
    - Em um problema de inadimplência de crédito (previsão de se uma pessoa irá pagar ou não uma dívida), por exemplo, pode-se ter a intuição de que, para pessoas de baixa renda, a existência de uma dívida não quitada anterior pode ser crucial para prever se elas pagarão o próximo empréstimo. Enquanto isso, para pessoas de alta renda, esse fator pode ser menos relevante. Esse tipo de pergunta pode ser testado diretamente, e essa hipótese pode se tornar uma regra de negócio que funcione como um benchmark que você desejará superar com o modelo. Será que essa regra, sozinha, já apresenta uma performance boa o suficiente que sequer justifique a criação de um modelo?
    - Além disso, a hipótese pode ajudá-lo a compreender onde está o maior foco de interesse na modelagem. Por exemplo, se você tem o valor da dívida, pode agrupar os dados para identificar quais grupos, caso sejam mal identificados pelo modelo, podem gerar maiores prejuízos. Imagine que, embora pessoas de alta renda sejam minoria na base de desenvolvimento (10%, por exemplo), elas estejam associadas a 90% do valor total em empréstimos. Identificar corretamente os maus pagadores nesse grupo pode ser muito mais importante do que identificá-los de forma geral na população. Isso pode orientar as etapas de modelagem e avaliação, levando a segmentar métricas, usar `sample_weights` para valorizar grupos de interesse e até mesmo dividir o modelo por faixa de renda.

- Estudar sua amostra de desenvolvimento, utilizando gráficos e estatísticas, pode fornecer insights sobre **quais variáveis estão relacionadas ao problema e possíveis técnicas de feature engineering** que podem ser aplicadas, além de **identificar problemas que precisam ser resolvidos** para uma modelagem adequada.
    - É comum tentar filtrar algumas variáveis nesta etapa com base em medidas de correlação. Embora isso possa ser mais eficaz em etapas posteriores do pipeline de desenvolvimento, pode ser necessário fazê-lo durante a exploração se o problema tiver muitas colunas e o tempo for curto, exigindo o foco nas mais relevantes. Rodar análises de correlação simples ou utilizar algoritmos que medem a importância das variáveis (como algoritmos baseados em árvores) pode ser muito útil, mas deve ser feito com cautela.
    - Conhecer as variáveis disponíveis, aliado a um entendimento do negócio, pode inspirá-lo a criar variáveis que simplifiquem o trabalho do modelo (especialmente árvores, que só conseguem fazer cortes paralelos aos eixos). Por exemplo, uma variável muito relevante em crédito é o "comprometimento de renda", ou seja, a porcentagem do salário que precisaria ser reservada para pagar a parcela de um financiamento ativo (`valor_parcela/valor_salario`). Criá-la nesta etapa e avaliá-la por meio de gráficos pode ser uma atividade exploratória muito útil.
    - Bases de dados frequentemente apresentam problemas como valores vazios ou, pior, valores preenchidos de forma incorreta. Se uma variável só deveria conter valores dentro de um certo intervalo e você encontra valores fora desse intervalo, é necessário investigar o motivo. Analisar criticamente os valores pode ajudar a traçar estratégias para tratá-los ou até mesmo descartá-los (com cuidado, pois excluir linhas no conjunto de teste só deve ser feito se aquilo realmente não ocorrer na vida real). (_Em um problema de Kaggle ou portfólio, essa resposta raramente estará disponível, mas em uma empresa, esse pode ser o momento de conversar com a engenharia e entender como os dados serão fornecidos para o modelo. Também pode ser o caso de alinhar com a área de negócios para adaptar a esteira de decisão quando faltarem informações, por exemplo._)

- Além disso, esse entendimento orienta a escolha de modelos mais apropriados para o tipo de dados disponíveis.
    - Por exemplo, escolher modelos que lidem nativamente bem com variáveis categóricas (como o CatBoost).
    - Para valores faltantes, dependendo do caso ([MCAR, MAR ou MNAR](https://youtu.be/YpqUbirqFxQ)), estratégias de imputação podem ser definidas nesta etapa (e testadas posteriormente na otimização de hiperparâmetros) ou pode-se optar por modelos que lidem nativamente com dados faltantes (como o LightGBM).

- Por fim, explorar tendências temporais nos dados é essencial. Aprendizado de máquina pressupõe que os dados sejam estáveis ao longo do tempo, o que nem sempre é a realidade. Em muitas aplicações, é importante segmentar os dados respeitando a lógica temporal, e a análise exploratória pode indicar se isso é relevante para o problema em questão.

### Opinião pessoal (também conhecido como desabafo):

- Pensando principalmente em aprendizado supervisionado, os pontos acima (validação de hipóteses de negócio, estudo de variáveis relevantes, criação de variáveis novas, identificação de problemas, entre outros) são os principais objetivos de uma análise exploratória de dados e devem ser o foco da exploração. Criar gráficos "por criar", que não trazem informações relevantes, deve ser evitado, pois tornam a análise prolixa e desviam do objetivo principal: o modelo final.

- É fácil se desviar desses objetivos, mas vale sempre se perguntar: "Por que estou criando este gráfico ou calculando esta estatística?" Se o motivo estiver claro, então provavelmente faz sentido fazê-lo.

- Pode parecer uma visão dura sobre as análises exploratórias, mas é muito comum observar, especialmente em cientistas de dados iniciantes (em cases de entrevistas e projetos de portfólio), notebooks extensos com inúmeros gráficos e comandos `.head()` que dificultam a navegação do código sem agregar valor ao processo de modelagem ou à narrativa do problema. Em 99% dos casos, esses elementos parecem desconectados do problema de interesse e, se excluídos do notebook, não alterariam o resultado final.

- Em casos em que não há necessariamente um modelo envolvido, aprofundar-se na análise exploratória pode fazer sentido, mas ainda assim raramente fugiria das motivações anteriores.

### 17.1) Requisitos sugeridos

Atividades 12.

### 17.2) Descrição

- [Exemplo de notebook que realiza uma análise exploratória focada em procurar erros nos dados](https://github.com/vitaliset/projetos-de-estudo/blob/main/New%20York%20City%20Taxi%20Fare%20Prediction/1_procurando_erros.ipynb).
- [Exemplo de notebook que gera gráficos interessantes das variáveis analisando quais fazem sentido para o modelo](https://github.com/vitaliset/projetos-de-estudo/blob/main/Credit%20Default%20without%20a%20target/provenir_carlo.ipynb).
- [Exemplo de notebook que cria hipóteses de negócio como modelos e depois compara performance dessas hipóteses contra modelos de ML](https://github.com/vitaliset/projetos-de-estudo/tree/main/Machine%20Failure%20Predictions).

Extras:
- Iremos discutir aspectos de validação out-of-time no futuro, mas este notebook apresenta uma análise exploratória para estudo de estabilidade temporal com uma [técnica interessante](https://vitaliset.github.io/covariate-shift-2-classificador-binario/):
    - [Exemplo de notebook que realiza uma análise exploratória focada em problemas de drift temporal nos dados](https://github.com/vitaliset/projetos-de-estudo/blob/main/New%20York%20City%20Taxi%20Fare%20Prediction/2_dinamica_temporal.ipynb).

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 18) Kaggle Challenge

Neste ponto, o mentorado já adquiriu o conhecimento necessário para realizar um ciclo completo de machine learning: desde a escolha de um dataset até a aplicação de um modelo, passando pela limpeza dos dados e avaliação dos resultados. Este desafio no Kaggle oferece uma oportunidade prática para consolidar e aplicar os aprendizados em um cenário prático.

### 18.1) Requisitos sugeridos

Atividades 16 e 17.

### 18.2) Descrição

Escolher um dataset do Kaggle e limpar os dados + aplicar um modelo de ML, avaliando os resultados (qual métrica utilizar pensando no problema que estou preocupado em resolver?)

- [Find Open Datasets and Machine Learning Projects - Kaggle](https://www.kaggle.com/datasets)
- Sugestão: [Heart Attack Analysis & Prediction Dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset)

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 19) Aula sobre dicas práticas da especialização (básica) de ML do Andrew

Esta atividade oferece uma visão prática e estratégica do ciclo de vida completo de um modelo de machine learning supervisionado, com foco em validação, análise de erros e iteração. Além disso, aborda tópicos fundamentais como MLOps e ética, ajudando o mentorado a conectar teoria e prática de forma holística.

### 19.1) Requisitos sugeridos

Atividade 18.

### 19.2) Descrição

Infelizmente, a partir do segundo curso do Andrew, é necessário se inscrever pelo Coursera porque os vídeos não estão disponíveis no YouTube. Não se preocupe, o conteúdo continua gratuito se você se inscrever como **ouvinte**.

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

As árvores de decisão são modelos fundamentais para resolver problemas tanto de classificação quanto de regressão, servindo de base para técnicas mais avançadas como Random Forests e XGBoost. Esta aula do Andrew Ng aprofunda os conceitos de aprendizado, medição de pureza e seleção de divisões, consolidando uma compreensão sólida do funcionamento e aplicações desse tipo de modelo.

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

O curso intermediário de Machine Learning do Kaggle é uma oportunidade de aprofundar conceitos essenciais, como imputação de valores ausentes, validação e ajuste de modelos. Além disso, a inclusão de referências sobre validação out-of-time e out-of-space ajuda a explorar técnicas avançadas para avaliar modelos em cenários mais desafiadores e próximos da realidade.

### 21.1) Requisitos sugeridos

Atividade 20.

### 21.2) Descrição

- [Kaggle Learn - Intermediate Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning) (4h).
- [Validação out of time - Experian (Serasa)](https://www.experian.com/blogs/insights/2018/06/understanding-validation-samples-within-model-development/).
- [Diagrama de validação out-of-space e out-of-time - Documentação do fklearn - Nubank](https://fklearn.readthedocs.io/en/latest/examples/fklearn_overview.html?highlight=out-of-ID#Spliting-the-dataset-into-train-and-holdout).

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 22) Otimização de hiperparâmetros

A otimização de hiperparâmetros é uma etapa crucial para maximizar o desempenho dos modelos de machine learning. Esta atividade apresenta as principais abordagens, como Grid Search e alternativas mais eficientes, ajudando o mentorado a compreender quando e como ajustar hiperparâmetros de maneira estratégica para diferentes tipos de problemas.

### 22.1) Requisitos sugeridos

Atividade 21.

### 22.2) Descrição

- [Hyperparameter optimization - Wikipedia](https://en.wikipedia.org/wiki/Hyperparameter_optimization)
- [Hyperparameter Optimization - The Math of Intelligence #7 - Siraj Raval](https://youtu.be/ttE0F7fghfk) (10min). Ele é meio impreciso em alguns poucos momentos não muito relevantes e entra em alguns tópicos que não são essenciais, mas é interessante para ver a ideia geral.
- [Nunca Mais Use Grid Search Para Ajustar Hiperparâmetros - Mario Filho](https://youtu.be/WhnkeasZNHI) (32min)

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 23) Curso de feature engineering do Kaggle Learn

O feature engineering é uma etapa essencial para aumentar a qualidade e o desempenho dos modelos de machine learning, permitindo extrair o máximo de informações úteis dos dados. Este curso do Kaggle Learn e as referências complementares ajudam o mentorado a dominar técnicas como encoding, manipulação de datas e criação de variáveis mais informativas, consolidando uma base sólida para modelagem avançada.

### 23.1) Requisitos sugeridos

Atividade 21.

### 23.2) Descrição

- [Kaggle Learn - Feature Engineering](https://www.kaggle.com/learn/feature-engineering) (5h).
- [Aula avulsa do curso de Data Cleaning sobre dataframes com colunas do tipo data](https://www.kaggle.com/code/alexisbcook/parsing-dates) (1h).
- [One-Hot, Label, Target e K-Fold Target Encoding, claramente explicados!!! - StatQuest](https://youtu.be/589nCGeWG1w)

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 24) Curso de explicabilidade do Kaggle Learn

Explicabilidade em machine learning é fundamental para entender como os modelos tomam decisões, aumentando a confiança e a transparência em suas previsões. Este curso do Kaggle Learn explora ferramentas como SHAP values, permitindo que o mentorado interprete modelos complexos e identifique os fatores mais relevantes na tomada de decisão, com aplicações práticas e acessíveis.

### 24.1) Requisitos sugeridos

Atividade 21.

### 24.2) Descrição

- [Kaggle Learn - Machine Learning Explainability](https://www.kaggle.com/learn/machine-learning-explainability) (4h).
- [SHAP Values Explained Exactly How You Wished Someone Explained to You - Samuele Mazzanti](https://towardsdatascience.com/shap-explained-the-way-i-wish-someone-explained-it-to-me-ab81cc69ef30)

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 25) Curso de ética do Kaggle Learn

A ética em IA é essencial para desenvolver soluções responsáveis, justas e alinhadas com os valores sociais.

### 25.1) Requisitos sugeridos

Atividade 21.

### 25.2) Descrição

- [Kaggle Learn - Intro to AI Ethics](https://www.kaggle.com/learn/intro-to-ai-ethics) (4h).

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 26) Aula aprendizado não supervisionado da especialização (básica) de ML do Andrew

O aprendizado não supervisionado é uma abordagem poderosa para descobrir padrões ocultos e identificar anomalias em dados sem rótulos. Esta aula do Andrew Ng apresenta fundamentos como clustering com K-means e detecção de anomalias, permitindo ao mentorado explorar aplicações práticas dessas técnicas.

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

Algoritmos baseados em distância, como KNN e K-Means, utilizam medidas de similaridade geométrica para classificação, regressão e clustering. Compreender esses métodos é essencial para explorar técnicas simples e eficazes em aprendizado de máquina.

### 27.1) Requisitos sugeridos

Atividade 26.

### 27.2) Descrição

- [K-nearest neighbors, Clearly Explained - StatQuest](https://youtu.be/HVXime0nQeI) (5min)
- [K-means clustering = StatQuest](https://youtu.be/4b5d3muPQmA) (9min)
- [Chapter 3 - GEOMETRY AND NEAREST NEIGHBORS - A Course in Machine Learning by Hal Daumé III](http://ciml.info/)
- [Generalizando distância - Carlo Lemos](https://vitaliset.github.io/distancia/)

<p><center><img src="{{ site.baseurl }}/assets/img/horizontal_line.png"></center></p>

## 28) Introdução a Version Control System (Git/GitHub)

O Git é uma ferramenta essencial para controle de versão, permitindo que você acompanhe e gerencie alterações em seu código de forma eficiente e segura. Ele é amplamente utilizado em projetos de software para colaborar em equipes, revisar mudanças e integrar diferentes partes de um projeto sem conflitos.

O GitHub, por sua vez, é uma plataforma que aproveita o poder do Git, oferecendo recursos adicionais como hospedagem de repositórios, integração contínua, controle de acesso e colaboração em projetos. Dominar essas ferramentas não é apenas importante para gerenciar seus próprios projetos, mas também essencial para trabalhar em equipes modernas, onde o versionamento e a rastreabilidade são cruciais para a produtividade e qualidade do trabalho.

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
Provavelmente o assunto mais importante que ficou faltando no currículo inicial da tutoria. Vale a pena ter um nível básico de SQL para aplicar para vagas.
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

## GenAI
- [ChatGPT Prompt Engineering for Developers - deeplearning.ai](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/)
- [Building Systems with the ChatGPT API - deeplearning.ai](https://www.deeplearning.ai/short-courses/building-systems-with-chatgpt/)
- [LangChain for LLM Application Development - deeplearning.ai](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/)
- [Functions, Tools and Agents with LangChain - deeplearning.ai](https://www.deeplearning.ai/short-courses/functions-tools-agents-langchain/)
- [Building and Evaluating Advanced RAG Applications - deeplearning.ai](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/)

## Cursos de Aprendizado de Máquina com um pouco mais de rigor
- [Introduction to Statistical Learning](https://youtube.com/playlist?list=PLOg0ngHtcqbPTlZzRHA2ocQZqB1D_qZ5V)
- [Learning from Data - Yaser Abu Mostafa - Caltech](https://youtube.com/playlist?list=PLnIDYuXHkit4LcWjDe0EwlE57WiGlBs08)

## Boosting Trees
- [Bagging vs Boosting - Ensemble Learning In Machine Learning Explained - WhyML](https://youtu.be/tjy0yL1rRRU)
- [AdaBoost, Clearly Explained](https://youtu.be/LsK-xG1cLYA)
- [Gradient Boost playlist - StatQuest](https://youtube.com/playlist?list=PLblh5JKOoLUJjeXUvUE0maghNuY2_5fY6)
- [XGBoost playlist - StatQuest](https://youtube.com/playlist?list=PLblh5JKOoLULU0irPgs1SnKO6wqVjKUsQ)
- [Why XGBoost is better than GBM? - Damien Benveniste](https://newsletter.theaiedge.io/p/why-xgboost-is-better-than-gbm)
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
- [Processamento Neural de Linguagem Natural em Português - Coursera USP](https://www.coursera.org/learn/processamento-neural-linguagem-natural-em-portugues-i)

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
- [sktime - A Unified Toolbox for ML with Time Series - Markus Löning - PyData Global 2021](https://youtu.be/ODspi8-uWgo)

## Robustez de Modelo
- [Como leakage de dados acaba com backtesting - Nubank ML Meetup - Tatyana Zabanova](https://youtu.be/qPYqeD2OUl4)
- [Generalização de domínio, invariância e a Floresta Temporalmente Robusta - Nubank ML Meetup - Luis Moneda](https://youtu.be/Gq20DI9punw)
- [Train/test mismatch e adaptação de domínio - A Course in Machine Learning by Hal Daumé III](http://ciml.info/dl/v0_99/ciml-v0_99-ch08.pdf)

## Inferência Causal
- [Causal Inference for The Brave and True - Matheus Facure](https://matheusfacure.github.io/python-causality-handbook/landing-page.html)
- [Notas de aula de Inferência Causal - Rafael Stern](https://github.com/rbstern/causality_book/blob/master/book.pdf)

## Feature Selection
- [Feature Selection - Machine Learning Interview Q&A for Data Scientists - Data Science Interviews - Emma Ding](https://youtu.be/7tW29jBceRw)
- [Feature Selection Usando Scikit-Learn - Mario Filho](https://www.youtube.com/live/Bcn5e7LYMhg?feature=share)
- [Como Remover Variáveis Irrelevantes de um Modelo de Machine Learning - Mario Filho](https://www.youtube.com/live/6-mKATDSQmk?feature=share)
- [Seleção de variáveis: uma utilização crítica do Boruta - Carlo Lemos](https://vitaliset.github.io/boruta/)

## Python
- Cursos de Python da USP no Coursera - Fábio Kon
    - [Introdução à Ciência da Computação com Python Parte 1](https://www.coursera.org/learn/ciencia-computacao-python-conceitos) 
    - [Introdução à Ciência da Computação com Python Parte 2](https://www.coursera.org/learn/ciencia-computacao-python-conceitos-2) 
    - [Laboratório de Programação Orientada a Objetos - Parte 1](https://www.coursera.org/learn/lab-poo-parte-1)
    - [Laboratório de Programação Orientada a Objetos - Parte 2](https://www.coursera.org/learn/lab-poo-parte-2)
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
