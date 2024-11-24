---
layout: page
title: Dicas de entrevista para big techs
permalink: /interview/
mathjax: true
---

_This page provides tips and insights for tackling LeetCode interview questions, helping folks prepare effectively for coding interviews._

___

Depois de receber algumas perguntas sobre minha experiência nas entrevistas do Google, resolvi consolidar um material com dicas de alguém que passou por esse processo.

Essas dicas não são oficiais do Google ou de outras empresas, apenas um compilado de aprendizados que tive. Espero que seja útil! :)

## Entrevista técnica (Leetcode-like)

### A dança da entrevista

- Priorize a comunicação durante todo o processo de entrevista. **Pense em voz alta!**
- Encare a entrevista como uma dança com o seu entrevistador, com etapas e ritmo a seguir.
    - Prepare uma introdução de no máximo 30 segundos sobre sua educação e experiência de trabalho. Talvez você nem precise se apresentar.
    - Parafraseie o problema apresentado pelo entrevistador e faça perguntas sobre ele. Garanta que você entendeu o que precisa resolver.
    - Utilize casos de teste para confirmar o entendimento do problema. Clarifique também os edge cases, por exemplo, entendendo o que você deve retornar se o input for vazio, por exemplo.
    - Se não é óbvio como proceder, lembre-se de que "código que resolve o problema" é mais importante do que "código otimizado". Pense em soluções de força bruta antes de tentar algo mais elegante.
    - Seja receptivo a dicas do seu entrevistador. Ele conhece detalhadamente aquele problema e já viu várias pessoas tentando resolvê-lo com abordagens diferentes. Ele quer te ajudar.
    - Nunca comece a codificar sem perguntar se pode antes. É a oportunidade do entrevistador te dizer se já está confortável com a abordagem que você está seguindo. Se ele ainda não estiver, ele pode te ajudar com dicas nesse momento.
    - Faça proativamente um [teste de mesa](https://pt.stackoverflow.com/questions/220474/o-que-%C3%A9-um-teste-de-mesa-como-aplic%C3%A1-lo) no final, com algum exemplo que você validou o input/output no início para garantir a lógica da sua solução.
    - Esteja preparado para responder perguntas sobre complexidade de tempo e espaço, escolha de estruturas de dados e algoritmos, e possíveis melhorias ou variações.
    - Mostre interesse pela empresa e separe algumas perguntas relevantes ao entrevistador para fazer nos minutos finais de entrevista. Preste atenção na introdução do entrevistador, se você conseguir conectar perguntas ao que ele falou, melhor ainda.
- A [entrevista de exemplo](https://youtu.be/XKu_SEDAykw?si=zp6YhHC8HhBiPU8x) que o Google disponibiliza demonstra muito be essa dança.
- Seja positivo e demonstre que está gostando. A entrevista não é para ser algo torturante, deve ser prazerosa (apesar de altamente estressante).

### Como estudar

- Ao estudar uma nova estrutura de dados, comece com exercícios fáceis para consolidar a compreensão básica e ganhar confiança. Quando estiver dominando os exercícios fáceis, avance para os de nível médio. A realização de exercícios difíceis pode ter menos valor, pois muitas vezes exigem dicas muito específicas que provavelmente seriam fornecidas pelo entrevistador, transformando o problema em algo semelhante a um problema de nível médio.
- Evite fazer exercícios de forma aleatória. É importante utilizar listas que abordem os tópicos mais frequentes em entrevistas. Eu, por exemplo, estudei quase exclusivamente com o [LeetCode75](https://leetcode.com/studyplan/leetcode-75/) (exceto os marcados como difíceis) e resolvi alguns exercícios específicos do [NeetCode150](https://neetcode.io/roadmap), [Blind75](https://leetcode.com/discuss/general-discussion/460599/blind-75-leetcode-questions) e [LeetCode150](https://leetcode.com/studyplan/top-interview-150/).
- Mesmo que você seja capaz de resolver os problemas, é importante revisar as soluções para garantir que está seguindo as melhores práticas. Se o exercício pertencer a alguma lista clássica, você poderá encontrar a solução no [NeetCode](https://www.youtube.com/c/neetcode) ou [NeetCodeIO](https://www.youtube.com/@NeetCodeIO).
- Certifique-se de que consegue implementar as estruturas de dados na linguagem escolhida. Em geral, é aceitável utilizar as bibliotecas nativas da linguagem, como o `collections` e `heapq` do Python, por exemplo.
- Não se preocupe em se aprofundar excessivamente em tópicos muito específicos, especialmente se estiver com pouco tempo disponível. Concentre-se nas listas clássicas.
- Pratique escrever código em um bloco de notas, para garantir que consiga escrever código sem depender de uma IDE.
- Simule entrevistas, resolvendo problemas em voz alta, preferencialmente em inglês para se familiarizar com o vocabulário técnico. Especialmente nos dias que antecedem a entrevista. Não é necessário fazer isso enquanto estiver aprendendo as estruturas de dados.
- Se tiver a oportunidade, pratique com alguém assumindo o papel de entrevistador para simular uma situação de entrevista. Isso pode ser uma experiência valiosa antes da sua primeira entrevista.

### Dicas sobre mentalidade

- Ser chamado para a entrevista já é uma conquista significativa, parabéns! Aproveite a oportunidade para aprender com o processo.
- Mantenha uma atitude positiva e comunique-se com o entrevistador. Ele está lá para te auxiliar. Se você estiver em uma etapa mais avançada, lembre-se de que não é necessário se sair bem em todas as entrevistas, pois você será avaliado de forma global pelo comitê de contratação.
- Esteja ciente de que há um elemento de sorte envolvido nas entrevistas das grandes empresas de tecnologia. As vezes a questão é muito complexa mesmo, as vezes o entrevistador não é o melhor. Isso é algo fora do seu controle, então não vale ficar ansioso por isso. A ideia é estar o mais preparado possível para quando a "sorte" aparecer.
- [Se sinta ok em relação à possibilidade de não ser contratado](https://steve-yegge.blogspot.com/2008/03/get-that-job-at-google.html). Mesmo que não seja bem-sucedido, mantenha a calma e absorva os aprendizados. Se o seu currículo foi selecionado para a entrevista, é provável que seja chamado para uma nova oportunidade em questão de meses!

## Entrevista comportamental

- Durante a entrevista comportamental, demonstre maturidade ao lidar com conflitos e mostre que é orientado por dados na resolução de problemas.
- Tenha exemplos da sua trajetória profissional para respaldar suas respostas.
- O [Jeff H Sipe](https://www.youtube.com/watch?v=tuL-WmYKBgo) tem ótimas dicas para esse tipo de entrevista (com foco na entrevista de "Googleyness and Leadership" do Google). O canal dele também é útil para entender sobre o [hiring committee](https://www.youtube.com/watch?v=SqnrXBVaCo8), [levels](https://www.youtube.com/watch?v=cC9V5IH4B6k), [team match](https://www.youtube.com/watch?v=fG3noON-IWo) e eventuais dúvidas sobre o Google.

Eu não tenho muitas dicas para entrevistas de system design porque não tive elas durante minhas etapas, mas ver alguns exemplos dos [conceitos principais](https://youtu.be/i53Gi_K3o7I?feature=shared) e [exemplos de entrevistas](https://youtu.be/jPKTo1iGQiE?feature=shared) do tipo podem ajudar a se preparar.

Boa sorte! Vai dar tudo certo! :D