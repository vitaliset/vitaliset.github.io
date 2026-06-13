---
layout: page
title: Interview tips for big techs
title_pt: Dicas de entrevista para big techs
permalink: /interview/
lang: en-US
bilingual: true
description: Tips and insights for cracking LeetCode-style coding interviews at big tech companies, from someone who went through Google's process.
description_pt: Dicas e aprendizados para mandar bem em entrevistas de código estilo LeetCode em big techs, de quem passou pelo processo do Google.
mathjax: true
---

<div class="i18n" lang="en" markdown="1">

After getting a few questions about my experience with Google's interviews, I decided to put together some tips from someone who has been through the process.

These tips aren't official from Google or any other company — just a compilation of things I learned. I hope it's useful! :)

## Technical interview (LeetCode-like)

### The interview dance

- Prioritize communication throughout the entire interview. **Think out loud!**
- Treat the interview as a dance with your interviewer, with steps and a rhythm to follow.
    - Prepare an introduction of at most 30 seconds about your education and work experience. You might not even need to introduce yourself.
    - Paraphrase the problem the interviewer presents and ask questions about it. Make sure you understand what you need to solve.
    - Use test cases to confirm your understanding of the problem. Clarify the edge cases too — for example, understanding what you should return if the input is empty.
    - If it's not obvious how to proceed, remember that "code that solves the problem" matters more than "optimized code". Think about brute-force solutions before trying something more elegant.
    - Be receptive to hints from your interviewer. They know that problem in detail and have already seen many people try to solve it with different approaches. They want to help you.
    - Never start coding without asking first. It's the interviewer's chance to tell you whether they're already comfortable with the approach you're following. If they aren't yet, they can help you with hints at that point.
    - Proactively do a dry run (manually trace through your code) at the end, using an example whose input/output you validated at the start, to make sure your solution's logic is correct.
    - Be ready to answer questions about time and space complexity, your choice of data structures and algorithms, and possible improvements or variations.
    - Show interest in the company and prepare a few relevant questions to ask the interviewer in the final minutes. Pay attention to the interviewer's introduction — if you can connect your questions to what they said, even better.
- The [sample interview](https://youtu.be/XKu_SEDAykw?si=zp6YhHC8HhBiPU8x) that Google provides demonstrates this dance really well.
- Be positive and show that you're enjoying it. The interview isn't meant to be torture — it should be enjoyable (even if highly stressful).

### How to study

- When studying a new data structure, start with easy exercises to consolidate the basics and build confidence. Once you're getting the hang of the easy ones, move on to medium-level problems. Doing hard exercises may be less valuable, since they often require very specific hints that the interviewer would likely provide anyway, turning the problem into something closer to a medium-level one.
- Avoid doing exercises at random. It's important to use lists that cover the topics most frequently seen in interviews. I, for example, studied almost exclusively with [LeetCode75](https://leetcode.com/studyplan/leetcode-75/) (except the ones marked as hard) and solved a few specific exercises from [NeetCode150](https://neetcode.io/roadmap), [Blind75](https://leetcode.com/discuss/general-discussion/460599/blind-75-leetcode-questions) and [LeetCode150](https://leetcode.com/studyplan/top-interview-150/).
- Even if you're able to solve the problems, it's important to review the solutions to make sure you're following best practices. If the exercise belongs to a classic list, you can find the solution on [NeetCode](https://www.youtube.com/c/neetcode) or [NeetCodeIO](https://www.youtube.com/@NeetCodeIO).
- Make sure you can implement the data structures in your chosen language. In general, it's acceptable to use the language's native libraries, such as Python's `collections` and `heapq`.
- Don't worry about going too deep into very specific topics, especially if you're short on time. Focus on the classic lists.
- Practice writing code in a plain text editor, to make sure you can write code without relying on an IDE.
- Simulate interviews by solving problems out loud, preferably in English to get familiar with the technical vocabulary. Especially in the days leading up to the interview. You don't need to do this while you're still learning the data structures.
- If you get the chance, practice with someone playing the role of interviewer to simulate a real interview situation. This can be a valuable experience before your first interview.

### Mindset tips

- Being called for the interview is already a significant achievement — congratulations! Take the opportunity to learn from the process.
- Keep a positive attitude and communicate with the interviewer. They are there to help you. If you're at a more advanced stage, remember that you don't need to do well in every single interview, since you'll be evaluated holistically by the hiring committee.
- Be aware that there's an element of luck involved in big tech interviews. Sometimes the question really is very hard, sometimes the interviewer isn't the best. That's outside your control, so it's not worth getting anxious about it. The idea is to be as prepared as possible for when "luck" shows up.
- [Be ok with the possibility of not getting hired](https://steve-yegge.blogspot.com/2008/03/get-that-job-at-google.html). Even if you don't succeed, stay calm and absorb the lessons. If your résumé was selected for the interview, you'll likely be called for a new opportunity within a few months!

## Behavioral interview

- During the behavioral interview, show maturity in handling conflict and show that you're data-driven when solving problems.
- Have examples from your professional journey to back up your answers.
- [Jeff H Sipe](https://www.youtube.com/watch?v=tuL-WmYKBgo) has great tips for this kind of interview (focused on Google's "Googleyness and Leadership" interview). His channel is also useful for understanding the [hiring committee](https://www.youtube.com/watch?v=SqnrXBVaCo8), [levels](https://www.youtube.com/watch?v=cC9V5IH4B6k), [team match](https://www.youtube.com/watch?v=fG3noON-IWo) and any other questions about Google.

I don't have many tips for system design interviews because I didn't have them during my stages, but watching some examples of the [core concepts](https://youtu.be/i53Gi_K3o7I?feature=shared) and [sample interviews](https://youtu.be/jPKTo1iGQiE?feature=shared) of that type can help you prepare.

Good luck! It's going to be fine! :D

</div>

<div class="i18n" lang="pt" markdown="1">

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
- A [entrevista de exemplo](https://youtu.be/XKu_SEDAykw?si=zp6YhHC8HhBiPU8x) que o Google disponibiliza demonstra muito bem essa dança.
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

</div>
