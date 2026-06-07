---
layout: page
title: Conteúdos sobre LLM
permalink: /awesome-llm/
mathjax: true
---

## Deep Learning fundamentals
- Pytorch
    - [Learn PyTorch for deep learning in a day. Literally.](https://youtu.be/Z_ikDlimN6A)
    - [PyTorch Tutorial (Sebastian Raschka)](https://youtu.be/B5GHmm3KN2A)
- NLP
    - [Neural Networks: Zero to Hero - Andrej Karpathy](https://youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
    - [Stanford CS224N: Natural Language Processing with Deep Learning](https://youtube.com/playlist?list=PLoROMvodv4rOSH4v6133s9LFPRHjEmbmJ)
    - [Deep Learning Fundamentals - PyTorch Lightning](https://lightning.ai/pages/courses/deep-learning-fundamentals/unit-8.0-natural-language-processing-and-large-language-models/)
    - DeepLearning.AI NLP course
    - [NYU Deep Learning SP20](https://www.youtube.com/playlist?list=PLLHTzKZzVU9eaEyErdV26ikyolxOsz6mq)
        - Week 6: CNN applications, RNN, and attention (lecture) ~2h
        - Week 12: NLP (lecture + practicum) ~2h
    - [MIT 6.S191: Recurrent Neural Networks, Transformers, and Attention](https://youtu.be/ySEx_Bqxvvo) ~1h
    - [LSTM StatQuest](https://youtu.be/YCzL96nL7j0) ~21min
    - [Attention Mechanism - Introduction to Deep Learning](https://youtu.be/aQQa_xl78Qw)
    - [CS324 - Large Language Models](https://stanford-cs324.github.io/winter2022/lectures/)
    - [Understanding Large Language Models - Sebastian Raschka](https://magazine.sebastianraschka.com/p/understanding-large-language-models?utm_source=profile&utm_medium=reader2)
    - [Hugging Face course](https://huggingface.co/learn/nlp-course/chapter1/1)
    - [ChatGPT by Paulo Finardi (Senior DS Itaú) - Muitos links interessantes!](https://github.com/finardi/tutos/blob/master/CharGPT_dev.ipynb)
- NLP (consulta)
    - [GPT-4 - How does it work, and how do I build apps with it? - CS50 Tech Talk](https://www.youtube.com/live/vw-KWfKwvTQ?feature=share) ~67min
    - [Livro Natural Language Processing with Transformers](https://www.oreilly.com/library/view/natural-language-processing/9781098136789/)
    - [NLP by Aman: Attention, Autoregressive vs. Autoencoder Models, Token Sampling Methods, Transformers, BERT](https://www.linkedin.com/posts/amanc_artificialintelligence-machinelearning-ai-activity-7055390952143663105-Ti-Z?utm_source=share&utm_medium=member_ios)
    - [A Survey of Large Language Models (27/04/23](https://arxiv.org/abs/2303.18223)
    - [What Are Transformer Models and How Do They Work? - Cohere](https://txt.cohere.com/what-are-transformer-models/)
    - [Transformers links by Damien](https://www.linkedin.com/posts/damienbenveniste_machinelearning-datascience-artificialintelligence-activity-7046508267467911168-i49y/?utm_source=share&utm_medium=member_ios)
- Parallel and Distributed Training (consulta)
    - [Demystifying Parallel and Distributed Deep Learning](https://youtu.be/xtxxLWZznBI) ~50min
    - [CUDA](https://youtu.be/nlGnKPpOpbE) ~1h20min
    - [MPI Reduce and Allreduce](https://mpitutorial.com/tutorials/mpi-reduce-and-allreduce/)
    - [PyTorch parallel](https://youtube.com/playlist?list=PL_lsbAsL_o2CSuhUhJIiW0IkdT5C2wGWj) ~1h
    - [DeepSpeed](https://github.com/microsoft/DeepSpeed)


## Prompt Engineering:
- [Prompt Engineering vs. Blind Prompting](https://mitchellh.com/writing/prompt-engineering-vs-blind-prompting)
- [Prompt Engineering Course by Andrew Ng and Open AI](https://lnkd.in/gQ_pkePr)
- [Intro to Prompt engineering by Microsoft](https://lnkd.in/gpzsWcgy)
- [Prompt Engineering Techniques by Microsoft](https://lnkd.in/gYRKEvaS)
- [Prompt Engineering Guide](https://lnkd.in/g4DbhQC5)
- [Guia de Engenharia Prompt](https://www.promptingguide.ai/pt)
    - [Few-Shot vs Zero-Shot prompting](https://www.promptingguide.ai/techniques/fewshot)


## LangChain
- [LangChain explained](https://youtu.be/RoR4XJw8wIc) ~3min
- [LangChain Crash Course](https://youtu.be/LbT1yp6quS8) ~15min
- [LangChain for Beginners](https://youtu.be/aywZrzNaKjs) ~13min
- [The easiest way to work with large language models](https://youtu.be/kmbS6FDQh7c) ~10min
- [The LangChain Cookbook - Beginner Guide To 7 Essential Concepts](https://youtu.be/2xxziIWmaSA) ~38min
- [LangChain Crash Course: Build a AutoGPT app in 25 minutes!](https://youtu.be/MlK6SIjcjE8)
- [Langchain PDF App (GUI) | Create a ChatGPT For Your PDF in Python](https://www.youtube.com/watch?v=wUAUdEw5oxM) ~40min
- [LangChain documentation](https://python.langchain.com/en/latest/index.html)
    - Models: The various model types and model integrations LangChain supports.
    - **Prompts**: This includes prompt management, prompt optimization, and prompt serialization.
    - Memory: Memory is the concept of persisting state between calls of a chain/agent. LangChain provides a standard interface for memory, a collection of memory implementations, and examples of chains/agents that use memory.
    - **Indexes**: Language models are often more powerful when combined with your own text data - this module covers best practices for doing exactly that.
    - Chains: Chains go beyond just a single LLM call, and are sequences of calls (whether to an LLM or a different utility). LangChain provides a standard interface for chains, lots of integrations with other tools, and end-to-end chains for common applications.
    - **Agents** (estilo plug-in do ChatGPT): Agents involve an LLM making decisions about which Actions to take, taking that Action, seeing an Observation, and repeating that until done. LangChain provides a standard interface for agents, a selection of agents to choose from, and examples of end to end agents.
    - Callbacks: It can be difficult to track all that occurs inside a chain or agent - callbacks help add a level of observability and introspection.
- [LlamaIndex with LangChain](https://www.linkedin.com/posts/pradipnichite_semantic-search-using-llamaindex-and-langchain-activity-7051411046770589696-gWJA?utm_source=share&utm_medium=member_ios)
- [Building a Document-based Question Answering System with LangChain, Pinecone, and LLMs like GPT-4](https://youtu.be/cVA1RPsGQcw)
- [FedLLM (concorrente)](https://blog.fedml.ai/releasing-fedllm-build-your-own-large-language-models-on-proprietary-data-using-the-fedml-platform/)
- [Conversational AI : Understanding the Technology behind Chat-GPT](https://youtu.be/JKoJ5YIr2O4)


## Finetuning/Training
- [Finetuning Large Language Models - Sebastian Raschka](https://magazine.sebastianraschka.com/p/finetuning-large-language-models?utm_source=profile&utm_medium=reader2)
- [LLaMA-Adapter](https://www.linkedin.com/feed/update/urn:li:activity:7059174832759869442?updateEntityUrn=urn%3Ali%3Afs_feedUpdate%3A%28V2%2Curn%3Ali%3Aactivity%3A7059174832759869442%29)
- [How to train your own Large Language Models](https://blog.replit.com/llm-training)
- [How to customize LLMs like ChatGPT with your own data and documents](https://bdtechtalks.com/2023/05/01/customize-chatgpt-llm-embeddings/)


## LLMOps
- [Building LLM applications for production by Chip Huyen](https://www.linkedin.com/posts/revanthmadamala_llm-productionengineering-promptengineering-activity-7054489651880857600-LVKX/?utm_source=share&utm_medium=member_ios)


## Reinforcement Learning from Human Feedback
- [RLHF - Chip Huyen](https://huyenchip.com/2023/05/02/rlhf.html)
- [RLHF by Aman](https://aman.ai/primers/ai/RLHF/)
- [Awesome RLHF](https://github.com/opendilab/awesome-RLHF)
- [Reinforcement Learning for Language Models](https://gist.github.com/yoavg/6bff0fecd65950898eba1bb321cfbd81#towards-rl-without-human-feedback)


## Restringindo o output
- [NeMo Guardrails from NVIDIA](https://www.linkedin.com/posts/rami-krispin_datascience-llm-nlp-activity-7056669315630055424-2njT/?utm_source=share&utm_medium=member_ios)


## Evaluation of LLM
- [PandaLM (ReProducible and Automated Language Model Assessment) is a open-source LLM that evaluates responses](https://github.com/WeOpenML/PandaLM) - LLaMA license
- [Linkedin post 1](https://www.linkedin.com/feed/update/urn:li:activity:7058755790391681024?utm_source=share&utm_medium=member_ios)
- [Linkedin post 2](https://www.linkedin.com/posts/jindong-wang_we-open-sourced-pandalm-reproducible-and-ugcPost-7058406955085664256-ZdL6?utm_source=share&utm_medium=member_ios)
- [LangChain evaluation with LLMs documentation](https://python.langchain.com/en/latest/use_cases/evaluation.html)
- [Plug-and-Play Bias Detection - HuggingFace](https://huggingface.co/spaces/avid-ml/bias-detection)


## Large Language Models Research - Weekly Digest:
- [Nikita Iserson - Lead Machine Learning Engineer at S&P Global](https://www.linkedin.com/in/nikita-iserson/)


## Outros
- Artigos e notícias
    - [Pytorch 2.0 alega ser mais eficiente (19/04/23)](https://www.linkedin.com/posts/michael-gschwind-3704222_pytorch-pytorch2-llm-activity-7054904878757736448-FbFX/?utm_source=share&utm_medium=member_ios)
    - [InstructGPT - Aligning language models to follow instructions - OpenAI (27/01/22)](https://openai.com/research/instruction-following)
    - [Segurança da informação: New ways to manage your data in ChatGPT](https://openai.com/blog/new-ways-to-manage-your-data-in-chatgpt)
    - [Amazon Science - LLM](https://www.amazon.science/tag/large-language-models)
    - [Chaining together Large Language Models with Large Knowledge Graphs](https://blog.diffbot.com/generating-company-recommendations-using-large-language-models-and-knowledge-graphs/)
    - [Sebastian Raschka's LinkedIn post on finetuning](https://www.linkedin.com/posts/sebastianraschka_ai-activity-7060220916240166912-UrTj/?utm_source=share&utm_medium=member_ios)
- Estudo
    - [ChatGPT and LLaMA by Aman](https://www.linkedin.com/posts/amanc_artificialintelligence-machinelearning-ai-activity-7058274203711537152-FhCK?utm_source=share&utm_medium=member_ios)
- Bibliotecas
    - [Run LLM natively on phones](https://www.linkedin.com/posts/sanyambhutani_we-can-now-run-llms-natively-on-phones-activity-7058622883782176768-QGAt?utm_source=share&utm_medium=member_ios)
    - [Toolformer: Language Models Can Teach Themselves to Use Tools](https://aman.ai/primers/ai/toolformer/)
    - [OpenAI API](https://youtu.be/uRQH2CFvedY)