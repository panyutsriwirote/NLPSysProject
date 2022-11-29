# Thai Dependency Parsing as Token Classification using WangchanBERTa

## Introduction
- Motivation: Why is this task interesting or important?
- What other people have done? What model have previous work have used? What did they miss?  (1-2 paragraphs)
- Summarize your idea
- Summarize your results

## Our Model
Explain your model here how it works.

- Input is ...
- Output is ...
- Model description
- Equation as necessary e.g. $\alpha_3$

## Dataset
- Annotation guidelines
- Results total how many tokens, sentences, etc.
- Label distribution

| Label | Frequency |
|-------|-----------|
|  good |    60%    |
|  bad  |    40%    |

## Experiment setup
- Which pre-trained model? How did you pretrain embeddings?
- Computer. How long?
- Hyperparameter tuning? Dropout? How many epochs?

## Results
How did it go?  + Interpret results.

### Model comparison
|         Model         |   UAS   |   LAS   |
|-----------------------|---------|---------|
|Wangchan-Token-5       |**78.02**|**71.39**|
|Wangchan-Token-10      |  74.96  |  69.69  |
|Thai Biaffine (Default)|  75.79  |  70.38  |
<!-- Wangchan-Token-10's UAS once reached 77.40 but failed to be reproduced -->

## Conclusion
- What task? What did we do?
- Summary of results.