# Thai Dependency Parsing as Token Classification using WangchanBERTa

## Introduction
- Nowadays Dependency parsing become a powerful tool for identifying the sentence structure and grammatical relation between the units in sentences, it gets a lot of attention from many people in the field of Natural Language Processing (NLP). Continually, There are a lot of Dependency parsing development which is efficiency for the international language like English and Chinese. However, The Dependency parsing for Thai still has the problem with some specific charateristic in Thai language. Therefore, we are interested to study about the method that can make the problems and lead the Thai dependency parser to become better. 
- The recent dependency parsers were developed based on two active algorithms such as transition-based parsing and graph-based parsing and the model they used are....
- As mentioned so far, our goal is to reformulate the dependency parser method by interpreting the problems to become the token-classification form which is much more easy than the previous mentioned method based on transition-based parsing and graph-based parsing algorithm.
- Summarize your results

## Our Model
Explain your model here how it works.

- Input is ...
- Output is ...
- Model description
- Equation as necessary e.g. $\alpha_3$

## Dataset
- The datasets we use were devided into two sections. The first section is the set of 50 sentences including 560 tokens that we recreate and annotate it using the universal independency annotation guidelines from [Thai Dependency Tree Bank](https://www.arts.chula.ac.th/ling/resources/publications/). The second section is the set of 1000 annotated sentences with 22322 tokens from [THAI PUD](https://universaldependencies.org/treebanks/th_pud/index.html).

-  The label we  

- Label distribution// 
head index-token id = relative head position// relation

| relative head position | Frequency |
|-------|-----------|
|  -10 |    0   |
|     -9|    1   |
|  -8 |  1 |
|   -7|  2 |
|   -6|  3 |
|   -5|  11 |
|   -4| 20 |
|   -3| 23  |
|   -2| 63  |
|   -1| 208  |
|   0|  50 |
|   1|   122|
|   2| 29  |
|   3|  8 |
|   4|  5 |
|   5| 4 |
|   6| 3  |
|   7|  0 |
|   8|  3 |
|   9|  2|
|   10|  0 |
|   out of range|  2 |
## Experimental setup
- Which pre-trained model? How did you pretrain embeddings?
- Computer. How long?
- Hyperparameter tuning? Dropout? How many epochs?

## Results
How did it go?  + Interpret results.

### Model comparison
|      Model      |   UAS   |   LAS   |
|-----------------|---------|---------|
|Wangchan-Token-5 |**78.02**|**71.39**|
|Wangchan-Token-10|  74.96  |  69.69  |
|Thai Biaffine    |  75.79  |  70.38  |

## Conclusion
- What task? What did we do?
- Summary of results.