# Thai Dependency Parsing as Token Classification using WangchanBERTa

## Introduction
**Dependency parsing** is a task of identifying sentence structure and grammatical relation between units in sentences. It has gained substantial attention in the field of Natural Language Processing (NLP). Many dependency parsers have been developed and achieved very high accuracy for high-resource languages, such as English and Chinese. However, a dependency parser for Thai capable of achieving the same level of accuracy has yet to be developed, which make us interested in the methods that could be used to improve Thai dependency parsing further.

A class of models that is frequently used in modern dependency parsing is **graph-based model**, where every possible head-dependent relation between every token in a sentence is assigned a score representing its likelihood of being an actual relation. The final dependency tree is then obtained by finding the **Maximum Spanning Tree (MST)** of the sentence, a set of relations that result in maximum overall score and still satisfy some grammatical constraints. One popular model that uses such architecture is proposed in the paper [**Deep Biaffine Attention for Neural Dependency Parsing**](https://arxiv.org/abs/1611.01734). In the proposed model, each token is converted into 2 representations, head embedding and dependent embedding, which are then combined in every possible combination to calculate scores.

With the advent of BERT encoder, and its Thai counterpart WangchanBERTa, many NLP tasks have been greatly improved by leveraging its pre-trained contextual embedding. We notice that one of the mechanisms used by BERT to achieve contextuality, namely the use of key and query embeddings, has some similarity to the scoring mechanism described above and can potentially be fine-tuned to achieve the same effect.

Our goal, therefore, is to reformulate the task of dependency parsing into simple token classification with pre-trained BERT as token encoder in hope that some part of the complex graph-based algorithm will be achieved internally by the encoder after fine-tuning.

Our results show that such model is capable of achieving competitive or even higher UAS and LAS compared to graph-based models. However, the fact that our model might require a large dataset to overcome a lack of training samples for longer-range dependencies prevents the model from potentially achieving very high accuracy.

## Our Model
Our dependency parser consists of 2 simple token classification sub-models, **head classifier** and **label classifier**, that use feedforward neural network (FFNN) to independently predict head position and dependency label for each token in a sentence using contextual token embeddings obtained from a pre-trained encoder. Input to both sub-models consists only of sequences of tokens. Output is **relative head positions** for head classifier and **dependency labels** for label classifier.

**Relative head position** is defined as the distance from the token to its head token. Negative distance means the head token is to the left of the token and vice versa. Said definition can be mathematically formulated as

$$ RelHeadPos = HeadPos - TokenPos $$

For root tokens, which have no heads, we define their relative head positions to be **0**.

Theoretically, **relative head position** can be any integer, resulting in an infinitely-large tagset and making the model untrainable. We, therefore, have to compromise by having a **window length** outside of which the model will label the head position as "Out of Range", a label that will be counted as incorrect in evaluation.

For comparison, we train 2 head classifier sub-models, one has a window length of 5 tokens to the left and right while another has 10.

## Dataset
Our dataset can be divided into two parts. The first is a dataset of 50 sentences, consisting of 560 tokens, that we create and annotate, following the [**Universal Dependency annotation guidelines**](https://www.arts.chula.ac.th/ling/resources/publications/). Due to time limitation, we only label the head position of each token. This part, therefore, will only be used to train the head classifier sub-models. The second part is [**Thai-PUD**](https://universaldependencies.org/treebanks/th_pud/index.html), which is a dataset of 1000 annotated sentences, consisting of 22322 tokens. We divide our dataset into train, dev, and test set using **sklearn.model_selection.train_test_split** with a ratio of **70:15:15** respectively.

Overall, our dataset has the following label distribution before splitting

<table><tr><td>

|Relative Head Position|Frequency|
|----------------------|---------|
|         -10          |   109   |
|          -9          |   124   |
|          -8          |   161   |
|          -7          |   251   |
|          -6          |   361   |
|          -5          |   506   |
|          -4          |   740   |
|          -3          |  1372   |
|          -2          |  2620   |
|          -1          |  6715   |
|           0          |  1050   |
|           1          |  5454   |
|           2          |  1231   |
|           3          |   529   |
|           4          |   324   |
|           5          |   218   |
|           6          |   169   |
|           7          |   109   |
|           8          |    91   |
|           9          |    77   |
|          10          |    65   |
|      Out of Range    |   606   |

</td><td>

|Dependency Label|Frequency|
|----------------|---------|
|      case      |  2413   |
|      obj       |  1734   |
|     nsubj      |  1622   |
|    compound    |  1563   |
|      obl       |  1415   |
|     advmod     |  1190   |
|      root      |  1000   |
|      acl       |   976   |
|      aux       |   929   |
|     xcomp      |   780   |
|      conj      |   662   |
|      amod      |   654   |
|       cc       |   619   |
|   acl:relcl    |   613   |
|      mark      |   560   |
|   flat:name    |   507   |
|      nmod      |   505   |
|   nmod:poss    |   438   |
|      cop       |   432   |
|      det       |   413   |
|     nummod     |   372   |
|  compound:prt  |   364   |
|     appos      |   362   |
|     advcl      |   341   |
|     fixed      |   334   |
|     ccomp      |   275   |
|     punct      |   272   |
|    obl:tmod    |   252   |
|      clf       |   247   |
|      dep       |   117   |
|    obl:poss    |   91    |
|    aux:pass    |   87    |
|   nsubj:pass   |   70    |
|     csubj      |   49    |
|   discourse    |   29    |
|   cc:preconj   |   20    |
|   parataxis    |    5    |
|    goeswith    |    4    |
|      iobj      |    2    |
|   dislocated   |    2    |
|   reparandum   |    1    |
|    vocative    |    1    |

</td></tr></table>

## Experimental setup
We use pre-trained **WangchanBERTa** ([airesearch/wangchanberta-base-att-spm-uncased](https://huggingface.co/airesearch/wangchanberta-base-att-spm-uncased)) as token encoder. Both sub-models in our parser are automatically generated using Huggingface's AutoModelForTokenClassification class. We train each sub-model for 50 epochs, which are enough for the models' accuracy to stabilize.

Each parser is then evaluated using **unlabeled attachment score (UAS)** and **labeled attachment score (LAS)**, which are defined as

$$ UAS = {number\ of\ head\ correct\ tokens \over number\ of\ tokens} * 100 $$

$$ LAS = {number\ of\ head\ and\ label\ correct\ tokens \over number\ of\ tokens} * 100 $$

For comparison, we also train and evaluate a Deep Biaffine Dependency Parser on the same dataset using a PyTorch implementation by [**Sattaya Singkul**](https://github.com/JoesSattes/Thai-Biaffine-Dependency-Parsing).

## Results
### Model comparison
|      Model     |   UAS   |   LAS   |
|----------------|---------|---------|
|Wangchan-5Token |**78.02**|**71.39**|
|Wangchan-10Token|  74.96  |  69.69  |
|Deep Biaffine   |  76.66  |  70.56  |

Our results show that simple token classification models that use WangchanBERTa as token encoder are capable of achieving competitive or even higher UAS and LAS compared to Deep Biaffine, which is a graph-based model.

The results also show that increasing the window length can significantly decrease model's accuracy.

After performing basic error analysis, we see that accuracy rate for each label seems to correlate with the number of training samples of that label in the dataset. Most notable is the fact that our models can "correctly" identify "Out of Range" heads with higher accuracy than other close-by shorter-range heads, such as -10 or 10. This effect is most evident in model with smaller window length since such model would have more "Out of Range" training samples.

This leads us to the conclusion that the distance between tokens and heads does not significantly affect model's performance. However, longer-range dependencies would still suffer from low accuracy due to the nature of natural languages that longer-range dependencies are less common and thus will have less training samples. This suggests that our models might require a significantly larger dataset to overcome a lack of training samples for longer-range dependencies.

Nevertheless, even if a larger dataset can be obtained, our models would still have an inherent hard limit that any heads outside of the window length can never be correctly parsed, but this can be kept to a minimum by selecting an appropriate window length that would result in an acceptable guaranteed error rate.

## Conclusion
- We reformulated the task of dependency parsing into simpler token classification problem that uses pre-trained WangchanBERTa as token encoder.
- Our results show that such models are capable of achieving competitive or even higher UAS and LAS compared to graph-based Deep Biaffine model.
- However, our models suffer from low accuracy in longer-range dependencies and might require large dataset to overcome a lack of training samples.
- Our models also have an inherent guaranteed error that must be tuned to be within an acceptable margin.

## Author
- Panyut Sriwirote 6340138322
- Apimon Jitaksorn 6340261822