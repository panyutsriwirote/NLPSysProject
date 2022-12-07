# Biaffine dependency parser
A PyTorch reimplementation from [daandouwe](https://github.com/daandouwe/biaffine-dependency-parser) of the neural dependency parser described as follow below:
1. [Deep Biaffine Attention for Neural Dependency Parsing](https://arxiv.org/abs/1611.01734)
2. [Parsing Thai Social Data: A New Challenge for Thai NLP](https://ieeexplore.ieee.org/abstract/document/9045639)
3. [KITTHAI: Thai Elementary Discourse Unit Segmentation](https://aiforthai.in.th/)
4. [Thai Dependency Parsing with Character Embedding](https://ieeexplore.ieee.org/document/8930002)

## Data
You can train on the Penn Treebank, converted to [Stanford Dependencies](https://nlp.stanford.edu/software/stanford-dependencies.shtml). We assume you have the PTB in standard train/dev/test splits in conll-format, stored somewhere in one directory, and that they are named `train.conll`, `dev.conll`, `test.conll`.

## Usage
First, extract a vocabulary:
```bash
mkdir vocab
./preprocess.py --data your/ptb/conll/dir --out vocab
```

Then, train a default model with the following arguments:
```bash
mkdir log checkpoints
./main.py train --data your/ptb/conll/dir
```
Training can be exited at any moment with Control-C and the current model will be evaluated on the development-set.

### Arguments
The following options are available:
```
usage: main.py {train,predict} [...]

Biaffine graph-based dependency parser

positional arguments:
  {train,predict}

optional arguments:
  -h, --help            show this help message and exit

Data:
  --data DATA           location of the data corpus
  --vocab VOCAB         location of the preprocessed vocabulary
  --disable-length-ordered
                        do not order sentences by length so batches have more
                        padding

Embedding options:
  --use-glove           use pretrained glove embeddings
  --use-chars           use character level word embeddings
  --char-encoder {rnn,cnn,transformer}
                        type of character encoder used for word embeddings
  --filter-factor FILTER_FACTOR
                        controls output size of cnn character embedding
  --disable-words       do not use words as input
  --disable-tags        do not use tags as input
  --word-emb-dim WORD_EMB_DIM
                        size of word embeddings
  --tag-emb-dim TAG_EMB_DIM
                        size of tag embeddings
  --emb-dropout EMB_DROPOUT
                        dropout used on embeddings

Encoder options:
  --encoder {rnn,cnn,transformer,none}
                        type of sentence encoder used

RNN options:
  --rnn-type {RNN,GRU,LSTM}
                        type of rnn
  --rnn-hidden RNN_HIDDEN
                        number of hidden units in rnn
  --rnn-num-layers RNN_NUM_LAYERS
                        number of layers
  --batch-first BATCH_FIRST
                        number of layers
  --rnn-dropout RNN_DROPOUT
                        dropout used in rnn

CNN options:
  --cnn-num-layers CNN_NUM_LAYERS
                        number convolutions
  --kernel-size KERNEL_SIZE
                        size of convolution kernel
  --cnn-dropout CNN_DROPOUT
                        dropout used in cnn

Transformer options:
  --N N                 transformer options
  --d-model D_MODEL     transformer options
  --d-ff D_FF           transformer options
  --h H                 transformer options
  --trans-dropout TRANS_DROPOUT
                        dropout used in transformer

Biaffine classifier arguments:
  --mlp-arc-hidden MLP_ARC_HIDDEN
                        number of hidden units in arc MLP
  --mlp-lab-hidden MLP_LAB_HIDDEN
                        number of hidden units in label MLP
  --mlp-dropout MLP_DROPOUT
                        dropout used in mlps

Training arguments:
  --multi-gpu           enable training on multiple GPUs
  --lr LR               initial learning rate
  --epochs EPOCHS       number of epochs of training
  --batch-size BATCH_SIZE
                        batch size
  --seed SEED           random seed
  --disable-cuda        disable cuda
  --print-every PRINT_EVERY
                        report interval
  --plot-every PLOT_EVERY
                        plot interval
  --logdir LOGDIR       directory to log losses
  --checkpoints CHECKPOINTS
                        path to save the final model
  ```

## Requirements
```
python>=3.6.0
torch>=0.3.0
numpy
```

## List of Implementation
- [x] Add MST algorithm for decoding.
- [x] Write predicted parses to conll file.
- [x] A couple of full runs of the model for results.
- [x] Enable multi-GPU training
- [x] Work on character-level embedding of words (CNN or LSTM).
- [x] Implement RNN options: RNN, GRU, (RAN?)
- [x] Character level word embeddings: CNN
- [x] Character level word embeddings: RNN
- [x] Different encoder: [Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html).
- [x] Different encoder: CNN (again see [spaCy's parser](https://spacy.io/api/)).

## Citation
```
@INPROCEEDINGS{9045639,
  author={Singkul, Sattaya and Khampingyot, Borirat and Maharattamalai, Nattasit and Taerungruang, Supawat and Chalothorn, Tawunrat},
  booktitle={2019 14th International Joint Symposium on Artificial Intelligence and Natural Language Processing (iSAI-NLP)}, 
  title={Parsing Thai Social Data: A New Challenge for Thai NLP}, 
  year={2019},
  volume={},
  number={},
  pages={1-7},
  doi={10.1109/iSAI-NLP48611.2019.9045639}}
  
@INPROCEEDINGS{8930002,
  author={Singkul, Sattaya and Woraratpanya, Kuntpong},
  booktitle={2019 11th International Conference on Information Technology and Electrical Engineering (ICITEE)}, 
  title={Thai Dependency Parsing with Character Embedding}, 
  year={2019},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/ICITEED.2019.8930002}}

@misc{dozat2017deep,
      title={Deep Biaffine Attention for Neural Dependency Parsing}, 
      author={Timothy Dozat and Christopher D. Manning},
      year={2017},
      eprint={1611.01734},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
