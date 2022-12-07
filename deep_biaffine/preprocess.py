#!/usr/bin/env python
import argparse
import os
from collections import Counter
import unicodedata

import numpy as np


def is_number(s):
    s = s.replace(',', '') # 10,000 -> 10000
    s = s.replace(':', '') # 5:30 -> 530
    s = s.replace('-', '') # 17-08 -> 1708
    s = s.replace('/', '') # 17/08/1992 -> 17081992
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def start_conll_count():
    word_counts = Counter()
    tag_counts = Counter()
    label_counts = Counter()
    return word_counts, tag_counts, label_counts

def process_conll(input_path, out_path, lower=True, clean=True, p=0.1):
    print(str(input_path))
    word_counts, tag_counts, label_counts = start_conll_count()
    with open(input_path, 'r', encoding="utf8") as f:
        for line in f:
            line = line.strip()
            # print(1,line)
            if not line:
                continue
            if line[0] == "#":
                continue
            
            # if not line:
            #     yield word_counts, tag_counts, label_counts
            #     word_counts, tag_counts, label_counts = start_conll_count()
            #     continue
            
            fields = line.split()
            if fields and ((len(fields)==10) and ('-' not in fields[0])):
                if (' ' != fields[1] and (fields[1] != '.' or fields[3] != 'PUNCT')):
                    assert len(fields) == 10, "invalid conllu line: %s" %line
                    word = fields[1].lower() if lower else fields[1]
                    tag = fields[3]
                    label = fields[7]
                    word_counts.update([word])
                    tag_counts.update([tag])
                    label_counts.update([label])
        print(word_counts, tag_counts, label_counts)
    with open(out_path + '.words.txt', 'w', encoding="utf8") as f:
        for word, count in word_counts.most_common():
            processed = word
            if count == 1:
                if is_number(word) and clean:
                    processed = '<num>'
                elif np.random.random() < p:
                    processed = '<unk>'
            # f.write('{} {} {}'.format(word, processed, count))
            # f.write('\n')
            print('{} {} {}'.format(word, processed, count), file=f)
    with open(out_path + '.tags.txt', 'w', encoding="utf8") as f:
        for tag, count in tag_counts.most_common():
            # f.write('{} {}'.format(tag, count))
            # f.write('\n')
            print('{} {}'.format(tag, count), file=f)
    with open(out_path + '.labels.txt', 'w', encoding="utf8") as f:
        for label, count in label_counts.most_common():
            # f.write('{} {}'.format(label, count))
            # f.write('\n')
            print('{} {}'.format(label, count), file=f)


def compare_vocabulary(train_path, dev_path, test_path):
    train_vocab = dict()
    dev_vocab = dict()
    test_vocab = dict()

    def read_dict(path, dict):
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                word, _, count = line.split()
                dict[word] = int(count)

    read_dict(train_path, train_vocab)
    read_dict(dev_path, dev_vocab)
    read_dict(test_path, test_vocab)

    nwords_train = len(train_vocab)
    ntokens_train = sum(train_vocab.values())
    nwords_dev = len(dev_vocab)
    ntokens_dev = sum(dev_vocab.values())
    nwords_test = len(test_vocab)
    ntokens_test = sum(test_vocab.values())
    unseen_words = list(set(dev_vocab.keys()) - (set(train_vocab.keys()) & set(dev_vocab.keys())))
    num_unseen_tokens = sum([dev_vocab[w] for w in unseen_words])
    with open('vocab/data-statistics.csv', 'w', encoding="utf8") as g:
        print('dataset,nwords,ntokens', file=g)
        print('train,{},{}'.format(nwords_train, ntokens_train), file=g)
        print('dev,{},{}'.format(nwords_dev, ntokens_dev), file=g)
        print('test,{},{}'.format(nwords_test, ntokens_test), file=g)
        print('unseen,{},{}'.format(len(unseen_words), num_unseen_tokens), file=g)
    with open('vocab/unseen.txt', 'w', encoding="utf8") as f:
        for word in unseen_words:
            print('{} {}'.format(word, dev_vocab[word]), file=f)


def main(args):
    data = os.path.expanduser(args.data)
    train_conll_path = os.path.join(data, 'train.conll')
    dev_conll_path = os.path.join(data, 'dev.conll')
    test_conll_path = os.path.join(data, 'test.conll')

    train_vocab_path = os.path.join(args.out, 'train')
    dev_vocab_path = os.path.join(args.out, 'dev')
    test_vocab_path = os.path.join(args.out, 'test')

    # print(train_conll_path)
    process_conll(train_conll_path, train_vocab_path, p=0.5, clean=False)
    process_conll(dev_conll_path, dev_vocab_path, p=0.0)
    process_conll(test_conll_path, test_vocab_path, p=0.0)

    compare_vocabulary(
        train_vocab_path + '.words.txt', dev_vocab_path + '.words.txt', test_vocab_path + '.words.txt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='~/data/ptb-stanford')
    parser.add_argument('--out', default='vocab')
    args = parser.parse_args()

    main(args)
