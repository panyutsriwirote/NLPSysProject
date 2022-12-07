import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

from data import Dictionary, Corpus, PAD_INDEX
from mst import mst
import os
from train import evaluate_predict, arc_accuracy, lab_accuracy
import inspect

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def plot(args, S_arc, heads):
    fig, ax = plt.subplots()
    # Make a 0/1 gold adjacency matrix.
    n = heads.size(1)
    G = np.zeros((n, n))
    heads = heads.squeeze().data.numpy()
    G[heads, np.arange(n)] = 1.
    im = ax.imshow(G, vmin=0, vmax=1)
    fig.colorbar(im)
    plt.savefig('img'+"/"+args.data+'/gold.pdf')
    plt.cla()
    # Plot the predicted adjacency matrix
    A = F.softmax(S_arc.squeeze(0), dim=0)
    fig, ax = plt.subplots()
    im = ax.imshow(A.data.numpy(), vmin=0, vmax=1)
    fig.colorbar(im)
    plt.savefig('img'+"/"+args.data+'/a.pdf')
    plt.cla()
    plt.clf()


def predict(model, words, tags):
    assert type(words) == type(tags)
    if type(words) == type(tags) == list:
        # Convert the lists into input for the PyTorch model.
        words = Variable(torch.LongTensor([words]))
        tags = Variable(torch.LongTensor([tags]))
    # Dissable dropout.
    model.eval()
    # Predict arc and label score matrices.
    S_arc, S_lab = model(words=words, tags=tags)

    # Predict heads
    S = S_arc[0].data.numpy()
    heads = mst(S)

    # Predict labels
    S_lab = S_lab[0]
    select = torch.LongTensor(heads).unsqueeze(0).expand(S_lab.size(0), -1)
    select = Variable(select)
    selected = torch.gather(S_lab, 1, select.unsqueeze(1)).squeeze(1)
    _, labels = selected.max(dim=0)
    labels = labels.data.numpy()

    return heads, labels

def predict_a(model, words, tags, oriheads, orilabels, arc_acc_ls, lab_acc_ls):
    assert type(words) == type(tags)
    if type(words) == type(tags) == list:
        # Convert the lists into input for the PyTorch model.
        words = Variable(torch.LongTensor([words]))
        tags = Variable(torch.LongTensor([tags]))
    arc_acc, lab_acc = 0, 0
    # Dissable dropout.
    model.eval()
    # Predict arc and label score matrices.
    S_arc1, S_lab1 = model(words=words, tags=tags)

    # Predict heads
    S = S_arc1[0].data.numpy()
    heads = mst(S)

    # Predict labels
    S_lab = S_lab1[0]
    select = torch.LongTensor(heads).unsqueeze(0).expand(S_lab.size(0), -1)
    select = Variable(select)
    selected = torch.gather(S_lab, 1, select.unsqueeze(1)).squeeze(1)
    _, labels = selected.max(dim=0)
    labels = labels.data.numpy()

    arc_acc += arc_accuracy(S_arc1, oriheads)
    lab_acc += lab_accuracy(S_lab1, oriheads, orilabels)

    arc_acc_ls.append(arc_acc)
    lab_acc_ls.append(lab_acc)

    return heads, labels, arc_acc_ls, lab_acc_ls

def predict_all(args, model, test_batches):
    # heads_list, labels_list = [], []

    print("Start Predict & Evaluate  in {} ...".format('Test'))
    model.eval()
    # test_batches = batch
    # test_batches = corpus.test.batches(256, length_ordered=True)
    arc_acc, lab_acc = [], []
    arc_word, arc_pos, arc_pred, arc_ans, lab_pred, lab_ans = [], [], [], [], [], []
    # k_out = 0
    # print('Test BACTH: '+str(test_batches)
    # for i in test_batches:
        # print(i)
    for k, batch in enumerate(test_batches, 1):
        words, tags, heads, labels = batch
        if args.cuda:
            words, tags, heads, labels = words.cuda(), tags.cuda(), heads.cuda(), labels.cuda()

        
        arcpred, labpred, arc_acc, lab_acc = predict_a(model, words, tags, heads, labels, arc_acc, lab_acc)


        # S_arc, S_lab = model(words=words, tags=tags)
        # print(words, S_arc)
        # _, arcpred = S_arc.max(dim=-2)
        arc_pred.append(arcpred)
        arc_ans.append(heads)
        arc_word.append(words)
        arc_pos.append(tags)
        lab_ans.append(labels)

        # select = torch.LongTensor(mst(S_arc[0].data.numpy())).unsqueeze(0).expand(S_lab.size(0), -1)
        # select = Variable(select)
        # selected = torch.gather(S_lab, 1, select.unsqueeze(1)).squeeze(1)
        # _, labpred = selected.max(dim=0)
        lab_pred.append(labpred)

        # arc_acc += arc_accuracy(S_arc, heads)
        # lab_acc += lab_accuracy(S_lab, heads, labels)
        # k_out += k
    # print(k_out)
        print(k)
    print(arc_pred)
    print(arc_ans[0].data[0])
    print(arc_acc, lab_acc)
    arc_acc[0] /= k
    lab_acc[0] /= k


    # heads, labels = predict(model, words, tags)
    # heads_list.append(heads)
    # labels_list.append(labels)
    return arc_acc, lab_acc, arc_pred, lab_pred, arc_ans, lab_ans


def predict_batch(S_arc, S_lab, tags):
    # Predict heads
    S = S_arc.data.numpy()
    heads = mst(S)

    # Predict labels
    select = torch.LongTensor(heads).unsqueeze(0).expand(S_lab.size(0), -1)
    select = Variable(select)
    selected = torch.gather(S_lab, 1, select.unsqueeze(1)).squeeze(1)
    _, labels = selected.max(dim=0)
    labels = labels.data.numpy()
    return heads, labels

def predict_define(args):
    # print(args)
    args.cuda = torch.cuda.is_available()
    data_path = args.data
    vocab_path = args.vocab
    model_path = args.checkpoints
    # data_path = 'data/ud/UD_English-EWT'
    # vocab_path = 'vocab/train'
    # model_path = 'checkpoints/enmodel.pt'

    corpus = Corpus(data_path=data_path, vocab_path=vocab_path)
    index2word = corpus.dictionary.i2w
    index2pos = corpus.dictionary.i2t
    index2label = corpus.dictionary.i2l
    model = torch.load(model_path, map_location=torch.device('cuda' if args.cuda else 'cpu'))
    print(model)
    batches = corpus.test.batches(1, shuffle=False)
    
    # print(index2word[0])
    # print([(step, batch) for step, batch in enumerate(batches, 1)])
    word_all, pos_all, head_all, label_all, label_predall = [], [], [], [], []
    arc_correct = 0
    arc_label_correct = 0
    total = 0
    for i in batches:
        # print(i)
        # words, tags, heads, labels = next(batches)
        words, tags, heads, labels = i
    # print(words, tags, heads, labels)

        S_arc, S_lab = model(words=words, tags=tags)

        # arc_acc, lab_acc, _, _, _, _ = evaluate_predict(args, model, corpus)

        plot(args, S_arc, heads)

        heads_pred, labels_pred = predict(model, words, tags)

        word_data, pos_data, label_data, label_pred = [], [], [], []

        for i in words[0].data.numpy():
            word_data.append(index2word[i])
        for j in tags[0].data.numpy():
            pos_data.append(index2pos[j])
        for k in labels[0].data.numpy():
            label_data.append(index2label[k])
        for l in labels_pred:
            label_pred.append(index2label[l])
        # print("Word: ", word_data, '\n', 'POS :', pos_data)
        print("Head Pred: ", heads_pred, '\n', 'Head Data :', heads[0].data.numpy())
        arc_correct += list(heads_pred == heads[0].data.numpy()).count(True)
        print("Label Pred: ", label_pred, '\n', 'Label Data :', label_data)
        arc_label_correct += list((heads_pred == heads[0].data.numpy()) & (np.array(label_pred) == np.array(label_data))).count(True)
        total += len(heads_pred)
        
        word_all.append(word_data)
        pos_all.append(pos_data)
        head_all.append(heads_pred)
        label_all.append(label_data)
        label_predall.append(label_pred)
        # break
    arc_acc, lab_acc, _, _, _, _ = evaluate_predict(args, model, corpus)
    # print('Arc Accuracy: {} , Label Accuracy: {}'.format(arc_acc, lab_acc))
    print("UAS:", arc_correct / total, "LAS:", arc_label_correct / total)

    to_conllu(word_all, pos_all, head_all, label_predall)

def to_conllu(word_sen, pos_sen, head_sen, label_predsen):
    with open("prediction.conllu", "w", encoding="utf-8") as out:
        for i in range(len(word_sen)):
            for idx, word in enumerate(word_sen[i]):
                # print(idx)
                out.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(idx, word, '_', pos_sen[i][idx], '_', '_', head_sen[i][idx], label_predsen[i][idx], '_', '_'))
            out.write('\n')
        out.write('\n')


# if __name__ == '__main__':

#     # parser = argparse.ArgumentParser()
#     # parser.add_argument('--data', default='~/data/ptb-stanford')
#     # parser.add_argument('--out', default='vocab')
#     # args = parser.parse_args()
#     # main(args)

#     data_path = 'data/ud/UD_English-EWT'
#     vocab_path = 'vocab/train'
#     model_path = 'checkpoints/enmodel.pt'
#     main(args)
