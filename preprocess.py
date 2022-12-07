import re, csv, pandas as pd
from sklearn.model_selection import train_test_split

with open("dataset/Thai-PUD/th_pud-ud-test.conllu", encoding="utf-8") as file,\
    open("dataset/Thai-PUD/th_pud-corpus.tsv", "w", encoding="utf-8") as out:

    out.write("tree_id\ttoken_index\ttoken\thead_index\tdep_type\n")#\tpos\tnote\n")

    tree_id = 0
    line = file.readline()
    num_line = 1
    while line.startswith('#'):

        match = re.match(r"# sent_id = (.+)\n", line)
        assert match, f"Unexpected line: {num_line}; Expect '# sent_id = ...'"
        sent_id = match[1]

        line = file.readline()
        num_line += 1
        match = re.match(r"# text = (.+)\n", line)
        assert match, f"Unexpected line: {num_line}; Expect '# text = ...'"
        text = match[1]

        line = file.readline()
        num_line += 1
        match = re.match(r"# translit = (.+)\n", line)
        assert match, f"Unexpected line: {num_line}; Expect '# translit = ...'"
        translit = match[1]

        line = file.readline()
        num_line += 1
        match = re.match(r"# text_en = (.+)\n", line)
        assert match, f"Unexpected line: {num_line}; Expect '# text_en = ...'"
        text_en = match[1]

        line = file.readline()
        num_line += 1
        while re.match(r"\d", line):
            match = re.match(r"(\d+)\t(.+)\t(.+)\t(.+)\t(.+)\t(.+)\t(\d+)\t(.+)\t(.+)\t(.+)\n", line)
            assert match, f"Unexpected line: {num_line}; Expect corpus body"
            token_index, token, _, pos, _, _, head_index, dep_type, _, note =\
            match[1], match[2], match[3], match[4], match[5], match[6], match[7], match[8], match[9], match[10]
            out.write(f"{tree_id}\t{int(token_index) - 1}\t{token}\t{int(head_index) - 1 if head_index != '0' else 'ROOT'}\t{dep_type}\n")#\t{pos}\t{note}\n")
            line = file.readline()
            num_line += 1
        assert line == '\n', f"Expect a newline character at line {num_line}"
        tree_id += 1
        line = file.readline()
        num_line += 1

our_corpus = pd.read_csv("dataset/50_sentence/50_sentence-corpus.tsv", sep='\t', quoting=csv.QUOTE_NONE)
our_corpus["corpus"] = "50_sentence"

ud_corpus = pd.read_csv("dataset/Thai-PUD/th_pud-corpus.tsv", sep='\t', quoting=csv.QUOTE_NONE)
ud_corpus["corpus"] = "th_pud"

ud_len = len(set(ud_corpus["tree_id"]))
our_corpus["tree_id"] = our_corpus["tree_id"].apply(lambda tree_id: tree_id + ud_len)

combined = pd.concat([ud_corpus, our_corpus])
combined.reset_index(drop=True, inplace=True)
combined["index"] = combined.index
combined["head_index"] = combined["index"].apply(lambda i: int(combined.at[i, "head_index"]) if combined.at[i, "head_index"] != "ROOT" else combined.at[i, "token_index"])
combined.drop(columns=["index"], inplace=True)
combined["rel_head"] = combined["head_index"] - combined["token_index"]

trees = []
LIMIT = 10
for i in range(len(set(combined["tree_id"]))):
    sentence = combined[combined["tree_id"] == i]
    assert len(set(sentence["corpus"])) == 1, f"Sentence comes from 2 different corpora: {i}"
    tree = {}
    tree["tokens"] = sentence["token"].to_list()
    tree["heads"] = sentence["rel_head"].apply(lambda rel_head: str(rel_head) if -LIMIT <= rel_head <= LIMIT else "OUT_OF_RANGE").to_list()
    tree["dep_types"] = sentence["dep_type"].to_list()
    corpus = sentence["corpus"].iloc[0]
    tree["corpus"] = corpus
    tree["tree_id"] = i if corpus == "th_pud" else i - ud_len
    trees.append(tree)

pre_split = pd.DataFrame(trees)

train, rest = train_test_split(pre_split, train_size=0.7)
dev, test = train_test_split(rest, train_size=0.5)
train.to_csv("our_model/train.tsv", sep='\t', index=False)
dev.to_csv("our_model/dev.tsv", sep='\t', index=False)
test.to_csv("our_model/test.tsv", sep='\t', index=False)

train = pd.read_csv("our_model/train.tsv", sep='\t')
dev = pd.read_csv("our_model/dev.tsv", sep='\t')
test = pd.read_csv("our_model/test.tsv", sep='\t')

train_id = set(train[train["corpus"] == "th_pud"]["tree_id"])
dev_id = set(dev[dev["corpus"] == "th_pud"]["tree_id"])
test_id = set(test[test["corpus"] == "th_pud"]["tree_id"])

with open("dataset/Thai-PUD/th_pud-ud-test.conllu", encoding="utf-8") as file:
    data = file.read().split("\n\n")
    if data[-1] == '':
        del data[-1]
    length = len(data)

train = (data[i] for i in range(len(data)) if i in train_id)
dev = (data[i] for i in range(len(data)) if i in dev_id)
test = (data[i] for i in range(len(data)) if i in test_id)

with open("deep_biaffine/train.conllu", "w", encoding="utf-8") as train_set:
    train_set.write("\n\n".join(train))
    train_set.write("\n\n")
with open("deep_biaffine/dev.conllu", "w", encoding="utf-8") as dev_set:
    dev_set.write("\n\n".join(dev))
    dev_set.write("\n\n")
with open("deep_biaffine/test.conllu", "w", encoding="utf-8") as test_set:
    test_set.write("\n\n".join(test))
    test_set.write("\n\n")
