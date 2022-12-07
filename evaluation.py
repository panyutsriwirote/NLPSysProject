import numpy as np, pandas as pd
from sklearn.metrics import classification_report
from ast import literal_eval
from datasets import Dataset, DatasetDict, Features, Sequence, ClassLabel, Value

def evaluate(head_classifier, head_label_list: list, head_dataset, label_classifier, dep_type_list: list, label_dataset):

    predictions, _, _ = head_classifier.predict(head_dataset["test"])
    predictions = np.argmax(predictions, axis=2)

    bert_prediction = [
        [(id, token, prediction) for (id, token, prediction) in zip(row_id, row_token, row_predictions)]
        for row_id, row_token, row_predictions in zip(head_dataset["test"]["word_ids"], head_dataset["test"]["tokens_bert"], predictions)
    ]

    gold_label = [
        [label for label in row_label ] for row_label in head_dataset["test"]["rel_heads"]
    ]

    wrap_predict = [] 
    for row in bert_prediction:
        predict_row = []
        previous_word = None
        for item in row:
            if item[0] is not None:
                #skip special token
                if item[0] != previous_word:
                    predict_row.append(item[-1])
                    previous_word = item[0]
        wrap_predict.append(predict_row)
    
    head_predictions = [head_label_list[item] for sublist in wrap_predict for item in sublist]
    head_golds = [head_label_list[item] for sublist in gold_label for item in sublist]

    result = classification_report(head_golds, head_predictions, output_dict=True, labels=head_label_list)
    result_df = pd.DataFrame(result).transpose()
    print(result_df)

    correct, total = 0, 0
    for pred, gold in zip(head_predictions, head_golds):
        if pred != "OUT_OF_RANGE" and pred == gold:
            correct += 1
        total += 1
    UAS = (correct / total) * 100

    predictions, _, _ = label_classifier.predict(label_dataset["test"])
    predictions = np.argmax(predictions, axis=2)

    bert_prediction = [
        [(id, token, prediction) for (id, token, prediction) in zip(row_id, row_token,row_predictions) ]
        for row_id, row_token, row_predictions in zip(label_dataset["test"]["word_ids"], label_dataset["test"]["tokens_bert"], predictions)
    ]

    gold_label = [
        [label for label in row_label ] for row_label in label_dataset["test"]["dep_types"]
    ]

    wrap_predict = [] 
    for row in bert_prediction:
        predict_row = []
        previous_word = None
        for item in row:
            if item[0] is not None:
                #skip special token
                if item[0] != previous_word:
                    predict_row.append(item[-1])
                    previous_word = item[0]
        wrap_predict.append(predict_row)

    label_predictions = [dep_type_list[item] for sublist in wrap_predict for item in sublist]
    label_golds = [dep_type_list[item] for sublist in gold_label for item in sublist]

    result = classification_report(label_golds, label_predictions, output_dict=True, labels=dep_type_list)
    result_df = pd.DataFrame(result).transpose()
    print(result_df)

    assert len(head_predictions) == len(head_golds) == len(label_predictions) == len(label_golds), "Predictions and Golds have unequal length"

    correct, total = 0, 0
    for h_pred, h_gold, l_pred, l_gold in zip(head_predictions, head_golds, label_predictions, label_golds):
        if h_pred != "OUT_OF_RANGE" and h_pred == h_gold and l_pred == l_gold:
            correct += 1
        total += 1
    LAS = correct / total

    return UAS, LAS

def create_eval_dataset(train: pd.DataFrame, dev: pd.DataFrame, test: pd.DataFrame, rel_head_eval, rel_head_list: list, dep_type_list: list):

    train = train.copy()
    dev = dev.copy()
    test = test.copy()

    train = train[train["corpus"] == "th_pud"]
    dev = dev[dev["corpus"] == "th_pud"]
    test = test[test["corpus"] == "th_pud"]

    train["tokens"] = train["tokens"].apply(literal_eval)
    dev["tokens"] = dev["tokens"].apply(literal_eval)
    test["tokens"] = test["tokens"].apply(literal_eval)

    train["dep_types"] = train["dep_types"].apply(literal_eval)
    dev["dep_types"] = dev["dep_types"].apply(literal_eval)
    test["dep_types"] = test["dep_types"].apply(literal_eval)

    train["rel_heads"] = train["rel_heads"].apply(rel_head_eval)
    dev["rel_heads"] = dev["rel_heads"].apply(rel_head_eval)
    test["rel_heads"] = test["rel_heads"].apply(rel_head_eval)

    head_train = train[["tokens", "rel_heads"]].to_dict(orient="series")
    head_dev = dev[["tokens", "rel_heads"]].to_dict(orient="series")
    head_test = test[["tokens", "rel_heads"]].to_dict(orient="series")

    features = Features({
        "tokens": Sequence(Value("string")),
        "rel_heads": Sequence(feature=ClassLabel(names=rel_head_list)),
    })

    head_dataset = DatasetDict({
        "train": Dataset.from_dict(head_train, features=features), 
        "dev": Dataset.from_dict(head_dev, features=features),
        "test": Dataset.from_dict(head_test, features=features)
    })

    label_train = train[["tokens", "dep_types"]].to_dict(orient="series")
    label_dev = dev[["tokens", "dep_types"]].to_dict(orient="series")
    label_test = test[["tokens", "dep_types"]].to_dict(orient="series")

    features = Features({
        "tokens": Sequence(Value("string")),
        "dep_types": Sequence(feature=ClassLabel(names=dep_type_list)),
    })

    label_dataset = DatasetDict({
        "train": Dataset.from_dict(label_train, features=features), 
        "dev": Dataset.from_dict(label_dev, features=features),
        "test": Dataset.from_dict(label_test, features=features)
    })

    return head_dataset, label_dataset
