import pandas as pd, numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import load_metric, Dataset, DatasetDict, Features, Sequence, ClassLabel, Value
from ast import literal_eval

def train_classifier(train: pd.DataFrame, dev: pd.DataFrame, test: pd.DataFrame, label_name: str, label_list: list, window_length: int, encoder: str, batch_size: int, num_epoch: int, save_strategy: str):

    train = train.copy()
    dev = dev.copy()
    test = test.copy()

    if label_name == "rel_heads":
        labelize = lambda x: [h if h == "OUT_OF_RANGE" or -window_length <= int(h) <= window_length else "OUT_OF_RANGE" for h in literal_eval(x)]
        train["rel_heads"] = train["rel_heads"].apply(labelize)
        dev["rel_heads"] = dev["rel_heads"].apply(labelize)
        test["rel_heads"] = test["rel_heads"].apply(labelize)
    elif label_name == "dep_types":
        train = train[train["corpus"] == "th_pud"]
        dev = dev[dev["corpus"] == "th_pud"]
        test = test[test["corpus"] == "th_pud"]
        train["dep_types"] = train["dep_types"].apply(literal_eval)
        dev["dep_types"] = dev["dep_types"].apply(literal_eval)
        test["dep_types"] = test["dep_types"].apply(literal_eval)
    else:
        raise Exception(f"Invalid label name: {label_name}; Expect 'rel_heads' or 'dep_types'")
    
    train["tokens"] = train["tokens"].apply(literal_eval)
    dev["tokens"] = dev["tokens"].apply(literal_eval)
    test["tokens"] = test["tokens"].apply(literal_eval)

    train = train[["tokens", label_name]].to_dict(orient="series")
    dev = dev[["tokens", label_name]].to_dict(orient="series")
    test = test[["tokens", label_name]].to_dict(orient="series")

    features = Features({
        "tokens": Sequence(Value('string')),
        label_name: Sequence(feature=ClassLabel(names=label_list))
    })

    data = DatasetDict({
        "train": Dataset.from_dict(train, features=features), 
        "dev": Dataset.from_dict(dev, features=features),
        "test": Dataset.from_dict(test, features=features)
    })

    tokenizer = AutoTokenizer.from_pretrained(encoder)
    tokenized_datasets = data.map(lambda x: tokenize_and_align_labels(x, tokenizer, label_name), batched=True)

    label_list = data['train'].features[label_name].feature.names
    model = AutoModelForTokenClassification.from_pretrained(encoder, num_labels=len(label_list))
    args = TrainingArguments(
        f"wangchan-head-{window_length}token" if label_name == "rel_heads" else "wangchan-label",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epoch,
        save_strategy=save_strategy,
        # save_total_limit = 2,
        # load_best_model_at_end=True,
        weight_decay=0.01,
        push_to_hub=False,
    )
    data_collator = DataCollatorForTokenClassification(tokenizer)
    metric = load_metric("seqeval")

    CLASSIFIER = Trainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["dev"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda x: compute_metrics(x, metric, label_list)
    )
    CLASSIFIER.train()
    # CLASSIFIER.save_model()
    return CLASSIFIER

def tokenize_and_align_labels(d, tokenizer, feature_to_align: str):
    label_all_tokens = True
    tokenized_inputs = tokenizer(d["tokens"], truncation=True, is_split_into_words=True)
    tokens = [tokenizer.convert_ids_to_tokens(x) for x in tokenized_inputs["input_ids"] ]
    ids=[]
    labels = []
    for i, label in enumerate(d[feature_to_align]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # Set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)
        ids.append(word_ids)

    tokenized_inputs["labels"] = labels
    tokenized_inputs["word_ids"] = ids
    tokenized_inputs["tokens_bert"] = tokens
    return tokenized_inputs

def compute_metrics(p, metric, label_list):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
