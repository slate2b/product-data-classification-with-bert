"""
Trainer

This script...

This file can also be imported as a module and contains the following
functions:

    * function_name - brief desc
    * function_name - brief desc
"""

import torch
from preprocessor import prepare_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from datasets import DatasetDict
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score

"""
Original dataset taken from https://www.kaggle.com/datasets/lokeshparab/amazon-products-dataset/data
"""

source_filepath = "./source_data/amazon-products-text-and-label_ids.csv"
llm = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(llm)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_labels = 20
model = (AutoModelForSequenceClassification
         .from_pretrained(llm, num_labels=num_labels)
         .to(device))


def tokenize_data(batch):
    """

    :param batch:
    :return:
    """
    return tokenizer(batch['text'], padding=True, truncation=True)


def get_accuracy(preds):
    """

    :param preds:
    :return:
    """

    predictions = preds.predictions.argmax(axis=-1)
    labels = preds.label_ids
    accuracy = accuracy_score(preds.label_ids, preds.predictions.argmax(axis=-1))

    return {'accuracy': accuracy}


def get_training_args(encoded_data, model_name):
    """

    :param encoded_data:
    :param model:
    :return:
    """

    batch_size = 8
    logging_steps = len(encoded_data["train"]) // batch_size
    output_directory = f"{model_name}-finetuned-tiny-product-data"
    training_args = TrainingArguments(output_dir=output_directory,
                                      num_train_epochs=3,
                                      learning_rate=2e-5,
                                      per_device_train_batch_size=batch_size,
                                      per_device_eval_batch_size=batch_size,
                                      weight_decay=0.01,
                                      eval_strategy="epoch",
                                      disable_tqdm=False,
                                      logging_steps=logging_steps,
                                      log_level="error",
                                      optim='adamw_torch'
                                      )
    return training_args


def train_model(model, training_args, encoded_data, tokenizer):
    """

    :param model:
    :param training_args:
    :param encoded_data:
    :param tokenizer:
    :return:
    """

    torch.cuda.empty_cache()

    trainer = Trainer(model=model,
                      args=training_args,
                      compute_metrics=get_accuracy,
                      train_dataset=encoded_data['train'],
                      eval_dataset=encoded_data['validation'],
                      tokenizer=tokenizer)

    trainer.train()

    trainer.evaluate()

    preds = trainer.predict(encoded_data['test'])
    print("\n")
    print(preds)

    accuracy = get_accuracy(preds)
    print("\n")
    print(accuracy)

    trainer.save_model()


# Create train, test, and validation splits from the source dataset
data = prepare_dataset(source_filepath)

# Tokenize the data
#data_encoded = data.map(tokenize_data, batched=True, batch_size=None)

#print(data_encoded['train'][0])

tiny_data = DatasetDict()
tiny_data['train'] = data['train'].shuffle(seed=1).select(range(7000))
tiny_data['validation'] = data['validation'].shuffle(seed=1).select(range(1500))
tiny_data['test'] = data['test'].shuffle(seed=1).select(range(1500))

tiny_data_encoded = tiny_data.map(tokenize_data, batched=True, batch_size=None)

training_arguments = get_training_args(encoded_data=tiny_data_encoded, model_name=llm)

train_model(model=model, training_args=training_arguments, encoded_data=tiny_data_encoded, tokenizer=tokenizer)
