"""
Trainer

This is the main script in a project designed to test the effectiveness
of fine-tuned BERT model for the use-case of classifying a product
record within a product taxonomy.

Script begins by asking user whether they want to perform a test run
using a tiny version of the dataset.  If user declines, then script
will perform fine-tuning operations using the full dataset.

Be sure to install the following libraries prior to use:

    transformers[sentencepiece]
    datasets
    torch
    matplotlib
    scikit-learn
    accelerate

This file can also be imported as a module and contains the following
functions:

    * tokenize_data - tokenize the dataset
    * get_accuracy - returns a simple calculated accuracy value
    * get_training_args - returns training arguments well-suited for fine-tuning
    * finetune_model - meant to be called from use_tiny_data or use_full_dataset
    * use_tiny_dataset - executes fine-tuning operations using a tiny version of the dataset
    * use_full_dataset - executes fine-tuning operations using the full dataset

    Original dataset taken from https://www.kaggle.com/datasets/lokeshparab/amazon-products-dataset/data

"""

import torch
from preprocessor import prepare_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from datasets import DatasetDict
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score

#####
# Global Configuration Variables - change as needed/desired
#

_source_filepath = "./source_data/amazon-products-text-and-label_ids.csv"
_llm = 'bert-base-uncased'
_tokenizer = AutoTokenizer.from_pretrained(_llm)
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_num_labels = 20
_model = (AutoModelForSequenceClassification
          .from_pretrained(_llm, num_labels=_num_labels)
          .to(_device))
_batch_size = 8


def tokenize_data(batch):
    """
    Tokenizes the given batch of data using the tokenizer defined
    in the trainer.py configuration variables section.

    Use this function as an argument in the DatasetDict.map function.

    :param batch: the DatasetDict batch from the map function
    :return: the encoded batch
    """
    return _tokenizer(batch['text'], padding=True, truncation=True)


def get_accuracy(preds):
    """
    Takes model predictions and calculates the accuracy score.

    :param preds: The predictions from the model
    :return: python dictionary - the accuracy of the model
    """

    predictions = preds.predictions.argmax(axis=-1)
    labels = preds.label_ids
    accuracy = accuracy_score(labels, predictions)

    return {'accuracy': accuracy}


def get_training_args(encoded_data, model_name):
    """
    Uses the encoded_data and model_name arguments to
    configure training arguments well-suited for the
    classification task.

    :param encoded_data: DatasetDict - the encoded data to be used
    :param model_name: str - the name of the pretrained model
    :return: TrainingArguments object
    """

    logging_steps = len(encoded_data["train"]) // _batch_size
    output_directory = f"{model_name}-finetuned-product-data"
    training_args = TrainingArguments(output_dir=output_directory,
                                      num_train_epochs=2,
                                      learning_rate=2e-5,
                                      per_device_train_batch_size=_batch_size,
                                      per_device_eval_batch_size=_batch_size,
                                      weight_decay=0.01,
                                      eval_strategy="epoch",
                                      disable_tqdm=False,
                                      logging_steps=logging_steps,
                                      log_level="error",
                                      optim='adamw_torch'
                                      )
    return training_args


def finetune_model(model, training_args, encoded_data, tokenizer):
    """
    Fine-tunes a pretrained model, performs some simple
    evaluations, and saves the fine-tuned model.

    :param model: Pretrained model via transformers AutoModelForSequenceClassification
    :param training_args: transformers TrainingArguments object
    :param encoded_data: DatasetDict - the encoded data for fine-tuning
    :param tokenizer: transformers AutoTokenizer from_pretrained

    :return: None
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

    accuracy = get_accuracy(preds)
    print("\n")
    print(accuracy)

    print("\nSaving model...")
    trainer.save_model()


def use_tiny_dataset(dataset):
    """
    Performs fine-tuning operations using a tiny version of the
    given dataset.

    :param dataset: DatasetDict - T=the full dataset
    :return: None
    """

    data = dataset

    tiny_data = DatasetDict()
    tiny_data['train'] = data['train'].shuffle(seed=1).select(range(70))
    tiny_data['validation'] = data['validation'].shuffle(seed=1).select(range(15))
    tiny_data['test'] = data['test'].shuffle(seed=1).select(range(15))

    print("Tokenizing the data...\n")
    tiny_data_encoded = tiny_data.map(tokenize_data, batched=True, batch_size=None)

    print("\nTokenization completed.  Proceeding with fine-tuning...\n")
    training_arguments = get_training_args(encoded_data=tiny_data_encoded, model_name=_llm)

    finetune_model(model=_model, training_args=training_arguments, encoded_data=tiny_data_encoded, tokenizer=_tokenizer)


def use_full_dataset(dataset):
    """
    Performs fine-tuning operations using the full dataset.

    :param dataset: DatasetDict - the full dataset
    :return: None
    """

    data = dataset

    print("Tokenizing the data...\n")
    data_encoded = data.map(tokenize_data, batched=True, batch_size=None)

    print("\nTokenization completed.  Proceeding with fine-tuning...\n")
    training_arguments = get_training_args(encoded_data=data_encoded, model_name=_llm)

    finetune_model(model=_model, training_args=training_arguments, encoded_data=data_encoded, tokenizer=_tokenizer)


def main():
    """
    The main function of the script.

    :return: None
    """

    # Create train, test, and validation splits from the source dataset
    data = prepare_dataset(_source_filepath)

    input_valid = False

    # Prompt user for input related to generator mode
    while not input_valid:
        user_input = input("\nWant to perform a test run using a tiny dataset?  (Y/n)?")
        user_input = user_input.lower()

        if user_input == "y" or user_input == "n" or user_input == "yes" or user_input == "no":

            input_valid = True

            if user_input == "y" or user_input == "yes":
                print("\nRunning test run using a tiny version of the dataset...\n")
                use_tiny_dataset(data)
            else:
                print("\nProceeding with fine-tuning using the full dataset...\n")
                use_full_dataset(data)
        else:
            print("\n  - Invalid Input - Please try again.")

    exit()


if __name__ == "__main__":
    main()
