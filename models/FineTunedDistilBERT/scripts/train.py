import sys
import os

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from models.FineTunedDistilBERT.model.utils import load_config
from models.FineTunedDistilBERT.scripts.preprocess import data_load, get_mapping_POS, get_bert_model, tokenize_data, prepare_POS, prepare_words, get_bad_tokens_and_map, align_labels, get_dataset, get_data_collator
import evaluate
from transformers import TrainingArguments, Trainer
import numpy as np

def compute_metrics(p):
        pred, labels = p
        pred = np.argmax(pred, axis=2)
        # ignore predictions where we don't have a valid label
        mask = labels != -100
        pred = pred[mask]
        labels = labels[mask]
        return metric.compute(predictions=pred, references=labels)

if __name__ == "__main__":
    # load the config
    config = load_config()

    # load the data
    sentence_tokens, sentence_POS, val_sentence_tokens, val_sentence_POS, test_sentence_tokens, test_sentence_POS, vocab, vocab_POS = data_load(config['train_file'], config['val_file'], config['test_file'])

    # get the mapping of the POS tags to indices and vice versa
    POS_to_idx, idx_to_POS = get_mapping_POS(vocab_POS)

    # get the BERT model for token classification and the tokenizer
    model, tokenizer = get_bert_model(config['model_name'], POS_to_idx, idx_to_POS)

    # tokenize the data
    sentence_POS_idx, val_sentence_POS_idx, test_sentence_POS_idx = prepare_POS(sentence_POS, val_sentence_POS, test_sentence_POS, POS_to_idx)
    text_sents, val_text_sents, test_text_sents = prepare_words(sentence_tokens, val_sentence_tokens, test_sentence_tokens)

    tokenized_data = tokenize_data(text_sents, tokenizer)
    val_tokenized_data = tokenize_data(val_text_sents, tokenizer)
    test_tokenized_data = tokenize_data(test_text_sents, tokenizer)

    # get the bad tokens and the mapping
    bad_tokens, mapping, val_mapping, test_mapping = get_bad_tokens_and_map(tokenized_data, sentence_tokens, val_tokenized_data, val_sentence_tokens, test_tokenized_data, test_sentence_tokens, tokenizer)

    # align the labels
    all_labels, val_all_labels, test_all_labels = align_labels(mapping, sentence_POS_idx, val_mapping, val_sentence_POS_idx, test_mapping, test_sentence_POS_idx)

    # get the dataset
    dataset = get_dataset(sentence_tokens, sentence_POS_idx, val_sentence_tokens, val_sentence_POS_idx, test_sentence_tokens, test_sentence_POS_idx, all_labels, val_all_labels, test_all_labels, tokenized_data, val_tokenized_data, test_tokenized_data)

    # get the data collator and the metric
    data_collator = get_data_collator(tokenizer)
    metric = evaluate.load("accuracy")

    training_args = TrainingArguments(
        disable_tqdm=False,
        output_dir="models/FineTunedDistilBERT/saved_models",
        learning_rate=config['learning_rate'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        num_train_epochs=config['epochs'],
        weight_decay=config['weight_decay'],
        evaluation_strategy="epoch"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    result = trainer.train()
    predictions = trainer.predict(dataset["validation"])
    test_results = compute_metrics((predictions.predictions, predictions.label_ids))

    pred = np.argmax(predictions.predictions, axis=2)
    print(test_results)

    # save the predictions
    with open("models/FineTunedDistilBERT/predictions/predictions.txt", "w") as f:
        for pred in predictions:
            f.write(str(pred) + "\n")

