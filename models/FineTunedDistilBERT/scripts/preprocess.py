import sys
import os


# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)


from transformers import AutoTokenizer, AutoModelForTokenClassification
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import DataCollatorForTokenClassification

def data_load(train_file, val_file, test_file):
    """function to load the training, validation and test data
        the function returns the sentences and the POS tags for each sentence
        the function also returns the word and POS vocabularies"""
    sentence_tokens = []
    test_sentence_tokens = []
    sentence_POS = []
    val_sentence_tokens = []
    val_sentence_POS = []
    test_sentence_POS = []
    vocab = []
    vocab_POS = []
    x = open(train_file)
    # reading training data
    for line in x.readlines():
        tokens = []
        POS = []
        pairs = line.split(" ")
        if len(pairs) < 51: # we ignore sentences longer than 50 words
            for pair in pairs:
                pair = pair.strip('\n')
                if pair != "\n":
                    word = pair.split("|")[0]
                    tag = pair.split("|")[1]
                    tokens.append(word)
                    vocab.append(word)
                    vocab_POS.append(tag)
                    POS.append(tag)
            sentence_tokens.append(tokens)
            sentence_POS.append(POS)
    x.close()

    # reading validation data
    x = open(val_file)
    for line in x.readlines():
        tokens = []
        POS = []
        pairs = line.split(" ")
        if len(pairs) < 51:
            for pair in pairs:
                pair = pair.strip('\n')
                if pair != "\n":
                    word = pair.split("|")[0]
                    tag = pair.split("|")[1]
                    tokens.append(word)
                    vocab.append(word)
                    vocab_POS.append(tag)
                    POS.append(tag)
            val_sentence_tokens.append(tokens)
            val_sentence_POS.append(POS)
    x.close()

    # reading test data
    x = open(test_file)
    for line in x.readlines():
        tokens = []
        POS = []
        pairs = line.split(" ")
        if len(pairs) < 51:
            for pair in pairs:
                pair = pair.strip('\n')
                if pair != "\n":
                    word = pair.split("|")[0]
                    tag = pair.split("|")[1]
                    tokens.append(word)
                    vocab.append(word)
                    vocab_POS.append(tag)
                    POS.append(tag)
            test_sentence_tokens.append(tokens)
            test_sentence_POS.append(POS)
    x.close()

    return sentence_tokens, sentence_POS, val_sentence_tokens, val_sentence_POS, test_sentence_tokens, test_sentence_POS, vocab, vocab_POS

def get_bert_model(model_name, POS_to_idx, idx_to_POS):
    """function to get the BERT model for token classification and the tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(POS_to_idx), id2label=idx_to_POS, label2id=POS_to_idx
    )
    return model, tokenizer

def get_mapping_POS(vocab_POS):
    """function to get the mapping of the POS tags to indices and vice versa"""
    vocab_POS = sorted(list(set(vocab_POS)))
    POS_to_idx = {tag: i for i, tag in enumerate(vocab_POS)}
    idx_to_POS = {i: tag for i, tag in enumerate(vocab_POS)}
    return POS_to_idx, idx_to_POS



def prepare_POS(sentence_POS, val_sentence_POS, test_sentence_POS, POS_to_idx):
    """function to prepare the POS tags for the sentences, adding start and pad tokens and tokenize the POS tags"""
    """the function handles all training, validation and test data"""
    sentence_POS_idx = []
    for sentence in sentence_POS:
        sentence_idx = [POS_to_idx[tag] for tag in sentence]
        sentence_POS_idx.append(sentence_idx)

    val_sentence_POS_idx = []
    for sentence in val_sentence_POS:
        sentence_idx = [POS_to_idx[tag] for tag in sentence]
        val_sentence_POS_idx.append(sentence_idx)

    test_sentence_POS_idx = []
    for sentence in test_sentence_POS:
        sentence_idx = [POS_to_idx[tag] for tag in sentence]
        test_sentence_POS_idx.append(sentence_idx)

    return sentence_POS_idx, val_sentence_POS_idx, test_sentence_POS_idx

def prepare_words(sentence_tokens, val_sentence_tokens, test_sentence_tokens):
    """function to prepare the words for processing"""
    text_sents = []
    for sent in sentence_tokens:
        text_sents.append(' '.join(sent))

    val_text_sents = []
    for sent in val_sentence_tokens:
        val_text_sents.append(' '.join(sent))

    test_text_sents = []
    for sent in test_sentence_tokens:
        test_text_sents.append(' '.join(sent))

    return text_sents, val_text_sents, test_text_sents

def tokenize_data(example, tokenizer):
  # example is a string
  inputs = tokenizer(example, padding="max_length", truncation=True, max_length=100, return_tensors="pt")
  return inputs
        
def bad_tokens_and_map(tokenized_sentence, original_sentence, tokenizer, cased=True):
    bad_tokens = []
    detokenized_sentence = tokenizer.convert_ids_to_tokens(tokenized_sentence)
    mapping_to_original = [None] # For the [CLS] token
    detokenized_index = 1 # Skip the [CLS] token
    for i, word in enumerate(original_sentence):
        word = word if cased else word.lower()
        detokenized_word = detokenized_sentence[detokenized_index]
        next_detokenized_word = detokenized_sentence[detokenized_index+1]
        if word != detokenized_word:
            if not next_detokenized_word.startswith('##') and not detokenized_word.startswith('##'):
                bad_tokens.append(word)
            reconstructed_word = detokenized_word
            while word != reconstructed_word:
                detokenized_index += 1
                reconstructed_word += detokenized_sentence[detokenized_index].strip('##')
                mapping_to_original.append(i)
            mapping_to_original.append(i)
        else:
            mapping_to_original.append(i)
        detokenized_index += 1
    while len(mapping_to_original) < len(detokenized_sentence):
        mapping_to_original.append(None)
    return bad_tokens, mapping_to_original

def get_bad_tokens_and_map(tokenized_data, sentence_tokens, val_tokenized_data, val_sentence_tokens, test_tokenized_data, test_sentence_tokens, tokenizer):
    all_bad_tokens = []
    mappings = []
    for i in range(len(tokenized_data['input_ids'])):
        bad_tokens, mapping = bad_tokens_and_map(tokenized_data['input_ids'][i], sentence_tokens[i], tokenizer)
        all_bad_tokens.append(bad_tokens)
        mappings.append(mapping)

    val_all_bad_tokens = []
    val_mappings = []
    for i in range(len(val_tokenized_data['input_ids'])):
        bad_tokens, mapping = bad_tokens_and_map(val_tokenized_data['input_ids'][i], val_sentence_tokens[i], tokenizer)
        val_all_bad_tokens.append(bad_tokens)
        val_mappings.append(mapping)

    test_all_bad_tokens = []
    test_mappings = []
    for i in range(len(test_tokenized_data['input_ids'])):
        bad_tokens, mapping = bad_tokens_and_map(test_tokenized_data['input_ids'][i], test_sentence_tokens[i], tokenizer)
        test_all_bad_tokens.append(bad_tokens)
        test_mappings.append(mapping)

    all_bad_tokens = sorted(list(set([word for sublist in all_bad_tokens for word in sublist])))
    return all_bad_tokens, mappings, val_mappings, test_mappings

def align_labels(mappings, sentence_POS_idx, val_mappings, val_sentence_POS_idx, test_mappings, test_sentence_POS_idx):
    all_labels = []
    for i, mapping in enumerate(mappings):
        labels = []
        previous_word_idx = None
        orig_j = 0
        for j, idx in enumerate(mapping):
            if idx is None:
                labels.append(-100)
            elif idx == previous_word_idx:
                labels.append(-100)
            else:
                labels.append(sentence_POS_idx[i][orig_j])
                orig_j += 1
                previous_word_idx = idx
        all_labels.append(labels)

    val_all_labels = []
    for i, mapping in enumerate(val_mappings):
        labels = []
        previous_word_idx = None
        orig_j = 0
        for j, idx in enumerate(mapping):
            if idx is None:
                labels.append(-100)
            elif idx == previous_word_idx:
                labels.append(-100)
            else:
                labels.append(val_sentence_POS_idx[i][orig_j])
                orig_j += 1
                previous_word_idx = idx
        val_all_labels.append(labels)

    test_all_labels = []
    for i, mapping in enumerate(test_mappings):
        labels = []
        previous_word_idx = None
        orig_j = 0
        for j, idx in enumerate(mapping):
            if idx is None:
                labels.append(-100)
            elif idx == previous_word_idx:
                labels.append(-100)
            else:
                labels.append(test_sentence_POS_idx[i][orig_j])
                orig_j += 1
                previous_word_idx = idx
        test_all_labels.append(labels)

    return all_labels, val_all_labels, test_all_labels

def get_dataset(train_tokens, train_POS, val_tokens, val_POS, test_tokens, test_POS, all_labels, val_all_labels, test_all_labels, tokenized_data, val_tokenized_data, test_tokenized_data):
    # create a dictionart with the training data
    data = {'id': list(range(len(train_tokens))), 'tokens': train_tokens, 'POS': train_POS, 'attention_mask': tokenized_data['attention_mask'].tolist(), 'input_ids': tokenized_data['input_ids'].tolist(), 'labels': all_labels}
    df = pd.DataFrame(data)
    train_dataset = Dataset.from_pandas(df)

    # create a dictionary with the validation data
    val_data = {'id': list(range(len(val_tokens))), 'tokens': val_tokens, 'POS': val_POS, 'attention_mask': val_tokenized_data['attention_mask'].tolist(), 'input_ids': val_tokenized_data['input_ids'].tolist(), 'labels': val_all_labels}
    val_df = pd.DataFrame(val_data)
    val_dataset = Dataset.from_pandas(val_df)

    # create a dictionary with the test data
    test_data = {'id': list(range(len(test_tokens))), 'tokens': test_tokens, 'POS': test_POS, 'attention_mask': test_tokenized_data['attention_mask'].tolist(), 'input_ids': test_tokenized_data['input_ids'].tolist(), 'labels': test_all_labels}
    test_df = pd.DataFrame(test_data)
    test_dataset = Dataset.from_pandas(test_df)

    # create a DatasetDict with both datasets
    datasets = DatasetDict({'train': train_dataset, 'validation': val_dataset, 'test': test_dataset})

    return datasets

def get_data_collator(tokenizer):
    return DataCollatorForTokenClassification(tokenizer=tokenizer)
