import random
import os
import sys

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

import torch
from transformers import DistilBertTokenizer, AutoModel
import random
from models.TagInsert.model.utils import Batch


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

def get_bert_model(model_name, device):
    """function to get the BERT model and the tokenizer"""
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    # We want the model to output all hidden states for summing up the last four layers to calculate embeddings
    bert_model = AutoModel.from_pretrained(model_name,
                                    output_hidden_states = True,
                                    )

    bert_model = bert_model.to(device)

    return bert_model, tokenizer

def get_mapping_POS(vocab_POS):
    """function to get the POS vocabulary and the mapping from POS to index"""
    vocab_POS = sorted(list(set(vocab_POS)))
    POS_to_idx = {tag: i+1 for i, tag in enumerate(vocab_POS)}

    # add padding, unknown, and start tokens
    POS_to_idx["<PAD>"] = 0
    POS_to_idx["<START>"] = len(POS_to_idx)
    POS_to_idx["<END>"] = len(POS_to_idx)
    POS_to_idx["<UNK>"] = len(POS_to_idx)
    idx_to_POS = {k: v for v, k in POS_to_idx.items()}

    return POS_to_idx, idx_to_POS

def get_mapping_words(vocab):
    """function to get the word vocabulary and the mapping from word to index"""
    # create mapping for words
    vocab = sorted(list(set(vocab)))
    word_to_idx = {word: i+1 for i, word in enumerate(vocab)}

    # add pad
    word_to_idx["<PAD>"] = 0
    idx_to_word = {k: v for v, k in word_to_idx.items()}

    return word_to_idx, idx_to_word, vocab

def create_mapping(sentence, tokens_tensor, tokenizer, cased = False, subword_manage = 'prefix'):
    """function to create the mapping from the tokenized sentence to the detokenized sentence"""
    """the function takes the sentence, the tokens tensor, the tokenizer, and the casing and subword management options"""
    """It is essentially useful to calculate the word embeddings from the BERT model so that we get the desired subword"""
    mapping = [None] # for the [CLS] token
    detokenized_index = 1 # for the [CLS] token
    # detokenize the sentence
    detokenized_sentence = tokenizer.convert_ids_to_tokens(tokens_tensor)

    for i, word in enumerate(sentence):
        word = word if cased else word.lower()
        # get the detokenized token
        detokenized_token = tokenizer.convert_ids_to_tokens(tokens_tensor[detokenized_index].item())
        # if the word is not equal to the detokenized token, we need to reconstruct the word
        if word != detokenized_token:
            reconstructed_word = detokenized_token
            # while the reconstructed word is not equal to the word, we need to add the index to the mapping
            while reconstructed_word != word:
                detokenized_index += 1
                reconstructed_word += detokenized_sentence[detokenized_index].strip('##')
                mapping.append(i)
            mapping.append(i) 
        else: 
            mapping.append(i)
        # move to the next token
        detokenized_index += 1
    # if the mapping is shorter than the detokenized sentence, we need to add None to the mapping
    while len(mapping) < len(detokenized_sentence):
        mapping.append(None)
    # if the subword management is prefix, we need to merge the subwords
    if subword_manage == 'prefix':
        for i, idx in enumerate(mapping):
            if idx is not None:
                j = i+1
                while mapping[j] == idx:
                    mapping[j] = None
                    j += 1
    # if the subword management is suffix, we need to merge the subwords
    elif subword_manage == 'suffix':
        for i, idx in enumerate(mapping):
            if idx is not None:
                j = i
                while mapping[j+1] == idx:
                    mapping[j] = None
                    j += 1
    return mapping

def prepare_POS(sentence_POS, val_sentence_POS, test_sentence_POS, POS_to_idx, BLOCK_SIZE):
    """function to prepare the POS tags for the sentences, adding start and pad tokens and tokenize the POS tags"""
    """the function handles all training, validation and test data"""
    sentence_POS_idx = []
    for sentence in sentence_POS:
        sentence_idx = [POS_to_idx[tag] for tag in sentence]
        # pad sentence to BLOCK_SIZE
        sentence_idx += [POS_to_idx['<PAD>']] * (BLOCK_SIZE - len(sentence_idx))
        sentence_POS_idx.append(sentence_idx)

    val_sentence_POS_idx = []
    for sentence in val_sentence_POS:
        sentence_idx = [POS_to_idx[tag] for tag in sentence]
        # pad sentence to BLOCK_SIZE
        sentence_idx += [POS_to_idx['<PAD>']] * (BLOCK_SIZE - len(sentence_idx))
        val_sentence_POS_idx.append(sentence_idx)

    test_sentence_POS_idx = []
    for sentence in test_sentence_POS:
        sentence_idx = [POS_to_idx[tag] for tag in sentence]
        # pad sentence to BLOCK_SIZE
        sentence_idx += [POS_to_idx['<PAD>']] * (BLOCK_SIZE - len(sentence_idx))
        test_sentence_POS_idx.append(sentence_idx)

    return sentence_POS_idx, val_sentence_POS_idx, test_sentence_POS_idx

def prepare_words(sentence_tokens, val_sentence_tokens, test_sentence_tokens, word_to_idx, BLOCK_SIZE):
    """function to prepare the words for the sentences, adding pad tokens and tokenizing the words"""
    """the function handles all training, validation and test data"""
    sentence_tokens_idx = []
    for sentence in sentence_tokens:
        sentence_idx = [word_to_idx[word] for word in sentence]
        # pad sentence to BLOCK_SIZE
        sentence_idx += [word_to_idx['<PAD>']] * (BLOCK_SIZE - len(sentence_idx))
        sentence_tokens_idx.append(sentence_idx)

    val_sentence_tokens_idx = []
    for sentence in val_sentence_tokens:
        sentence_idx = [word_to_idx[word] for word in sentence]
        # pad sentence to BLOCK_SIZE
        sentence_idx += [word_to_idx['<PAD>']] * (BLOCK_SIZE - len(sentence_idx))
        val_sentence_tokens_idx.append(sentence_idx)

    test_sentence_tokens_idx = []
    for sentence in test_sentence_tokens:
        sentence_idx = [word_to_idx[word] for word in sentence]
        # pad sentence to BLOCK_SIZE
        sentence_idx += [word_to_idx['<PAD>']] * (BLOCK_SIZE - len(sentence_idx))
        test_sentence_tokens_idx.append(sentence_idx)

    return sentence_tokens_idx, val_sentence_tokens_idx, test_sentence_tokens_idx

def get_splits(sentence_tokens_idx, sentence_POS_idx, val_sentence_tokens_idx, val_sentence_POS_idx, test_sentence_tokens_idx, test_sentence_POS_idx):
    """function to get well-structured data for training, validation and test"""
    train_data = []
    for i in range(len(sentence_tokens_idx)):
        train_data.append((sentence_tokens_idx[i], sentence_POS_idx[i]))

    val_data = []
    for i in range(len(val_sentence_tokens_idx)):
        val_data.append((val_sentence_tokens_idx[i], val_sentence_POS_idx[i]))

    test_data = []
    for i in range(len(test_sentence_tokens_idx)):
        test_data.append((test_sentence_tokens_idx[i], test_sentence_POS_idx[i]))

    return train_data, val_data, test_data

def extract_BERT_embs(sentences, bert_model, tokenizer, BLOCK_SIZE_BERT, BLOCK_SIZE, device):
    """function to extract the BERT embeddings for the sentences"""
    """Through the mapping previously created, we can extract the embeddings for the words depending on the subword management"""
    """The embeddings are calculated for the last four layers and summed up to get a single embedding for the word"""
    marked_text = [" ".join(sentence) for sentence in sentences]
    # use tokenizer to get tokenized text and pad up to BLOCK_SIZE_BERT
    tokens_tensor = [tokenizer(text, padding="max_length", truncation=True, max_length=BLOCK_SIZE_BERT, return_tensors="pt") for text in marked_text]
    # get attention mask to ignore padding tokens
    attention_mask = torch.stack([t['attention_mask'] for t in tokens_tensor])
    tokens_tensor = torch.stack([t['input_ids'] for t in tokens_tensor]).squeeze(1)
    # get the mapping to manage subwords
    mappings = [create_mapping(sentence, tokens_tensor[i], tokenizer, cased = True, subword_manage='prefix') for i, sentence in enumerate(sentences)]
    tokens_tensor = tokens_tensor.to(device)
    attention_mask = attention_mask.to(device)
    # forward pass to get the embeddings
    with torch.no_grad():
        bert_model = bert_model.to(device)
        outputs = bert_model(tokens_tensor, attention_mask = attention_mask)
        bert_model = bert_model.to('cpu')
        hidden_states = outputs[1]
        hidden_states = torch.stack(hidden_states, dim=0)
        hidden_states = hidden_states.to('cpu')
        hidden_states = hidden_states.permute(1,2,0,3)
    del tokens_tensor, attention_mask
    # sum the last four layers
    sentence_embeddings = []
    for i, token_embeddings in enumerate(hidden_states):
        token_vecs_sum = []
        for j, token in enumerate(token_embeddings):
            if mappings[i][j] is None:
                continue
            sum_vec = torch.sum(token[-4:], dim=0)
            token_vecs_sum.append(sum_vec)
        sentence_embeddings.append(token_vecs_sum)
    del hidden_states, mappings
    # pad sentence embeddings to BLOCK_SIZE 
    for i, sentence_embedding in enumerate(sentence_embeddings):
        sentence_embeddings[i] += [torch.zeros(sentence_embedding[0].size(0))] * (BLOCK_SIZE - len(sentence_embedding))

    sentence_embeddings = torch.stack([torch.stack(sentence) for sentence in sentence_embeddings])
    return sentence_embeddings

def get_batch_scan(data, sentences, batch_size, config, reached = 0, bert_model = None, tokenizer = None):
    """Function that builds batches for the training, and validation"""
    """The function uses an index to keep track of the current position in the data during the epoch"""
    """The function returns the words, the POS tags, and the BERT embeddings for the words"""
    # calculate end of batch
    end = reached+batch_size
    # if end is greater than the length of the data, set end to the length of the data
    if end > len(data):
        end = len(data)
    indices = range(reached, end)
    # build batches
    src = []
    tgt = []
    # get words and POS tags
    for idx in indices:
        src.append(data[idx][0])
        tgt.append(data[idx][1])
    src = torch.LongTensor(src)
    tgt = torch.LongTensor(tgt)
    # get BERT embeddings
    original_sentences = [sentences[i] for i in indices]
    src_embs = extract_BERT_embs(original_sentences, bert_model, tokenizer, config['model']['bert']['BLOCK_SIZE_BERT'], config['model']['BLOCK_SIZE'], config['model']['bert']['device'])
    return src, tgt, src_embs


def shuffle_data(data, raw_sentences):
    """Function to shuffle the data and the corresponding sentences"""
    """The function is used to randomize the data across the epochs"""
    combined = list(zip(data, raw_sentences))  # Pair corresponding elements
    random.shuffle(combined)  # Shuffle the pairs
    data, sentences = zip(*combined)  # Unzip back into separate lists
    return list(data), list(sentences)

def dataload_scan(data, raw_sentences, device, bert_model, tokenizer, config, POS_to_idx):
    """Function to load the data for the training, and validation"""
    """The function returns the words, the POS tags, and the BERT embeddings for the words and wraps them in a Batch object"""
    data, sentences = shuffle_data(data, raw_sentences)
    for i in range(0, len(data), config['model']['BATCH_SIZE']):
        # get the batch
        xb, yb, embs = get_batch_scan(data, sentences, config['model']['BATCH_SIZE'], config, reached = i, bert_model = bert_model, tokenizer = tokenizer)
        # detach the tensors to prevent gradient tracking
        src = xb.requires_grad_(False).clone().detach()
        tgt = yb.requires_grad_(False).clone().detach()
        embs = embs.requires_grad_(False).clone().detach()
        src = src.to(device)
        tgt = tgt.to(device)
        embs = embs.to(device)
        # wrap the data in a Batch object
        yield Batch(config, POS_to_idx, src, tgt, embs, pad = POS_to_idx['<PAD>'])
