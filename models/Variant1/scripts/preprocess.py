import os
import sys

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

import torch
import random
from torch.autograd import Variable
import numpy as np
from collections import Counter
from models.Variant1.model.CCGNode import CCGNode

def load_data(train_file, val_file, test_file):
    """ Load the data from the files. """
    sentence_tokens = []
    test_sentence_tokens = []
    sentence_CCG = []
    val_sentence_tokens = []
    val_sentence_CCG = []
    test_sentence_CCG = []
    vocab = []
    vocab_CCG = []

    x = open(train_file)
    for i, line in enumerate(x.readlines()):
        tokens = []
        CCG = []
        pairs = line.split(" ")
        if len(pairs) < 71:
            for pair in pairs:
                pair = pair.strip('\n')
                if pair != "\n" and pair != '':
                    word = pair.split("|")[0]
                    tag = pair.split("|")[-1]
                    tokens.append(word)
                    vocab.append(word)
                    vocab_CCG.append(tag)
                    CCG.append(tag)
            sentence_tokens.append(tokens)
            sentence_CCG.append(CCG)
    x.close()

    test_vocab_CCG = []
    x = open(val_file)
    for line in x.readlines():
        tokens = []
        CCG = []
        pairs = line.split(" ")
        if len(pairs) < 71: # ignore the sentences that are too long
            for pair in pairs:
                pair = pair.strip('\n')
                if pair != "\n" and pair != '':
                    word = pair.split("|")[0]
                    tag = pair.split("|")[-1]
                    tokens.append(word)
                    vocab.append(word)
                    test_vocab_CCG.append(tag)
                    CCG.append(tag)
            val_sentence_tokens.append(tokens)
            val_sentence_CCG.append(CCG)
    x.close()

    x = open(test_file)
    for line in x.readlines():
        tokens = []
        CCG = []
        pairs = line.split(" ")
        if len(pairs) < 71:
            for pair in pairs:
                pair = pair.strip('\n')
                if pair != "\n" and pair != '':
                    word = pair.split("|")[0]
                    tag = pair.split("|")[-1]
                    tokens.append(word)
                    vocab.append(word)
                    CCG.append(tag)
            test_sentence_tokens.append(tokens)
            test_sentence_CCG.append(CCG)
    x.close()

    # get the frequency of each CCG supertag
    freqs = Counter(vocab_CCG)
    # get the frequency of each CCG supertag in the test set
    test_freqs = Counter(test_vocab_CCG)
    # sort the CCG supertags and remove duplicates
    vocab_CCG = sorted(list(set(vocab_CCG)))

    return sentence_tokens, sentence_CCG, test_sentence_tokens, test_sentence_CCG, val_sentence_tokens, val_sentence_CCG, vocab, vocab_CCG, test_vocab_CCG, freqs, test_freqs

def get_atomic_labels(vocab_CCG):
    """ Get the atomic labels from the CCG supertags. """
    atomic_labels = []
    ignore_chars = set(['(', ')', '/',  '\\'])
    for supertag in vocab_CCG:
        current_label = []
        flag = 0
        for i, char in enumerate(supertag):
            if char not in ignore_chars:
                flag = 1
                current_label.append(char)
            if flag == 1 and (char in ignore_chars or i == len(supertag) - 1):
                # construct a string from the list
                string_label = ''.join(current_label)
                atomic_labels.append(string_label)
                current_label = []
                flag = 0
    return sorted(list(set(atomic_labels)))

def get_opaque_mapping(vocab_CCG):
    """ Get the opaque mapping from the CCG supertags. """
    CCG_to_idx = {tag: i+1 for i, tag in enumerate(vocab_CCG)}
    # add padding, unknown, and start tokens
    CCG_to_idx["<PAD>"] = 0
    CCG_to_idx["<START>"] = len(CCG_to_idx)
    CCG_to_idx["<END>"] = len(CCG_to_idx)
    CCG_to_idx["<UNK>"] = len(CCG_to_idx)
    CCG_to_idx['<U>'] = len(CCG_to_idx)
    idx_to_CCG = {k: v for v, k in CCG_to_idx.items()}
    return CCG_to_idx, idx_to_CCG

def get_atomic_mapping(atomic_labels):
    """ Get the atomic mapping from the atomic labels. """
    atomic_to_idx = {tag: i for i, tag in enumerate(atomic_labels)}
    atomic_to_idx["/"] = len(atomic_to_idx)
    atomic_to_idx["\\"] = len(atomic_to_idx)
    idx_to_atomic = {k: v for v, k in atomic_to_idx.items()}
    return atomic_to_idx, idx_to_atomic

def encode_CCG(sentence_CCG, val_sentence_CCG, test_sentence_CCG, CCG_to_idx, atomic_to_idx):
    """Function that takes all the splits data and encodes the CCG supertags so that each atomic label is mapped to an integer
    Parentheses are retained. For example, [(N/N), NP] will be encoded as [['(', 1, 2, 1, ')'], [3]]
    The format will be used to construct supertag trees"""
    ignore_chars = set(['(', ')', '/',  '\\'])
    encoded_sentence_CCG = []
    for sentence in sentence_CCG:
        converted_tag = []
        for supertag in sentence:
            current_label = []
            flag = 0
            string_in_progress = []
            for i, char in enumerate(supertag):
                if char not in ignore_chars:
                    flag = 1
                    current_label.append(char)
                if flag == 1 and (char in ignore_chars or i == len(supertag)-1):
                    string_label = ''.join(current_label)
                    integer_label = atomic_to_idx[string_label]
                    string_in_progress.append(integer_label)
                    current_label = []
                    flag = 0
                if char in ignore_chars:
                    string_in_progress.append(char)
            converted_tag.append(string_in_progress)
        encoded_sentence_CCG.append(converted_tag)

    for sentence in encoded_sentence_CCG:
        for tag in sentence:
            for i in range(len(tag)):
                if tag[i] == '/':
                    tag[i] = atomic_to_idx['/']
                elif tag[i] == '\\':
                    tag[i] = atomic_to_idx['\\']

    val_encoded_sentence_CCG = []
    for sentence in val_sentence_CCG:
        converted_tag = []
        for supertag in sentence:
            current_label = []
            flag = 0
            string_in_progress = []
            for i, char in enumerate(supertag):
                if char not in ignore_chars:
                    flag = 1
                    current_label.append(char)
                if flag == 1 and (char in ignore_chars or i == len(supertag)-1):
                    string_label = ''.join(current_label)
                    integer_label = atomic_to_idx[string_label]
                    string_in_progress.append(integer_label)
                    current_label = []
                    flag = 0
                if char in ignore_chars:
                    string_in_progress.append(char)
            converted_tag.append(string_in_progress)
        val_encoded_sentence_CCG.append(converted_tag)

    for sentence in val_encoded_sentence_CCG:
        for tag in sentence:
            for i in range(len(tag)):
                if tag[i] == '/':
                    tag[i] = atomic_to_idx['/']
                elif tag[i] == '\\':
                    tag[i] = atomic_to_idx['\\']

    test_encoded_sentence_CCG = []
    for sentence in test_sentence_CCG:
        converted_tag = []
        for supertag in sentence:
            current_label = []
            flag = 0
            string_in_progress = []
            for i, char in enumerate(supertag):
                if char not in ignore_chars:
                    flag = 1
                    current_label.append(char)
                if flag == 1 and (char in ignore_chars or i == len(supertag)-1):
                    string_label = ''.join(current_label)
                    integer_label = atomic_to_idx[string_label]
                    string_in_progress.append(integer_label)
                    current_label = []
                    flag = 0
                if char in ignore_chars:
                    string_in_progress.append(char)
            converted_tag.append(string_in_progress)
        test_encoded_sentence_CCG.append(converted_tag)

    for sentence in test_encoded_sentence_CCG:
        for tag in sentence:
            for i in range(len(tag)):
                if tag[i] == '/':
                    tag[i] = atomic_to_idx['/']
                elif tag[i] == '\\':
                    tag[i] = atomic_to_idx['\\']

    return encoded_sentence_CCG, val_encoded_sentence_CCG, test_encoded_sentence_CCG

def get_positional_mapping(MAX_DEPTH):
    """ Get the unique identifiers for each node in the tree, assuming the root is at depth 0; the tree is a full binary tree and the given MAX_DEPTH. """
    positions = ['0']
    for depth in range(1, MAX_DEPTH+1):
        length = depth
        # binary string of length `length` with all possible comninations
        for i in range(2 ** length):
            positions.append(f'{i:0{length}b}')

    # replace 0s with -1
    positions = [p.replace('0', '-1') for p in positions]
    positions[0] = '0'

    # copy positions list to another list
    slashes = positions.copy()
    all_encodings = [f'{p}{s}' for p in positions[1:] for s in slashes[1:]]
    all_encodings = ['00'] + all_encodings

    enc_to_idx = {}
    for i, enc in enumerate(sorted(list(set(all_encodings)))):
        enc_to_idx[enc] = i

    return enc_to_idx

def encode_trees(encoded_sentence_CCG, val_encoded_sentence_CCG, test_encoded_sentence_CCG, CCG_to_idx, BLOCK_SIZE, atomic_to_idx, idx_to_atomic):
    """ Encode the trees into CCGNodes. The function encodes all the splits data into CCGNodes. """
    sentence_trees = []
    for i, sentence in enumerate(encoded_sentence_CCG):
        sentence_tree = []
        for supertag in sentence:
            root = CCGNode(supertag)
            sentence_tree.append(root)
        # pad to block size
        sentence_tree += [CCG_to_idx['<PAD>']] * (BLOCK_SIZE - len(sentence_tree))
        sentence_trees.append(sentence_tree)

    val_sentence_trees = []
    for i, sentence in enumerate(val_encoded_sentence_CCG):
        sentence_tree = []
        for supertag in sentence:
            root = CCGNode(supertag)
            sentence_tree.append(root)
        sentence_tree += [CCG_to_idx['<PAD>']] * (BLOCK_SIZE - len(sentence_tree))
        val_sentence_trees.append(sentence_tree)

    test_sentence_trees = []
    for i, sentence in enumerate(test_encoded_sentence_CCG):
        sentence_tree = []
        for supertag in sentence:
            root = CCGNode(supertag)
            sentence_tree.append(root)
        sentence_tree += [CCG_to_idx['<PAD>']] * (BLOCK_SIZE - len(sentence_tree))
        test_sentence_trees.append(sentence_tree)

    return sentence_trees, val_sentence_trees, test_sentence_trees

def create_mapping(sentence, tokens_tensor, cased = False, subword_manage = 'prefix', tokenizer = None):
    """ Create the mapping from the tokens to the detokenized sentence.  The function can handle both prefix and suffix subword management. """
    mapping = [None] # for the [CLS] token
    detokenized_index = 1 # for the [CLS] token
    detokenized_sentence = tokenizer.convert_ids_to_tokens(tokens_tensor)
    for i, word in enumerate(sentence):
        word = word if cased else word.lower()
        detokenized_token = tokenizer.convert_ids_to_tokens(tokens_tensor[detokenized_index].item())
        next_detokenized_token = tokenizer.convert_ids_to_tokens(tokens_tensor[detokenized_index + 1].item())
        if word != detokenized_token:
            reconstructed_word = detokenized_token.strip('Ġ')
            while reconstructed_word != word:
                # print(word, reconstructed_word)
                detokenized_index += 1
                reconstructed_word += detokenized_sentence[detokenized_index].strip('Ġ')
                mapping.append(i)
            mapping.append(i)
        else:
            mapping.append(i)
        detokenized_index += 1
    while len(mapping) < len(detokenized_sentence):
        mapping.append(None)
    if subword_manage == 'prefix':
        for i, idx in enumerate(mapping):
            if idx is not None:
                j = i+1
                while mapping[j] == idx:
                    mapping[j] = None
                    j += 1
    elif subword_manage == 'suffix':
        for i, idx in enumerate(mapping):
            if idx is not None:
                j = i
                while mapping[j+1] == idx:
                    mapping[j] = None
                    j += 1
    return mapping

def get_words_mapping(vocab):
    """ Get the mapping from the words to the opaque ids. """
    vocab = sorted(list(set(vocab)))
    word_to_idx = {word: i+1 for i, word in enumerate(vocab)}
    word_to_idx["<PAD>"] = 0
    idx_to_word = {k: v for v, k in word_to_idx.items()}
    return word_to_idx, idx_to_word

def encode_words(sentence_tokens, val_sentence_tokens, test_sentence_tokens, word_to_idx, BLOCK_SIZE):
    """ Encode the words into opaque ids. The function encodes all the splits data into opaque ids. """
    sentence_tokens_idx = []
    for sentence in sentence_tokens:
        sentence_idx = [word_to_idx[word] for word in sentence]
        # pad sentence to 100
        sentence_idx += [word_to_idx['<PAD>']] * (BLOCK_SIZE - len(sentence_idx))
        sentence_tokens_idx.append(sentence_idx)

    val_sentence_tokens_idx = []
    for sentence in val_sentence_tokens:
        sentence_idx = [word_to_idx[word] for word in sentence]
        # pad sentence to 100
        sentence_idx += [word_to_idx['<PAD>']] * (BLOCK_SIZE - len(sentence_idx))
        val_sentence_tokens_idx.append(sentence_idx)

    test_sentence_tokens_idx = []
    for sentence in test_sentence_tokens:
        sentence_idx = [word_to_idx[word] for word in sentence]
        # pad sentence to 100
        sentence_idx += [word_to_idx['<PAD>']] * (BLOCK_SIZE - len(sentence_idx))
        test_sentence_tokens_idx.append(sentence_idx)

    return sentence_tokens_idx, val_sentence_tokens_idx, test_sentence_tokens_idx

def get_splits(sentence_tokens_idx, val_sentence_tokens_idx, test_sentence_tokens_idx, sentence_trees, val_sentence_trees, test_sentence_trees):
    """ Get the splits from the data. """
    train_data = []
    for i in range(len(sentence_tokens_idx)):
        train_data.append((sentence_tokens_idx[i], sentence_trees[i]))

    val_data = []
    for i in range(len(val_sentence_tokens_idx)):
        val_data.append((val_sentence_tokens_idx[i], val_sentence_trees[i]))

    test_data = []
    for i in range(len(test_sentence_tokens_idx)):
        test_data.append((test_sentence_tokens_idx[i], test_sentence_trees[i]))

    return train_data, val_data, test_data

def get_batch_scan(data, sentences, config, CCG_to_idx, reached = 0):
    """Prepare the batch for training for a whole epoch. """
    end = reached+config['model']['BATCH_SIZE']
    if end > len(data):
        end = len(data)
    indices = range(reached, end)
    # build batches
    src = []
    tgt = torch.zeros((config['model']['BATCH_SIZE'], config['model']['BLOCK_SIZE'], 2**(config['model']['MAX_DEPTH']+1)-1), dtype = torch.long)
    for i, idx in enumerate(indices):
        src.append(data[idx][0])
        for j, tree in enumerate(data[idx][1]):
            if tree != CCG_to_idx['<PAD>']:
                list_of_atomic = tree.get_nodes()
                tgt[i][j] = torch.LongTensor(list_of_atomic)


    src = torch.LongTensor(src)
    original_sentences = [sentences[i] for i in indices]
    return src, tgt, original_sentences

def shuffle_data(data, raw_sentences):
    """Function to shuffle the data and the corresponding sentences"""
    """The function is used to randomize the data across the epochs"""
    combined = list(zip(data, raw_sentences))  # Pair corresponding elements
    random.shuffle(combined)  # Shuffle the pairs
    data, sentences = zip(*combined)  # Unzip back into separate lists
    return list(data), list(sentences)

def dataload_scan(batch, data, raw_sentences, device, CCG_to_idx, config):
    """Wraps the data into batches. """
    data, sentences = shuffle_data(data, raw_sentences)
    for i in range(0, len(data), batch):
        # print(f'Doing sequence {i}-{i+batch}', flush = True)
        xb, yb, original_sentences = get_batch_scan(data, sentences, config, CCG_to_idx, reached = i)
        src = Variable(xb, requires_grad=False)#.clone().detach()
        tgt = Variable(yb, requires_grad = False)#.clone().detach()
        src = src.to(device)
        tgt = tgt.to(device)
        yield Batch(original_sentences, src, config, tgt, pad = CCG_to_idx['<PAD>'])

class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, encoded_src, config, tgt=None, pad=0): 
        self.src = src
        self.src_mask = (encoded_src != pad).unsqueeze(-2)
        self.sequence_lengths = [torch.nonzero(encoded_src[i] == 0, as_tuple = False)[0,0].item() if torch.nonzero(encoded_src[i] == 0, as_tuple = False).numel() > 0 else config['model']['BLOCK_SIZE']-1 for i in range(encoded_src.size(0))]
        if tgt is not None:
            self.tgt_y = tgt
            self.tgt_mask = self.src_mask
            self.ntokens = np.sum(self.sequence_lengths)
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask
        return tgt_mask
