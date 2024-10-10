import os
import sys

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

import torch
from models.Variant1.model.utils import greedy_decode, load_config
from models.Variant1.scripts.preprocess import load_data, get_atomic_mapping, get_words_mapping, get_opaque_mapping, get_splits, get_positional_mapping, get_atomic_labels, encode_CCG, encode_trees, encode_words
from models.Variant1.scripts.train import resume_training

def evaluate(model, val_data, val_sentence_tokens, CCG_to_idx, config):
    """ Evaluate the model on the validation data. """
    with torch.no_grad():
        model.eval()
        test_predictions = []
        for i in range(0, len(val_data), config['model']['BATCH_SIZE']):
            if i % 256 == 0:
                print(f'Predicting sentence {i+1}/{len(val_data)}.', flush = True)
            batch = val_data[i:i+config['model']['BATCH_SIZE']]
            src = []
            # initialize the target tensor
            tgt = torch.zeros((config['model']['BATCH_SIZE'], config['model']['BLOCK_SIZE'], 2**(config['model']['MAX_DEPTH']+1)-1), dtype = torch.long)
            # for each sentence in the batch
            for j, (sentence, trees) in enumerate(batch):
                # store the sentence
                src.append(sentence)
                # store the trees
                for t, tree in enumerate(trees):
                    if tree != CCG_to_idx['<PAD>']:
                        list_of_atomic = tree.get_nodes()
                        tgt[j][t] = torch.LongTensor(list_of_atomic)
            src = torch.LongTensor(src)
            src = src.to(config['model']['device'])
            tgt = tgt.to(config['model']['device'])
            start = i
            end = i + config['model']['BATCH_SIZE']
            if end > len(val_data):
                end = len(val_data)
            # get the original sentences
            original_sentences = [val_sentence_tokens[i] for i in range(start, end)]
            # get the sequence lengths
            sequence_lengths = []
            for sent in original_sentences:
                sequence_lengths.append(len(sent))
            # greedy decode the trees
            trg = greedy_decode(model, original_sentences, sequence_lengths, config['model']['BLOCK_SIZE'])
            # store the predictions
            test_predictions += trg

        # get the golden target
        test_targets = []
        for i, (sentence, trees) in enumerate(val_data):
            sentence_trees_tgt = []
            for j, tree in enumerate(trees):
                if tree != CCG_to_idx['<PAD>']:
                    sentence_trees_tgt.append(tree)
            test_targets.append(sentence_trees_tgt)


        # calculate accuracy
        correct = 0
        total = 0
        for i in range(len(test_predictions)):
            for j in range(len(test_predictions[i])):
                total += 1
                if test_predictions[i][j].is_equal(test_targets[i][j]):
                    correct += 1

        accuracy = correct / total
        return accuracy, test_predictions, test_targets
    

if __name__ == "__main__":
    # load the config
    config = load_config()

    # load the data
    sentence_tokens, sentence_CCG, test_sentence_tokens, test_sentence_CCG, val_sentence_tokens, val_sentence_CCG, vocab, vocab_CCG, test_vocab_CCG, freqs, test_freqs = load_data(config['model']['train_file'], config['model']['val_file'], config['model']['test_file'])

    # get the atomic labels
    atomic_labels = get_atomic_labels(vocab_CCG)

    # get the opaque mapping
    CCG_to_idx, idx_to_CCG = get_opaque_mapping(vocab_CCG)

    # get the atomic labels mapping
    atomic_to_idx, idx_to_atomic = get_atomic_mapping(atomic_labels)

    # encode the CCG supertags atomic labels
    encoded_sentence_CCG, val_encoded_sentence_CCG, test_encoded_sentence_CCG = encode_CCG(sentence_CCG, val_sentence_CCG, test_sentence_CCG, CCG_to_idx, atomic_to_idx)

    # get the positional mapping
    encoder_to_idx = get_positional_mapping(config['model']['MAX_DEPTH'])

    # encode the trees
    sentence_trees, val_sentence_trees, test_sentence_trees = encode_trees(encoded_sentence_CCG, val_encoded_sentence_CCG, test_encoded_sentence_CCG, CCG_to_idx, config['model']['BLOCK_SIZE'], atomic_to_idx, idx_to_atomic)

    # get the words mapping
    words_to_idx, idx_to_words = get_words_mapping(vocab)

    # encode the words
    sentence_words, val_sentence_words, test_sentence_words = encode_words(sentence_tokens, val_sentence_tokens, test_sentence_tokens, words_to_idx, config['model']['BLOCK_SIZE'])

    # get the splits
    train_data, val_data, test_data = get_splits(sentence_words, val_sentence_words, test_sentence_words, sentence_trees, val_sentence_trees, test_sentence_trees)

    # load model
    path = 'models/Variant1/saved_models/Variant1'
    model, optimizer, lr_scheduler, trained_epochs, train_losses, val_losses = resume_training(path, atomic_to_idx, config, train_data, encoder_to_idx)    
    # evaluate model
    accuracy, test_predictions, test_targets = evaluate(model, val_data, val_sentence_tokens, CCG_to_idx, config)
    print(f'Accuracy: {accuracy}')

    # save predictions to a file
    list_tgt = []
    for i, sent_tgts in enumerate(test_targets):
        sent_tgt = []
        for tgt in sent_tgts:
            sent_tgt.append(tgt.to_opaque(atomic_to_idx, idx_to_atomic))
        list_tgt.append(sent_tgt)

    list_preds = []
    for tree_preds in test_predictions:
        sent_preds = []
        for tree in tree_preds:
            try:
                sent_preds.append(tree.to_opaque(atomic_to_idx, idx_to_atomic))
            except:
                sent_preds.append('<BAD_TREE>')
        list_preds.append(sent_preds)

    with open('models/Variant1/predictions/predictions.txt', 'w') as file:
        for i, sent_tgts in enumerate(list_tgt):
            for j, tgt in enumerate(sent_tgts):
                file.write(f'{tgt}|{list_preds[i][j]} ')
            file.write('\n')


