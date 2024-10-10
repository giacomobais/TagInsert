import os
import sys

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

import torch
from torch.autograd import Variable
import csv
from models.ConstructiveTagInsert.model.utils import greedy_decode, load_config
from models.ConstructiveTagInsert.scripts.preprocess import load_data, get_atomic_mapping, get_words_mapping, get_opaque_mapping, get_splits, get_positional_mapping, get_atomic_labels, encode_CCG, encode_trees, encode_words, encode_CCGs, get_bert_model, extract_BERT_embs
from models.ConstructiveTagInsert.scripts.train import resume_training

def evaluate(model, val_data, val_sentence_tokens, CCG_to_idx, config, atomic_to_idx, idx_to_atomic):
    """ Evaluate the model on the validation data. """
    with torch.no_grad():
        model.eval()
        test_predictions = []
        test_targets = []
        test_orders = []
        BATCH_SIZE = config['model']['BATCH_SIZE']
        for i in range(0, len(val_data), BATCH_SIZE):
            if i % 256 == 0:
                print(f'Predicting sentence {i+BATCH_SIZE}/{len(val_data)}.')
            batch = val_data[i:i+BATCH_SIZE]
            src = []
            ccg_tgt = []
            tree_tgt = torch.zeros((BATCH_SIZE, config['model']['BLOCK_SIZE'], 2**(config['model']['MAX_DEPTH']+1)-1), dtype = torch.long)
            for words, tags, trees in batch:
                src.append(words)
                ccg_tgt.append(tags)
            src = torch.LongTensor(src)
            ccg_tgt = torch.LongTensor(ccg_tgt)

            src = src.to(config['model']['device'])
            start = i
            end = i + BATCH_SIZE
            if end > len(val_data):
                end = len(val_data)
            original_sentences = [val_sentence_tokens[i] for i in range(start, end)]
            src_embs = extract_BERT_embs(original_sentences, bert_model=bert_model, tokenizer=tokenizer, BLOCK_SIZE_BERT=config['model']['BLOCK_SIZE_BERT'], BLOCK_SIZE=config['model']['BLOCK_SIZE'], device=config['model']['device'])
            src_mask = Variable((src != 0).unsqueeze(-2)).to(config['model']['device'])
            src_mask = src_mask.to(config['model']['device'])
            src_embs = src_embs.to(config['model']['device'])
            seq_lengths= [torch.nonzero(ccg_tgt[i] == 0, as_tuple = False)[0,0].item() if ccg_tgt[i][-1] == 0 else config['model']['BLOCK_SIZE'] for i in range(len(original_sentences))]
            preds, orders = greedy_decode(model, src, src_mask, src_embs, max_len = config['model']['BLOCK_SIZE'],seq_length = seq_lengths, CCG_to_idx=CCG_to_idx, config=config, device=config['model']['device'], atomic_to_idx=atomic_to_idx, idx_to_atomic=idx_to_atomic)
            # orders is a list of indeces, orders[0][0] is the time step in which the first word of the first sentence got inserted
            for i in range(len(preds)):
                preds[i] = preds[i][1:]

            preds = [preds[i][:seq_lengths[i]] for i in range(len(seq_lengths))]
            ccg_tgt = [ccg_tgt[i, :seq_lengths[i]] for i in range(len(seq_lengths))]
            orders = [orders[i, :seq_lengths[i]] for i in range(len(seq_lengths))]
            test_predictions+=preds
            test_targets += ccg_tgt
            test_orders += orders

        correct = 0
        total = 0
        for i in range(len(test_predictions)):
            for j in range(len(test_predictions[i])):
                total += 1
                if test_predictions[i][j] == test_targets[i][j]:
                    correct += 1

        accuracy = correct / total
    return accuracy, test_predictions, test_targets, test_orders
    
def save_predictions(test_predictions, test_targets, test_orders):
    """Save the predictions to a csv file."""
    list_preds = [tensor for tensor in test_predictions]
    list_tgt = [tensor.tolist() for tensor in test_targets]
    list_orders = [tensor.tolist() for tensor in test_orders]

    with open('models/ConstructiveTagInsert/predictions/sample_predictions.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Targets'])
        writer.writerows(list_tgt)
        writer.writerow(['Predictions'])
        writer.writerows(list_preds)
        writer.writerow(['Orders'])
        writer.writerows(list_orders)

        

def read_predictions(file_path):
    """Read the predictions from a csv file."""
    test_targets = []
    test_predictions = []
    test_orders = []
    is_targets = True
    is_predictions = False
    is_orders = False
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == 'Targets':
                is_targets = True
                continue
            elif row[0] == 'Predictions':
                is_targets = False
                is_predictions = True
                continue
            elif row[0] == 'Orders':
                is_targets = False
                is_predictions = False
                is_orders = True
                continue
            if is_targets:
                test_targets.append(list(map(int, row)))
            elif is_predictions:
                preds = []
                for pred in row:
                    try:
                        preds.append(int(pred))
                    except:
                        preds.append(pred)
                test_predictions.append(preds)
            elif is_orders:
                test_orders.append(list(map(int, row)))
    return test_predictions, test_targets, test_orders


if __name__ == "__main__":
    # load the config
    config = load_config()

    # load the data
    sentence_tokens, sentence_CCG, test_sentence_tokens, test_sentence_CCG, val_sentence_tokens, val_sentence_CCG, vocab, vocab_CCG, test_vocab_CCG, freqs, test_freqs = load_data(config['model']['train_file'], config['model']['val_file'], config['model']['test_file'])

    # get the bert model
    bert_model, tokenizer = get_bert_model(config['model']['bert_model'], config['model']['device'])

    # get the atomic labels
    atomic_labels = get_atomic_labels(vocab_CCG)

    # get the opaque mapping
    CCG_to_idx, idx_to_CCG = get_opaque_mapping(vocab_CCG)

    # get the atomic labels mapping
    atomic_to_idx, idx_to_atomic = get_atomic_mapping(atomic_labels)
    ATOMIC_VOCAB = len(atomic_to_idx)
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

    # encode the CCGs into ids for the TagInsert model
    sentence_CCG_idx, val_sentence_CCG_idx, test_sentence_CCG_idx = encode_CCGs(sentence_CCG, val_sentence_CCG, test_sentence_CCG, CCG_to_idx, config['model']['BLOCK_SIZE'])

    # get the splits
    train_data, val_data, test_data = get_splits(sentence_words, val_sentence_words, test_sentence_words, sentence_trees, val_sentence_trees, test_sentence_trees, sentence_CCG_idx, val_sentence_CCG_idx, test_sentence_CCG_idx)

    # load the model
    path = 'models/ConstructiveTagInsert/saved_models/ConstructiveTagInsert'
    model, optimizer, lr_scheduler, trained_epochs, train_losses, val_losses = resume_training(path, words_to_idx, CCG_to_idx, config, train_data, ATOMIC_VOCAB, encoder_to_idx, atomic_to_idx)
    
    # evaluate model
    accuracy, test_predictions, test_targets, test_orders = evaluate(model, val_data, val_sentence_tokens, CCG_to_idx, config, atomic_to_idx, idx_to_atomic)
    print(f'Accuracy: {accuracy}')

    # save the predictions to a file
    save_predictions(test_predictions, test_targets, test_orders)


