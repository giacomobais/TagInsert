import random
import os
import sys

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

import torch
from torch.autograd import Variable
from models.TagInsert.scripts.preprocess import extract_BERT_embs, data_load, get_mapping_POS, get_mapping_words, get_splits, prepare_POS, prepare_words, get_bert_model
from models.TagInsert.model.utils import load_config, greedy_decode
from models.TagInsert.scripts.train import resume_training

def evaluate(model, val_data, val_sentence_tokens, POS_to_idx, bert_model, config, device): 
    """The function takes the model, the validation data, the validation sentence tokens, the index to POS mapping, the index to word mapping, the POS to index mapping, the BERT model, the config, and the device"""
    """It evalutes the model on a dataset of choice, and it returns the accuracy, the predictions and the gold targets"""
    with torch.no_grad():
        test_predictions = []
        model.eval()
        test_targets = []
        all_orders = []
        for i in range(0, len(val_data), config['model']['BATCH_SIZE']):
            if i % 256 == 0:
                print(f'Predicting sentence {i+1}/{len(val_data)}.')
            batch = val_data[i:i+config['model']['BATCH_SIZE']]
            src = []
            tgt = []
            for words, tags in batch:
                src.append(words)
                tgt.append(tags)
            src = torch.LongTensor(src)
            tgt = torch.LongTensor(tgt)
            src = src.to(device)
            start = i
            end = i + config['model']['BATCH_SIZE']
            if end > len(val_data):
                end = len(val_data)
            original_sentences = [val_sentence_tokens[i] for i in range(start, end)]
            src_embs = extract_BERT_embs(original_sentences, bert_model=bert_model, tokenizer=tokenizer, BLOCK_SIZE_BERT=config['model']['bert']['BLOCK_SIZE_BERT'], BLOCK_SIZE=config['model']['BLOCK_SIZE'], device=config['model']['device'])
            src_mask = Variable((src != 0).unsqueeze(-2)).to(device)
            src_mask = src_mask.to(device)
            src_embs = src_embs.to(device)
            seq_lengths= [torch.nonzero(tgt[i] == 0, as_tuple = False)[0,0].item() if tgt[i][-1] == 0 else config['model']['BLOCK_SIZE'] for i in range(len(original_sentences))]
            preds, orders = greedy_decode(model, src, src_mask, src_embs, max_len = config['model']['BLOCK_SIZE'],seq_length = seq_lengths, POS_to_idx=POS_to_idx, device=device)
            preds = preds[:, 1:]
            preds = [preds[i, :seq_lengths[i]] for i in range(len(seq_lengths))]
            tgt = [tgt[i, :seq_lengths[i]] for i in range(len(seq_lengths))]
            test_predictions+=preds
            test_targets += tgt
            all_orders += orders
        correct = 0
        total = 0
        for i in range(len(test_predictions)):
            for j in range(len(test_predictions[i])):
                total += 1
                if test_predictions[i][j] == test_targets[i][j]:
                    correct += 1

        accuracy = correct / total
        return accuracy, test_predictions, test_targets, all_orders
    
if __name__ == "__main__":
    # load config for hyperparameters
    config = load_config()

    # load bert model
    bert_model, tokenizer = get_bert_model(config['model']['bert']['model_name'], config['model']['bert']['device'])

    # get data
    train_sentence_tokens, train_sentence_POS, val_sentence_tokens, val_sentence_POS, val_sentence_tokens, test_sentence_POS, vocab, vocab_POS = data_load(config['model']['train_file'], config['model']['val_file'], config['model']['test_file'])

    # get mapping
    POS_to_idx, idx_to_POS = get_mapping_POS(vocab_POS)
    word_to_idx, idx_to_word, vocab = get_mapping_words(vocab)

    # prepare POS
    train_sentence_POS_idx, val_sentence_POS_idx, test_sentence_POS_idx = prepare_POS(train_sentence_POS, val_sentence_POS, test_sentence_POS, POS_to_idx, config['model']['BLOCK_SIZE'])

    # prepare tokens
    train_sentence_tokens_idx, val_sentence_tokens_idx, test_sentence_tokens_idx = prepare_words(train_sentence_tokens, val_sentence_tokens, val_sentence_tokens, word_to_idx, config['model']['BLOCK_SIZE'])
    # get splits
    train_data, val_data, val_data = get_splits(train_sentence_tokens_idx, train_sentence_POS_idx, val_sentence_tokens_idx, val_sentence_POS_idx, test_sentence_tokens_idx, test_sentence_POS_idx)

    # load model
    path = 'models/TagInsert/saved_models/TagInsert_POS'
    model, optimizer, lr_scheduler, trained_epochs, train_losses, val_losses = resume_training(path, word_to_idx, POS_to_idx, config, config['model']['device'])
    model.eval()
    # evaluate model
    accuracy, test_predictions, test_targets, ordering = evaluate(model, val_data, val_sentence_tokens, POS_to_idx, bert_model, config, config['model']['device'])
    print(f'Accuracy: {accuracy}')

    test_predictions = [[idx_to_POS[idx.item()] for idx in sentence] for sentence in test_predictions]
    test_targets = [[idx_to_POS[idx.item()] for idx in sentence] for sentence in test_targets]
    # Trim ordering tensors to match the length of predictions/targets
    trimmed_ordering = []
    for i, order in enumerate(ordering):
        trimmed_order = order[:len(test_predictions[i])]
        trimmed_ordering.append(trimmed_order)

    ordering = trimmed_ordering
    # save predictions to a file
    with open('models/TagInsert/predictions/sample_predictions.txt', 'w') as file:
        for i in range(len(test_predictions)):
            file.write(f'Sentence {i+1}:\n')
            file.write(f'Golden: {test_targets[i]}\n')
            file.write(f'Predicted: {test_predictions[i]}\n')
            file.write(f'Ordering: {ordering[i]}\n')
            file.write('\n')
        
