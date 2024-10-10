import os
import sys

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

import torch
from transformers import AutoModel, RobertaTokenizer

import torch.nn as nn
import numpy as np
from models.ConstructiveTagInsert.scripts.preprocess import  load_data, get_atomic_labels, get_opaque_mapping, get_atomic_mapping, encode_CCG, get_positional_mapping, encode_trees, get_words_mapping, encode_words, get_splits, dataload_scan, encode_CCGs, get_bert_model
from models.ConstructiveTagInsert.model.utils import load_config, load_model, SimpleLossCompute, save_model
from models.ConstructiveTagInsert.model.CCGNode import CCGNode, pretty_print_tree
from models.ConstructiveTagInsert.model.model import TrainState, make_model

def run_epoch_scan(
data_iter,
model,
loss_compute,
optimizer,
scheduler,
config,
mode="train",
accum_iter=1,
train_state=TrainState(),
current_losses = []
):
    """Train a single epoch"""
    total_tokens = 0
    total_loss = 0
    # for each batch in the data iterator
    for i, batch in enumerate(data_iter):
        # get the max length of the sequences in the batch
        max_len = np.max(batch.sequence_lengths)
        # initialize the losses tensor with zeros
        losses = torch.zeros(max_len).to(config['model']['device'])
        # for each trajectory step 
        for traj_step in range(max_len):
            # forward pass, returns the logits of all predicted trees in the batch and their locations
            roots, logits, locations = model.forward(batch.src, batch.trg, batch.tree_tgt,
                                batch.src_mask, batch.trg_mask, batch.embs, batch.sequence_lengths, mode)
            # compute the loss for the predicted trees
            loss, loss_node = loss_compute(logits, batch.tree_tgt, batch.sequence_lengths, config, batch.inserted, locations)
            # if the mode is train or train+log, compute the gradient and update the parameters
            if mode == "train" or mode == "train+log":
                # backward pass, compute the gradient of the loss with respect to the model parameters
                loss_node.backward()
                if i % accum_iter == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            norm = batch.ntokens
            if norm == 0:
                norm = 1
            # store the loss for the current trajectory step
            losses[traj_step] = loss_node
            # update the target with the next trajectory step
            batch.next_trajectory(locations)
        # delete the embeddings tensor
        del batch.embs
        # update the learning rate
        scheduler.step()
        # compute the mean loss for the current batch
        loss_node_backward = torch.mean(losses)
        # store the loss for the current batch
        current_losses.append(loss_node_backward.item())
        # update the total loss and the total number of tokens
        total_loss += loss_node_backward.data
        total_tokens += batch.ntokens
        del loss
        del loss_node
        # log the loss every 2 batches
        if i % 2 == 0:
            print(f'Train loss at step {i*batch.src.shape[0]}/{(i+1)*batch.src.shape[0]}: {loss_node_backward.data.item()} in mode {mode}', flush=True)
    return total_loss / total_tokens, train_state, current_losses


def resume_training(path, word_to_idx, CCG_to_idx, config, train_data, ATOMIC_VOCAB, encoder_to_idx, atomic_to_idx):
    """Resume training from a pre-existent model. If no model is found, create a new model."""
    try:
        model, model_opt, lr_scheduler, logs = load_model(path, word_to_idx, CCG_to_idx, config, train_data, ATOMIC_VOCAB, encoder_to_idx, atomic_to_idx)
        trained_epochs = logs['epochs']
        train_losses = logs['train_losses']
        val_losses = logs['val_losses']
        print('Loading a pre-existent model.')
    except:
        model =  make_model(len(word_to_idx), len(CCG_to_idx), ATOMIC_VOCAB, encoder_to_idx, atomic_to_idx, CCG_to_idx, N = 6, d_model = 768,d_ff = 768*4, h = 8)
        model_opt = torch.optim.AdamW(model.parameters(), lr=config['model']['lr'], betas=(config['model']['beta1'], config['model']['beta2']), eps=config['model']['eps'])
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer = model_opt,
                                                max_lr=config['model']['lr'], epochs = config['model']['EPOCHS'], steps_per_epoch=len(train_data) // config['model']['BATCH_SIZE'] + int(len(train_data) % config['model']['BATCH_SIZE'] != 0))
        trained_epochs = 0
        train_losses = []
        val_losses = []
        print('No pre-trained model found.')

    model.train()
    model = model.to(config['model']['device'])
    return model, model_opt, lr_scheduler, trained_epochs, train_losses, val_losses


def train(model, bert_model, tokenizer, train_data, sentence_tokens, CCG_to_idx, config, optimizer, lr_scheduler, trained_epochs, all_train_losses):
    """Train the model and log relevant information."""
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config['model']['EPOCHS']):
        model.train()
        all_train_losses = run_epoch_scan(dataload_scan(config['model']['BATCH_SIZE'], train_data, sentence_tokens, config['model']['device'], CCG_to_idx, config, bert_model, tokenizer),model,SimpleLossCompute(criterion),optimizer,lr_scheduler, config, mode="train", current_losses=all_train_losses)[2]
        trained_epochs += 1
    return model, optimizer, lr_scheduler, all_train_losses, trained_epochs


if __name__ == '__main__':
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

    # train the model
    print("Training the model...")
    path = 'models/ConstructiveTagInsert/saved_models/ConstructiveTagInsert'
    model, optimizer, lr_scheduler, trained_epochs, train_losses, val_losses = resume_training(path, words_to_idx, CCG_to_idx, config, train_data, ATOMIC_VOCAB, encoder_to_idx, atomic_to_idx)
    model, optimizer, lr_scheduler, train_losses, trained_epochs = train(model, bert_model, tokenizer, train_data, sentence_tokens, CCG_to_idx, config, optimizer, lr_scheduler, trained_epochs, train_losses)

    # save the model
    save_model(model, optimizer, lr_scheduler, train_losses, val_losses, trained_epochs, 'models/ConstructiveTagInsert/saved_models/ConstructiveTagInsert')


