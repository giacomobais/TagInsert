import os
import sys

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

import torch
from transformers import AutoModel, RobertaTokenizer

import torch.nn as nn
from models.Variant1.scripts.preprocess import  load_data, get_atomic_labels, get_opaque_mapping, get_atomic_mapping, encode_CCG, get_positional_mapping, encode_trees, get_words_mapping, encode_words, get_splits, dataload_scan
from models.Variant1.model.utils import load_config, load_model, make_model, SimpleLossCompute, save_model
from models.Variant1.model.CCGNode import CCGNode, pretty_print_tree
from models.Variant1.model.model import TrainState

def run_epoch_scan(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=TrainState(),
    losses = []
    ):
    """Train a single epoch"""
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        # Forward pass
        trees, logits = model.forward(
                batch.src, batch.sequence_lengths, batch.tgt_y, mode
            )
        # Compute the loss
        loss, loss_node = loss_compute(logits, batch.tgt_y, batch.sequence_lengths, batch.ntokens)
        losses.append(loss_node.item())
        # Backward pass
        if mode == "train" or mode == "train+log":

            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.tgt_y.shape[0]
            train_state.tokens += batch.ntokens
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        # Log the loss
        if i % 2 == 0:
            print(f'Train loss at step {i*batch.tgt_y.shape[0]}/{(i+1)*batch.tgt_y.shape[0]}: {loss_node.data.item()} in mode {mode}', flush=True)
        del loss
        del loss_node
    return total_loss / total_tokens, train_state, losses

def resume_training(path, atomic_to_idx, config, train_data, encoder_to_idx):
    """Resume training from a pre-existent model. If no model is found, it initializes a new model."""
    try:
        model, model_opt, lr_scheduler, logs = load_model(path, atomic_to_idx, config, train_data, encoder_to_idx)
        trained_epochs = logs['epochs']
        train_losses = logs['train_losses']
        val_losses = logs['val_losses']
        print('Loading a pre-existent model.', flush=True)
    except:
        # Load pre-trained model (weights)
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        bert_model = AutoModel.from_pretrained('roberta-base',
                                            output_hidden_states = True,
                                            )
        bert_model = bert_model.to(config['model']['device'])
        model = make_model(len(atomic_to_idx), encoder_to_idx, atomic_to_idx, bert_model, tokenizer)
        model_opt = torch.optim.AdamW(model.parameters(), lr=config['model']['lr'], betas=(config['model']['beta1'], config['model']['beta2']), eps=config['model']['eps'])
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer = model_opt,
                                                  max_lr=config['model']['lr'], epochs = config['model']['EPOCHS'], steps_per_epoch=len(train_data) // config['model']['BATCH_SIZE'] + int(len(train_data) % config['model']['BATCH_SIZE'] != 0))

        trained_epochs = 0
        train_losses = []
        val_losses = []
        print('No pre-trained model found.', flush=True)

    model.train()
    model = model.to(config['model']['device'])
    return model, model_opt, lr_scheduler, trained_epochs, train_losses, val_losses

def train(model, train_data, sentence_tokens, CCG_to_idx, config, optimizer, lr_scheduler, trained_epochs, all_train_losses):
    """Train the model. Also logs relevant information."""
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config['model']['EPOCHS']):
        model.train()
        all_train_losses = run_epoch_scan(dataload_scan(config['model']['BATCH_SIZE'], train_data, sentence_tokens, config['model']['device'], CCG_to_idx, config),model,SimpleLossCompute(criterion),optimizer,lr_scheduler,mode="train", losses = all_train_losses)[2]
        trained_epochs += 1
    return model, optimizer, lr_scheduler, all_train_losses, trained_epochs


if __name__ == '__main__':
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

    # train the model
    print("Training the model...")
    path = 'models/Variant1/saved_models/Variant1'
    model, optimizer, lr_scheduler, trained_epochs, train_losses, val_losses = resume_training(path, atomic_to_idx, config, train_data, encoder_to_idx)
    model, optimizer, lr_scheduler, train_losses, trained_epochs = train(model, train_data, sentence_tokens, CCG_to_idx, config, optimizer, lr_scheduler, trained_epochs, train_losses)

    # save the model
    save_model(model, optimizer, lr_scheduler, train_losses, val_losses, trained_epochs, 'models/Variant1/saved_models/Variant1')


