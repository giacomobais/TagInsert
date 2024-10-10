import os
import sys

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import yaml
import math
from models.ConstructiveTagInsert.model.CCGNode import CCGNode

def load_config():
    """The function loads the config from the config.yaml file"""
    with open('models/ConstructiveTagInsert/config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, addrmlp, CCG_to_idx):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.addrmlp = addrmlp
        self.CCG_to_idx = CCG_to_idx

    def forward(self, src, tgt, tree_tgt, src_mask, tgt_mask, embs, sequence_lengths, mode):
        "Take in and process masked src and target sequences."
        roots, logits, locations = self.decode(self.encode(src, src_mask, embs), src, embs, src_mask, tgt, tree_tgt, tgt_mask, sequence_lengths, mode, self.CCG_to_idx)
        return roots, logits, locations

    def encode(self, src, src_mask, embs):
        """ Encode the source sentence."""
        emb = self.src_embed(embs)
        out = self.encoder(emb, src_mask)
        return out

    def decode(self, memory, src, embs, src_mask, tgt, tree_tgt, tgt_mask, sequence_lengths, mode, CCG_to_idx, done_positions = None):
        """ Decode the target sentence. The words embeddings are added to the target embeddings. The decoder outputs the logits for the target sentence.
          The AddrMLP takes the argmaxed logit (position to build next) and outputs the root of the tree and the logits for the next position. """
        self.config = load_config()
        # get the embeddings for the source and target sentences
        src_emb = self.src_embed(embs)
        emb = self.tgt_embed(tgt)
        # add the source embeddings to the target embeddings
        emb = self.addsrc(emb, src_emb)
        # decode the target sentence
        out = self.decoder(emb, memory, src_mask, tgt_mask)
        # get the logits for the target sentence
        prob = self.generator(out)

        # evaluation setting, the number of done positions needs to be initialized depending on the sentence length at that step
        if done_positions is None:
            done_positions = torch.zeros((prob.size(0), self.config['model']['BLOCK_SIZE']), dtype=torch.long)
        # iterate over the target sentence
        for i, sent in enumerate(tgt):
            for j, tag in enumerate(sent):
                # if the tag is unknown, mark the position as not done
                if tag.item() == CCG_to_idx['<UNK>']:
                    done_positions[i][j] = 1
        # create a mask for the done positions
        mask = done_positions.bool()
        # expand the mask to match the shape of the logits
        mask_expanded = mask.unsqueeze(-1).expand_as(prob)
        # set the logits for the done positions to -inf
        masked_probs = prob.clone()
        masked_probs[~mask_expanded] = float('-inf')
        # flatten the logits and get the argmax
        masked_probs = masked_probs.view(prob.size(0), -1)
        flattened_indices = torch.argmax(masked_probs, dim = 1)
        locations = flattened_indices // prob.size(2)
        # perform the AddrMLP forward pass, return the built tree and the logits for the next position
        roots, logits = self.addrmlp(out, sequence_lengths, tree_tgt, mode, single_tree = locations)
        return roots, logits, locations


    def addsrc(self, emb, src_emb):
        """ Add the source embeddings to the target embeddings. """
        emb[:, 1:] += src_emb[:, :-1]
        return emb


class AddrMLP(nn.Module):
    """ The AddrMLP module from Prange """
    def __init__(self, tgt_vocab, enc_to_idx, atomic_to_idx):
        super(AddrMLP, self).__init__()
        self.config = load_config()
        self.dropout = nn.Dropout(0.2)
        self.layernorm = nn.LayerNorm(self.config['model']['d_model'])
        # Linear layer to transform the features of each node
        self.feats = nn.Sequential(
            nn.Linear(self.config['model']['d_model'], self.config['model']['d_model']),
            nn.GELU(),
            self.dropout
        )
        # MLP to predict the root of the tree and the logits for the next position
        self.mlp = nn.Sequential(
            self.layernorm,
            nn.Linear(self.config['model']['d_model'], self.config['model']['d_model']),
            nn.GELU(),
            self.dropout,
            self.layernorm,
            nn.Linear(self.config['model']['d_model'], self.config['model']['d_model']),
            nn.GELU(),
            self.dropout
        )
        # Softmax layer to get the probabilities for the next position
        self.softmax_layer = nn.Sequential(
            nn.Linear(self.config['model']['d_model'], tgt_vocab, bias=False),
            nn.Softmax(dim=-1)
        )

        self.tgt_vocab = tgt_vocab
        self.enc_to_idx = enc_to_idx
        self.atomic_to_idx = atomic_to_idx
        # Embedding layer for the position of each node in the tree
        self.feat_embs = nn.Embedding(num_embeddings=len(self.enc_to_idx), embedding_dim=self.config['model']['d_model'])

    def predict_data(self, node, memory, current_depth):
        """ Predict the root of the tree and return the logits for the built tree. """
        # get the position of the node in the tree
        pos = node.pos
        if len(pos) == 0:
            pos = '0'
        else:
            pos = ''.join(map(str, pos))
        # get the ancestors slashes
        slashes = node.get_ancestors_slashes(self.atomic_to_idx)
        if len(slashes) == 0:
            slashes = '0'
        else:
            slashes =  ''.join(map(str, slashes))
        # get the features of the node
        features = torch.tensor(self.enc_to_idx[pos+slashes], requires_grad = False).to(self.config['model']['device'])
        # embed the features
        out = self.feat_embs(features)
        # pass the features through the MLP
        out = self.feats(out)
        # add the TagInsert decoder representation to the features
        out = out + memory
        # pass the features through the MLP
        out = self.mlp(out)
        # pass the features through the softmax layer
        out = self.softmax_layer(out)
        # get the predicted root of the tree
        pred = torch.argmax(out)
        return pred, out


    def build_tree_depth(self, root, memory, tgt, mode, current_depth):
        """ Build the tree on a specific position of the sequence. """
        # initialize the nodes at the current depth
        nodes_at_d = [root]
        # initialize the logits for the current depth
        logits = torch.zeros((2**(self.config['model']['MAX_DEPTH']+1)-1, self.tgt_vocab)).to(self.config['model']['device'])
        # predict the root of the tree
        atomic, distribution = self.predict_data(root, memory, 0)
        # set the logits for the root
        logits[0] = distribution
        # set the data for the root
        root.data = atomic.item()
        # if the mode is train, set the teacher data for the root
        if mode == "train":
            label = tgt[0].item()
            root.teacher_data = label
        else:
            # if the mode is not train, set the teacher data to the data
            label = root.data
            root.teacher_data = root.data
        # if the label is a slash or backslash, add the children to the root
        if label == self.atomic_to_idx['/'] or label == self.atomic_to_idx['\\']:
            # iterate over the depth of the tree
            for d in range(self.config['model']['MAX_DEPTH']):
                # initialize the nodes at the current depth
                new_nodes_at_d = [None] * (2**(d+1))
                # iterate over the nodes at the current depth
                for n, node in enumerate(nodes_at_d):
                    if node is None:
                        continue
                    # if the mode is train, set the label to the teacher data
                    if mode == "train":
                        label = node.teacher_data
                    else:
                        # if the mode is not train, set the label to the data
                        label = node.data
                    # if the label is a slash or backslash, add the children to the root
                    if label == self.atomic_to_idx['/'] or label == self.atomic_to_idx['\\']:
                        left_child = CCGNode(supertag = None)
                        right_child = CCGNode(supertag = None)
                        # add the children to the root
                        node.add_children(left_child, right_child)
                        # predict the data for the children
                        left_atomic, left_distribution = self.predict_data(left_child, memory, d+1)
                        right_atomic, right_distribution = self.predict_data(right_child, memory, d+1)
                        # set the data for the children
                        left_child.data = left_atomic.item()
                        right_child.data = right_atomic.item()
                        # add the children to the new nodes at the current depth
                        new_nodes_at_d[n*2] = left_child
                        new_nodes_at_d[(n*2)+1] = right_child
                        # set the logits for the children
                        logits[(2**(d+1)-1) + n*2] = left_distribution
                        logits[(2**(d+1)-1) + (n*2)+1] = right_distribution
                        # if the mode is train, set the teacher data for the children
                        if mode == 'train':
                            left_child.teacher_data = tgt[(2**(d+1)-1) + n*2].item()
                            right_child.teacher_data = tgt[(2**(d+1)-1) + (n*2)+1].item()
                        else:
                            # if the mode is not train, set the teacher data to the predicteddata
                            left_child.teacher_data = left_child.data
                            right_child.teacher_data = right_child.data
                       
                # check if the new nodes at the current depth are all slashes or backslashes, if so, stop the loop to save computation
                for n, node in enumerate(new_nodes_at_d):
                    if node is None:
                        continue
                    flag = 0
                    if node.teacher_data == self.atomic_to_idx['/'] or node.teacher_data == self.atomic_to_idx['\\']:
                        flag = 1
                        break
                if flag == 0:
                    break
                # update the nodes at the current depth
                nodes_at_d = new_nodes_at_d
        return root, logits


    def forward(self, memory, seq_length, tgt, mode, single_tree = None):
        """ The forward pass of the AddrMLP. """
        # build the tree on a specific position of the sequence
        if single_tree is not None:
            sequence_trees = []
            # initialize the logits for the current depth for all the trees in the batch
            logits = torch.zeros((memory.size(0), 1, 2**(self.config['model']['MAX_DEPTH']+1)-1, self.tgt_vocab)).to(self.config['model']['device'])
            # build the trees across the batch
            for i, loc in enumerate(single_tree):
                CCG_tree = CCGNode(supertag = None)
                # if the mode is train, there is a target tag for the root, otherwise, there is no target tag for the root and the teacher data is set to the predicted data
                if mode == 'train':
                    tree, tag_logits = self.build_tree_depth(CCG_tree, memory[i, loc], tgt[i, loc-1], mode, current_depth=0)
                else:
                    tree, tag_logits = self.build_tree_depth(CCG_tree, memory[i, loc], None, mode, current_depth=0)
                sequence_trees.append(tree)
                # set the logits for the root
                logits[i][0] = tag_logits
            return sequence_trees.copy(), logits
        # if the single tree is not specified, build the trees across the batch
        else:
            out = []
            # initialize the logits for the current depth for all the trees in the batch
            logits = torch.zeros((memory.size(0), self.config['model']['BLOCK_SIZE']-1, 2**(self.config['model']['MAX_DEPTH']+1)-1, self.tgt_vocab)).to(self.config['model']['device'])
            for i, sequence in enumerate(memory):
                sequence_trees = []
                for j in range(seq_length[i]+1):
                    if j != 0: # skip start token and end
                        CCG_tree = CCGNode(supertag = None)
                        if mode == 'train':
                            CCG_tree, tag_logits = self.build_tree_depth(CCG_tree, memory[i, j], tgt[i, j-1], mode, current_depth=0)
                        else:
                            CCG_tree, tag_logits = self.build_tree_depth(CCG_tree, memory[i, j], None, mode, current_depth=0)
                        sequence_trees.append(CCG_tree)
                        logits[i][j-1] = tag_logits
                out.append(sequence_trees.copy())
        return out, logits

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        out = F.softmax(self.proj(x), dim=-1)
        return out

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

class POS_Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(POS_Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PretrainedEmbeddings(nn.Module): # modify this so it returns the  bert embeddings calculated on the fly for the current x
    def __init__(self, d_model, vocab):
        super(PretrainedEmbeddings, self).__init__()
        # self.lut = nn.Embedding.from_pretrained(glove_vectors, freeze = False)
        self.d_model = d_model

    def forward(self, embs):
        # print(embs.shape)
        return embs

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

def make_model(
    src_vocab, tgt_vocab, ATOMIC_VOCAB, encoder_to_idx, atomic_to_idx, CCG_to_idx, N=6, d_model=768, d_ff=768*4, h=8, dropout=0.1
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(PretrainedEmbeddings(d_model, src_vocab), c(position)),
        nn.Sequential(POS_Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
        AddrMLP(ATOMIC_VOCAB, encoder_to_idx, atomic_to_idx),
        CCG_to_idx
    )

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model