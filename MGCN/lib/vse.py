"""VSE model"""
import os
import torch.nn.init
from torch.nn.utils import clip_grad_norm_
from torch.nn import Parameter
from lib.encoders import get_image_encoder, get_text_encoder
from lib.loss import ContrastiveLoss
from lib.MMGCN import MMGCN_Enc

from models.model.decoder import ConceptDecoder
from models.layers.multi_head_attention import MyMultiHeadAttention
import logging

from collections import OrderedDict
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.init
import torchtext
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm

from lib.utils import *
from lib.C_GCN import C_GCN
import pickle
from scipy import linalg
logger = logging.getLogger(__name__)


class MKVSE(nn.Module):
    def __init__(self, opt):
        super(MKVSE, self).__init__()
        self.opt = opt
        self.GT_label_ratio = opt.Concept_label_ratio
        self.fuse_weight = opt.fuse_weight
        self.score_weight = opt.score_weight
        self.cat_weight = opt.cat_weight

        # concept embeddings
        text_concept_nodes = np.load(os.path.join(opt.data_path, opt.data_name, 'Concept_annotations',
                                                 opt.data_name + '_concepts_word2vec_{}.pkl'.format(str(opt.num_attribute))), allow_pickle=True)
        visual_concept_nodes = np.load(os.path.join(opt.data_path, opt.data_name, 'Concept_annotations',
                                                   opt.data_name + '_concepts_visual2vec_{}.pkl'.format(str(opt.num_attribute))), allow_pickle=True)
        self.text_concept_nodes = Parameter(torch.Tensor(text_concept_nodes), requires_grad = False)
        self.visual_concept_nodes =Parameter(torch.Tensor(visual_concept_nodes), requires_grad = False)
        self.num_text_nodes = self.text_concept_nodes.size(0)
        self.num_visual_nodes = self.visual_concept_nodes.size(0)

        # concept adj matrix
        with open(os.path.join(opt.data_path, opt.data_name, 'Concept_annotations',
                               'adj_matrix_{}.pkl'.format(str(opt.num_attribute))), 'rb') as f:
            adj_matrix = torch.Tensor(pickle.load(f))

        diag = torch.diag_embed(torch.diag(adj_matrix))
        adj_matrix = adj_matrix - diag
        self.adj_all = Parameter(adj_matrix / adj_matrix.sum(dim=1,keepdim=True), requires_grad = False)

        # TODO
        with open(os.path.join(opt.data_path, opt.data_name, 'Concept_annotations',
                               'similarity_matrix_{}.pkl'.format(str(opt.num_attribute))), 'rb') as f:
            similarity_matrix = torch.Tensor(pickle.load(f))
            mask = torch.eye(similarity_matrix.size(0),dtype=int)
            similarity_matrix = similarity_matrix.masked_fill(mask == 1, 1.0)

        self.adj_text = Parameter(similarity_matrix[:self.num_text_nodes, :self.num_text_nodes], requires_grad = False)
        self.adj_visual = Parameter(similarity_matrix[self.num_text_nodes:, self.num_text_nodes:], requires_grad = False)

        # encoders
        self.img_enc = get_image_encoder(opt.data_name, opt.img_dim, opt.embed_size,
                                         precomp_enc_type=opt.precomp_enc_type,
                                         backbone_source=opt.backbone_source,
                                         backbone_path=opt.backbone_path,
                                         no_imgnorm=opt.no_imgnorm)
        self.txt_enc = get_text_encoder(opt.embed_size, no_txtnorm=opt.no_txtnorm)

        self.text_linear = nn.Linear(self.text_concept_nodes.size(-1), opt.embed_size)
        self.visual_linear = nn.Linear(self.visual_concept_nodes.size(-1), opt.embed_size)
        nn.init.xavier_normal_(self.text_linear.weight, gain=1.414)
        nn.init.xavier_normal_(self.visual_linear.weight, gain=1.414)


        # MMGCN
        # TODO: l2norm before input
        self.concept_enc = MMGCN_Enc(opt.embed_size, bias = True)
        self.text_dec = ConceptDecoder(d_output = opt.embed_size,
                                      d_model = opt.embed_size,
                                      ffn_hidden = 2*opt.embed_size,
                                      n_head = opt.n_head_concept_encoder,
                                      n_layers = opt.n_layers_concept_decoder,
                                      drop_prob = opt.dropout_rate_concept_encoder)
        self.visual_dec = ConceptDecoder(d_output = opt.embed_size,
                                      d_model = opt.embed_size,
                                      ffn_hidden = 2*opt.embed_size,
                                      n_head = opt.n_head_concept_encoder,
                                      n_layers = opt.n_layers_concept_decoder,
                                      drop_prob = opt.dropout_rate_concept_encoder)

        # matrix = adj_matrix.numpy()
        # nums = matrix.sum(axis=1)
        # matrix = matrix + np.diag(nums)
        # matrix = matrix / np.sqrt(np.expand_dims(nums, axis=0)) / np.sqrt(np.expand_dims(nums, axis=1))
        # U = linalg.cholesky(matrix, lower=False)
        # self.W=nn.Linear(2*opt.num_attribute, 2*opt.num_attribute, bias=False)
        # self.W.weight.data = torch.Tensor(U)

        # Set up the lr for different parts of the VSE model
        decay_factor = 1e-4
        all_text_params = list(self.txt_enc.parameters())
        bert_params = list(self.txt_enc.bert.parameters())
        bert_params_ptr = [p.data_ptr() for p in bert_params]
        text_params_no_bert = list()
        for p in all_text_params:
            if p.data_ptr() not in bert_params_ptr:
                text_params_no_bert.append(p)
        self.optimizer = torch.optim.AdamW([
            {'params': text_params_no_bert, 'lr': opt.learning_rate},
            {'params': bert_params, 'lr': opt.learning_rate * 0.1},
            {'params': self.img_enc.parameters(), 'lr': opt.learning_rate},
            {'params': self.text_linear.parameters(), 'lr': opt.learning_rate/2},
            {'params': self.visual_linear.parameters(), 'lr': opt.learning_rate/2},
            {'params': self.concept_enc.parameters(), 'lr': opt.learning_rate / 2},
            {'params': self.text_dec.parameters(), 'lr': opt.learning_rate/2},
            {'params': self.visual_dec.parameters(), 'lr': opt.learning_rate/2},
            # {'params': self.W.parameters(), 'lr': opt.learning_rate},
        ],
            lr=opt.learning_rate, weight_decay=decay_factor)
        #
        # logger.info('Use {} as the optimizer, with init lr {}'.format('Adam', opt.learning_rate))
        # logger.info('Use {} as the optimizer, with init lr {}'.format('Adam', opt.learning_rate))
        # logger.info('image encoder trainable parameters: {}'.format(count_params(self.img_enc)))
        # logger.info('txt encoder trainable parameters: {}'.format(count_params(self.txt_enc)))
        # logger.info('text linear trainable parameters: {}'.format(count_params(self.text_linear)))
        # logger.info('visual linear trainable parameters: {}'.format(count_params(self.visual_linear)))
        # logger.info('concept encoder trainable parameters: {}'.format(count_params(self.concept_enc)))
        # logger.info('text dec trainable parameters: {}'.format(count_params(self.text_dec)))
        # logger.info('visual dec trainable parameters: {}'.format(count_params(self.visual_dec)))



    def forward(self, images, image_lengths, captions, lengths, concept_labels):
        img_emb, img_features = self.img_enc(images, image_lengths) # B*dim_emb, B*max_num_img_features*dim_emb
        cap_emb, cap_features = self.txt_enc(captions, lengths)  # B*dim_emb, B*L*dim_emb, L is the max length of caps in mini-batch

        text_concept_basis = self.text_linear(self.text_concept_nodes)
        visual_concept_basis = self.visual_linear(self.visual_concept_nodes)
        concept_basis = self.concept_enc(text_concept_basis, visual_concept_basis, self.adj_text, self.adj_visual, self.adj_all)
        concept_basis = concept_basis.unsqueeze(dim=0)
        concept_v, attention_v = self.visual_dec(img_emb.unsqueeze(dim=1),
                                                concept_basis.expand(img_emb.size(0),
                                                concept_basis.size(1),
                                                concept_basis.size(2)),
                                                trg_mask=None, src_mask=None)
        concept_t, attention_t = self.text_dec(cap_emb.unsqueeze(dim=1),
                                                concept_basis.expand(cap_emb.size(0),
                                                concept_basis.size(1),
                                                concept_basis.size(2)),
                                                trg_mask=None, src_mask=None)

        concept_v = concept_v.squeeze(dim=1)
        concept_t = concept_t.squeeze(dim=1)




        emb_v = torch.cat([torch.sqrt(torch.tensor(self.cat_weight))*img_emb,
                           torch.sqrt(torch.tensor(1-self.cat_weight))*concept_v], dim=1)
        emb_t = torch.cat([torch.sqrt(torch.tensor(self.cat_weight))*cap_emb,
                           torch.sqrt(torch.tensor(1-self.cat_weight))*concept_t], dim=1)
        emb_v=l2norm(emb_v ,dim=-1)
        emb_t=l2norm(emb_t, dim=-1)

        return emb_v, emb_t, attention_v, attention_t


class MKVSEModel(object):
    def __init__(self, opt):
        # Build Models
        self.opt = opt
        self.grad_clip = opt.grad_clip
        self.feature_fuse_type = opt.feature_fuse_type

        # forward loss
        self.model=MKVSE(opt)

        if torch.cuda.is_available():
            self.model.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(opt=opt,
                                         margin=opt.margin,
                                         max_violation=opt.max_violation)
        self.criterion_KL_softmax = KL_loss_softmax()
        self.params = self.model.parameters()

        self.Eiters = 0
        self.model = nn.DataParallel(self.model)
        self.optimizer = self.model.module.optimizer
        logger.info('The model is data paralleled now.')

    def set_max_violation(self, max_violation):
        if max_violation:
            self.criterion.max_violation_on()
        else:
            self.criterion.max_violation_off()

    def state_dict(self):
        return self.model.module.state_dict()

    def load_state_dict(self, state_dict):
        self.model.module.load_state_dict(state_dict)

    def train_start(self):
        self.model.train()

    def val_start(self):
        self.model.eval()

    def freeze_backbone(self):
        if 'backbone' in self.opt.precomp_enc_type:
            if isinstance(self.model, nn.DataParallel):
                self.model.module.img_enc.freeze_backbone()
            else:
                self.model.img_enc.freeze_backbone()

    def unfreeze_backbone(self, fixed_blocks):
        if 'backbone' in self.opt.precomp_enc_type:
            if isinstance(self.model, nn.DataParallel):
                self.model.module.img_enc.unfreeze_backbone(fixed_blocks)
            else:
                self.model.img_enc.unfreeze_backbone(fixed_blocks)


    def forward_emb(self, images, image_lengths, captions, lengths, concept_labels):

        """
        Compute the image and caption embeddings
        fdd: add params: concept_labels, concept_input_embs, alpha,
        """
        # Set mini-batch dataset
        if torch.cuda.is_available():
            images = images.cuda()  # B*C*D, C<=36, D is the dimension BUTD feature(default 2048)
            captions = captions.cuda()  # B*L, int, L is the max length of caps in mini-batch
            image_lengths = image_lengths.cuda()  # B, int
            lengths = torch.Tensor(lengths).cuda()  # B, int
            concept_labels = concept_labels.cuda()  # B*num_concepts, 0/1 int
        if self.opt.precomp_enc_type == 'basic':
            emb_v, emb_t, predict_score_v, predict_score_t = self.model(images, image_lengths, captions, lengths, concept_labels)
        else:
            raise ValueError("opt.precomp_enc_type must be 'basic' currently, it is {} now".format(self.opt.precomp_enc_type))

        return emb_v, emb_t, predict_score_v, predict_score_t

    def forward_loss(self, v_emb, t_emb):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(v_emb, t_emb)

        self.logger.update('Le', loss.item(), v_emb.size(0))
        return loss

    def train_emb(self, images, img_lengths, captions, lengths, concept_labels, ids):
        """One training step given images and captions.
        """

        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        v_emb, t_emb, predict_score_v, predict_score_t = \
            self.forward_emb(images, img_lengths, captions, lengths, concept_labels)

        # img_emb, cap_emb = self.forward_emb(images, captions, lengths, image_lengths=image_lengths)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        # loss = self.forward_loss(img_em
        loss = self.forward_loss(v_emb, t_emb)

        # compute gradient and update
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()


'''Image Encoder'''


def EncoderImage(data_name, img_dim, embed_size, precomp_enc_type='basic',
                 no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    if precomp_enc_type == 'basic':
        img_enc = EncoderImagePrecomp(
            img_dim, embed_size, no_imgnorm)
    elif precomp_enc_type == 'weight_norm':
        img_enc = EncoderImageWeightNormPrecomp(
            img_dim, embed_size, no_imgnorm)
    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))

    return img_enc


class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        '''features_mean: visual initial memory'''
        features_mean = torch.mean(features, 1)

        '''choose whether to l2norm'''
        # if not self.no_imgnorm:
        #     features_mean = l2norm(features_mean)

        return features, features_mean

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)


class EncoderImageWeightNormPrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImageWeightNormPrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = weight_norm(nn.Linear(img_dim, embed_size), dim=None)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param
        super(EncoderImageWeightNormPrecomp, self).load_state_dict(new_state)


''' Text encoder'''


class EncoderText(nn.Module):
    '''This func can utilize w2v initialization for word embedding'''

    def __init__(self, wemb_type, word2idx, opt, vocab_size, word_dim, embed_size, num_layers,
                 use_bidirectional_RNN=True, no_txtnorm=False,
                 use_abs=False, RNN_type='GRU'):

        super(EncoderText, self).__init__()
        self.use_abs = use_abs
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm
        self.vocab_size = vocab_size
        self.word_dim = word_dim

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        self.use_bidirectional_RNN = use_bidirectional_RNN
        self.RNN_type = RNN_type
        if RNN_type == 'GRU':
            self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bidirectional_RNN)
        elif RNN_type == 'LSTM':
            self.rnn = nn.LSTM(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bidirectional_RNN)

        self.dropout = nn.Dropout(opt.dropout_rate)

        # self.init_weights()
        '''change here'''
        self.init_weights(wemb_type, word2idx, word_dim)

    def init_weights(self, wemb_type, word2idx, word_dim):
        if wemb_type.lower() == 'random_init':
            nn.init.xavier_uniform_(self.embed.weight)
        else:
            # Load pretrained word embedding
            if 'fasttext' == wemb_type.lower():
                wemb = torchtext.vocab.FastText()
            elif 'glove' == wemb_type.lower():
                wemb = torchtext.vocab.GloVe()
            else:
                raise Exception('Unknown word embedding type: {}'.format(wemb_type))
            assert wemb.vectors.shape[1] == word_dim

            # quick-and-dirty trick to improve word-hit rate
            missing_words = []
            for word, idx in word2idx.items():
                if word not in wemb.stoi:
                    word = word.replace('-', '').replace('.', '').replace("'", '')
                    if '/' in word:
                        word = word.split('/')[0]
                if word in wemb.stoi:
                    self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
                else:
                    missing_words.append(word)
            print('Words: {}/{} found in vocabulary; {} words missing'.format(
                len(word2idx) - len(missing_words), len(word2idx), len(missing_words)))

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        x = self.dropout(x)

        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        if self.use_bidirectional_RNN:
            cap_emb = (cap_emb[:, :, : int(cap_emb.size(2) / 2)] + cap_emb[:, :, int(cap_emb.size(2) / 2):]) / 2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        # take absolute value, used by order embeddings
        if self.use_abs:
            cap_emb = torch.abs(cap_emb)

        cap_emb_mean = torch.mean(cap_emb, 1)
        if not self.no_txtnorm:
            cap_emb_mean = l2norm(cap_emb_mean)

        return cap_emb, cap_emb_mean


'''Image Encoder'''
def EncoderImage(data_name, img_dim, embed_size, precomp_enc_type='basic',
                 no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    if precomp_enc_type == 'basic':
        img_enc = EncoderImagePrecomp(
            img_dim, embed_size, no_imgnorm)
    elif precomp_enc_type == 'weight_norm':
        img_enc = EncoderImageWeightNormPrecomp(
            img_dim, embed_size, no_imgnorm)
    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))

    return img_enc


class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        '''features_mean: visual initial memory'''
        features_mean = torch.mean(features, 1)

        '''choose whether to l2norm'''
        # if not self.no_imgnorm:
        #     features_mean = l2norm(features_mean)

        return features, features_mean

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)


class EncoderImageWeightNormPrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImageWeightNormPrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = weight_norm(nn.Linear(img_dim, embed_size), dim=None)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param
        super(EncoderImageWeightNormPrecomp, self).load_state_dict(new_state)


''' Text encoder'''
class EncoderText(nn.Module):
    '''This func can utilize w2v initialization for word embedding'''

    def __init__(self, wemb_type, word2idx, opt, vocab_size, word_dim, embed_size, num_layers,
                 use_bidirectional_RNN=True, no_txtnorm=False,
                 use_abs=False, RNN_type='GRU'):

        super(EncoderText, self).__init__()
        self.use_abs = use_abs
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm
        self.vocab_size = vocab_size
        self.word_dim = word_dim

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        self.use_bidirectional_RNN = use_bidirectional_RNN
        self.RNN_type = RNN_type
        if RNN_type == 'GRU':
            self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bidirectional_RNN)
        elif RNN_type == 'LSTM':
            self.rnn = nn.LSTM(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bidirectional_RNN)

        self.dropout = nn.Dropout(opt.dropout_rate)

        # self.init_weights()
        '''change here'''
        self.init_weights(wemb_type, word2idx, word_dim)


    def init_weights(self, wemb_type, word2idx, word_dim):
        if wemb_type.lower() == 'random_init':
            nn.init.xavier_uniform_(self.embed.weight)
        else:
            # Load pretrained word embedding
            if 'fasttext' == wemb_type.lower():
                wemb = torchtext.vocab.FastText()
            elif 'glove' == wemb_type.lower():
                wemb = torchtext.vocab.GloVe()
            else:
                raise Exception('Unknown word embedding type: {}'.format(wemb_type))
            assert wemb.vectors.shape[1] == word_dim

            # quick-and-dirty trick to improve word-hit rate
            missing_words = []
            for word, idx in word2idx.items():
                if word not in wemb.stoi:
                    word = word.replace('-', '').replace('.', '').replace("'", '')
                    if '/' in word:
                        word = word.split('/')[0]
                if word in wemb.stoi:
                    self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
                else:
                    missing_words.append(word)
            print('Words: {}/{} found in vocabulary; {} words missing'.format(
                len(word2idx) - len(missing_words), len(word2idx), len(missing_words)))


    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        x = self.dropout(x)

        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        if self.use_bidirectional_RNN:
            cap_emb = (cap_emb[:, :, : int(cap_emb.size(2) / 2)] + cap_emb[:, :, int(cap_emb.size(2) / 2):]) / 2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        # take absolute value, used by order embeddings
        if self.use_abs:
            cap_emb = torch.abs(cap_emb)

        cap_emb_mean = torch.mean(cap_emb, 1)
        if not self.no_txtnorm:
            cap_emb_mean = l2norm(cap_emb_mean)

        return cap_emb, cap_emb_mean


''' Visual self-attention module '''
class V_single_modal_atten(nn.Module):
    """
    Single Visual Modal Attention Network.
    """

    def __init__(self, image_dim, embed_dim, activation_type, dropout_rate):
        """
        param image_dim: dim of visual feature
        param embed_dim: dim of embedding space
        """
        super(V_single_modal_atten, self).__init__()

        self.fc1 = nn.Linear(image_dim, embed_dim)  # embed visual feature to common space

        self.fc2 = nn.Linear(image_dim, embed_dim)  # embed memory to common space
        self.fc2_2 = nn.Linear(embed_dim, embed_dim)

        self.fc3 = nn.Linear(embed_dim, 1)  # turn fusion_info to attention weights
        self.fc4 = nn.Linear(image_dim, embed_dim)  # embed attentive feature to common space


        if activation_type == 'tanh':
            self.embedding_1 = nn.Sequential(self.fc1,
                                             nn.Tanh(),
                                             nn.Dropout(dropout_rate))
            self.embedding_2 = nn.Sequential(self.fc2,
                                             nn.Tanh(),
                                             nn.Dropout(dropout_rate))
            self.embedding_2_2 = nn.Sequential(self.fc2_2,
                                               nn.Tanh(),
                                               nn.Dropout(dropout_rate))
            self.embedding_3 = nn.Sequential(self.fc3,
                                               nn.Tanh(),
                                               nn.Dropout(dropout_rate))
        else:
            self.embedding_1 = nn.Sequential(self.fc1,
                                             nn.Sigmoid(),
                                             nn.Dropout(dropout_rate))
            self.embedding_2 = nn.Sequential(self.fc2,
                                             nn.Sigmoid(),
                                             nn.Dropout(dropout_rate))
            self.embedding_2_2 = nn.Sequential(self.fc2_2,
                                               nn.BatchNorm1d(embed_dim),
                                               nn.Sigmoid(),
                                               nn.Dropout(dropout_rate))
            self.embedding_3 = nn.Sequential(self.fc3)

        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, v_t, m_v):
        """
        Forward propagation.
        :param v_t: encoded images, shape: (batch_size, num_regions, image_dim)
        :param m_v: previous visual memory, shape: (batch_size, image_dim)
        :return: attention weighted encoding, weights
        """
        W_v = self.embedding_1(v_t)

        if m_v.size()[-1] == v_t.size()[-1]:
            W_v_m = self.embedding_2(m_v)
        else:
            W_v_m = self.embedding_2_2(m_v)

        W_v_m = W_v_m.unsqueeze(1).repeat(1, W_v.size()[1], 1)

        h_v = W_v.mul(W_v_m)

        a_v = self.embedding_3(h_v)
        a_v = a_v.squeeze(2)
        weights = self.softmax(a_v)

        v_att = ((weights.unsqueeze(2) * v_t)).sum(dim=1)

        # l2 norm
        v_att = l2norm((v_att))

        return v_att, weights


''' Textual self-attention module '''
class T_single_modal_atten(nn.Module):
    """
    Single Textual Modal Attention Network.
    """

    def __init__(self, embed_dim, activation_type, dropout_rate):
        """
        param embed_dim: dim of embedding space/ dim of input text features
        """
        super(T_single_modal_atten, self).__init__()

        self.fc1 = nn.Linear(embed_dim, embed_dim)  # embed visual feature to common space
        self.fc2 = nn.Linear(embed_dim, embed_dim)  # embed memory to common space
        self.fc3 = nn.Linear(embed_dim, 1)  # turn fusion_info to attention weights

        if activation_type == 'tanh':
            self.embedding_1 = nn.Sequential(self.fc1,
                                             nn.Tanh(),
                                             nn.Dropout(dropout_rate))
            self.embedding_2 = nn.Sequential(self.fc2,
                                             nn.Tanh(),
                                             nn.Dropout(dropout_rate))
            self.embedding_3 = nn.Sequential(self.fc3)
        elif activation_type == 'sigmoid':
            self.embedding_1 = nn.Sequential(self.fc1,
                                             nn.Sigmoid(),
                                             nn.Dropout(dropout_rate))
            self.embedding_2 = nn.Sequential(self.fc2,
                                             nn.Sigmoid(),
                                             nn.Dropout(dropout_rate))
            self.embedding_3 = nn.Sequential(self.fc3)

        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, u_t, m_u):
        """
        Forward propagation.
        :param v_t: encoded images, shape: (batch_size, num_regions, image_dim)
        :param m_v: previous visual memory, shape: (batch_size, image_dim)
        :return: attention weighted encoding, weights
        """

        W_u = self.embedding_1(u_t)

        W_u_m = self.embedding_2(m_u)
        W_u_m = W_u_m.unsqueeze(1).repeat(1, W_u.size()[1], 1)

        h_u = W_u.mul(W_u_m)

        a_u = self.embedding_3(h_u)
        a_u = a_u.squeeze(2)
        weights = self.softmax(a_u)

        u_att = ((weights.unsqueeze(2) * u_t)).sum(dim=1)

        # l2 norm
        u_att = l2norm(u_att)

        return u_att, weights


'''Fusing instance-level feature and consensus-level feature'''
class Multi_feature_fusing(nn.Module):
    """
    Emb the features from both modalities to the joint attribute label space.
    """

    def __init__(self, embed_dim, fuse_type='weight_sum'):
        """
        param image_dim: dim of visual feature
        param embed_dim: dim of embedding space
        """
        super(Multi_feature_fusing, self).__init__()

        self.fuse_type = fuse_type
        self.embed_dim = embed_dim
        if fuse_type == 'concat':
            input_dim = int(2*embed_dim)
            self.joint_emb_v = nn.Linear(input_dim, embed_dim)
            self.joint_emb_t = nn.Linear(input_dim, embed_dim)
            self.init_weights_concat()
        if fuse_type == 'adap_sum':
            self.joint_emb_v = nn.Linear(embed_dim, 1)
            self.joint_emb_t = nn.Linear(embed_dim, 1)
            self.init_weights_adap_sum()

    def init_weights_concat(self):
        """Xavier initialization"""
        r = np.sqrt(6.) / np.sqrt(self.embed_dim + 2*self.embed_dim)
        self.joint_emb_v.weight.data.uniform_(-r, r)
        self.joint_emb_v.bias.data.fill_(0)
        self.joint_emb_t.weight.data.uniform_(-r, r)
        self.joint_emb_t.bias.data.fill_(0)

    def init_weights_adap_sum(self):
        """Xavier initialization"""
        r = np.sqrt(6.) / np.sqrt(self.embed_dim + 1)
        self.joint_emb_v.weight.data.uniform_(-r, r)
        self.joint_emb_v.bias.data.fill_(0)
        self.joint_emb_t.weight.data.uniform_(-r, r)
        self.joint_emb_t.bias.data.fill_(0)

    def forward(self, v_emb_instance, t_emb_instance, v_emb_concept, t_emb_concept, alpha=0.75):
        """
        Forward propagation.
        :param v_emb_instance, t_emb_instance: instance-level visual or textual features, shape: (batch_size, emb_dim)
        :param v_emb_concept, t_emb_concept: consensus-level concept features, shape: (batch_size, emb_dim)
        :return: joint embbeding features for both modalities
        """
        if self.fuse_type == 'multiple':
            v_fused_emb = v_emb_instance.mul(v_emb_concept);
            v_fused_emb = l2norm(v_fused_emb)
            t_fused_emb = t_emb_instance.mul(t_emb_concept);
            t_fused_emb = l2norm(t_fused_emb)

        elif self.fuse_type == 'concat':
            v_fused_emb = torch.cat([v_emb_instance, v_emb_concept], dim=1)
            v_fused_emb = self.joint_emb_instance_v(v_fused_emb)
            v_fused_emb = l2norm(v_fused_emb)

            t_fused_emb = torch.cat([t_emb_instance, t_emb_concept], dim=1)
            t_fused_emb = self.joint_emb_instance_v(t_fused_emb)
            t_fused_emb = l2norm(t_fused_emb)

        elif self.fuse_type == 'adap_sum':
            v_mean = (v_emb_instance + v_emb_concept) / 2
            v_emb_instance_mat = self.joint_emb_instance_v(v_mean)
            alpha_v = F.sigmoid(v_emb_instance_mat)
            v_fused_emb = alpha_v * v_emb_instance + (1 - alpha_v) * v_emb_concept
            v_fused_emb = l2norm(v_fused_emb)

            t_mean = (t_emb_instance + t_emb_concept) / 2
            t_emb_instance_mat = self.joint_emb_instance_t(t_mean)
            alpha_t = F.sigmoid(t_emb_instance_mat)
            t_fused_emb = alpha_t * t_emb_instance + (1 - alpha_t) * t_emb_concept
            t_fused_emb = l2norm(t_fused_emb)

        elif self.fuse_type == 'weight_sum':
            # alpha = 0.75

            v_fused_emb = alpha * v_emb_instance + (1 - alpha) * v_emb_concept
            v_fused_emb = l2norm(v_fused_emb)
            t_fused_emb = alpha * t_emb_instance + (1 - alpha) * t_emb_concept
            t_fused_emb = l2norm(t_fused_emb)

        return v_fused_emb, t_fused_emb


''' Consensus-level feature learning module '''
class Consensus_level_feature_learning(nn.Module):
    """
    Consensus-level feature learning module .
    """
    def __init__(self, image_dim, embed_dim, use_bn, activation_type, dropout_rate, attribute_num,
                 no_imgnorm=False, ):
        """
        param image_dim: dim of visual feature
        param embed_dim: dim of embedding space
        """
        super(Consensus_level_feature_learning, self).__init__()

        self.no_imgnorm = no_imgnorm
        self.fc1 = nn.Linear(image_dim, embed_dim)  # embed visual feature to common space
        self.fc2 = nn.Linear(embed_dim, embed_dim)  # embed attribute to common space
        self.fc3 = nn.Linear(embed_dim, 1)  # turn fusion_info to attention weights

        if use_bn == True and activation_type == 'tanh':
            self.embedding_1 = nn.Sequential(
                                             self.fc1,
                                             nn.BatchNorm1d(embed_dim),
                                             nn.Tanh()
                                             )

            self.embedding_2 = nn.Sequential(
                                             self.fc2,
                                             nn.BatchNorm1d(embed_dim),
                                             nn.Tanh()
                                             )
            self.embedding_3 = nn.Sequential(self.fc3)
        elif use_bn == False and activation_type == 'tanh':
            self.embedding_1 = nn.Sequential(
                                             self.fc1,
                                             nn.Tanh()
                                             )
            self.embedding_2 = nn.Sequential(
                                             self.fc2,
                                             nn.Tanh()
                                             )
            self.embedding_3 = nn.Sequential(self.fc3)
        elif use_bn == True and activation_type == 'sigmoid':
            self.embedding_1 = nn.Sequential(
                                             self.fc1,
                                             nn.BatchNorm1d(embed_dim),
                                             nn.Sigmoid()
                                             )
            self.embedding_2 = nn.Sequential(
                                             self.fc2,
                                             nn.BatchNorm1d(embed_dim),
                                             nn.Sigmoid()
                                             )
            self.embedding_3 = nn.Sequential(self.fc3)
        else:
            self.embedding_1 = nn.Sequential(
                                             self.fc1,
                                             nn.Dropout(dropout_rate)
                                             )
            self.embedding_2 = nn.Sequential(
                                             self.fc2,
                                             nn.Dropout(dropout_rate)
                                             )
            self.embedding_3 = nn.Sequential(self.fc3)

        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights
        self.smooth_coef = 10


    def forward(self, emb_instance, concept_feature, input_modal, GT_label, GT_label_ratio):
        """
        Forward propagation.
        :param emb_instance: encoded images or text, shape: (batch_size, emb_dim)
        :param concept_feature: concept feature, shape: (att_num, emb_dim)
        :return: emb_concept: consensus-level feature
                 weights_u, weights_v: predicted concept score
        """
        W_s = self.embedding_1(concept_feature)  # (concept_num, emb_dim) 600*1024

        W_v_m = self.embedding_2(emb_instance)   # (bs, emb_dim)
        W_v_m = W_v_m.unsqueeze(1).repeat(1, W_s.size()[0], 1)   # (bs, att_num, emb_dim)

        h_s = W_s.mul(W_v_m)    # (bs, concept_num, emb_dim)

        a_s = self.embedding_3(h_s) # (bs, concept_num, 1)
        a_s = a_s.squeeze(2)        # (bs, concept_num)

        weights = self.softmax(a_s * self.smooth_coef) #B*concept_num
        concept_feature = l2norm(concept_feature)
        emb_concept = (weights.unsqueeze(2) * concept_feature).sum(dim=1)
        if not self.no_imgnorm:
            emb_concept = l2norm(emb_concept) # B*emb_dim
        return emb_concept, weights



'''KL regularizer for softmax prob distribution'''
class KL_loss_softmax(nn.Module):
    """
    Compute KL_divergence between all prediction score (already sum=1, omit softmax function)
    """
    def __init__(self):
        super(KL_loss_softmax, self).__init__()

        self.KL_loss = nn.KLDivLoss(reduce=False)

    def forward(self, im, s):
        img_prob = torch.log(im)
        s_prob = s
        KL_loss = self.KL_loss(img_prob, s_prob)
        loss = KL_loss.sum()

        return loss


def count_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params