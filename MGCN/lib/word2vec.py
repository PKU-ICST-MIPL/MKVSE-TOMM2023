import os
import numpy as np
from transformers import BertTokenizer
import pickle
from lib.datasets import image_caption
import torch
import torchtext
import arguments


torch.multiprocessing.set_sharing_strategy('file_system')


def word2vec(opt):
    categroy_concepts_dir = os.path.join(opt.data_path, opt.data_name,
                                         'Concept_annotations', 'concepts_{}.pkl'.format(str(opt.num_attribute)))
    with open(categroy_concepts_dir, "rb") as names_concepts:
        concepts = pickle.load(names_concepts)
        text_concepts = concepts['text_concepts']

    glove = torchtext.vocab.GloVe(name="840B", dim=300)
    word_embeddings = torch.zeros(len(text_concepts),300)
    for i, word in enumerate(text_concepts):
        word_embeddings[i,:]=glove[word]

    outfile = os.path.join(opt.data_path, opt.data_name, 'Concept_annotations',
                           opt.data_name + '_concepts_word2vec_{}.pkl'.format(opt.num_attribute))
    with open(outfile, 'wb') as file:
        pickle.dump(word_embeddings, file)



if __name__ == '__main__':
    # Hyper Parameters
    parser = arguments.get_argument_parser()
    opt = parser.parse_args()
    outfile = os.path.join(opt.data_path, opt.data_name, 'Concept_annotations',
                           opt.data_name + '_concepts_word2vec_{}.pkl'.format(opt.num_attribute))

    try:
        data_load = np.load(outfile, allow_pickle=True)
        print('Success to load visual2vec.pkl')
    except:
        word2vec(opt)
        data_load = np.load(outfile, allow_pickle=True)
        print('Success to load visual2vec.pkl')
