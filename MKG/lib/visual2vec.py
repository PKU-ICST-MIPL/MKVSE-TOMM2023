import os
import numpy as np
import torch
from transformers import BertTokenizer
import pickle
from lib.datasets import image_caption
import arguments
torch.multiprocessing.set_sharing_strategy('file_system')

def visual2vec(opt, tokenizer):
    print('Begin to generate visual2vec.pkl')
    # train=False to avoid data augmentation
    train_loader = image_caption.get_loader(opt.data_path, opt.data_name, 'train', tokenizer, opt,
                                            1000, shuffle=True, num_workers=opt.workers, train=False)

    # initialize
    data_concepts_visual2vec=torch.zeros(300,2048).cuda()
    count = torch.zeros(300,1).cuda() + 1e-10
    log_step=len(train_loader)//10

    # calculate visual2vec
    for i, train_data in enumerate(train_loader):
        all_images, img_lengths, targets, lengths, attribute_labels, ids = train_data
        all_images=all_images.cuda()
        attribute_labels=attribute_labels.cuda()
        images_pooled=all_images.mean(dim=1,keepdim=True)
        attribute_labels = attribute_labels.unsqueeze(dim=2)
        new_memory=images_pooled * attribute_labels
        new_memory = new_memory.sum(dim=0, keepdim=False) # 300*2048
        new_count = attribute_labels.sum(dim=0, keepdim=False) # 300*1
        # update
        data_concepts_visual2vec = torch.div((data_concepts_visual2vec * count) + new_memory,(count+new_count)) # 300*2048
        count += new_count # 300*1
        # logger
        if i % log_step ==0:
            print(100*i/len(train_loader),'% has been done.')

    # save to .pkl
    data_concepts_visual2vec = data_concepts_visual2vec.cpu().numpy()
    outfile=os.path.join(opt.data_path,opt.data_name,'Concept_annotations',opt.data_name+'_concepts_visual2vec.pkl')
    with open(outfile, 'wb') as file:
        pickle.dump(data_concepts_visual2vec, file)

    # check for debug
    # data_load=np.load(outfile, allow_pickle=True)
    # print(np.abs(data_load-data_concepts_visual2vec).max())


if __name__ == '__main__':
    # Hyper Parameters
    parser = arguments.get_argument_parser()
    opt = parser.parse_args()

    # Load Tokenizer and Vocabulary
    tokenizer = BertTokenizer.from_pretrained('../bert-base-uncased')
    vocab = tokenizer.vocab
    opt.vocab_size = len(vocab)
    outfile = os.path.join(opt.data_path, opt.data_name, 'Concept_annotations',
                           opt.data_name + '_concepts_visual2vec.pkl')

    try:
        data_load = np.load(outfile, allow_pickle=True)
        print('Success to load visual2vec.pkl')
    except:
        visual2vec(opt, tokenizer)
        data_load = np.load(outfile, allow_pickle=True)
        print('Success to load visual2vec.pkl')
