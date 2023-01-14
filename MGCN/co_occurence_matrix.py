import os
import json
import os.path as osp
from transformers import BertTokenizer
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
lemmatizer = WordNetLemmatizer()
import numpy as np
import pickle
import arguments


def lemmatize(tokens):
    results=[]
    words, tags = zip(*nltk.pos_tag(tokens))
    for word, tag in zip(words,tags):
        if tag.startswith('NN'):
            results.append(lemmatizer.lemmatize(word, pos='n'))
        elif tag.startswith('VB'):
            results.append(lemmatizer.lemmatize(word, pos='v'))
        elif tag.startswith('JJ'):
            results.append(lemmatizer.lemmatize(word, pos='a'))
        elif tag.startswith('R'):
            results.append(lemmatizer.lemmatize(word, pos='r'))
        else:
            results.append(word)
    return results

def wn_similarity(x,y):
    try:
        word1=wn.synsets(x)[0]
        word2=wn.synsets(y)[0]
        return word1.path_similarity(word2)
    except:
        return 0.0

def adj_matrix_generate (num_nodes=300):
    ## n most frequent objects and motions in captions, image_id->objects+caption->one hot concept vector->co-occurence matrix
    n=num_nodes

    cwd=os.getcwd()
    print(cwd)
    vg_path = os.path.join('../data/vse_infty/VisualGenome/')

    with open(os.path.join(vg_path,'objects.json'),'r',encoding='utf8')as fp:
        objects = json.load(fp)
        print('the type of data in objects.json：',type(objects))

    with open(os.path.join(vg_path,'image_data.json'),'r',encoding='utf8')as fp:
        image_data = json.load(fp)
        print('the type of data in image_data.json：',type(image_data))

    # image_id is the same in image_data and objects
    for i,j in zip(objects,image_data):
        if i['image_id']!=j['image_id']:
            print(i,j)
            break

    # images in both coco and flickr
    count = 0
    for item in image_data:
        if item['coco_id'] != None and item['flickr_id'] != None:
            count +=1

    print(count)


    ## image_id in coco_train_id or flickr_train_id with coco_caption/flickr_caption
    base_path=osp.join('../data/vse_infty/')
    # Read Captions
    captions={}
    for data_name in ['coco','f30k']:
        captions[data_name] = []
        with open(osp.join(base_path,data_name,'precomp', '%s_caps.txt' % 'train'), 'r') as f:
            for line in f:
                captions[data_name].append(line.strip())

    # Get the train image ids
    image_ids={}
    image_id2captions={}
    for data_name in ['coco','f30k']:
        with open(osp.join(base_path,data_name,'precomp', '{}_ids.txt'.format('train')), 'r') as f:
            ids = f.readlines()
            image_ids[data_name] = [int(x.strip()) for x in ids]
        # image_id to captions
        image_id2captions[data_name]={}
        for i, id in enumerate(image_ids[data_name]):
            image_id2captions[data_name][id]=[captions[data_name][k] for k in range(5 * i, 5 * i + 5)]




    ## select the VG image in training datasets, and add the captions(the first one)
    print('select the vg images in training sets')
    select_data_path = osp.join(base_path,'coco','Concept_annotations','select_data.pkl')
    if os.path.exists(select_data_path):
        with open(select_data_path, 'rb') as f:
            select_data = pickle.load(f)
    else:
        select_data=[]
        log_step=1000
        all_step=len(image_data)
        for i,item in enumerate(image_data):
            if i%log_step ==0:
                print('{}/{}'.format(i,all_step))
            if item['coco_id'] in image_ids['coco'] or item['flickr_id'] in image_ids['f30k']:
                now_data=item
                if item['coco_id'] in image_ids['coco']:
                    now_data['caption']=image_id2captions['coco'][item['coco_id']]
                else:
                    now_data['caption'] = captions['f30k'][5 * item['flickr_id']]
                now_data['objects']=[object['names'][0] for object in objects[i]['objects']]
                select_data.append(now_data)

        with open(select_data_path,'wb') as f:
            pickle.dump(select_data,f)






    # word->concept
    # Load Tokenizer and Vocabulary
    print('calculate the n most frequent concepts')
    tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
    num_text_concepts={}
    num_visual_objects={}
    step_all = len(select_data)
    log_step = 1000
    for i,item in enumerate(select_data):
        if i%log_step ==0:
            print('{}/{}'.format(i,step_all))
        for caption in item['caption']:
            caption_tokens = tokenizer.basic_tokenizer.tokenize(caption)
            try:
                words = lemmatize(caption_tokens)
                for word in words:
                    if word in num_text_concepts:
                        num_text_concepts[word]+=1
                    else:
                        num_text_concepts[word] = 1
            except:
                continue

        try:
            objects = lemmatize(item['objects'])
            for object in objects:
                if object in num_visual_objects:
                    num_visual_objects[object] +=1
                else:
                    num_visual_objects[object] = 1
        except:
            continue

    visual_objects = list(num_visual_objects.items())
    visual_objects.sort(key=lambda x:x[1], reverse=True)
    visual_objects = [item[0] for item in visual_objects[:n]]

    text_concepts = list(num_text_concepts.items())
    text_concepts = [word for word in text_concepts if word[0] not in ['be','is','are']]
    text_concepts.sort(key=lambda x:x[1], reverse=True)

    text_motions=[]
    text_objects=[]
    pos_tags = nltk.pos_tag([item[0] for item in text_concepts])
    motion_tags = ['VB','VBD','VBG','VBN','VBP','VBZ']
    object_tags = ['NN','NNS']
    count_motion=0
    count_object=0
    for item in pos_tags:
        if count_motion < n and item[1] in motion_tags:
            text_motions.append(item[0])
            count_motion+=1
        if count_object<n and item[1] in object_tags:
            text_objects.append(item[0])
            count_object+=1



    ''' 
    co-occurence matrix 
    '''
    print('calculate the co-occurence matrix')
    text_concepts = text_motions+text_objects
    all_step = len(select_data)
    log_step = 1000
    co_matrix=np.zeros((3*n,3*n),dtype=int)
    for i,item in enumerate(select_data):
        if i%log_step ==0:
            print('{}/{}'.format(i,all_step))
        concept_labels = np.zeros(3*n,dtype=int)
        for caption in item['caption']:
            caption_tokens = tokenizer.basic_tokenizer.tokenize(caption)
            try:
                words = lemmatize(caption_tokens)
                for word in words:
                    if word in text_concepts:
                        ind_concept = text_concepts.index(word)
                        concept_labels[ind_concept] = 1
            except:
                continue
        try:
            objects = lemmatize(item['objects'])
            for object in objects:
                if object in visual_objects:
                    ind_concept = visual_objects.index(object)
                    concept_labels[ind_concept+2*n] = 1
        except:
            continue

        co_matrix = co_matrix + np.expand_dims(concept_labels,axis=0)*np.expand_dims(concept_labels,axis=1)

    print(co_matrix)


    ## matrix
    with open(osp.join(base_path,'coco','Concept_annotations','adj_matrix_{}.pkl'.format(str(n))),'wb') as f:
        pickle.dump(co_matrix,f)

    with open(osp.join(base_path,'f30k','Concept_annotations','adj_matrix_{}.pkl'.format(str(n))),'wb') as f:
        pickle.dump(co_matrix,f)


    # with open(osp.join(base_path,'coco','Concept_annotations','adj_matrix.pkl'),'rb') as f:
    #     hhh=pickle.load(f)

    ## concepts
    with open(osp.join(base_path,'coco','Concept_annotations','concepts_{}.pkl'.format(str(n))),'wb') as f:
        pickle.dump({'text_concepts':text_concepts, 'visual_objects':visual_objects},f)

    with open(osp.join(base_path,'f30k','Concept_annotations','concepts_{}.pkl'.format(str(n))),'wb') as f:
        pickle.dump({'text_concepts':text_concepts, 'visual_objects':visual_objects},f)




    '''
    concept similarity using wordnet (3n)*(3n) matrix
    '''
    print('concept similarity using wordnet')


    concepts_all = text_concepts + visual_objects
    concepts_all = [word.split(' ')[-1] for word in concepts_all]
    concepts_all = lemmatize(concepts_all)
    similarity_matrix=np.zeros((3*n,3*n))

    for i in range(3*n):
        if i %(3*n//20) == 0:
            print('{}/{}'.format(i,3*n))
        for j in range(3*n):
            try:
                similarity_matrix[i,j]=wn_similarity(concepts_all[i],concepts_all[j])
            except:
                similarity_matrix[i, j] = 0.0
    similarity_matrix[np.isnan(similarity_matrix)] = 0.0
    print(similarity_matrix)


    ## save similarity
    with open(osp.join(base_path,'coco','Concept_annotations','similarity_matrix_{}.pkl'.format(str(n))),'wb') as f:
        pickle.dump(similarity_matrix,f)

    with open(osp.join(base_path,'f30k','Concept_annotations','similarity_matrix_{}.pkl'.format(str(n))),'wb') as f:
        pickle.dump(similarity_matrix,f)
    

if __name__ == '__main__':
    # Hyper Parameters
    parser = arguments.get_argument_parser()
    opt = parser.parse_args()
    adj_matrix_generate(num_nodes = opt.num_attribute)