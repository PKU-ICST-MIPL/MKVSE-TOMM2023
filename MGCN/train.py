"""Training script"""
import os
import time
import numpy as np
import torch
from transformers import BertTokenizer

from lib.datasets import image_caption
from lib.vse import MKVSEModel
from lib.evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data, compute_sim
from lib.visual2vec import visual2vec
from lib.word2vec import word2vec

import logging
import tensorboard_logger as tb_logger

import arguments


# try to fix the bug: deadlock in dataloader
torch.multiprocessing.set_sharing_strategy('file_system')


def main():
    # Hyper Parameters
    parser = arguments.get_argument_parser()
    opt = parser.parse_args()

    if opt.debug:
        opt.validate = False
        opt.batch_size = 16
        opt.log_step = 10
        opt.embedding_warmup_epochs = -1
        opt.num_epochs = 100
        opt.lr_update = 50
        opt.learning_rate = 1e-4
        opt.lr_MLGCN = 1e-4


    opt.text_channel = 768 # dimension of initial text concept embedding
    opt.visual_channel = 2048  # dimension of initial visual concept embedding
    opt.adj_file = os.path.join(opt.data_path, opt.data_name, 'Concept_annotations',
                                'adj_matrix_{}.pkl'.format(str(opt.num_attribute)))# file for matirx of co-occurrence numbers

    if not os.path.exists(opt.model_name):
        os.makedirs(opt.model_name)
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    logger = logging.getLogger(__name__)
    logger.info(opt)

    # Load Tokenizer and Vocabulary
    tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
    vocab = tokenizer.vocab
    opt.vocab_size = len(vocab)

    # generate visual2vec
    outfile = os.path.join(opt.data_path, opt.data_name, 'Concept_annotations',
                           opt.data_name + '_concepts_visual2vec_{}.pkl'.format(str(opt.num_attribute)))
    if os.path.exists(outfile):
        print(outfile + ' exits.')
    else:
        visual2vec(opt, tokenizer)
        print('Success to generate visual2vec.pkl')

    # generate word2vec
    outfile = os.path.join(opt.data_path, opt.data_name, 'Concept_annotations',
                           opt.data_name + '_concepts_word2vec_{}.pkl'.format(str(opt.num_attribute)))
    if os.path.exists(outfile):
        print(outfile + ' exits.')
    else:
        word2vec(opt)
        print('Success to generate word2vec.pkl')


    train_loader = image_caption.get_loader(opt.data_path, opt.data_name, 'train', tokenizer, opt,
                              opt.batch_size, shuffle=True, num_workers=opt.workers, train=True)
    val_loader = image_caption.get_loader(opt.data_path, opt.data_name, 'dev', tokenizer, opt,
                              opt.batch_size, shuffle=False, num_workers=opt.workers, train=False)


    model = MKVSEModel(opt)

    lr_schedules = [opt.lr_update, ]

    # optionally resume from a checkpoint
    start_epoch = 0
    if opt.resume:
        if os.path.isfile(opt.resume):
            logger.info("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another training
            model.Eiters = checkpoint['Eiters']
            logger.info("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                        .format(opt.resume, start_epoch, best_rsum))
            # validate(opt, val_loader, model)
            if opt.reset_start_epoch:
                start_epoch = 0
        else:
            logger.info("=> no checkpoint found at '{}'".format(opt.resume))


    # Train the Model
    best_rsum = 0
    logger.info(opt.logger_name)
    logger.info(opt.model_name)
    for epoch in range(start_epoch, opt.num_epochs):
        adjust_learning_rate(opt, model.model.module.optimizer, epoch, lr_schedules)

        if epoch >= opt.vse_mean_warmup_epochs:
            opt.max_violation = True
            model.set_max_violation(opt.max_violation)

        # Set up the all warm-up options
        if opt.precomp_enc_type == 'backbone':
            if epoch < opt.embedding_warmup_epochs:
                model.freeze_backbone()
                logger.info('All backbone weights are frozen, only train the embedding layers')
            else:
                model.unfreeze_backbone(3)

            if epoch < opt.embedding_warmup_epochs:
                logger.info('Warm up the embedding layers')
            elif epoch < opt.embedding_warmup_epochs + opt.backbone_warmup_epochs:
                model.unfreeze_backbone(3)  # only train the last block of resnet backbone
            elif epoch < opt.embedding_warmup_epochs + opt.backbone_warmup_epochs * 2:
                model.unfreeze_backbone(2)
            elif epoch < opt.embedding_warmup_epochs + opt.backbone_warmup_epochs * 3:
                model.unfreeze_backbone(1)
            else:
                model.unfreeze_backbone(0)

        # train for one epoch
        train(opt, train_loader, model, epoch)

        # do not evaluate in debug mode
        if opt.debug:
            if opt.validate == False:
                continue

        # evaluate on validation set
        rsum = validate(opt, val_loader, model)

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        if not os.path.exists(opt.model_name):
            os.mkdir(opt.model_name)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'Eiters': model.Eiters,
        }, is_best, filename='checkpoint.pth'.format(epoch), prefix=opt.model_name + '/')




def train(opt, train_loader, model, epoch):
    # average meters to record the training statistics
    logger = logging.getLogger(__name__)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    # model.model.module.img_enc
    # logger.info('image encoder trainable parameters: {}'.format(count_params(model.img_enc)))
    # logger.info('txt encoderopout_rater trainable parameters: {}'.format(count_params(model.txt_enc)))
    # logger.info('V_self_atten_enhance trainable parameters: {}'.format(count_params(model.V_self_atten_enhance)))
    # logger.info('T_self_atten_enhance trainable parameters: {}'.format(count_params(model.T_self_atten_enhance)))
    # logger.info('V_consensus_level_embedding trainable parameters: {}'.format(count_params(model.V_consensus_level_embedding)))
    # logger.info('T_consensus_level_embedding trainable parameters: {}'.format(count_params(model.T_consensus_level_embedding)))
    # logger.info('C_GCN trainable parameters: {}'.format(count_params(model.C_GCN)))
    # logger.info('Multi_feature_fusing trainable parameters: {}'.format(count_params(model.Multi_feature_fusing)))


    num_loader_iter = len(train_loader.dataset) // train_loader.batch_size + 1

    end = time.time()
    for i, train_data in enumerate(train_loader):
        if opt.debug:
            if 'train_data_debug' not in opt:
                opt.train_data_debug = train_data
            else:
                train_data = opt.train_data_debug
        # switch to train mode
        model.train_start()

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        if opt.precomp_enc_type == 'basic':
            # all_images, img_lengths, targets, lengths, concept_labels, ids = train_data
            model.train_emb(*train_data)
        else:
            images, captions, lengths, _ = train_data
            if epoch == opt.embedding_warmup_epochs:
                warmup_alpha = float(i) / num_loader_iter
                model.train_emb(images, captions, lengths, warmup_alpha=warmup_alpha)
            else:
                model.train_emb(images, captions, lengths)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # logger.info log info
        if model.Eiters % opt.log_step == 0:
            if opt.precomp_enc_type == 'backbone' and epoch == opt.embedding_warmup_epochs:
                logging.info('Current epoch-{}, the first epoch for training backbone, warmup alpha {}'.format(epoch,
                                                                                                               warmup_alpha))
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(
                    epoch, i, len(train_loader.dataset) // train_loader.batch_size + 1, batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)

        if opt.debug:
            break



def validate(opt, val_loader, model):
    logger = logging.getLogger(__name__)
    model.val_start()
    with torch.no_grad():
        # compute the encoding for all the validation images and captions
        img_embs, cap_embs = encode_data(
            model, val_loader, opt.log_step, logging.info, backbone=opt.precomp_enc_type == 'backbone')

    img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])

    start = time.time()
    sims = compute_sim(img_embs, cap_embs)
    end = time.time()
    logger.info("calculate similarity time: {}".format(end - start))

    # caption retrieval
    npts = img_embs.shape[0]
    # (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, cap_lens, sims)
    (r1, r5, r10, medr, meanr) = i2t(npts, sims)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, meanr))
    # image retrieval
    # (r1i, r5i, r10i, medri, meanr) = t2i(img_embs, cap_embs, cap_lens, sims)
    (r1i, r5i, r10i, medri, meanr) = t2i(npts, sims)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanr))
    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r10 + r1i + r5i + r10i
    logger.info('Current rsum is {}'.format(currscore))

    # record metrics in tensorboard
    tb_logger.log_value('r1', r1, step=model.Eiters)
    tb_logger.log_value('r5', r5, step=model.Eiters)
    tb_logger.log_value('r10', r10, step=model.Eiters)
    tb_logger.log_value('medr', medr, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('r1i', r1i, step=model.Eiters)
    tb_logger.log_value('r5i', r5i, step=model.Eiters)
    tb_logger.log_value('r10i', r10i, step=model.Eiters)
    tb_logger.log_value('medri', medri, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('rsum', currscore, step=model.Eiters)

    return currscore


def save_checkpoint(state, is_best, filename='checkpoint.pth', prefix=''):
    logger = logging.getLogger(__name__)
    tries = 15

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            torch.save(state, prefix + filename)
            if is_best:
                torch.save(state, prefix + 'model_best.pth')
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        logger.info('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error


def adjust_learning_rate(opt, optimizer, epoch, lr_schedules):
    logger = logging.getLogger(__name__)
    """Sets the learning rate to the initial LR
       decayed by 10 every opt.lr_update epochs"""
    if epoch in lr_schedules:
        logger.info('Current epoch num is {}, decrease all lr by 10'.format(epoch, ))
        for param_group in optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = old_lr * 0.1
            param_group['lr'] = new_lr
            logger.info('new lr {}'.format(new_lr))


def count_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


if __name__ == '__main__':
    main()
