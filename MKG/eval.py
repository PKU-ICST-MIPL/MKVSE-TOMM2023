import os
import argparse
import logging
from lib import evaluation
import torch

# try to fix the bug: deadlock in dataloader
torch.multiprocessing.set_sharing_strategy('file_system')


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main():
    DATA_PATH = '../data/vse_infty/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='coco',
                        help='coco or f30k')
    parser.add_argument('--data_path', default='../data/vse_infty/')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--weights_base_path', type=str, default='', help='The path that the best model locates')
    parser.add_argument('--backbone_path', type=str, default=DATA_PATH+'weights/original_updown_backbone.pth', help='path to the pre-trained backbone net')

    opt = parser.parse_args()
    # if opt.dataset == 'coco':
    #     weights_bases = [
    #         'runs/coco_butd_region_bert',
    #         'runs/coco_butd_grid_bert',
    #         'runs/coco_wsl_grid_bert',
    #     ]
    # elif opt.dataset == 'f30k':
    #     weights_bases = [
    #         'runs/f30k_butd_region_bert',
    #         'runs/f30k_butd_grid_bert',
    #         'runs/f30k_wsl_grid_bert',
    #     ]
    # else:
    #     raise ValueError('Invalid dataset argument {}'.format(opt.dataset))
    # weights from trained model
    base=opt.weights_base_path
    logger.info('Evaluating {}...'.format(base))
    model_path = os.path.join(base, 'model_best.pth')
    if opt.save_results:  # Save the final results for computing ensemble results
        save_path = os.path.join(base, 'results_{}.npy'.format(opt.dataset))
    else:
        save_path = None

    if opt.dataset == 'coco':
        # Evaluate COCO 5-fold 1K
        evaluation.evalrank(model_path, data_path=opt.data_path, split='testall', fold5=True)
        # Evaluate COCO 5K
        evaluation.evalrank(model_path, data_path=opt.data_path, split='testall', fold5=False, save_path=save_path)
    elif opt.dataset == 'f30k':
        # Evaluate Flickr30K
        evaluation.evalrank(model_path, data_path=opt.data_path, split='test', fold5=False, save_path=save_path)
    else:
        raise ValueError('{} is not coco or f30k'.format(opt.dataset))

if __name__ == '__main__':
    main()
