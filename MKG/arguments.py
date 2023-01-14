import argparse

# default data path
DATASET_NAME='f30k' # {coco,f30k}_precomp
BACKBONE_NAME='_butd_region_bert'
PARAMS_NAME='_debug'
DATA_PATH='../data/vse_infty/'

def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default=DATA_PATH,
                        help='path to datasets')
    parser.add_argument('--data_name', default=DATASET_NAME,
                        help='{coco,f30k}_precomp')
    parser.add_argument('--vocab_path', default='./vocab/',
                        help='Path to saved vocabulary json files.')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=25, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--learning_rate', default=5e-4, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=15, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--backbone_lr_factor', default=0.01, type=float,
                        help='The lr factor for fine-tuning the backbone, it will be multiplied to the lr of '
                             'the embedding layers')
    parser.add_argument('--optim', default='adam', type=str,
                        help='the optimizer')
    parser.add_argument('--workers', default=0, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=200, type=int,
                        help='Number of steps to logger.info and record the log.')
    parser.add_argument('--val_step', default=500, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='../runs/TKVSE_'+DATASET_NAME+BACKBONE_NAME+PARAMS_NAME+"/log",
                        help='Path to save Tensorboard log.')
    parser.add_argument('--model_name', default='../runs/TKVSE_'+DATASET_NAME+BACKBONE_NAME+PARAMS_NAME,
                        help='Path to save the model.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--no_txtnorm', action='store_true',
                        help='Do not normalize the text embeddings.')
    parser.add_argument('--precomp_enc_type', default='basic',
                        help='basic|backbone')
    parser.add_argument('--backbone_path', type=str, default='',
                        help='path to the pre-trained backbone net')
    parser.add_argument('--backbone_source', type=str, default='detector',
                        help='the source of the backbone model, detector|imagenet')
    parser.add_argument('--vse_mean_warmup_epochs', type=int, default=0,
                        help='The number of warmup epochs using mean vse loss')
    parser.add_argument('--reset_start_epoch', action='store_true',
                        help='Whether restart the start epoch when load weights')
    parser.add_argument('--backbone_warmup_epochs', type=int, default=5,
                        help='The number of epochs for warmup')
    parser.add_argument('--embedding_warmup_epochs', type=int, default=2,
                        help='The number of epochs for warming up the embedding layers')
    parser.add_argument('--input_scale_factor', type=float, default=1.0,
                        help='The factor for scaling the input image')


    parser.add_argument('--activation_type', default='tanh',
                        help='choose type of activation functions.')
    parser.add_argument('--use_BatchNorm', action='store_false',
                        help='Whether to use BN in Consensus_level_feature_learning.')
    parser.add_argument('--dropout_rate', default=0.4, type=float,
                        help='dropout rate.')
    parser.add_argument('--feature_fuse_type', default='weight_sum',
                        help='choose the fusing type for raw feature and attribute feature '
                             '(multiple|concat|adap_sum|weight_sum))')
    parser.add_argument('--fuse_weight', default=0.9, type=float,
                        help='fusing weight for raw feature and attribute feature')
    parser.add_argument('--cat_weight', default=0.7, type=float,
                        help='emb_final = emb_origin * cat_weight + '
                             'emb_fuse * (1-cat_weight) + score_concept * score_weight')
    parser.add_argument('--score_weight', default=0.2, type=float,
                        help='emb_final = emb_origin * cat_weight + '
                             'emb_fuse * (1-cat_weight) + score_concept * score_weight')
    parser.add_argument('--lr_MLGCN', default=.0002, type=float,
                        help='learning rate of module of MLGCN.')
    parser.add_argument('--Concept_label_ratio', default=0.4, type=float,
                        help='The ratio of concept label.')

    parser.add_argument('--n_head_concept_encoder', type=int, default=8,
                        help='The number of heads for concept encoder layers')
    parser.add_argument('--n_layers_concept_encoder', type=int, default=1,
                        help='The number of layers for concept encoder')
    parser.add_argument('--dropout_rate_concept_encoder', type=float, default=0.1,
                        help='The dropout rate of concept encoder layers')


    parser.add_argument('--debug', action='store_true',
                        help='Debug mode will train only one batch size.')

    return parser


