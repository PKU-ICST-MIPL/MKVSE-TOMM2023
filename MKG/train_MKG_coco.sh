export PYTHONPATH=${PROJECT_PATH}/MKG
DATASET_NAME='coco'
DATA_PATH='../data/vse_infty/'
PARAM_NAME='MKG'

MODEL_NAME=MKVSE_${DATASET_NAME}_butd_region_bert_${PARAM_NAME}
CUDA_VISIBLE_DEVICES=2,3 python train.py \
  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} \
  --logger_name ../runs/${MODEL_NAME}/log --model_name ../runs/${MODEL_NAME} \
  --num_epochs 25 --lr_update 15 --learning_rate .0005 --precomp_enc_type basic --workers 2 \
  --log_step 1000 --embed_size 1024 --vse_mean_warmup_epochs 1 --batch_size 128 \
  --lr_MLGCN 0.0002  --cat_weight 0.95 --n_layers_concept_encoder 1 \
  --dropout_rate_concept_encoder 0.1 --n_head_concept_encoder 1

CUDA_VISIBLE_DEVICES=2,3 python eval.py \
  --dataset ${DATASET_NAME} --data_path ${DATA_PATH} --save_results \
  --weights_base_path ../runs/${MODEL_NAME} --backbone_path ${DATA_PATH}weights/original_updown_backbone.pth

