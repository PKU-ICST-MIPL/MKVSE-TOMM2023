export PYTHONPATH=${PROJECT_PATH}
DATA_PATH='../data/vse_infty/'


python eval_ensemble.py --data_path ${DATA_PATH}  \
  --result1_path 'runs/MKVSE_f30k_butd_region_bert_MKG/results_f30k.npy' \
  --result2_path 'runs/MKVSE_f30k_butd_region_bert_MGCN/results_f30k.npy'


python eval_ensemble.py --data_path ${DATA_PATH} --fold5 \
  --result1_path 'runs/MKVSE_coco_butd_region_bert_MKG/results_coco.npy' \
  --result2_path 'runs/MKVSE_coco_butd_region_bert_MGCN/results_coco.npy'


python eval_ensemble.py --data_path ${DATA_PATH} \
  --result1_path 'runs/MKVSE_coco_butd_region_bert_MKG/results_coco.npy' \
  --result2_path 'runs/MKVSE_coco_butd_region_bert_MGCN/results_coco.npy'


