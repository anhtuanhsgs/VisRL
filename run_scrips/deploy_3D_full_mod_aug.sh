cat $0
python main.py --env deploy_3D_1rgba_FullModAug \
--gpu-id 0 1 2 3 --workers 12 --valid-gpu 0 \
--data 3DVols \
--lr 1e-4 \
--num-actions 4 --color-step 32 \
--num-steps 8 --max-episode-length 8 \
--size 32 32 \
--feats 64 64 128 128 512 \
--save-sample \
--save-period 100 --log-period 1 --train-log-period 100 \
--log-dir logs/Dec2020/ --save-model-dir logs/trained_models/Dec2020/ \
--load logs/trained_models/Dec2020/3D_1rgba_FullModAug/900.dat \
\
