cat $0
python main.py --env DEBUG \
--gpu-id 0 1 2 3 --workers 3 --valid-gpu 0 \
--data Random \
--lr 1e-5 \
--num-steps 3 --max-episode-length 3 \
--size 32 32 \
--feats 64 64 128 128 1024 \
--save-period 100 --log-period 100 --train-log-period 100 \
--log-dir logs/Dec2020/ --save-model-dir logs/trained_models/Dec2020/ \
