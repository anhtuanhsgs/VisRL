cat $0
python main.py --env DEBUG \
--gpu-id 0 1 2 3 --workers 4 --valid-gpu 0 \
--data Random \
--num-steps 3 --max-episode-length 3 \
--size 32 32 \
--feats 32 32 64 64 1024 \
--save-period 50 --log-period 50 \
--log-dir logs/Dec2020/ --save-model-dir logs/trained_models/Dec2020/ \
