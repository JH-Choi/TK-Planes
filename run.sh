# pip install -e . 
rm -rf outputs
DATA_PATH=/mnt/hdd/data/Okutama_Action/Chris_data/1.1.1/training_set/train
ns-train kplanes-dynamic --data $DATA_PATH \
    --vis tensorboard