#!/bin/bash -l

# ADD CLUSTER COMPUTE SPECIFIC CONFIGURATIONS (e.g. memory request, gpu request)
# FOR TESTING PURPOSES THESE ARE IGNORED

# ENV VARIABLES
# -- email configs
export EMAIL_PASSWORD="password"
export EMAIL_RECIPIENTS="recipient@mail.com"

# -- model specific configs
export MODEL_NAME="google/bigbird-roberta-base"
export MAX_LENGTH=32

# -- training configs
export EPOCHS=1
export BATCH_SIZE=2
export VERBOSE=2
export SAVE_FREQ=5
export TEST_SIZE="0.3"

# -- dataset configs
export DATASET="Persuade"

# -- experiment configs
export STRATEGY_LEVEL="standard"
export STRATEGY_NAME="bieo"
export STRATEGY="${STRATEGY_LEVEL}_${STRATEGY_NAME}"
export RUN_INFERENCE=1

# -- inferred variables
export JSON_FILE_NAME="${DATASET}_postprocessed.json"
export DATA_PATH="data/${STRATEGY_NAME}/${JSON_FILE_NAME}"

# TODO this is a hotfix due to darmstadt processor tts. Needs cleaning
export TTS_FILE="train-test-split.csv"
export TTS_PATH="data/${TTS_FILE}"
cp -r $TTS_PATH $TMPDIR/$TTS_FILE

# COPY NECESSARY FILES
cp -r job_files/run.py $TMPDIR/run.py
cp -r $DATA_PATH $TMPDIR/$JSON_FILE_NAME
cp -r venv $TMPDIR/venv

cd $TMPDIR


# LOAD MODULES
module unload compilers mpi
module load compilers/gnu/4.9.2
module load python/3.7.4
module load cuda/10.1.243/gnu-4.9.2
module load cudnn/7.5.0.56/cuda-10.1

# venv should have the most recent version of argminer installed
source venv/bin/activate



python3 -c "import torch; print(f'GPU Availability: {torch.cuda.is_available()}')"
python3 run.py $DATASET $STRATEGY $MODEL_NAME $MAX_LENGTH -test-size=$TEST_SIZE -b=$BATCH_SIZE -e=$EPOCHS -save-freq=$SAVE_FREQ -verbose=$VERBOSE -i=$RUN_INFERENCE
python3 -c "from argminer.utils import send_job_completion_report; send_job_completion_report('${JOB_ID}')"


# nvidia-smi

tar -zcvf $HOME/Scratch/files_from_job_$JOB_ID.tar.gz $TMPDIR
