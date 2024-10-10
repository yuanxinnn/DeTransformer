

train_batch_size=${1:-8192} 
learning_rate=${2:-"6e-3"}
precision=${3:-"fp32"}  
num_gpus=${4:-2} 
warmup_proportion=${5:-"0.2843"}
train_steps=${6:-10}
save_checkpoint_steps=${7:-200}
resume_training=${8:-"false"}
create_logfile=${9:-"true"}
accumulate_gradients=${10:-"true"}
gradient_accumulation_steps=${11:-128}
seed=${12:-12439}
job_name=${13:-"bert_lamb_pretraining"}
allreduce_post_accumulation=${14:-"true"}
allreduce_post_accumulation_fp16=${15:-"false"}


DATA_DIR_PHASE=${21:-/your_data_dir/DeBert/data}
CODEDIR=${23:-/your_data_dir/DeBert}

init_checkpoint=${24:-"None"}
VOCAB_FILE=${CODEDIR}/vocab/vocab
RESULTS_DIR=${CODEDIR}/results
CHECKPOINTS_DIR=${RESULTS_DIR}/checkpoints_DeBert  #temp

num_dask_workers=${26:-$(nproc)}
num_shards_per_worker=${27:-128}
num_workers=${28:-4} 
num_nodes=1
sample_ratio=${29:-0.9}
phase2_bin_size=${30:-64}
masking=${31:-static}
BERT_CONFIG=${32:-/your_data_dir/DeBert/model_configs/base_shuffle.json} 


if [ ! -d "$RESULTS_DIR" ] ; then
   echo "Error! $RESULTS_DIR directory missing."
   exit -1
fi

if [ ! -d "$CHECKPOINTS_DIR" ] ; then
   echo "Warning! $CHECKPOINTS_DIR directory missing."
   echo "Checkpoints will be written to $RESULTS_DIR instead."
   CHECKPOINTS_DIR=$RESULTS_DIR
fi

if [ ! -f "$BERT_CONFIG" ] ; then
   echo "Error! BERT base configuration file not found at $BERT_CONFIG"
   exit -1
fi


#Start Phase1
PREC=""
if [ "$precision" = "fp16" ] ; then
   PREC="--fp16"
elif [ "$precision" = "fp32" ] ; then
   PREC=""
elif [ "$precision" = "tf32" ] ; then
   PREC=""
else
   echo "Unknown <precision> argument"
   exit -2
fi

ACCUMULATE_GRADIENTS=""
if [ "$accumulate_gradients" == "true" ] ; then
   ACCUMULATE_GRADIENTS="--gradient_accumulation_steps=$gradient_accumulation_steps"
fi

CHECKPOINT=""
if [ "$resume_training" == "true" ] ; then
   CHECKPOINT="--resume_from_checkpoint"
fi

ALL_REDUCE_POST_ACCUMULATION=""
if [ "$allreduce_post_accumulation" == "true" ] ; then
   ALL_REDUCE_POST_ACCUMULATION="--allreduce_post_accumulation"
fi

ALL_REDUCE_POST_ACCUMULATION_FP16=""
if [ "$allreduce_post_accumulation_fp16" == "true" ] ; then
   ALL_REDUCE_POST_ACCUMULATION_FP16="--allreduce_post_accumulation_fp16"
fi

INIT_CHECKPOINT=""
if [ "$init_checkpoint" != "None" ] ; then
   INIT_CHECKPOINT="--init_checkpoint=$init_checkpoint"
fi

echo $DATA_DIR_PHASE
INPUT_DIR=$DATA_DIR_PHASE
CMD=" $CODEDIR/run_pretraining.py"
CMD+=" --input_dir=$DATA_DIR_PHASE"
CMD+=" --output_dir=$CHECKPOINTS_DIR"
CMD+=" --config_file=$BERT_CONFIG"
CMD+=" --vocab_file=$VOCAB_FILE"
CMD+=" --train_batch_size=$train_batch_size"
CMD+=" --max_seq_length=128"
CMD+=" --max_predictions_per_seq=20"
CMD+=" --max_steps=$train_steps"
CMD+=" --warmup_proportion=$warmup_proportion"
CMD+=" --num_steps_per_checkpoint=$save_checkpoint_steps"
CMD+=" --learning_rate=$learning_rate"
CMD+=" --seed=$seed"
CMD+=" $PREC"
CMD+=" $ACCUMULATE_GRADIENTS"
CMD+=" $CHECKPOINT"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION_FP16"
CMD+=" $INIT_CHECKPOINT"
CMD+=" --do_train"
CMD+=" --json-summary ${RESULTS_DIR}/dllogger_DeBert.json "  #temp 
CMD+=" --disable_progress_bar"
CMD+=" --num_workers=${num_workers}"

# FutureWarning: The module torch.distributed.launch is deprecated and will be removed in future. Use torchrun
# CMD="python3 -m torch.distributed.launch --nproc_per_node=$num_gpus $CMD"
CMD="torchrun --standalone --nnodes 1 --nproc_per_node=$num_gpus $CMD"


if [ "$create_logfile" = "true" ] ; then
  export GBS=$(expr $train_batch_size \* $num_gpus)
  printf -v TAG "pyt_bert_pretraining_phase1_%s_gbs%d" "$precision" $GBS
  DATESTAMP=`date +'%y%m%d%H%M%S'`
  LOGFILE=$RESULTS_DIR/$job_name.$TAG.$DATESTAMP.log
  printf "Logs written to %s\n" "$LOGFILE"
fi

set -x
if [ -z "$LOGFILE" ] ; then
   $CMD
else
   (
     $CMD
   ) |& tee $LOGFILE
fi

set +x

echo "finished pretraining 1"