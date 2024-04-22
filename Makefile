APP = pulp_llama2

# set WEIGHTS_PATH=NULL for random weights generation
WEIGHTS_PATH ?= stories260K.bin

# if weights are random generated manually set the config parameter
DIM ?= 64
HIDDEN_DIM ?= 172
N_LAYERS ?= 1 	
N_HEADS ?= 8
N_KV_HEADS ?= 4
VOCAB_SIZE ?= 512
SEQ_LEN ?= 512

# other option 
STEPS ?= 256
TEMPERATURE ?= 1.0
RND_SEED ?= 100
PROMPT ?= "Tim was very happy"

DATA_TYPE ?= 'fp16'

get_golden:
	cd $(DATA_TYPE)/utils && gcc genWeights.c -o genWeights -O3
	cd $(DATA_TYPE)/utils && gcc run.c -o run -lm -O3 
	python3 $(DATA_TYPE)/utils/GM.py --weights_path $(WEIGHTS_PATH) --dim $(DIM) --hidden_dim $(HIDDEN_DIM) --n_layers $(N_LAYERS) --n_heads $(N_HEADS) --n_kv_heads $(N_KV_HEADS) --vocab_size $(VOCAB_SIZE) --seq_len $(SEQ_LEN) --steps $(STEPS) --temperature $(TEMPERATURE) --rnd_seed $(RND_SEED) --prompt $(PROMPT)

TRAIN_LIB= /home/andrea/PULP-TrainLib-Tutorial/pulp-trainlib/lib
TRAIN_LIB_SRCS=$(TRAIN_LIB)/sources

APP_SRCS = main.c

ifeq ($(DATA_TYPE), 'fp32')
	APP_SRCS += fp32/net.c fp32/llama2_utils.c
	APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_matmul_fp32.c
	APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_train_utils_fp32.c
	APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_rmsnorm_fp32.c
	APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_act_fp32.c
else
ifeq ($(DATA_TYPE), 'fp16')
	APP_SRCS += fp16/net.c fp16/llama2_utils_fp16.c
	APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_matmul_fp16.c
	APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_train_utils_fp16.c
	APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_rmsnorm_fp16.c
	APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_act_fp16.c
endif
endif

APP_LDFLAGS += -lm

APP_CFLAGS += -I. -I$(TRAIN_LIB)/include
APP_CFLAGS += -DCLUSTER -DFABRIC -O3 -g3
APP_CFLAGS += -DDATA_TYPE=$(DATA_TYPE)

# OTTIMIZZAZIONI: 

NUM_CORES ?= 8
APP_CFLAGS += -DNUM_CORES=$(NUM_CORES)
#APP_CFLAGS += -DFASTEXPF # usa la fast expf al posto della expf standard
#APP_CFLAGS += -DQ_RSQRT

APP_CFLAGS += -DOUTPUT
APP_CFLAGS += -DSTATS=1

include $(RULES_DIR)/pmsis_rules.mk

clean_all:
	rm -f $(DATA_TYPE)/utils/genWeights
	rm -f $(DATA_TYPE)/utils/rnd_weights.bin
	rm -f $(DATA_TYPE)/utils/run
	rm -f token_and_logits.h
	rm -f $(DATA_TYPE)/conf_and_weights.h
	rm -rf BUILD/
