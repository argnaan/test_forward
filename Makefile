APP = test_forward

# set WEIGHTS_PATH=NULL for random weights generation
WEIGHTS_PATH ?= utils/stories260K.bin
# if weights are random generated manually set the config parameter
DIM ?= 64
HIDDEN_DIM ?= 172
N_LAYERS ?= 5
N_HEADS ?= 8
N_KV_HEADS ?= 4
VOCAB_SIZE ?= 512
SEQ_LEN ?= 512

# other option 
STEPS ?= 64
TEMPERATURE ?= 0
RND_SEED ?= 42
PROMPT ?= "Tim was so happy"


get_golden:
	cd utils && gcc genWeights.c -o genWeights -O3
	cd utils && gcc run.c -o run -lm -O3
	python3 utils/GM.py --weights_path $(WEIGHTS_PATH) --dim $(DIM) --hidden_dim $(HIDDEN_DIM) --n_layers $(N_LAYERS) --n_heads $(N_HEADS) --n_kv_heads $(N_KV_HEADS) --vocab_size $(VOCAB_SIZE) --seq_len $(SEQ_LEN) --steps $(STEPS) --temperature $(TEMPERATURE) --rnd_seed $(RND_SEED) --prompt $(PROMPT)


TRAIN_LIB=/home/andrea/PULP-TrainLib-Tutorial/pulp-trainlib/lib
TRAIN_LIB_SRCS=$(TRAIN_LIB)/sources
APP_SRCS = main.c net.c quicksort.c

APP_LDFLAGS += -lm 
APP_CFLAGS += -DNUM_CORES=${NUM_CORES}

DATA_TYPE?='fp32'
APP_CFLAGS += -I. -I$(TRAIN_LIB)/include
APP_CFLAGS += -O3 -g3 
APP_CFLAGS += -DFABRIC 
APP_CFLAGS += -DCLUSTER
APP_CFLAGS += -DNUM_CORES=$(NUM_CORES)
APP_CFLAGS += -DPROF_NET
APP_CFLAGS += -DMEMOCC_COMP
APP_CFLAGS += -mhwloopalign
APP_LDFLAGS += -lm 


include $(RULES_DIR)/pmsis_rules.mk

clean_all:
	rm -f utils/genWeights
	rm -f utils/rnd_weights.bin
	rm -f utils/run
	rm -f token_and_logits.h
	rm -f conf_and_weights.h
	rm -rf BUILD/
