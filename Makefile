DIM ?= 64
HIDDEN_DIM ?= 172
N_LAYERS ?= 1
N_HEADS ?= 8
N_KV_HEADS ?= 4
VOCAB_SIZE ?= 512
SEQ_LEN ?= 512 	

NUM_CORES?=1

get_golden:
	gcc utils/genRndWeights.c -o utils/genRndWeights 
	python3 utils/GM.py --dim $(DIM) --hidden_dim $(HIDDEN_DIM) --n_layers $(N_LAYERS) --n_heads $(N_HEADS) --n_kv_heads $(N_KV_HEADS) --vocab_size $(VOCAB_SIZE) --seq_len $(SEQ_LEN)

APP_SRCS = main.c net.c
APP_LDFLAGS += -lm 
APP_CFLAGS += -DNUM_CORES=${NUM_CORES}
include $(RULES_DIR)/pmsis_rules.mk