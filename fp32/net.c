#include "pmsis.h"
#include <math.h>
#include <string.h>
#include <ctype.h>
#include "stdlib.h"
#include "stdio.h"
#include "llama2_utils.h"
#include "conf_and_weights.h"
#include "stats.h"
#include "pulp_train.h" // include tutti gli altri header di trainlib
#include "pulp_rmsnorm_fp32.h"

long unsigned tmp, cycle_matmul = 0;

PI_L1 float buffer_n_cores[NUM_CORES];

// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    float* wq; // (layer, dim, n_heads * head_size)
    float* wk; // (layer, dim, n_kv_heads * head_size)
    float* wv; // (layer, dim, n_kv_heads * head_size)
    float* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits (vocab_size, )
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    //float* initial_ptr;
    //initial_ptr = ptr;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    //printf("Numero di weights: %ld\n", ptr + p->dim - initial_ptr);
    ptr += p->dim;
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

void read_checkpoint(Config* config, TransformerWeights* weights, int* fd, float** data, ssize_t* file_size) {
    config->dim = DIM;
    config->hidden_dim = HIDDEN_DIM;
    config->n_heads = N_HEADS;
    config->n_kv_heads = N_KV_HEADS;
    config->n_layers = N_LAYERS;
    config->seq_len = SEQ_LEN;
    config->vocab_size = VOCAB_SIZE;

    int shared_weights;
    if(config->vocab_size > 0)
        shared_weights = 1;
    else{
        shared_weights = 0;
        config->vocab_size = - config->vocab_size;
    }
    float* weights_ptr = weights_list;
    memory_map_weights(weights, config, weights_ptr, shared_weights);
}

void malloc_run_state(RunState* s, Config* p) {
    /*
    s->x = X;
    s->xb = XB;
    s->xb2 = XB2;
    s->hb = HB;
    s->hb2 = HB2;
    s->q = Q;
    s->att = ATT;
    s->logits = LOGITS;
    */
    s->key_cache = KEY_CACHE;
    s->value_cache = VALUE_CACHE;
}

void build_transformer(Transformer *t) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(&t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

void matmul(float* xout, float* x, float* w, int n, int d) {

    long unsigned cycle_tmp = pi_perf_read (PI_PERF_CYCLES);;

    struct matMul_args mm_args;
    mm_args.A = w;
    mm_args.B = x;
    mm_args.C = xout; 
    mm_args.N = d;
    mm_args.K = n;
    mm_args.M = 1;
    mm_args.trans_B = 0;

    struct mm_manager_args man_args1;
    man_args1.mm_args = &mm_args;
    man_args1.layer_type = LAYER_LINEAR;
    man_args1.step_type = STEP_FW;
    man_args1.matmul_type = 7; //MATMUL_TYPE 11: 4x2, 7: 4x1, 8: 8x1
    pi_cl_team_fork(NUM_CORES, mm_manager, &man_args1);

    cycle_matmul += pi_perf_read (PI_PERF_CYCLES) - cycle_tmp;
}

void softmax_original(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {

        #ifdef FASTEXPF
        x[i] = fastexp_gist(x[i] - max_val);
        #else
        x[i] = expf(x[i] - max_val);
        #endif

        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

float* forward(Transformer* transformer, int token, int pos) {

    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    int dim = p->dim;
    // float* x = s->x;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    // printf("kv_dim : %d\n", kv_dim);
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;
  
    // copy the token embedding into x
    float* content_row = w->token_embedding_table + token * dim;

    // memcpy(x, content_row, dim*sizeof(*x));
    float* x = BUFF1;
    
    // trasferimento di memora dalla token embedding table al vettore x (BUFF1)
    pi_cl_dma_copy_t token_emb_table_to_x;
    token_emb_table_to_x.ext = (uint32_t) content_row;
    token_emb_table_to_x.loc = (uint32_t) x;
    token_emb_table_to_x.size = dim*sizeof(*x);
    token_emb_table_to_x.dir = PI_CL_DMA_DIR_EXT2LOC;
    token_emb_table_to_x.merge = 0;
    pi_cl_dma_memcpy(&token_emb_table_to_x);

    // trasferimento dei pesi della rmsnorm
    pi_cl_dma_copy_t rms_weight;
    rms_weight.ext = (uint32_t) w->rms_att_weight;
    rms_weight.loc = (uint32_t) BUFF4;
    rms_weight.size = dim* sizeof(*w->rms_att_weight);
    rms_weight.dir = PI_CL_DMA_DIR_EXT2LOC;
    rms_weight.merge = 0;
    pi_cl_dma_memcpy(&rms_weight);       
 

    // forward all the layers
    for(unsigned long long l = 0; l < p->n_layers; l++) {
        // key and value point to the k cache
        int loff = l * STEPS * kv_dim; // kv cache layer offset for convenience
        // s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;
        s->xb = BUFF2;
        s->q = BUFF3;

        // trasferimento dei pesi della matmul per v
        pi_cl_dma_copy_t kv_weight;
        kv_weight.ext = (uint32_t) (w->wv + l*dim*kv_dim);
        kv_weight.loc = (uint32_t) BUFF_W_2;
        kv_weight.size = dim*kv_dim*sizeof(*w->wv);
        kv_weight.dir = PI_CL_DMA_DIR_EXT2LOC;
        kv_weight.merge = 0;        
        pi_cl_dma_memcpy(&kv_weight);

        pi_cl_dma_wait(&token_emb_table_to_x);
        pi_cl_dma_wait(&rms_weight);

        tmp = pi_perf_read (PI_PERF_CYCLES);

        rmsnorm_parallelized_fp32(s->xb, x, BUFF4, buffer_n_cores, dim);
        // rmsnorm_parallelized_fp32(s->xb, x, w->rms_att_weight + l*dim, dim);
        #ifdef STATS
        if(pos==STEPS-1)
            printf("\nforward_l%llu_rmsorm: %lu\n", l, pi_perf_read (PI_PERF_CYCLES) - tmp);
        #endif
        // qkv matmuls for this position

        // trasferimento dei pesi della matmul di q
        pi_cl_dma_copy_t q_weight;
        q_weight.ext = (uint32_t) (w->wq + l*dim*dim);
        q_weight.loc = (uint32_t) BUFF_W_1;
        q_weight.size = dim*dim*sizeof(*w->wq);
        q_weight.dir = PI_CL_DMA_DIR_EXT2LOC;
        q_weight.merge = 0;
        pi_cl_dma_memcpy(&q_weight);
        
        pi_cl_dma_wait(&kv_weight);

        tmp = pi_perf_read (PI_PERF_CYCLES);

        matmul(BUFF4, s->xb, BUFF_W_2, dim, kv_dim);
        // matmul(s->v, s->xb, BUFF_W_2, dim, kv_dim);
        #ifdef STATS
        if(pos==STEPS-1)
            printf("forward_l%llu_matmul_v: %lu\n", l, pi_perf_read (PI_PERF_CYCLES) - tmp);
        #endif
        kv_weight.ext = (uint32_t) (w->wk + l*dim*kv_dim);
        pi_cl_dma_memcpy(&kv_weight);

        // trasferimento del vettore v nella value cache
        pi_cl_dma_copy_t kv_to_L2;
        kv_to_L2.ext = (uint32_t) s->v;
        kv_to_L2.loc = (uint32_t) BUFF4;
        kv_to_L2.size = kv_dim*sizeof(*s->v);
        kv_to_L2.dir = PI_CL_DMA_DIR_LOC2EXT;
        kv_to_L2.merge = 0;
        pi_cl_dma_memcpy(&kv_to_L2);
        

        pi_cl_dma_wait(&q_weight);

        tmp = pi_perf_read (PI_PERF_CYCLES);
        matmul(s->q, s->xb, BUFF_W_1, dim, dim);
        // matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
        #ifdef STATS
        if(pos==STEPS-1)
            printf("forward_l%llu_matmul_q: %lu\n", l, pi_perf_read (PI_PERF_CYCLES) - tmp);
        #endif
        // spostamento della key cache in BUFF_W_1 (tranne per la possima posizione)
        pi_cl_dma_copy_t k_cache_to_L1;
        k_cache_to_L1.ext = (uint32_t) (s->key_cache + loff);
        k_cache_to_L1.loc = (uint32_t) BUFF_W_1;
        k_cache_to_L1.size = kv_dim * pos * sizeof(s->key_cache);
        k_cache_to_L1.dir = PI_CL_DMA_DIR_EXT2LOC;
        pi_cl_dma_memcpy(&k_cache_to_L1);

        s->k = BUFF_W_1 + kv_dim*pos;
        pi_cl_dma_wait(&kv_weight);

        tmp = pi_perf_read (PI_PERF_CYCLES);

        matmul(s->k, s->xb, BUFF_W_2, dim, kv_dim);
        // matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
        #ifdef STATS
        if(pos==STEPS-1)
            printf("forward_l%llu_matmul_k: %lu\n", l, pi_perf_read (PI_PERF_CYCLES) - tmp);
        #endif
        // spostamento della value cache in BUFF_W_2
        pi_cl_dma_wait(&kv_to_L2);
        pi_cl_dma_copy_t v_cache_to_L1;
        v_cache_to_L1.ext = (uint32_t) (s->value_cache + loff);
        v_cache_to_L1.loc = (uint32_t) BUFF_W_2;
        v_cache_to_L1.size = kv_dim * (pos+1) * sizeof(s->value_cache);
        v_cache_to_L1.dir = PI_CL_DMA_DIR_EXT2LOC;
        pi_cl_dma_memcpy(&v_cache_to_L1);
        
        tmp = pi_perf_read (PI_PERF_CYCLES);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head    
        struct rope_args ra;
        ra.q = s->q;
        ra.k = s->k;
        ra.dim = dim;
        ra.head_size = head_size;
        ra.pos = pos;
        ra.kv_dim = kv_dim;

        pi_cl_team_fork(NUM_CORES, rope_parallelized_fp32_cl, &ra);
        /*
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }
        */
        #ifdef STATS
        if(pos==STEPS-1)
            printf("forward_l%llu_RoPE: %lu \n", l, pi_perf_read (PI_PERF_CYCLES) - tmp);
        #endif
        // trasferimento del vettore k nella key cache
        kv_to_L2.loc = (uint32_t) s->k;
        kv_to_L2.ext = (uint32_t) (s->key_cache + loff + pos * kv_dim);
        pi_cl_dma_memcpy(&kv_to_L2);

        // multihead attention. iterate over all heads
        int h;

        struct llama2_mhsa_args mhsa_args;

        mhsa_args.q = s->q;         // BUFF3
        mhsa_args.att = BUFF4;
        mhsa_args.key_cache = BUFF_W_1;
        mhsa_args.value_cache = BUFF_W_2;
        // mhsa_args.att = s->att;
        // mhsa_args.key_cache = s->key_cache + loff;
        // mhsa_args.value_cache = s->value_cache + loff ;
        mhsa_args.xb = s->xb;       // BUFF2
        mhsa_args.pos = pos;
        mhsa_args.kv_dim = kv_dim;
        mhsa_args.kv_mul = kv_mul;
        mhsa_args.head_size = head_size;
        mhsa_args.n_heads = p->n_heads;
        mhsa_args.steps = STEPS;

        pi_cl_dma_wait(&k_cache_to_L1);
        pi_cl_dma_wait(&v_cache_to_L1);

        tmp = pi_perf_read (PI_PERF_CYCLES);
        
        pi_cl_team_fork(NUM_CORES, llama2_mhsa_fp32_cl, &mhsa_args);

        #ifdef STATS
        if(pos==STEPS-1)
            printf("forward_l%llu_mhsa: %lu\n", l, pi_perf_read(PI_PERF_CYCLES) - tmp);
        #endif

        pi_cl_dma_wait(&kv_to_L2);
        
        pi_cl_dma_copy_t wo_to_L1;
        wo_to_L1.loc = (uint32_t) BUFF_W_1;
        wo_to_L1.ext = (uint32_t) (w->wo + l*dim*dim);
        wo_to_L1.size = dim * dim * sizeof(*w->wo);
        wo_to_L1.dir = PI_CL_DMA_DIR_EXT2LOC;
        pi_cl_dma_memcpy(&wo_to_L1);
        
        s->xb2 = BUFF3;

        pi_cl_dma_copy_t rms_ffn_weight_to_L1;
        rms_ffn_weight_to_L1.loc = (uint32_t) BUFF4;
        rms_ffn_weight_to_L1.ext = (uint32_t) (w->rms_ffn_weight + l*dim);
        rms_ffn_weight_to_L1.size = dim * sizeof(*w->rms_ffn_weight);
        rms_ffn_weight_to_L1.dir = PI_CL_DMA_DIR_EXT2LOC;
        pi_cl_dma_memcpy(&rms_ffn_weight_to_L1);

        pi_cl_dma_copy_t mm1_ffn_weight_to_L1;
        mm1_ffn_weight_to_L1.loc = (uint32_t) BUFF_W_2;
        mm1_ffn_weight_to_L1.ext = (uint32_t) (w->w1 + l*dim*hidden_dim);
        mm1_ffn_weight_to_L1.size = dim * hidden_dim * sizeof(*w->w1);
        mm1_ffn_weight_to_L1.dir = PI_CL_DMA_DIR_EXT2LOC;
        pi_cl_dma_memcpy(&mm1_ffn_weight_to_L1);

        // final matmul to get the output of the attention
        pi_cl_dma_wait(&wo_to_L1);

        tmp = pi_perf_read (PI_PERF_CYCLES);
        matmul(s->xb2, s->xb, BUFF_W_1, dim, dim);
        // matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

        #ifdef STATS
        if(pos==STEPS-1)
            printf("forward_l%llu_att_mm: %lu\n", l, pi_perf_read (PI_PERF_CYCLES) - tmp);
        #endif

        // residual connection back into x
        struct vect_sum_args vsa;
        vsa.op_1 = s->xb2;          // BUFF3
        vsa.op_2 = x;               // BUFF1
        vsa.dest = x;               // BUFF1
        vsa.size = dim;

        tmp = pi_perf_read (PI_PERF_CYCLES);
        pi_cl_team_fork(NUM_CORES, vect_sum, &vsa);

        #ifdef STATS
        if(pos==STEPS-1)
            printf("forward_l%llu_residual_conn: %lu\n", l, pi_perf_read (PI_PERF_CYCLES) - tmp);
        #endif 

        // ffn rmsnorm
        pi_cl_dma_wait(&rms_ffn_weight_to_L1);

        tmp = pi_perf_read (PI_PERF_CYCLES);
        rmsnorm_parallelized_fp32(s->xb, x, BUFF4, buffer_n_cores, dim);
        // rmsnorm_parallelized_fp32(s->xb, x, w->rms_ffn_weight + l*dim, dim);
        
        #ifdef STATS
        if(pos==STEPS-1)
            printf("forward_l%llu_ffn_rmsnorm: %lu\n", l, pi_perf_read (PI_PERF_CYCLES) - tmp);
        #endif

        s->hb = BUFF3;
        s->hb2 = BUFF4;

        pi_cl_dma_copy_t mm2_ffn_weight_to_L1;
        mm2_ffn_weight_to_L1.loc = (uint32_t) BUFF_W_1;
        mm2_ffn_weight_to_L1.ext = (uint32_t) (w->w3 + l*dim*hidden_dim);
        mm2_ffn_weight_to_L1.size = dim * hidden_dim * sizeof(*w->w3);
        mm2_ffn_weight_to_L1.dir = PI_CL_DMA_DIR_EXT2LOC;
        pi_cl_dma_memcpy(&mm2_ffn_weight_to_L1);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        pi_cl_dma_wait(&mm1_ffn_weight_to_L1);

        tmp = pi_perf_read (PI_PERF_CYCLES);
        matmul(s->hb, s->xb, BUFF_W_2, dim, hidden_dim);

        #ifdef STATS
        if(pos==STEPS-1)
            printf("forward_l%llu_ffn_mm1: %lu\n", l, pi_perf_read (PI_PERF_CYCLES) - tmp);
        #endif 

        mm1_ffn_weight_to_L1.ext = (uint32_t) (w->w2 + l*dim*hidden_dim);
        pi_cl_dma_memcpy(&mm1_ffn_weight_to_L1);

        pi_cl_dma_wait(&mm2_ffn_weight_to_L1);

        tmp = pi_perf_read (PI_PERF_CYCLES);
        matmul(s->hb2, s->xb, BUFF_W_1, dim, hidden_dim);

        #ifdef STATS
        if(pos==STEPS-1)
            printf("forward_l%llu_ffn_mm2: %lu\n", l, pi_perf_read (PI_PERF_CYCLES) - tmp);
        #endif
        
        // matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        // matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);
        
        // SwiGLU non-linearity
        struct swiglu_args sa;
        sa.in1 = s->hb;             // BUFF3
        sa.in2 = s->hb2;            // BUFF4
        sa.out = s->hb;             // BUFF3
        sa.dim = hidden_dim;

        tmp = pi_perf_read (PI_PERF_CYCLES);

        pi_cl_team_fork(NUM_CORES, pulp_swiglu_fp32_cl, &sa);

	#ifdef STATS
        if(pos==STEPS-1)
            printf("forward_l%llu_ffn_SwiGLU: %lu\n", l, pi_perf_read (PI_PERF_CYCLES) - tmp);
	#endif
	
        if(l < p->n_layers - 1){
            rms_weight.ext = (uint32_t) (w->rms_att_weight + (l+1)*dim);
            pi_cl_dma_memcpy(&rms_weight);
        }

        // final matmul to get the output of the ffn
        pi_cl_dma_wait(&mm1_ffn_weight_to_L1);

        tmp = pi_perf_read (PI_PERF_CYCLES);

        matmul(s->xb, s->hb, BUFF_W_2, hidden_dim, dim);
        // matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);
        #ifdef STATS
        if(pos==STEPS-1)
            printf("forward_l%llu_ffn_mm3: %lu\n", l, pi_perf_read (PI_PERF_CYCLES) - tmp);
        #endif
        
        // residual connection
        vsa.op_1 = s->xb;         // BUFF2
        vsa.op_2 = x;             // BUFF1
        vsa.dest = x;             // BUFF1
        vsa.size = dim;

        tmp = pi_perf_read (PI_PERF_CYCLES);

        pi_cl_team_fork(NUM_CORES, vect_sum, &vsa);

	#ifdef STATS
        if(pos==STEPS-1)
            printf("forward_l%llu_final_residual_conn: %lu\n", l, pi_perf_read (PI_PERF_CYCLES) - tmp);
        #endif
    }
    
    int mm_div = 4;   // deve essere un divisore di vocab_size
    int part = p->vocab_size / mm_div;
    s->logits = BUFF4;
    
    pi_cl_dma_copy_t mm_weights_to_BUFF_W_1, mm_weights_to_BUFF_W_2;
    mm_weights_to_BUFF_W_1.ext = (uint32_t) w->wcls;
    mm_weights_to_BUFF_W_1.loc = (uint32_t) BUFF_W_1;
	mm_weights_to_BUFF_W_1.size = dim * part * sizeof(*w->wcls);
	mm_weights_to_BUFF_W_1.dir = PI_CL_DMA_DIR_EXT2LOC;
    pi_cl_dma_memcpy(&mm_weights_to_BUFF_W_1);

    mm_weights_to_BUFF_W_2.loc = (uint32_t) BUFF_W_2;
    mm_weights_to_BUFF_W_2.size = dim * part * sizeof(*w->wcls);
    mm_weights_to_BUFF_W_2.dir = PI_CL_DMA_DIR_EXT2LOC;
    
    // final rmsnorm

    tmp = pi_perf_read (PI_PERF_CYCLES);

    rmsnorm_parallelized_fp32(s->xb, x, w->rms_final_weight, buffer_n_cores, dim);
	
    #ifdef STATS
    if(pos==STEPS-1)
        printf("\nforward_final_rmsnorm: %lu\n", pi_perf_read (PI_PERF_CYCLES) - tmp);
    #endif
    
    tmp = pi_perf_read (PI_PERF_CYCLES);

    for(int i=0; i<mm_div; i+=2){
        mm_weights_to_BUFF_W_2.ext = (uint32_t) (w->wcls + (i+1)*part*dim);
        pi_cl_dma_memcpy(&mm_weights_to_BUFF_W_2);

        pi_cl_dma_wait(&mm_weights_to_BUFF_W_1);
        matmul(s->logits+i*part, s->xb, BUFF_W_1, p->dim, part);

        if(i < mm_div - 2){
            mm_weights_to_BUFF_W_1.ext = (uint32_t) (w->wcls + (i+2)*part*dim);
            pi_cl_dma_memcpy(&mm_weights_to_BUFF_W_1);
        }
        
        pi_cl_dma_wait(&mm_weights_to_BUFF_W_2);
        matmul(s->logits+(i+1)*part, s->xb, BUFF_W_2, p->dim, part);
    }
	
    #ifdef STATS
    if(pos==STEPS-1)
        printf("forward_final_matmul: %lu\n\n", pi_perf_read (PI_PERF_CYCLES) - tmp);
    #endif 
    // classifier into logits
    // matmul(s->logits, s->xb, w->wcls, p->dim, p->vocab_size);

    return s->logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens
#ifndef _TokenIndex_
#define _TokenIndex_
typedef struct {
    char *str;
    int id;
} TokenIndex;
#endif

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

void build_tokenizer(Tokenizer* t, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = VOCAB;
    t->vocab_scores = VOCAB_SCORES;
    t->sorted_vocab = SORTED_VOCAB;
    
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }

    t->max_token_length = MAX_TOKEN_LENGTH;
    int len;
    int j=0;
    for (int i = 0; i < vocab_size; i++) {
        t->vocab[i] = &VOCAB_DATA[j];
        while(VOCAB_DATA[j] != '\0' && i < vocab_size-1)
            j++;
        j++;
    }
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    /*
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    */
   // codice in sostituzione della sscanf, specifico per questo formato:
    if(piece[0]=='<' && piece[1] == '0' && piece[2]=='x' && piece[5]=='>'){
        int cifra1, cifra2;
        if('0' <= piece[3] && piece[3]<= '9')
            cifra1 = piece[3] - '0';
        else
            cifra1 = piece[3] - 'A' + 10; 
        if('0' <= piece[4] && piece[4] <= '9')
            cifra2 = piece[4] - '0';
        else
            cifra2 = piece[4] - 'A' + 10;
        byte_val = cifra1*16 + cifra2;

        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    //printf("token: %d piece: %s\n", token, piece);
    return piece;
}

void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
}

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL)
        exit(1);
    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    char* str_buffer = (char*) BUFF1;
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;
        
        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(*n_tokens)++] = 2;
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling
#ifndef _ProbIndex_
#define _ProbIndex_
typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling
#endif

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    //printf("token: %3d prob: %f\n", max_i, probabilities[max_i]);
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}


int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    //qsort(probindex, n0, sizeof(ProbIndex), compare);
    quickSort(probindex, 0, n0-1);
    
    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = (ProbIndex*)PROB_INDEX;
}

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits, char isLastPos) {
    // sample the token given the logits and some hyperparameters
    int next;
    //printf("sampler->temperature: %f\n", sampler->temperature);
    if (sampler->temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        // apply the temperature to the logits
        for (int q=0; q<sampler->vocab_size; q++)
            logits[q] /= sampler->temperature; 
        
        tmp = pi_perf_read (PI_PERF_CYCLES);

        // apply softmax to the logits to get the probabilities for next token
        pulp_vector_softmax_fp32(logits, logits, buffer_n_cores, sampler->vocab_size);

        if(isLastPos)
            printf("sample_softmax: %lu\n", pi_perf_read(PI_PERF_CYCLES)-tmp);

        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            tmp = pi_perf_read(PI_PERF_CYCLES);
            
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);

            if(isLastPos)
                printf("sample_sample_topp: %lu\n", pi_perf_read(PI_PERF_CYCLES)-tmp);
        }
    }
    return next;
}

void net_step(){
    #ifdef STATS
    INIT_STATS();
    PRE_START_STATS();
    START_STATS();
    #endif

    int steps = STEPS;
    float temperature = TEMPERATURE;
    float topp = 0.9f;
    unsigned long long rng_seed = RND_SEED;

    Transformer transformer;
    build_transformer(&transformer);

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, transformer.config.vocab_size);

    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    float* log;
    int token, next;
    int num_prompt_tokens=0;
    int* prompt_tokens = PROMPT_TOKENS;

    tmp = pi_perf_read (PI_PERF_CYCLES);

    encode(&tokenizer, PROMPT, 1, 0, prompt_tokens, &num_prompt_tokens);
	
    #ifdef STATS
        printf("encode: %lu\n", pi_perf_read (PI_PERF_CYCLES) - tmp);
    #endif 
    
    token = prompt_tokens[0];


    for(int pos = 0; pos < steps; pos++ ) {
        
        log = forward(&transformer, token, pos);

        if(pos < num_prompt_tokens -1)
            next = prompt_tokens[pos+1];
        else{
            next = sample(&sampler, log, pos==STEPS-1);
        }
        
        /*
        if(next==1)
            break; 
        */
        tmp = pi_perf_read (PI_PERF_CYCLES);
        
        char* piece = decode(&tokenizer, token, next);

	#ifdef STATS
        if(pos==STEPS-1)
            printf("decode: %lu\n\n\n", pi_perf_read( PI_PERF_CYCLES)-tmp);
	#endif 
	
        token = next;

        safe_printf(piece);
    }

    printf("\n\n");

    #ifdef STATS
    STOP_STATS();
    #endif

    printf("\n\n-------------------------------------------------------\n\n");
    return;
}
