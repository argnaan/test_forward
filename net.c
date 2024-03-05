#include "pmsis.h"
#include <math.h>
#include <string.h>
#include <ctype.h>
#include "stdlib.h"
#include "stdio.h"
#include "quicksort.h"
#include "conf_and_weights.h"
#include "stats.h"
#include "pulp_train.h"

#ifdef DEBUG_PRINT
#include "token_and_logits.h"
#endif

long unsigned cycle_softmax=0, cycle_matmul=0, tmp;

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
    s->x = X;
    s->xb = XB;
    s->xb2 = XB2;
    s->hb = HB;
    s->hb2 = HB2;
    s->q = Q;
    s->key_cache = KEY_CACHE;
    s->value_cache = VALUE_CACHE;
    s->att = ATT;
    s->logits = LOGITS;
}

void build_transformer(Transformer *t) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(&t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    long unsigned tmp = pi_perf_read (PI_PERF_CYCLES);
/*
    struct matMul_args mm_args;
    mm_args.A = w;
    mm_args.B = x;
    mm_args.C = xout; 
    mm_args.N = d;
    mm_args.K = n;
    mm_args.M = 1;
    mm_args.trans_B = 1;

    pi_cl_team_fork(NUM_CORES, mm, &mm_args);
    
*/ 
    // Original matmul: 

    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
    cycle_matmul += pi_perf_read (PI_PERF_CYCLES) - tmp;
}

void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax_here(float* x, int size) {
    long unsigned tmp = pi_perf_read (PI_PERF_CYCLES);
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
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }

    cycle_softmax += pi_perf_read (PI_PERF_CYCLES) - tmp;
}

float* forward(Transformer* transformer, int token, int pos) {

    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;
  
    // copy the token embedding into x
    float* content_row = w->token_embedding_table + token * dim;

    tmp = pi_perf_read (PI_PERF_CYCLES);
    memcpy(x, content_row, dim*sizeof(*x));
    if(pos == STEPS-1)
        printf("\n\nforward_memcpy: %lu\n", pi_perf_read (PI_PERF_CYCLES) - tmp);

    // forward all the layers
    for(unsigned long long l = 0; l < p->n_layers; l++) {
        
        // attention rmsnorm
        tmp = pi_perf_read (PI_PERF_CYCLES);
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);
        if(pos == STEPS-1)
            printf("\nforward_l%llu_rmsorm: %lu\n", l, pi_perf_read (PI_PERF_CYCLES) - tmp);

        // key and value point to the k cache
        int loff = l * STEPS * kv_dim; // kv cache layer offset for convenience
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        // qkv matmuls for this position
        tmp = pi_perf_read (PI_PERF_CYCLES);
        matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
        if(pos == STEPS-1)
            printf("forward_l%llu_matmul_q: %lu\n", l, pi_perf_read (PI_PERF_CYCLES) - tmp);
        
        tmp = pi_perf_read (PI_PERF_CYCLES);
        matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
        if(pos == STEPS-1)
            printf("forward_l%llu_matmul_k: %lu\n", l, pi_perf_read (PI_PERF_CYCLES) - tmp);

        tmp = pi_perf_read (PI_PERF_CYCLES);
        matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);
        if(pos == STEPS-1)
            printf("forward_l%llu_matmul_v: %lu\n", l, pi_perf_read (PI_PERF_CYCLES) - tmp);
        
        tmp = pi_perf_read (PI_PERF_CYCLES);
        // RoPE relative positional encoding: complex-valued rotate q and k in each head
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
        if(pos == STEPS-1)
            printf("forward_l%llu_RoPE: %lu\n", l, pi_perf_read (PI_PERF_CYCLES) - tmp);
        
        // multihead attention. iterate over all heads
        unsigned long cycle_qk_prod = 0, cycle_softmax = 0, cycle_att_per_v;
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++) {
            // get the query vector for this head
            float* q = s->q + h * head_size;
            // attention scores for this head
            float* att = s->att + h * STEPS;
            // iterate over all timesteps, including the current one

            tmp =  pi_perf_read (PI_PERF_CYCLES);

            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                // save the score to the attention buffer
                att[t] = score;
            }

            cycle_qk_prod += pi_perf_read (PI_PERF_CYCLES) - tmp;
            tmp = pi_perf_read (PI_PERF_CYCLES);
        
            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax_here(att, pos + 1);

            cycle_softmax += pi_perf_read (PI_PERF_CYCLES) - tmp;
            tmp = pi_perf_read (PI_PERF_CYCLES);

            // weighted sum of the values, store back into xb
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // get the attention weight for this timestep
                float a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }

            cycle_att_per_v += pi_perf_read (PI_PERF_CYCLES) - tmp;
        }
        if(pos == STEPS-1){
            printf("forward_l%llu_qk_prod: %lu\n", l, cycle_qk_prod);
            printf("forward_l%llu_softmax: %lu\n", l, cycle_softmax);
            printf("forward_l%llu_att_per_v: %lu\n", l, cycle_att_per_v);
        }

        // final matmul to get the output of the attention
        tmp = pi_perf_read (PI_PERF_CYCLES);
        matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

        if(pos == STEPS-1)
            printf("forward_l%llu_att_mm: %lu\n", l, pi_perf_read (PI_PERF_CYCLES) - tmp);

        // residual connection back into x
        tmp = pi_perf_read (PI_PERF_CYCLES);
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        if(pos == STEPS-1)
            printf("forward_l%llu_residual_conn: %lu\n", l, pi_perf_read (PI_PERF_CYCLES) - tmp);

        // ffn rmsnorm
        tmp = pi_perf_read (PI_PERF_CYCLES);
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);
        
        if(pos == STEPS-1)
            printf("forward_l%llu_ffn_rmsnorm: %lu\n", l, pi_perf_read (PI_PERF_CYCLES) - tmp);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        tmp = pi_perf_read (PI_PERF_CYCLES);
        matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        if(pos == STEPS-1)
            printf("forward_l%llu_ffn_mm1: %lu\n", l, pi_perf_read (PI_PERF_CYCLES) - tmp);

        tmp = pi_perf_read (PI_PERF_CYCLES);
        matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);
        if(pos == STEPS-1)
            printf("forward_l%llu_ffn_mm2: %lu\n", l, pi_perf_read (PI_PERF_CYCLES) - tmp);

        // SwiGLU non-linearity
        tmp = pi_perf_read (PI_PERF_CYCLES);
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }
        if(pos == STEPS-1)
            printf("forward_l%llu_ffn_SwiGLU: %lu\n", l, pi_perf_read (PI_PERF_CYCLES) - tmp);

        // final matmul to get the output of the ffn
        tmp = pi_perf_read (PI_PERF_CYCLES);
        matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

        if(pos == STEPS-1)
            printf("forward_l%llu_ffn_mm3: %lu\n", l, pi_perf_read (PI_PERF_CYCLES) - tmp);

        // residual connection
        tmp = pi_perf_read (PI_PERF_CYCLES);
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }

        if(pos == STEPS-1)
            printf("forward_l%llu_final_residual_conn: %lu\n", l, pi_perf_read (PI_PERF_CYCLES) - tmp);
    }
    
    // final rmsnorm
    tmp = pi_perf_read (PI_PERF_CYCLES);
    rmsnorm(x, x, w->rms_final_weight, dim);
    
    if(pos == STEPS-1)
        printf("\nforward_final_rmsnorm: %lu\n", pi_perf_read (PI_PERF_CYCLES) - tmp);

    // classifier into logits
    tmp = pi_perf_read (PI_PERF_CYCLES);
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
    
    if(pos == STEPS-1)
        printf("forward_final_matmul: %lu\n\n", pi_perf_read (PI_PERF_CYCLES) - tmp);

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
    char* str_buffer = STR_BUFFER;
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

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
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
        tmp = pi_perf_read (PI_PERF_CYCLES);
        
        next = sample_argmax(logits, sampler->vocab_size);

        if(isLastPos)
            printf("sample_argmax: %lu\n", pi_perf_read(PI_PERF_CYCLES)-tmp);

    } else {
        // apply the temperature to the logits
        for (int q=0; q<sampler->vocab_size; q++)
            logits[q] /= sampler->temperature; 

        tmp = pi_perf_read(PI_PERF_CYCLES);
        // apply softmax to the logits to get the probabilities for next token
        softmax_here(logits, sampler->vocab_size);

        if(isLastPos)
            printf("sample_softmax: %lu\n", pi_perf_read(PI_PERF_CYCLES)-tmp);

        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            tmp = pi_perf_read(PI_PERF_CYCLES);
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
    float diff_pos;
    float d, d_tot=0;
    int token, next, tok_diversi=0;
    int num_prompt_tokens=0;
    int* prompt_tokens = PROMPT_TOKENS;

    tmp = pi_perf_read (PI_PERF_CYCLES);

    encode(&tokenizer, PROMPT, 1, 0, prompt_tokens, &num_prompt_tokens);
    
    printf("encode: %lu\n", pi_perf_read (PI_PERF_CYCLES) - tmp);

    token = prompt_tokens[0];
    for(int pos = 0; pos < steps; pos++ ) {
        
        log = forward(&transformer, token, pos);

        if(pos < num_prompt_tokens -1)
            next = prompt_tokens[pos+1];
        else{
            next = sample(&sampler, log, pos==steps-1);

            #ifdef DEBUG_PRINT
            int next2 = sample(&sampler, &LOGITS_RUN[pos*VOCAB_SIZE]);
            printf("next: %3d next2: %3d \n", next, next2);
            printf("log[next]:         %.10f log[next2]          %.10f\n", log[next], log[next2]);
            printf("LOGITS_RUN [next]: %.10f LOGITS_RUN [next2]: %.10f\n\n", LOGITS_RUN[pos*512 + next], LOGITS_RUN[pos*512 + next2]);
            #endif
        }
        
        /*
        if(next==1)
            break;
        */
        tmp = pi_perf_read (PI_PERF_CYCLES);

        char* piece = decode(&tokenizer, token, next);
        token = next;

        if(pos==steps-1)
            printf("decode: %lu\n\n\n", pi_perf_read( PI_PERF_CYCLES)-tmp);

        #ifndef DEBUG_PRINT
        
        safe_printf(piece);
        
        #else
        
        diff_pos = 0;
        for(int j=0;j<VOCAB_SIZE;j++){
            d = log[j] - LOGITS_RUN[pos*VOCAB_SIZE+ j];
            if(d>0)
                diff_pos+=d;
            else
                diff_pos-=d;
        }
        diff_pos = diff_pos / VOCAB_SIZE;
        d_tot+=diff_pos;
        printf("Differenza media allo step %3d: %f\n", pos, diff_pos);       
        
        if(pos<steps-1){
            if(token != TOKEN[pos+1])
                tok_diversi++;
            printf("Predict token: %4d Token vero: %4d\n", token, TOKEN[pos+1]);
        }
        #endif
    }

    #ifdef DEBUG_PRINT
    printf("\nDifferenza media: %f\n", d_tot/steps);
    printf("Token diversi: %d/%d\n\n", tok_diversi, steps);
    #endif
/*
    cycle_tot = pi_perf_read (PI_PERF_CYCLES);
    instr_tot = pi_perf_read (PI_PERF_INSTR);
     STOP_STATS();

    printf("\n\n\nSTATS:\n\n");
    printf("Cycle tot: %lu\n", cycle_tot);
    printf("Cycle encode: %lu (%lu per token)\n", cycle_encode, cycle_encode/num_prompt_tokens);
    printf("Cycle forward: %lu (%lu per step)\n", cycle_forward, cycle_forward/steps);
    printf("Cycle sample: %lu (%lu per each sampled token)\n", cycle_sample, cycle_sample/(steps - num_prompt_tokens));
    printf("Cycle decode: %lu (%lu per step)\n", cycle_decode, cycle_decode/steps);
    printf("Instr tot: %lu\n", instr_tot);
    printf("Instr encode: %lu (%lu per token)\n", instr_encode, instr_encode/num_prompt_tokens);
    printf("Instr forward: %lu (%lu per step)\n", instr_forward, instr_forward/steps);
    printf("Instr sample: %lu (%lu per each sampled token)\n", instr_sample, instr_sample/(steps - num_prompt_tokens));
    printf("Instr decode: %lu (%lu per step)\n", instr_decode, instr_decode/steps);
*/
    #ifdef STATS
    STOP_STATS();
    printf("\nTotal cycle matmul = %lu\n", cycle_matmul);
    printf("Total cycle softmax = %lu\n", cycle_softmax);
    #endif

    printf("\n\n-------------------------------------------------------\n\n");
    //check_decode(&tokenizer, transformer.config.vocab_size);    
    return;
}


// funzione per verificare che la decodifica avviene correttamente
void check_decode(Tokenizer* tokenizer, int vocab_size){
    for(int tok=0;tok<vocab_size;tok++){
        char* piece = decode(tokenizer, 2, tok);
        safe_printf(piece);
        printf("\n");
    }
}