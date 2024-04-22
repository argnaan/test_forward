#include "pmsis.h"

typedef float16alt fp16;                                    // Standard IEEE FP16 format
typedef fp16 v2f16 __attribute__((vector_size (4)));        // Vectorized fp16 for SIMD

#ifndef _ProbIndex_
#define _ProbIndex_
typedef struct {
    fp16 prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling
#endif

#ifndef _TokenIndex_
#define _TokenIndex_
typedef struct {
    char *str;
    int id;
} TokenIndex;
#endif

struct llama2_mhsa_args_fp16{
    fp16* q;
    fp16* att;
    fp16* key_cache;
    fp16* value_cache;
    fp16* xb;
    int pos;
    int kv_dim;
    int kv_mul;
    int head_size;
    int n_heads;
    int steps;
};


void quickSort(ProbIndex* a, int l, int h);

void *bsearch (const void *key, const void *base0, size_t nmemb, size_t size, int (*compar)(const void *, const void *));

void softmax_original_fp16(fp16* x, int size);

void llama2_mhsa_fp16_cl(void *llama2_mhsa_args);