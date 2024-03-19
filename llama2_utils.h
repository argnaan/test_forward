#include "pmsis.h"

#ifndef _ProbIndex_
#define _ProbIndex_
typedef struct {
    float prob;
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

struct llama2_mhsa_args{
    float* q;
    float* att;
    float* key_cache;
    float* value_cache;
    float* xb;
    int pos;
    int loff;
    int kv_dim;
    int kv_mul;
    int head_size;
    int n_heads;
    int steps;
};


void quickSort(ProbIndex* a, int l, int h);
void quickSort_vocab(TokenIndex* a, int l, int h);

void *bsearch (const void *key, const void *base0, size_t nmemb, size_t size, int (*compar)(const void *, const void *));

void llama2_mhsa_fp32_cl(void *llama2_mhsa_args);