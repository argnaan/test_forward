#include "llama2_utils.h"
#include <string.h>
#include <math.h>

void softmax_original(float* x, int size);

int part(ProbIndex* a, int l, int h){
    float p = a[h].prob;
    int i = l-1;
    ProbIndex tmp;
    for(int j=l;j<h;j++){
        if(a[j].prob>=p){
            i++;
            tmp = a[j];
            a[j] = a[i];
            a[i] = tmp;
        }
    }
    tmp = a[i+1];
    a[i+1] = a[h];
    a[h] = tmp;
    return i+1;
}

void quickSort(ProbIndex* a, int l, int h){
    if(l < h){
        int p = part(a, l, h);
        quickSort(a, l, p-1);
        quickSort(a, p+1, h);
    }
}

void *bsearch (const void *key, const void *base0, size_t nmemb, size_t size, int (*compar)(const void *, const void *))
{
	const char *base = (const char *) base0;
	int lim, cmp;
	const void *p;

	for (lim = nmemb; lim != 0; lim >>= 1) {
		p = base + (lim >> 1) * size;
		cmp = (*compar)(key, p);
		if (cmp == 0)
			return (void *)p;
		if (cmp > 0) {	/* key > p: move right */
			base = (const char *)p + size;
			lim--;
		} /* else move left */
	}
	return (NULL);
}

void llama2_mhsa_fp32_cl(void *llama2_mhsa_args){
    struct llama2_mhsa_args* args = (struct llama2_mhsa_args*) llama2_mhsa_args;

    int pos = args->pos;
    int kv_dim = args->kv_dim;
    int kv_mul = args->kv_mul;
    int head_size = args->head_size;
    int n_heads = args->n_heads;
    int STEPS = args->steps;


    const uint32_t blockSize = (n_heads + NUM_CORES-1) / NUM_CORES;
    const uint32_t start = pi_core_id()*blockSize;
    const uint32_t stop = start+blockSize > n_heads ? n_heads : start+blockSize;


    for (int h = start; h < stop; h++) {
            // get the query vector for this head
            float* q = args->q + h * head_size;
            // attention scores for this head
            float* att = args->att + h * STEPS;
            // iterate over all timesteps, including the current one

            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float* k = args->key_cache + t * kv_dim + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                
                score /= sqrtf(head_size);
                // score *= q_rsqrt(head_size);             // non migliora le prestazioni

                // save the score to the attention buffer
                att[t] = score;
            }
        
            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax_original(att, pos + 1);

            // weighted sum of the values, store back into xb
            float* xb = args->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float* v = args->value_cache + t * kv_dim + (h / kv_mul) * head_size;
                // get the attention weight for this timestep
                float a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
    }
}
