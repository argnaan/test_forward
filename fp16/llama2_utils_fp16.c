#include <string.h>
#include <math.h>
#include "llama2_utils_fp16.h"
#include "pulp_train_utils_fp16.h"

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


void softmax_original_fp16(fp16* x, int size) {
    // find max value (for numerical stability)
    fp16 max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    fp16 sum = 0.0f;
    for (int i = 0; i < size; i++) {

        #ifdef FASTEXPF
        x[i] = (fp16) fastexp_gist_fp16((float) (x[i] - max_val));
        #else
        x[i] = (fp16) expf((float) (x[i] - max_val));
        #endif

        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}


void llama2_mhsa_fp16_cl(void *llama2_mhsa_args){
    struct llama2_mhsa_args_fp16* args = (struct llama2_mhsa_args_fp16*) llama2_mhsa_args;

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
            fp16* q = args->q + h * head_size;
            // attention scores for this head
            fp16* att = args->att + h * (STEPS+1);
            // iterate over all timesteps, including the current one

            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                fp16* k = args->key_cache + t * kv_dim + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                fp16 score = 0.0;
                
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                //printf("\t qk prod done, pos = %d\n", pos);
                score /= (fp16) sqrtf(head_size);
                // score *= q_rsqrt(head_size);             // non migliora le prestazioni

                // save the score to the attention buffer
                att[t] = score;
            }
            
            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax_original_fp16(att, pos + 1);

            // weighted sum of the values, store back into xb
            fp16* xb = args->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(*xb));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                fp16* v = args->value_cache + t * kv_dim + (h / kv_mul) * head_size;
                // get the attention weight for this timestep
                fp16 a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
    }
}