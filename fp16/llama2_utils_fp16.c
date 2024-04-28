#include <string.h>
#include <math.h>
#include "llama2_utils_fp16.h"
#include "pulp_train.h" 

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
                
                /*
                fp16 score;
                for (int i = 0; i < head_size; i++) {
                    score += q[i]*k[i];
                }
                */
                v2f16 temp = (v2f16) {0, 0};
                v2f16 A,B;
                for (int i = 0; i < head_size; i+=2) {
                    A = *((v2f16*) &q[i]);
                    B = *((v2f16*) &k[i]);
                    temp += A * B;
                }
                fp16 score = temp[0] + temp[1];
                
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


void rope_parallelized_fp16_cl(void* void_args){
    struct rope_args_fp16* args = (struct rope_args_fp16* ) void_args;
    int head_size = args->head_size;
    int dim = args->dim;
    int kv_dim = args->kv_dim;
    int pos = args->pos;

    int id = pi_core_id();

    int head_dim = (id*2) % head_size;
    #ifdef FASTEXPF
    fp16 freq = 1.0f / fastexp_gist_fp16(9.21034037198 * head_dim / (float)head_size);
    #else
    fp16 freq = 1.0f / powf(10000.0f, head_dim/ (float)head_size);
    #endif

    fp16 val = pos*freq;
    fp16 fcr, fci;

    if(val <= 200){
        fcr = (fp16) cosf((float) val);
        fci = (fp16) sinf((float) val);
    } else
       cordic_cos_sin_fp16(val, &fcr, &fci);

    for(int i=id*2; i < dim ; i+=2*NUM_CORES){
        int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                fp16* vec = v == 0 ? args->q : args->k; // the vector to rotate (query or key)
                fp16 v0 = vec[i];
                fp16 v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
    }
}