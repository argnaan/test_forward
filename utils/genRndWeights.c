#include <stdlib.h>
#include <stdio.h>
#include <time.h>

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

int main(int argc, char* argv[]){
    srand(time(NULL));

    if(argc!=8){
        printf("Errore nella generazione dei pesi\nargc=%d\n", argc);
        exit(1);
    }

    Config c;
    sscanf(argv[1], "%d", &c.dim);
    sscanf(argv[2], "%d", &c.hidden_dim);
    sscanf(argv[3], "%d", &c.n_layers);
    sscanf(argv[4], "%d", &c.n_heads);
    sscanf(argv[5], "%d", &c.n_kv_heads);
    sscanf(argv[6], "%d", &c.vocab_size);
    sscanf(argv[7], "%d", &c.seq_len);

    int head_size = c.dim / c.n_heads;
    // numero di pesi da generare: calcolato in base alla descrizione della struttura dati TransformerWeights
    int w_dim = c.dim * (c.vocab_size + c.n_layers*(2 + 2*c.dim + head_size*c.n_kv_heads*2 + 3*c.hidden_dim) + 1);
    
    int file_dim = (w_dim*sizeof(float) + sizeof(Config))/1024;
    printf("Dimensione totale dati: %d kB\n", file_dim);
    printf("Nota: memoria L2 di 512 KB\n");
    
    FILE *fo = fopen("rnd_weights.bin", "wb");
    if(fo == NULL){
        printf("Errore nella creazione del file rnd_weights.bin\n");
        exit(1);
    }
    // Scrittura della configurazione c
    fwrite(&c, sizeof(Config), 1, fo);
    float* w = malloc(w_dim*sizeof(float));

    // generazione di float casuali tra -2 e 2
    for(int i=0;i<w_dim;i++)
        w[i] = ((float)rand() / (RAND_MAX/4)) - 2;
    fwrite(w, sizeof(float), w_dim, fo);
    fclose(fo);
    printf("Scrittura di rnd_weights.bin completata\n");

    FILE* fh = fopen("conf_and_weights.h", "w");
    if(fh==NULL){
        printf("Errore: impossible creare il file conf_and_weights.h\n");
        exit(1);
    }

    fprintf(fh, "#define DIM %d\n", c.dim);
    fprintf(fh, "#define HIDDEN_DIM %d\n", c.hidden_dim);
    fprintf(fh, "#define N_LAYERS %d\n", c.n_layers);
    fprintf(fh, "#define N_HEADS %d\n", c.n_heads);
    fprintf(fh, "#define N_KV_HEADS %d\n", c.n_kv_heads);
    fprintf(fh, "#define VOCAB_SIZE %d\n", c.vocab_size);
    fprintf(fh, "#define SEQ_LEN %d\n", c.seq_len);

    int i;
    fprintf(fh, "PI_L2 float weights_list[%d] = { ", w_dim);
    for(i=0;i<w_dim-1;i++){
        fprintf(fo, "%.10f, ", w[i]);
        if(i%10 == 9)
            fprintf(fh, "\n");
    }
    fprintf(fh, "%.10f};", w[i]);
    fclose(fh);
    printf("Scrittura di conf_and_weights.h completata\n");
    free(w);
    return 0;
}
