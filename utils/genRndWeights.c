#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define WEIGHTS_BIN "utils/rnd_weights.bin"
#define WEIGHTS_HEDAER "conf_and_weights.h"
#define TOKENIZER_PATH "utils/tokenizer.bin"

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
    float prob;
    int index;
} ProbIndex;

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int main(int argc, char* argv[]){
    srand(time(NULL));

    if(argc!=11){
        printf("Errore nella generazione dei pesi\nargc è %d ma deve essere 11\n", argc);
        exit(1);
    }

    Config c;
    int steps, rnd_seed;
    float temperature;
    sscanf(argv[1], "%d", &c.dim);
    sscanf(argv[2], "%d", &c.hidden_dim);
    sscanf(argv[3], "%d", &c.n_layers);
    sscanf(argv[4], "%d", &c.n_heads);
    sscanf(argv[5], "%d", &c.n_kv_heads);
    sscanf(argv[6], "%d", &c.vocab_size);
    sscanf(argv[7], "%d", &c.seq_len);
    sscanf(argv[8], "%d", &steps);
    sscanf(argv[9], "%f", &temperature);
    sscanf(argv[10], "%d", &rnd_seed);

    int head_size = c.dim / c.n_heads;
    // numero di pesi da generare: calcolato in base alla descrizione della struttura dati TransformerWeights
    int w_dim = c.dim * (c.vocab_size + c.n_layers*(2 + 2*c.dim + head_size*c.n_kv_heads*2 + 3*c.hidden_dim) + 1);
    
    int file_dim = (w_dim*sizeof(float) + sizeof(Config))/1024;
    printf("Numero pesi generati: %d (%d kB)\n", w_dim, file_dim);
    
    FILE *fo = fopen(WEIGHTS_BIN, "wb");
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

    FILE* fh = fopen(WEIGHTS_HEDAER, "w");
    if(fh==NULL){
        printf("Errore: impossible creare il file conf_and_weights.h\n");
        exit(1);
    }

    fprintf(fh, "// Definizioni per Config\n");
    fprintf(fh, "#define DIM %d\n", c.dim);
    fprintf(fh, "#define HIDDEN_DIM %d\n", c.hidden_dim);
    fprintf(fh, "#define N_LAYERS %d\n", c.n_layers);
    fprintf(fh, "#define N_HEADS %d\n", c.n_heads);
    fprintf(fh, "#define N_KV_HEADS %d\n", c.n_kv_heads);
    fprintf(fh, "#define VOCAB_SIZE %d\n", c.vocab_size);
    fprintf(fh, "#define SEQ_LEN %d\n", c.seq_len);
    fprintf(fh, "#define KV_DIM %d\n", c.dim * c.n_kv_heads / c.n_heads);
    fprintf(fh, "#define STEPS %d\n", steps);
    fprintf(fh, "#define TEMPERATURE %f\n", temperature);
    fprintf(fh, "#define RND_SEED %d\n\n", rnd_seed);

    fprintf(fh,"// Allocazioni per il RunState\n");
    fprintf(fh, "PI_L2 float X [DIM];\n");
    fprintf(fh, "PI_L2 float XB [DIM];\n");
    fprintf(fh, "PI_L2 float XB2 [DIM];\n");
    fprintf(fh, "PI_L2 float HB [HIDDEN_DIM];\n");
    fprintf(fh, "PI_L2 float HB2 [HIDDEN_DIM];\n");
    fprintf(fh, "PI_L2 float Q [DIM];\n");
    fprintf(fh, "PI_L2 float KEY_CACHE [N_LAYERS*SEQ_LEN*KV_DIM];\n");
    fprintf(fh, "PI_L2 float VALUE_CACHE [N_LAYERS*SEQ_LEN*KV_DIM];\n");
    fprintf(fh, "PI_L2 float ATT[N_HEADS*SEQ_LEN];\n");
    fprintf(fh, "PI_L2 float LOGITS [VOCAB_SIZE];\n");
    fprintf(fh, "PI_L2 char PROB_INDEX [VOCAB_SIZE*%ld];\n\n", sizeof(ProbIndex));

    // Lettura valori dal file tokenizer
    Tokenizer t;
    FILE *file_tok = fopen(TOKENIZER_PATH, "rb");
    t.vocab = (char**) malloc(c.vocab_size*sizeof(char*));
    t.vocab_scores = (float*) malloc(c.vocab_size*sizeof(float));
    for (int i = 0; i < 256; i++) {
        t.byte_pieces[i * 2] = (unsigned char)i;
        t.byte_pieces[i * 2 + 1] = '\0';
    }
    if(file_tok == NULL){
        printf("Errore: impossible aprire il file %s\n", TOKENIZER_PATH);
        exit(1);
    }
    fread(&t.max_token_length, sizeof(int), 1, file_tok);
    int len, tot_size=0;
    for(int i=0; i<c.vocab_size; i++){
        fread(t.vocab_scores+i, sizeof(float), 1, file_tok);
        fread(&len, sizeof(int), 1, file_tok);
        t.vocab[i] = (char *)malloc(len+1);
        tot_size += len+1;
        fread(t.vocab[i], len, sizeof(char), file_tok); 
        t.vocab[i][len] = '\0';
    }
    fclose(file_tok);
    
    // Scrittura sul file di header
    fprintf(fh, "// Allocazioni per il tokenizer\n");
    fprintf(fh, "#define MAX_TOKEN_LENGTH %d\n", t.max_token_length);
    fprintf(fh, "PI_L2 float VOCAB_SCORES [VOCAB_SIZE] = {\n");
    int i;
    for(i=0;i<c.vocab_size-1;i++){
        fprintf(fh, "%f, ", t.vocab_scores[i]);
        if(i%10==9)
            fprintf(fh, "\n");
    }
    fprintf(fh, "%f};\n\n", t.vocab_scores[i]);

    fprintf(fh, "PI_L2 char* VOCAB[VOCAB_SIZE];\n");
    fprintf(fh, "PI_L2 unsigned char VOCAB_DATA [%d] = {\n", tot_size);
    for(i=0;i<c.vocab_size;i++){
        int j=0;
        while(t.vocab[i][j]!='\0')
            fprintf(fh, "0x%02x, ", (unsigned char) t.vocab[i][j++]);
        if(i<c.vocab_size-1)
            fprintf(fh, "0x%02x, \n", (unsigned char)'\0');
        else
            fprintf(fh, "0x%02x };\n\n", (unsigned char)'\0');
    }

    fprintf(fh, "// Allocazioni per i pesi del modello\n");
    fprintf(fh, "PI_L2 unsigned int weights_list[%d] = { ", w_dim);
    for(i=0;i<w_dim-1;i++){
        fprintf(fo, "%#x, ", *(unsigned int*)&w[i]);
        if(i%10 == 9)
            fprintf(fh, "\n");
    }
    fprintf(fh, "%#x};", *(unsigned int*)&w[i]);
    fclose(fh);
    printf("Scrittura di conf_and_weights.h completata\n");
    free(w);
    return 0;
}
