#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>     
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#define WEIGHTS_BIN "fp16/utils/rnd_weights.bin"
#define WEIGHTS_HEADER "fp16/conf_and_weights_fp16.h"
#define TOKENIZER_PATH "fp16/utils/tokenizer.bin"

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

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

int main(int argc, char* argv[]){
    srand(time(NULL));

    Config c;
    float* w;
    int steps, rnd_seed, head_size, w_dim;
    float temperature;
    char* weights_path;
    char* prompt;
    weights_path = argv[1];
    sscanf(argv[9], "%d", &steps);
    sscanf(argv[10], "%f", &temperature);
    sscanf(argv[11], "%d", &rnd_seed);
    prompt = argv[12];

    if(!strcmp(weights_path, "NULL") || weights_path==NULL){
        printf("Pesi generati casualmente\n");
        sscanf(argv[2], "%d", &c.dim);
        sscanf(argv[3], "%d", &c.hidden_dim);
        sscanf(argv[4], "%d", &c.n_layers);
        sscanf(argv[5], "%d", &c.n_heads);
        sscanf(argv[6], "%d", &c.n_kv_heads);
        sscanf(argv[7], "%d", &c.vocab_size);
        sscanf(argv[8], "%d", &c.seq_len);

        head_size = c.dim / c.n_heads;
        // numero di pesi da generare: calcolato in base alla descrizione della struttura dati TransformerWeights
        w_dim = c.dim * (c.vocab_size + c.n_layers*(2 + 2*c.dim + head_size*c.n_kv_heads*2 + 3*c.hidden_dim) + 1);
        
        int file_dim = (w_dim*sizeof(float) + sizeof(Config))/1024;
        printf("Numero pesi generati: %d (%d kB)\n", w_dim, file_dim);
        
        FILE *fo = fopen(WEIGHTS_BIN, "wb");
        if(fo == NULL){
            printf("Errore nella creazione del file %s\n", WEIGHTS_BIN);
            exit(1);
        }
        // Scrittura della configurazione c
        fwrite(&c, sizeof(Config), 1, fo);
        w = malloc(w_dim*sizeof(float));

        // generazione di float casuali tra -4 e 4
        for(int i=0;i<w_dim;i++)
            w[i] = ((float)rand() / (RAND_MAX/8)) - 4;
        fwrite(w, sizeof(float), w_dim, fo);
        fclose(fo);
        printf("Scrittura di %s completata\n", WEIGHTS_BIN);
    }else{
        printf("Using pretrained weights\n");
        // lettura dei pesi e della configurazione dal file weights_path
        FILE* fw = fopen(weights_path, "rb");
        if(fw == NULL){
            printf("Errore: Impossibile aprire il file %s\n\n", weights_path);
            exit(1);
        }
        if(fread(&c, sizeof(Config), 1, fw)!=1){
            fprintf(stderr, "Errore nella lettura di %s\n", weights_path);
            exit(1);
        }
        fseek(fw, 0, SEEK_END);
        int file_size = ftell(fw);
        fclose(fw);
        int fwd = open(weights_path, O_RDONLY);
        float* data;
        data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fwd, 0);
        if(data == NULL){
            printf("Errore: mmap failed\n");
            exit(1);
        }
        head_size = c.dim / c.n_heads;
        w = data + sizeof(Config)/sizeof(float);
        w_dim = c.dim * (c.vocab_size + c.n_layers*(2 + 2*c.dim + head_size*c.n_kv_heads*2 + 3*c.hidden_dim) + 1);
    }
    printf("w_dim: %d\n", w_dim);
    FILE* fh = fopen(WEIGHTS_HEADER, "w");
    if(fh==NULL){
        printf("Errore: impossible creare il file conf_and_weights.h\n");
        exit(1);
    }
    // Aggiunto: override c.seq_len con steps
    c.seq_len = steps;   

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

    fprintf(fh, "PI_L2 char* PROMPT = \"%s\";\n", prompt);
    fprintf(fh, "PI_L2 int PROMPT_TOKENS[%ld];\n", strlen(prompt)+3);

    fprintf(fh,"\n// Allocazioni per il RunState\n");
 /*
    fprintf(fh, "PI_L2 fp16 X [DIM];\n");
    fprintf(fh, "PI_L2 fp16 XB [DIM];\n");
    fprintf(fh, "PI_L2 fp16 XB2 [DIM];\n");
    fprintf(fh, "PI_L2 fp16 HB [HIDDEN_DIM];\n");
    fprintf(fh, "PI_L2 fp16 HB2 [HIDDEN_DIM];\n");
    fprintf(fh, "PI_L2 fp16 Q [DIM];\n");
 */
    fprintf(fh, "PI_L2 fp16 KEY_CACHE [N_LAYERS*STEPS*KV_DIM];\n");
    fprintf(fh, "PI_L2 fp16 VALUE_CACHE [N_LAYERS*STEPS*KV_DIM];\n");
/*
    fprintf(fh, "PI_L2 fp16 ATT[N_HEADS*STEPS];\n");
    fprintf(fh, "PI_L2 fp16 LOGITS [VOCAB_SIZE];\n\n");
*/
    int size_1, size_2, size_4;
    if(c.hidden_dim > c.dim){
        size_1 = c.hidden_dim;
        size_2 = c.hidden_dim > steps ? c.hidden_dim : steps;
    }
    else{
        size_1 = c.dim;
        size_2 = c.dim > steps ? c.dim : steps;
    }
    size_4 = c.dim;
/*
    fprintf(fh, "PI_L1 fp16 BUFF1[%d];\n", size_1);
    fprintf(fh, "PI_L1 fp16 BUFF2[%d];\n", size_2);
    fprintf(fh, "PI_L1 fp16 BUFF3[%d];\n", size_2);
    fprintf(fh, "PI_L1 fp16 BUFF4[%d];\n", size_4);
    fprintf(fh, "PI_L1 fp16 BUFF_W_1[%d];\n", c.dim*c.hidden_dim);
    fprintf(fh, "PI_L1 fp16 BUFF_W_2[%d];\n\n", c.dim*c.hidden_dim);
 */   

    fprintf(fh, "\nPI_L2 char PROB_INDEX [VOCAB_SIZE*%ld];\n\n", sizeof(ProbIndex));

    // Lettura valori dal file tokenizer
    Tokenizer t;
    t.vocab_size = c.vocab_size;
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
    if(fread(&t.max_token_length, sizeof(int), 1, file_tok)!=1){
        printf("Errore nella lettura di %s\n", TOKENIZER_PATH);
        exit(1);
    }
    int len, tot_size=0;
    for(int i=0; i<c.vocab_size; i++){
        if(fread(t.vocab_scores+i, sizeof(float), 1, file_tok)!=1){
            printf("Errore in fread\n");
            exit(1);
        }
        if(fread(&len, sizeof(int), 1, file_tok)!=1){
            printf("Errore in fread\n");
            exit(1);
        }
        t.vocab[i] = (char *)malloc(len+1);
        tot_size += len+1;
        if(fread(t.vocab[i], sizeof(char), len, file_tok)!=len){
            printf("Errore in fread\n");
            exit(1);
        } 
        t.vocab[i][len] = '\0';
    }
    fclose(file_tok);
    
    // Scrittura sul file di header
    fprintf(fh, "// Allocazioni per il tokenizer\n");
    fprintf(fh, "#define MAX_TOKEN_LENGTH %d\n", t.max_token_length);
    fprintf(fh, "PI_L2 fp16 VOCAB_SCORES [VOCAB_SIZE] = {\n");
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

    t.sorted_vocab = malloc(t.vocab_size * sizeof(TokenIndex));
    for (int i = 0; i < t.vocab_size; i++) {
        t.sorted_vocab[i].str = t.vocab[i];
        t.sorted_vocab[i].id = i;
    }
    qsort(t.sorted_vocab, t.vocab_size, sizeof(TokenIndex), compare_tokens);

    fprintf(fh, "PI_L2 TokenIndex SORTED_VOCAB [VOCAB_SIZE] = {\n");
    for(i=0; i<t.vocab_size; i++){
        fprintf(fh, "{\"");
        for(int k=0; t.sorted_vocab[i].str[k] != '\0'; k++)
            fprintf(fh, "\\x%02X", (unsigned char) t.sorted_vocab[i].str[k]);
        if(i< t.vocab_size-1)
            fprintf(fh, "\", %d},\n", t.sorted_vocab[i].id);
        else
            fprintf(fh, "\", %d}\n", t.sorted_vocab[i].id);
    }
    fprintf(fh, "};\n\n");
    

    fprintf(fh, "PI_L2 char STR_BUFFER [MAX_TOKEN_LENGTH*2 + 1+2];\n\n");

    fprintf(fh, "// Allocazioni per i pesi del modello\n");
    fprintf(fh, "PI_L2 fp16 weights_list[%d] = { ", w_dim);
    for(i=0;i<w_dim-1;i++){
        fprintf(fh, "%.10f, ", w[i]);
        if(i%10 == 9)
            fprintf(fh, "\n");
    }
    fprintf(fh, "%.10f};", w[i]);
    fclose(fh);
    printf("Scrittura di %s completata\n", WEIGHTS_HEADER);
    return 0;
}

