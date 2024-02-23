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

void quickSort(ProbIndex* a, int l, int h);
void quickSort_vocab(TokenIndex* a, int l, int h);

void *bsearch (const void *key, const void *base0, size_t nmemb, size_t size, int (*compar)(const void *, const void *));
void _qsort(void* v, int size, int left, int right, int (*comp)(void*, void*));