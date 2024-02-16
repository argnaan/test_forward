#ifndef _ProbIndex_
#define _ProbIndex_
typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling
#endif

void quickSort(ProbIndex* a, int l, int h);