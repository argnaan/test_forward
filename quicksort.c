#include "quicksort.h"

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
