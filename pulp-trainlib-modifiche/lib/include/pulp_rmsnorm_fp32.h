

struct weighted_scaling_args {
    float* out;
    float* in;
    float* w;
    float scaling_factor;
    unsigned int size;
};

struct sum_of_squares_args {
    float* out;
    float* in;
    unsigned int size;
};

void weighted_scaling_fp32_cl(void* weighted_scaling_args);

void sum_of_squares_fp32_cl(void* sum_of_squares_args);

void rmsnorm_parallelized_fp32(float* o, float* x, float* weight, float* buffer_n_cores, int size);
