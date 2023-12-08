int gemm_init(void);
int gemm_deinit(void);
void gemm_fpga(int m, int n, int k_, int *a, int *b, int *c);