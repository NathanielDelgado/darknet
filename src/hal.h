void hal_init(void);

void gemm_fpga(int* A, int* B, int* C, int M, int N, int K, int M_BLOCK, int N_BLOCK, int K_BLOCK);

void hal_deinit(void);