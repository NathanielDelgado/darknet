int fpga_init(void);
int fpga_deinit(void);
int fpga_gemm(int m, int n, int k_, int *a, int *b, int *c);