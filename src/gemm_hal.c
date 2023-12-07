#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>

/* FPGA matrix size */
#define BLOCK_LENGTH 16
#define BLOCK_SIZE (BLOCK_LENGTH * BLOCK_LENGTH)

#define min( i, j ) ( (i) < (j) ? (i) : (j) )

#define MEM_SIZE 0x8000UL
#define MEM_MASK (MEM_SIZE - 1)
#define MEM_BASE   0xa0000000ul
#define ACCEL_BASE 0xa0010000ul

#define STATUS_REG 0x0
#define M_REG      0x4
#define N_REG      0x6
#define K_REG      0x8

#define START      0x1
#define DONE       0x2
#define IDLE       0x4

#define BRAM_SIZE 8192

volatile int *mem, *accel;
volatile int *A, *B, *C;

int fd_mem;

int fpga_init(void)
{
  /* open memory as a file */
  int fd_mem = open("/dev/mem", O_RDWR|O_SYNC);
  if (!fd_mem) {
    printf("Unable to open /dev/mem\n");
    return -1;
  }

  /* map physical base address for memory */
  mem = (volatile int *)mmap(NULL, MEM_SIZE, PROT_READ|PROT_WRITE,
                                MAP_SHARED, fd_mem, MEM_BASE & ~MEM_MASK);	
  if (mem == MAP_FAILED) {
    printf("Memory mapping failed\n");
    fflush(stdout);
  }

  accel = (volatile int *)mmap(NULL, 256, PROT_READ|PROT_WRITE,
                                MAP_SHARED, fd_mem, ACCEL_BASE);	
  if (accel == MAP_FAILED) {
    printf("Memory mapping failed\n");
    fflush(stdout);
  }

  A = &mem[(0*BRAM_SIZE) >> 2];
  B = &mem[(1*BRAM_SIZE) >> 2];
  C = &mem[(2*BRAM_SIZE) >> 2];

  return 0;
}

int fpga_deinit(void)
{
  //In the end, close the device driver
  munmap((void*)mem, MEM_SIZE);
  munmap((void*)accel, 256);
  close(fd_mem);

  return 0;
}

void gemm_blocked(int M, int N, int K, int *A, int *B, int *C)
{
  int i, j, k;
  for (i = 0; i < M; ++i) {
    for (k = 0; k < K; ++k) {
      for (j = 0; j < N; ++j) {
          C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
}

void load_row(int *dst, int *src, int cols)
{
  for(int i = 0; i < cols; i++) {
    dst[i] = src[i];
  }
}

int fpga_gemm(int m, int n, int k_, int *a, int *b, int *c)
{
  unsigned long val, result;

  int a_buff[BLOCK_LENGTH * BLOCK_LENGTH] = {0};
  int b_buff[BLOCK_LENGTH * BLOCK_LENGTH] = {0};
  int c_buff[BLOCK_LENGTH * BLOCK_LENGTH] = {0};

  for (int i = 0; i < m; i += BLOCK_LENGTH) {
    int mm = min(m - i, BLOCK_LENGTH);
    for (int j = 0; j < n; j += BLOCK_LENGTH) {
      int nn = min(n - j, BLOCK_LENGTH);
      for (int k = 0; k < k_; k += BLOCK_LENGTH) {
        int kk = min(k_ - k, BLOCK_LENGTH);
        
        /* load a, b, & c into buffers to send */
        for (int x = 0; x < mm; x++) {
          load_row(a_buff + (x * BLOCK_LENGTH), a + ((i + x) * k_) + k, kk);
        }
        for (int x = 0; x < kk; x++) {
          load_row(b_buff + (x * BLOCK_LENGTH), b + ((k + x) * n) + j, nn);
        }
        for (int x = 0; x < mm; x++) {
          load_row(c_buff + (x * BLOCK_LENGTH), c + ((i + x) * n) + j, nn);
        }

        accel[M_REG] = mm;
        accel[N_REG] = nn;
        accel[K_REG] = kk;

        // put in A
        for (int i = 0; i < mm; i++) {
          for (int j = 0; j < kk; j++) {
            A[i * kk + j] = a_buff[i * BLOCK_LENGTH + j];
          }
        }

        // put in B
        for (int i = 0; i < kk; i++) {
          for (int j = 0; j < nn; j++) {
            B[i * nn + j] = b_buff[i * BLOCK_LENGTH + j];
          }
        }

        // put in C
        for (int i = 0; i < mm; i++) {
          for (int j = 0; j < nn; j++) {
            C[i * nn + j] = c_buff[i * BLOCK_LENGTH + j];
          }
        }

        // do gemm
        accel[STATUS_REG] = START;

        // wait for output
        __asm("dmb sy");
        while (!(accel[STATUS_REG] & DONE));
        __asm("dmb sy");

        // read gemm output
        for (int i = 0; i < mm; i++) {
          for (int j = 0; j < nn; j++) {
            c_buff[i * BLOCK_LENGTH + j] = C[i * nn + j];
          }
        }

        // gemm_blocked(a_buff, b_buff, c_buff);
        
        /* load result */
        for (int x = 0; x < mm; x++) {
          load_row(c + ((i + x) * n) + j, c_buff + (x * BLOCK_LENGTH), nn);
        }
      }
    }
  }

  return 0;
}
