#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>
#include <omp.h>
#include <time.h>

#define M_MAX 128
#define N_MAX 128
#define K_MAX 128

#define min( i, j ) ( (i) < (j) ? (i) : (j) )

#define MEM_SIZE   0x30000UL
#define MEM_BASE   0xa0000000ul
#define ACCEL_BASE 0xa0030000ul

#define STATUS_REG 0x0
#define M_REG      0x4
#define N_REG      0x6
#define K_REG      0x8

#define START      0x1
#define DONE       0x2
#define IDLE       0x4

#define BRAM_SIZE (8192*8)

volatile int *mem, *accel;
volatile int *A, *B, *C;

int fd_mem;

int gemm_init(void)
{
  /* open memory as a file */
  fd_mem = open("/dev/mem", O_RDWR|O_SYNC);
  if (!fd_mem) {
    printf("Unable to open /dev/mem\n");
    return -1;
  }

  /* map physical base address for memory */
  mem = (volatile int *)mmap(NULL, MEM_SIZE, PROT_READ|PROT_WRITE,
                                MAP_SHARED, fd_mem, MEM_BASE);	
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

int gemm_deinit(void)
{
  //In the end, close the device driver
  munmap((void*)mem, MEM_SIZE);
  munmap((void*)accel, 256);
  close(fd_mem);

  return 0;
}

void load_row(int *dst, int *src, int cols)
{
  for(int i = 0; i < cols; i++) {
    dst[i] = src[i];
  }
}

void gemm_fpga(int m, int n, int k_, int *a, int *b, int *c)
{
  for (int i = 0; i < m; i += M_MAX) {
    int mm = min(m - i, M_MAX);
    for (int j = 0; j < n; j += N_MAX) {
      int nn = min(n - j, N_MAX);
      for (int k = 0; k < k_; k += K_MAX) {
        int kk = min(k_ - k, K_MAX);

        /* set blocked matrix pointers */
        int *aa = a + (i * k_) + k;
        int *bb = b + (k * n) + j;
        int *cc = c + (i * n) + j;

        accel[M_REG] = M_MAX;
        accel[N_REG] = N_MAX;
        accel[K_REG] = K_MAX;

        // put in A
        for (int i = 0; i < M_MAX; i++) {
          if (i < mm) {
            for (int j = 0; j < K_MAX; j++) {
              if (j < kk) {
                A[i * K_MAX + j] = aa[i * k_ + j];
              } else {
                A[i * K_MAX + j] = 0;
              }
            }
          } else {
            for (int j = 0; j < K_MAX; j++) {
              A[i * K_MAX + j] = 0;
            }
          }
        }

        // put in B
        for (int i = 0; i < K_MAX; i++) {
          if (i < kk) {
            for (int j = 0; j < N_MAX; j++) {
              if (j < nn) {
                B[i * N_MAX + j] = bb[i * n + j];
              } else {
                B[i * N_MAX + j] = 0;
              }
            }
          } else {
            for (int j = 0; j < N_MAX; j++) {
              B[i * N_MAX + j] = 0;
            }
          }
        }

        // do gemm
        accel[STATUS_REG] = START;

        // wait for output
        __asm("dmb sy");
        while (!(accel[STATUS_REG] & DONE));
        __asm("dmb sy");

        for (int i = 0; i < mm; i++) {
          for (int j = 0; j < nn; j++) {
            cc[i * n + j] += C[i * N_MAX + j];
          }
        }
      }
    }
  }
}
