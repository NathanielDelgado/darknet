#include <stdlib.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <signal.h>
#include <sys/types.h>
#include "HAL.h"

//When writing: Fills up matrix elements starting from A->B. After writing last element, write once more to run gemm.
//When reading: Sends out C matrix elements. Read once more to reset C ptr.
#define GEMM (0x00)
#define DONE (0x04 >> 2)
#define MAP_SIZE 512UL
#define MAP_MASK (MAP_SIZE - 1)

#define A_BRAM 0xa0000000uL
#define B_BRAM 0xa0002000uL
#define C_BRAM 0xa0004000uL

#define READ_CMD  (0x0 << 31)
#define WRITE_CMD (0x1 << 31)

static volatile int det_int = 0;
static volatile int* A_base;
static volatile int* B_base;
static volatile int* C_base;
static volatile int fd; //file descriptor for /dev/mem
static volatile int fdi; //file descriptor for /dev/fpga
unsigned long iie, gie, trig;

// signal handler for receiving events from hardware driver
void sighandler(int signo){
    if(signo==SIGIO) {
        det_int++;
        printf("\nInterrupt detected\n");
    }   
    return;
}

void set_pl_freq() {
    int dh = open("/dev/mem", O_RDWR | O_SYNC);
    if(dh == -1) {
        printf("Unable to open /dev/mem\n");
        exit(1);
    }
    uint32_t* clk_reg = mmap(NULL, 0x1000, PROT_READ|PROT_WRITE, MAP_SHARED, dh, 0xFF5E0000);
    int i = 0;
    uint32_t* pl0 = clk_reg;
    pl0 += 0xC0; // PL0_REF_CTRL reg offset 0xC0
    *pl0 = (1<<24) // bit 24 enables clock
     | (1<<16) // bit 23:16 is divisor 1
     | (6<<8); // bit 15:0 is clock divisor 0
    // frequency = 1.5Ghz/divisor0/divisor1
    // = 1.5Ghz/6=250MHz
    munmap(clk_reg, 0x1000);

}

//sets cpu frequency to 1500
void set_ps_freq() {
    int dh = open("/dev/mem", O_RDWR | O_SYNC);
    if(dh < 0) {
        printf("Unable to open /dev/mem.  Ensure it exists!\n");
        exit(1);
    }

    int* ps_pll = (unsigned int *)mmap(NULL, MAP_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, dh, (0x00FD1A0000) & ~MAP_MASK);              
    int* apll_ctrl = ps_pll + 8; //at 0x00fd1a0020
    int* apll_cfg = ps_pll + 9; //at 0x00fd1a0024
    int* pll_status = ps_pll + 17; //at 0x00fd1a0044
    // *apll_ctrl = 0x00014800;
    // *apll_cfg = 0x7E4B0C6C;
    // *apll_ctrl = 0x00014808;
    // *apll_ctrl = 0x00014809;
    // *apll_ctrl = 0x00014808;
    *apll_ctrl= 0x00002D00;
    *apll_cfg = 0x7E672C6C;
    *apll_ctrl= 0x00002D08;
    *apll_ctrl= 0x00002D09;
    *apll_ctrl= 0x00002D08;
    while( ((*pll_status) & 1)!= 1) printf("Waiting\n");
    *apll_ctrl= 0x00002D00;
    // *apll_ctrl = 0x00014800;
    close(dh);
}

void hal_init() {
    struct sigaction action;
    static unsigned long addr0 = 0xa0000000ul; //physical address of the start of the FPGA address region.
    // install signal handler
    sigemptyset(&action.sa_mask);
    sigaddset(&action.sa_mask, SIGIO);

    action.sa_handler = sighandler;
    action.sa_flags=0;

    sigaction(SIGIO, &action, NULL);

    fdi=open("/dev/fpga", O_RDWR);
    if(fdi < 0) {
        printf("Unable to open /dev/fpga.  Ensure it exists!\n");
        exit(1);
    }
    fcntl(fdi, F_SETOWN, getpid());
    fcntl(fdi, F_SETFL, fcntl(fdi, F_GETFL)|O_ASYNC);
    // enable FPGA interrupts (global and IP)
    
    if(ioctl(fdi, READ_CMD + 0x1, &gie)) err(1, "Getting GIE");
    gie = gie | 0x00000001;
    if(ioctl(fdi, WRITE_CMD + 0x1, &gie)) err(1, "Setting GIE");
    iie = 0x1;
    if(ioctl(fdi, WRITE_CMD + 0x2, &iie)) err(1, "Setting IIE");

    fd = open("/dev/mem", O_RDWR|O_SYNC);
    if(fd == -1) {
        printf("Unable to open /dev/mem.  Ensure it exists (  major=1, minor=1)\n");
        exit(1);
    }

    A_base = (int *)mmap(NULL, MAP_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, fd, A_BRAM);
    printf("fpga_base: %p, MAP_SIZE: %x, OFFSET: %p\n", A_base, MAP_SIZE, A_BRAM);
    if (A_base == MAP_FAILED) {
        printf("mmap error %p\n", A_base);
        exit(1);
    }
    B_base = (int *)mmap(NULL, MAP_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, fd, B_BRAM);
    printf("fpga_base: %p, MAP_SIZE: %x, OFFSET: %p\n", B_base, MAP_SIZE, B_BRAM);
    if (B_base == MAP_FAILED) {
        printf("mmap error %p\n", B_base);
        exit(1);
    }
    C_base = (int *)mmap(NULL, MAP_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, fd, C_BRAM);
    printf("fpga_base: %p, MAP_SIZE: %x, OFFSET: %p\n", C_base, MAP_SIZE, C_BRAM);
    if (C_base == MAP_FAILED) {
        printf("mmap error %p\n", C_base);
        exit(1);
    }
}

void gemm_fpga(int* A, int* B, int* C, int M, int N, int K, int M_BLOCK, int N_BLOCK, int K_BLOCK) {
    volatile int x;
    for (int a = 0; a < MM_BLOCK_SIZE; a++){
        for (int b = 0; b < MM_BLOCK_SIZE; b++){
            if (b < K_BLOCK && a < M_BLOCK) *(A_base+ a * MM_BLOCK_SIZE + b) = A[a * K + b];
            else *(A_base + a * MM_BLOCK_SIZE + b) = 0;
        }
    }
    for (int a = 0; a < MM_BLOCK_SIZE; a++){
        for (int b = 0; b < MM_BLOCK_SIZE; b++){
            if (b < N_BLOCK && a < K_BLOCK) *(B_base+ a * MM_BLOCK_SIZE + b) = B[a * N + b];
            else *(B_base + a * MM_BLOCK_SIZE + b) = 0;
        }
    }
    for (int a = 0; a < MM_BLOCK_SIZE; a++){
        for (int b = 0; b < MM_BLOCK_SIZE; b++){
            if (b < N_BLOCK && a < M_BLOCK) *(C_base + a * MM_BLOCK_SIZE + b) = C[a * N + b];
            else *(C_base + a * MM_BLOCK_SIZE + b) = 0;
        }
    }
    trig = 0x1;
    if(ioctl(fdi, WRITE_CMD, &trig)) err(1, "Trigger");
    // Wait for interrupt
    while(!det_int) continue;
    det_int = 0;
    for (int a = 0; a < MM_BLOCK_SIZE; a++){
        for (int b = 0; b < MM_BLOCK_SIZE; b++){
            if (b < N_BLOCK && a < M_BLOCK) C[a * N + b] = *(C_base + a * MM_BLOCK_SIZE + b);
            else x = *(C_base + a * MM_BLOCK_SIZE + b);
            
        }
    }
}

void hal_deinit() {
    munmap((void*)A_base, MAP_SIZE);
    munmap((void*)B_base, MAP_SIZE);
    munmap((void*)C_base, MAP_SIZE);
    close(fd);
    close(fdi);
}