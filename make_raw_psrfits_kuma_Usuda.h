#ifndef GPR_RAW_DEDISPERSION_V3
#define GPR_RAW_DEDISPERSION_V3

#if defined(__cplusplus)
extern "C" {
#endif

#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <time.h>
#include <math.h>
#include <sys/stat.h>
#include <cuda_runtime.h>
#include <cufft.h>

#define N_dt 8192000 //67108864//(int)(pow(2,26)) //81920000//
#define FFTP 64000 //data points per 1 FFT
#define INPUT_RAWDATA  "FRB201124_2022049142130-142140.dat" // /mnt/c368762e-4376-4cbd-ab30-7e38fde56dcc/U21339A/LHCP/u21339a_hit3.vdif" //RHCP/u21339a4_hit2.vdif" //LHCP/u21339a_hit2.vdif"
#define INPUT_BITSIZE 4
#define CHNUM 8
#define SR 64000000LL//128000000LL
#define OUTPUT_DATA  "FRB201124_Usuda" //"../practice/crab_10m/GRB_dedispered_GPU_81920000_i.dat" //"../practice/crab_20ms/dedispersion_ikebe.dat"  //"../practice/crab_2s/dedispersion_ikebe.dat" //"./10m/GRB_dedispered_GPU_2^26.dat" //
#define observationdata "Usuda" //"Ibaraki" //"Kashima"
//data description
#define FREQ_DATA_WIDTH 32.0 //MHZ
#define FINE_TIME (7.8125*1e-9)
//for CUDA
#define SET_BLOCK 512 //de-dispersion
#define SET_BLOCK_TIME 512 //num bin of time

int getFileSize(const char* fileName);
int read_Usuda_to_ds(unsigned char *raw_pre, cufftDoubleReal **d_real, cufftDoubleReal *d_ho,FILE **fp, double *time_arranging_bit, double *time_fread);
void exec_reduce4(double *d_real_out, double *res_dev, int intgnumber, int timedelay, int max_repeat);
void sum64(double *d_real_2d, double *d_real_2dsummed);


#if defined(__cplusplus)
};
#endif

#endif




//--------end of this code
