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
#define INPUT_RAWDATA  "p21339a_ch4_ymg34cl.vdif" // /mnt/c368762e-4376-4cbd-ab30-7e38fde56dcc/U21339A/LHCP/u21339a_hit3.vdif" //RHCP/u21339a4_hit2.vdif" //LHCP/u21339a_hit2.vdif"
#define INPUT_BITSIZE 2
#define SR 1024000000LL//128000000LL
#define OUTPUT_DATA  "p21339a_ch4_ymg34cl_test_rev.dat" //"../practice/crab_10m/GRB_dedispered_GPU_81920000_i.dat" //"../practice/crab_20ms/dedispersion_ikebe.dat"  //"../practice/crab_2s/dedispersion_ikebe.dat" //"./10m/GRB_dedispered_GPU_2^26.dat" //
#define observationdata "Ibaraki" //"Ibaraki" //"Kashima"
//data description
#define FREQ_DATA_LOW 6600.0 //MHz
#define FREQ_DATA_WIDTH 512.0 //MHZ
#define FINE_TIME (7.8125*1e-9)
//for de dispersion
#define INPUT_DM_VAL 330.0 //56.743 //56.764 //56.764 //178 //56.764 //
//for averaging
#define INTG_TIME 1000 //us
#define mode 1 //0->sum by CPU, 1->sum by GPU
#define method 0
//for CUDA
#define SET_BLOCK 512 //de-dispersion
#define SET_BLOCK_TIME 512 //num bin of time

int getFileSize(const char* fileName);
int read_Kashima_to_d(unsigned char *raw_pre, cufftDoubleReal *d_real, FILE **fp, int length);
int read_Ibaraki_Y_to_d(unsigned char *raw_pre, double *d_real, FILE **fp, cufftDoubleReal *ho, double *time_arranging_bit, double *time_fread);
int read_Kashima(double *h_in, FILE **fp, int length);
void exec_reduce4(double *d_real_out, double *res_dev, int intgnumber, int timedelay, int max_repeat);
void sum64(double *d_real_2d, double *d_real_2dsummed);


#if defined(__cplusplus)
};
#endif

#endif




//--------end of this code
