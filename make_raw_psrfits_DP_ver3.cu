/*

This is a simple dedispersion code using CUDA.
double precision ver

Note : some values for de-dispersion was copied from Mikami-san's code

S.Ikebe,
*/

#include "make_raw_psrfits_kuma.h"
#include "utils/utils_si.cuh"



int getFileSize(const char* fileName)
{
  struct stat st_f;

  if (stat(fileName, &st_f) != 0) {
    return -1;
  }

  //check file type
  if ((st_f.st_mode & S_IFMT) != S_IFREG) {
    //S_IFMT : type
    //S_IFREG : regular file
    return -1;
  }

  return st_f.st_size;//file size in bite
}


__global__
void bitconversion(unsigned char *d_raw_pre, cufftDoubleReal *d_real, unsigned char *wordl, unsigned char *wordr, unsigned char *alword)
  {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if( i < N_dt/4){
     wordl[i] = (d_raw_pre[i] & 0xAA) >> 1;
     wordr[i] = (d_raw_pre[i] & 0x55) << 1;
     alword[i] = (wordl[i] | wordr[i]) ^ 0x33;
     d_real[i*4] = alword[i] & 0x03;
     d_real[i*4+1] = (alword[i] >> 2) & 0x03;
     d_real[i*4+2] = (alword[i] >> 4) & 0x03;
     d_real[i*4+3] = (alword[i] >> 6) & 0x03;
      }//for j
    }//if

__global__
void bitconversion_i(unsigned char *d_raw_pre, cufftDoubleReal *d_real, cufftDoubleReal *ho)
  {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int k;
    k = d_raw_pre[i];
    d_real[4*i] = ho[4*k];
    d_real[4*i+1] = ho[4*k+1];
    d_real[4*i+2] = ho[4*k+2];
    d_real[4*i+3] = ho[4*k+3];
    /*
    if (d_real[i]!=0){
    printf("d_real: %lf",d_real[i]);}*/
    //printf("d_real: %lf",ho[i]);
    }//if

__global__
void bitconversion_k(unsigned char *d_raw_pre, cufftDoubleReal *d_real)
  {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    d_real[i] = d_raw_pre[i];
    }//if


int read_Kashima_to_d(unsigned char *raw_pre, cufftDoubleReal *d_real, FILE **fp, int length){
      dim3 block_bit (SET_BLOCK, 1, 1);
      dim3 grid_bit  (length / block_bit.x, 1, 1);
      long long int err_read; cudaError_t err;
      double time_s[4];
      double time_e[4];

      time_s[0] = cputimeinsec();
      err_read = fread(raw_pre,sizeof(unsigned char),length,*fp);
      if(err_read!=length){
       printf("data is shorter than N_dt so, stop this roop...\n");
       return 1;
      }
      unsigned char *d_raw_pre;
      err = cudaMalloc((void**)&d_raw_pre, sizeof(unsigned char)*length);
       if (err != cudaSuccess) {
       exit(err);
       }
      cudaDeviceSynchronize();
      time_e[0] = cputimeinsec();

      time_s[1] = cputimeinsec();
      cudaMemcpy(d_raw_pre, raw_pre, length*sizeof(unsigned char), cudaMemcpyHostToDevice);
      cudaDeviceSynchronize();
      time_e[1] = cputimeinsec();

      time_s[2] = cputimeinsec();
      bitconversion_k<<<grid_bit,block_bit>>>(d_raw_pre, d_real);
      cudaDeviceSynchronize();
      time_e[2] = cputimeinsec();

      time_s[3] = cputimeinsec();
      cudaFree(d_raw_pre);
      cudaDeviceSynchronize();
      time_e[3] = cputimeinsec();

          /*
          printf("time for fread + malloc for bit conversion : %lf\n", time_e[0]-time_s[0]);
          printf("time for memcpy of raw data : %lf\n", time_e[1]-time_s[1]);
          printf("time for bit conversion : %lf\n", time_e[2]-time_s[2]);
          printf("time for free : %lf\n", time_e[3]-time_s[3]);
          */
      return 0;
    }



int read_Ibaraki_Y_to_d(unsigned char *raw_pre, double *d_real, FILE **fp, cufftDoubleReal *ho, double *time_arranging_bit, double *time_fread){
      unsigned char  *d_raw_pre; //*raw_pre
      dim3 block_bit (SET_BLOCK, 1, 1);
      dim3 grid_bit  (N_dt / block_bit.x / 4, 1, 1);
      long long int err_read;
      cudaError_t err;
      double time_s[5];
      double time_e[5];

      //start for Hitachi or Takahagi data
      printf("data is read as observation in Hitachi or Takahagi\n");
      time_s[0] = cputimeinsec();
      err = cudaMalloc((void**)&d_raw_pre, sizeof(unsigned char)*N_dt*2/8);
      if (err != cudaSuccess) {
        exit(err);
      }
      cudaDeviceSynchronize();
      time_e[0] = cputimeinsec();

      time_s[1] = cputimeinsec();
      err_read = fread(raw_pre,sizeof(unsigned char),(N_dt)*2/8,*fp);
      if(err_read!=(long long int)(N_dt)*INPUT_BITSIZE/8){
        printf("data is shorter than N_dt so, stop this roop...\n");
        //break;
        return 1;
        exit(0);
      }
      time_e[1] = cputimeinsec();

      time_s[2] = cputimeinsec();
      cudaMemcpy(d_raw_pre, raw_pre, N_dt*sizeof(unsigned char)*2/8, cudaMemcpyHostToDevice);
      cudaDeviceSynchronize();
      time_e[2] = cputimeinsec();

      time_s[3] = cputimeinsec();
      bitconversion_i<<<grid_bit,block_bit>>>(d_raw_pre, d_real,ho);
      cudaDeviceSynchronize();
      time_e[3] = cputimeinsec();

      time_s[4] = cputimeinsec();
      cudaFree(d_raw_pre); //free(raw_pre);
      cudaDeviceSynchronize();
      time_e[4] = cputimeinsec();

      //end for Hitachi or Takahagi data
      printf("time for malloc for bit conversion : %lf\n", time_e[0]-time_s[0]);
      printf("time for fread : %lf\n", time_e[1]-time_s[1]);
      printf("time for memcpy of raw data : %lf\n", time_e[2]-time_s[2]);
      printf("time for bit conversion : %lf\n", time_e[3]-time_s[3]);
      printf("time for free : %lf\n", time_e[4]-time_s[4]);

      *time_arranging_bit += (time_e[3] - time_s[3]);
      *time_fread += (time_e[1]-time_s[1]);

      return 0;
    }

int read_Kashima(double *h_in, FILE **fp, int length){
  unsigned char *raw;
  int err_read;
  raw = (unsigned char*) malloc(sizeof(unsigned char) * length);
  if(raw == NULL) {
    printf("memory cannot be allocated!!\n");
    exit(EXIT_FAILURE);
  }
  err_read = fread(raw,sizeof(unsigned char),length,*fp);
  if(err_read!=length){
    printf("data is shorter than N_dt so, stop this roop...\n");
    //break;
    return 1;
  }
  for(int i=0;i<length;i++){
    h_in[i] = raw[i];
  }
  free(raw);
  return 0;
}


__global__ void sum_time_gpu0(float *time_gpu_dev, float milliseconds){
  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if (idx == 0){
  time_gpu_dev[idx] = time_gpu_dev[idx] + milliseconds/1e3;
  }
}

__global__ void sum_time_gpu1(float *time_gpu_dev, float milliseconds){
  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if (idx == 1){
  time_gpu_dev[idx] = time_gpu_dev[idx] + milliseconds/1e3;
  }
}

__global__ void sum_time_gpu2(float *time_gpu_dev, float milliseconds){
  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if (idx == 2){
  time_gpu_dev[idx] = time_gpu_dev[idx] + milliseconds/1e3;
  }
}


//#ZZZ
__global__ void realtocomplex(cufftDoubleReal *in, cufftDoubleComplex *out, int bin_size){
  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if (idx < bin_size){
    out[idx].x   = in[idx];
    out[idx].y   = 0;//in[idx];
    //out[idx][1]   = 0;
  }
}
__global__ void realtocomplex_single(cufftReal *in, cufftComplex *out, int bin_size){
  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if (idx < bin_size){
    out[idx].x   = in[idx];
    out[idx].y   = 0;//in[idx];
    //out[idx][1]   = 0;
  }
}





__global__ void reduce4(double *g_idata, double *g_odata, unsigned int n)
{
    extern __shared__ double sdata[];

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 300)
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    double mysum = (i < n) ? g_idata[i] : 0;
    if (i + blockDim.x < n) mysum += g_idata[i + blockDim.x];
    sdata[tid] = mysum;
    __syncthreads();

    for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
        if (tid < s) {
            sdata[tid] = mysum = mysum + sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid < 32) {
       if(blockDim.x >= 64) mysum += sdata[tid + 32];
        for (int offset = 32/2; offset>0; offset>>=1) {
            mysum += __shfl_down(mysum, offset);
        }
    }
    if (tid == 0) g_odata[blockIdx.x] = mysum;
#else
#error "__shfl_down requires CUDA arch >= 300."
#endif
}

__global__
void cp_array(double *arr_dev, double *d_out, int repeat, int intgnumber){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i >= 0 && i < intgnumber){
  arr_dev[i] = d_out[repeat*intgnumber+i];
  }
}

__global__
void input_result(double *res_dev, double *out, int repeat){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i==repeat){
  	res_dev[i] = out[0];
  }
}

__global__
void sum_to_ave(double *ave_dev, double *res_dev, int intgnumber, int max_repeat){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i >= 0 && i < max_repeat){
        res_dev[i] = ave_dev[i] / intgnumber / N_dt;
  }
}


void exec_reduce4(double *d_real_out, double *res_dev, int intgnumber, int timedelay, int max_repeat){
  double *out1_dev, *out2_dev;
  int th = SET_BLOCK;
  int blocks = (intgnumber - 1) / (2 * th) + 1;
  int shared_mem_size = 2 * th * sizeof(double);
  cudaMalloc((void**)&out1_dev, sizeof(double) * blocks);
  cudaMalloc((void**)&out2_dev, sizeof(double) * blocks);
  int repeat;
  double *arr_dev, *ave_dev;
  cudaMalloc((void**)&arr_dev, sizeof(double)*intgnumber);
  cudaMalloc((void**)&ave_dev, sizeof(double)*max_repeat);
  //cudaMalloc((void**)&res_dev, sizeof(double)*max_repeat);
  int blocksize_cp = SET_BLOCK;//512;
  dim3 block_cp (blocksize_cp, 1, 1);
  dim3 grid_cp  (N_dt / block_cp.x, 1, 1);//here +1 is reqired as FFTed array has (N_dt/2+1) samples
  dim3 block_in (SET_BLOCK, 1, 1);
  dim3 grid_in (max_repeat / block_in.x+1, 1, 1);

  for(repeat = 0; repeat < max_repeat; repeat++){
  cp_array<<<grid_cp, block_cp>>>(arr_dev, d_real_out, repeat, intgnumber);
  double **in = &arr_dev, **out = &out1_dev;
  int n = intgnumber;
  //printf("%")
  while (blocks > 1) {
    reduce4<<<blocks, th, shared_mem_size>>>(*in, *out, intgnumber);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        puts(cudaGetErrorString(err));
    }
    if (*out == out1_dev) {
        out = &out2_dev; in = &out1_dev;
    }
    else {
        out = &out1_dev; in = &out2_dev;
    }
    n = blocks;
    blocks = (blocks - 1) / (2 * th) + 1;
    cudaDeviceSynchronize();
    }
    reduce4<<<blocks, th, shared_mem_size>>>(*in, *out, n);

    //dim3 block_in (SET_BLOCK, 1, 1);
    //dim3 grid_in (max_repeat / block_in.x+1, 1, 1);
    input_result<<<grid_in, block_in >>>(ave_dev, *out, repeat);
    //res_dev[repeat] = out;
    //cudaMemcpy(&result, *out, sizeof(double), cudaMemcpyDeviceToHost);
    blocks = (intgnumber - 1) / (2 * th) + 1;
  }
  cudaFree(out1_dev); cudaFree(out2_dev); cudaFree(arr_dev);
  sum_to_ave<<<grid_in, block_in>>>(ave_dev, res_dev, intgnumber, max_repeat);
  cudaFree(ave_dev);
}

__global__ void reduceUnrollWarps1 (double *g_idata, double *g_odata)
{
    int tid = threadIdx.x;
    //long idx = blockIdx.x * blockDim.x * 1 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    double *idata = g_idata + blockIdx.x * blockDim.x * 1;

    // unrolling 8
    /*
    if (idx + 1 * blockDim.x < n)
    {
        double a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1; // + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }
    */

    __syncthreads();

    // in-place reduction and complete unroll
    /*
    if (blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid + 512];
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256) idata[tid] += idata[tid + 256];
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid + 128];
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64) idata[tid] += idata[tid + 64];
    __syncthreads();
    */

    // unrolling warp
    if (tid < 32)
    {
        volatile double *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid +  8];
        vmem[tid] += vmem[tid +  4];
        vmem[tid] += vmem[tid +  2];
        vmem[tid] += vmem[tid +  1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
    //if (tid == 0) printf("summed %lf",g_odata[blockIdx.x]);
}

void sum64(double *d_real_2d, double *d_real_2dsummed){
  dim3 grid (64, 1, 1);
  dim3 block  (FFTP/ 2 / grid.x, 1, 1);
  reduceUnrollWarps1<<<block,grid>>>(d_real_2d, d_real_2dsummed);
}

//--------end of this code
