/*
This code makes psrfits data from the data observed by Japanese telescpoe.
*/

#include "make_raw_psrfits_kuma.h"
#include "utils/utils_si.cuh"

__global__
void make_ho(cufftDoubleReal *ho){
  unsigned char i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned char wordl, wordr, alword;
  wordl = (i & 0xAA) >> 1;
  wordr = (i & 0x55) << 1;
  alword = wordl | wordr;
  alword = alword ^ 0x33;
  ho[4*i] = alword & 0x03;
  ho[4*i+1] = (alword >> 2) & 0x03;
  ho[4*i+2] = (alword >> 4) & 0x03;
  ho[4*i+3] = (alword >> 6) & 0x03;
  //printf("d_ho: %lf",ho[i]);
}

__global__
void calcmag_to_double(cufftDoubleComplex *data, double *mag, int bin_size){
      int idx = threadIdx.x + blockDim.x*blockIdx.x;
      mag[idx]   = data[idx].x * data[idx].x + data[idx].y * data[idx].y;
      //if (isnan(mag[idx])){printf("nan in calcmag");}
    }

__global__
void cp_array_cufftDoubleReal(cufftDoubleReal *in, cufftDoubleReal *out, int in_start, int out_start){
      int i = blockIdx.x*blockDim.x + threadIdx.x;
      out[out_start+i] = in[in_start+i];
    }

__global__
void cp_array_double(double *in, double *out, int in_start, int out_start){
          int i = blockIdx.x*blockDim.x + threadIdx.x;
          out[out_start+i] = in[in_start+i];
    }

    __global__
    void double_devide(double *in, long length){ //割りたい配列と割りたい値
      int i = blockIdx.x*blockDim.x + threadIdx.x;
      in[i] = in[i]/length;
    }

int main(){

  cudaError_t err;

  //for time measure
  double t_start,t_end;
  t_start = cputimeinsec();
  clock_t start_clock_all = clock();
  time_t start_time_all, end_time_all;//for CPU time
  start_time_all = time(NULL);
  //for cpu
  int time_read, time_ave, time_out;
  time_read = 0; time_ave = 0; time_out = 0;
  time_t time_read_s, time_read_e, time_ave_s, time_ave_e, time_out_s, time_out_e;
  //for cudaMemcpy
  double time_s[1], time_e[1], time_res[1];
  time_res[0]=0.0;
  double time_arranging_bit = 0; double time_fread = 0;
  float time_arrange_bit_bycudaE = 0.0;
  float h_time_memcpy = 0.0; float h_time_memcpy_a = 0.0;

  //file pointer
  FILE *fp,*fp_out;
  printf("\n----------------------------\n");

  const char *readname = INPUT_RAWDATA;
  //sprintf(filename,"%s",inputname);
  if ((fp = fopen(readname, "rb")) == NULL){
    //  if ((fp = fopen("CRAB_2009222233800-233802.raw", "rb")) == NULL){
    printf("file open error!!\n");
    exit(EXIT_FAILURE);
  }
  printf("read data from %s \n" , INPUT_RAWDATA);

  long n_move;

  const char *writename = OUTPUT_DATA;
  if ((fp_out = fopen(writename, "wb")) == NULL){
          printf("file open error!!\n");
          exit(EXIT_FAILURE);
        }


  //size of input data
  fseek(fp,0,SEEK_END);
  long long length = ftell(fp);
  printf("Length in byte  = %lld\n", length);
  int totRp = length * 8/INPUT_BITSIZE / SR;
  printf("Observation period: %d seconds\n", totRp);
  rewind(fp);
  int loop_max = length * 8/INPUT_BITSIZE / N_dt;

  int fft_max = N_dt / FFTP;
  double dt = double(FFTP)/SR;
  printf("dt %lf\n",dt);

  dim3 block (SET_BLOCK, 1, 1); dim3 grid  (FFTP /2 / block.x, 1, 1);
  dim3 grid_ini  (FFTP / block.x, 1, 1);

  cufftHandle plan_f;
  cufftPlan1d(&plan_f, FFTP, CUFFT_D2Z, 1);

  unsigned char *raw_pre;
  raw_pre = (unsigned char*) malloc(sizeof(unsigned char) * N_dt * INPUT_BITSIZE/8);
  if(raw_pre == NULL) {
    printf("memory cannot be allocated!!\n");
    exit(EXIT_FAILURE);
  }
  double *res_host, *freq;
    res_host = (double*) malloc(sizeof(double) * FFTP /2/64);
    if(res_host == NULL) {
      printf("memory cannot be allocated in res_host!!\n");
      exit(EXIT_FAILURE);
    }
    freq = (double*) malloc(sizeof(double) * FFTP /2/64);
    if(freq == NULL) {
      printf("memory cannot be allocated in res_host!!\n");
      exit(EXIT_FAILURE);
    }
  /*
  for(int i=0; i<FFTP /2/64;i++){
    freq[i] = FREQ_DATA_LOW + (FREQ_DATA_WIDTH/(FFTP /2/64))*i; //MHz
  }
  */
  for(int i=0; i<FFTP /2/64;i++){
    freq[i] = FREQ_DATA_LOW + FREQ_DATA_WIDTH - (FREQ_DATA_WIDTH/(FFTP /2/64))*i; //MHz
  }
  double *d_real_out, *d_real_2d, *d_real_2dsummed, *d_real_out_sum;
    err = cudaMalloc((void**)&d_real_2d, sizeof(double)*N_dt/2);
      if (err != cudaSuccess) {printf("error in cudamalloc in d_real_2d\n"); exit(err);}
    err = cudaMalloc((void**)&d_real_out, sizeof(double)*FFTP/2);
      if (err != cudaSuccess) {
        printf("error in cudamalloc in d_real_out\n");
        exit(err);
      }
      err = cudaMalloc((void**)&d_real_out_sum, sizeof(double)*FFTP/2/64);
        if (err != cudaSuccess) {
          printf("error in cudamalloc in d_real_sum_out\n");
          exit(err);
        }
    err = cudaMalloc((void**)&d_real_2dsummed, sizeof(double)*N_dt/2/64);
      if (err != cudaSuccess) {
        printf("error in cudamalloc in d_real_2dsummed\n");
        exit(err);
    }

  cufftDoubleReal *d_real, *d_real_ini;
  cufftDoubleComplex *d_cplx;
    err = cudaMalloc((void**)&d_real, sizeof(cufftDoubleReal)*N_dt);
      if (err != cudaSuccess) {printf("error in cudamalloc in d_real\n"); exit(err);}
    err = cudaMalloc((void**)&d_cplx, sizeof(cufftDoubleComplex)*FFTP);
    if (err != cudaSuccess) {
      printf("error in cudamalloc in d_cplx\n");
      exit(err);
    }
    err = cudaMalloc((void**)&d_real_ini, sizeof(cufftDoubleReal)*FFTP);
    if (err != cudaSuccess) {
      printf("error in cudamalloc in d_real_ini\n");
      exit(err);
    }
  cufftDoubleReal *ho;
  cudaMalloc((void**)&ho, sizeof(cufftDoubleReal)*1024);
  make_ho<<<1,256>>>(ho);

  //roop for larger data
  for(int tt=0;tt<loop_max;tt++){
    if(tt==0){n_move = long(sizeof(unsigned char)*N_dt*tt*INPUT_BITSIZE/8);}
    //else{n_move = long(sizeof(unsigned char)*N_dt*tt-i_d_tp);}
    else{n_move = long(sizeof(unsigned char)*(N_dt)*tt*INPUT_BITSIZE/8);}
    printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
    printf("roop of %d %ld\n",tt,n_move);
    fseek(fp, n_move, SEEK_SET); //move position of pointer by n_move byte from start of the file

    int e;
    time_read_s = time(NULL);
      if(observationdata == "Ibaraki"){
      //start for Hitachi or Takahagi data
      e = read_Ibaraki_Y_to_d(raw_pre, d_real, &fp, ho, &time_arranging_bit,&time_fread);
      }
      else if(observationdata == "Kashima"){
      e = read_Kashima_to_d(raw_pre, d_real, &fp, N_dt);
      }
      else{
      printf("!!!error in reading data!!!!\n");
      break;
      }
      if(!(e == 0)){
      printf("data is shorter than N_dt so, stop this roop...\n");
      exit(0);
      break;
      }
      time_read_e = time(NULL);
      time_read += (time_read_e - time_read_s);


    //-----------------start
    time_s[0] = cputimeinsec();
    for(int i=0;i<fft_max;i++){ //i<8000
      cp_array_cufftDoubleReal<<<grid_ini, block>>>(d_real, d_real_ini, i*FFTP, 0);
      cufftExecD2Z(plan_f, d_real_ini, d_cplx);
      calcmag_to_double<<<grid, block>>>(d_cplx, d_real_out, FFTP/2);
      //cp_array_double<<<grid, block>>>(d_real_out, d_real_2d, 0, i*FFTP/2);
      time_ave_s = time(NULL);
      sum64(d_real_out,d_real_out_sum);
      time_ave_e = time(NULL);
      time_ave += (time_ave_e - time_ave_s);

      cudaMemcpy(res_host, d_real_out_sum, sizeof(double)*(FFTP/128), cudaMemcpyDeviceToHost);
      //printf("%lf",res_host[0]);
      time_out_s = time(NULL);
      
      for (long j=0;j<FFTP/128;j++){
     // fwrite(res_host, sizeof(double), FFTP/128, fp_out); // -> this seems to work well
      //fprintf(fp_out, "%hhu ", (unsigned char)(res_host[FFTP/128-j+1]));
      //fprintf(fp_out, "%f ", (float)(res_host[FFTP/128-j+1]));

      //fprintf(fp_out, "%e\t%lf\t%f\n", i*dt+tt*((double)N_dt/SR),freq[j],(float)(res_host[FFTP/128-j-1])); //for test
      fprintf(fp_out, "%f\n", (float)(res_host[FFTP/128-j-1]));

      //fprintf(fp_out, "%e\t%lf\t%hhu\n", i*dt+tt*((double)N_dt/SR),freq[j],(unsigned char)(res_host[j])); // unsigned char seems bad
      //fprintf(fp_out, "%e\t%lf\t%lf\n", i*dt+tt*((double)N_dt/SR),freq[j],res_host[j]);
      }

    }
    time_e[0] = cputimeinsec();
    time_res[0] += (time_e[0] - time_s[0]);


    }//tt
    cudaFree(d_real_out); free(res_host); free(freq); cudaFree(ho); cudaFree(d_real_out_sum);
    cufftDestroy(plan_f);
    fclose(fp); fclose(fp_out);


  clock_t end_clock_fwrite = clock();
  //printf("fwrite clock_t %2.5f sec\n",(double)(end_clock_fwrite - start_clock_fwrite) / CLOCKS_PER_SEC);
  clock_t end_clock_all = clock();
  end_time_all = time(NULL);


  printf("\n----------------------------\n");
  printf("summary\n");
  printf("in total %2.5f sec\n",(double)(end_clock_all - start_clock_all) / CLOCKS_PER_SEC);

  printf(
    "time total:%ld [s]\n",
    end_time_all - start_time_all);

  t_end = cputimeinsec();
  printf("%2.5f sec\n",t_end-t_start);
  char timefile[256];
  sprintf(timefile, "../crabGRP/time/calculation_time_u21339a4_ave500nsbyGPU.dat");
  FILE *fp_time;
  if ((fp_time = fopen(timefile, "wb")) == NULL){
         printf("file open error!!\n");
         exit(EXIT_FAILURE);
       }
  fprintf(fp_time, "in total %2.5f sec\ntime total:%ld [s]\noverall time:%2.5f sec\n",(double)(end_clock_all - start_clock_all) / CLOCKS_PER_SEC, end_time_all - start_time_all, t_end-t_start);
  fprintf(fp_time, "reading time : %d s\n", time_read);
  //fprintf(fp_time, "averaging + output time : %d s\n", time_aveout);
  fprintf(fp_time, "arranging bit time : %lf s\n", time_arranging_bit);
  fprintf(fp_time, "arranging bit time measured by cudaEvent: %e s\n", time_arrange_bit_bycudaE);
  fprintf(fp_time, "fread time : %lf s\n", time_fread);
  fprintf(fp_time, "averaging time : %d s\n", time_ave);
  fprintf(fp_time, "output time : %d s\n", time_out);
  fprintf(fp_time, "cudaMemcpy time : %e\n", h_time_memcpy+h_time_memcpy_a);
  fprintf(fp_time, "FFT + mag time : %e\n", time_res[0]);
  fclose(fp_time);

  char paramfile[256];
  sprintf(paramfile, "u21339a4_ave500nsbyGPU.param");
  FILE *fp_param;
  if ((fp_param = fopen(paramfile, "wb")) == NULL){
         printf("file open error!!\n");
         exit(EXIT_FAILURE);
       }
  fprintf(fp_param, "Input data : %s\noutput data : %s\n",INPUT_RAWDATA, OUTPUT_DATA);
  fprintf(fp_param, "bitsize : %d\ntelescope : %s\nSR : %lld", INPUT_BITSIZE, observationdata,SR);
  //fprintf(fp_time, "averaging + output time : %d s\n", time_aveout);
  fprintf(fp_param, "freq width : %f\n", FREQ_DATA_WIDTH);
  fprintf(fp_param, "\n------------------------------------\n");
  fprintf(fp_param, "ch number(chnum) : %d\n ", (FFTP/2/64));
  fprintf(fp_param, "lowest freq : %f\nfreq width : %f\n",FREQ_DATA_LOW, FREQ_DATA_WIDTH);
  fprintf(fp_param, "dt(tsamp) : %e[s]\n ", (double)(FFTP)/SR);
  fprintf(fp_param, "dnu(foff): %f\n[MHz]", (double)(FREQ_DATA_WIDTH)/(FFTP/2/64));
  fclose(fp_param);
}





//--------end of this code
