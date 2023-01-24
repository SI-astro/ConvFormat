/*
Code from a text book CUDA professonal programming.
I have no idea about the lisince, so DO NOT DISTRIBUTE THE CODE!!
I'll write better one later.
*/

#include "utils_si.cuh"

// int main(int argc, char **argv){
// 
//   printf("%s Starting...\n", argv[0]);
//   checkDeviceinfo();
// 
// }



double cputimeinsec(){
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.0e-6);
  
}

