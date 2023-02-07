#include <stdio.h>

#define tb 512
#define N 65535*tb
// tamaño bloque

__global__ void VecAdd(int* DA, int* DB, int* DC)
{
  int id = ((blockIdx.x * blockDim.x) + threadIdx.x) * tb;

  for (int i = 0; i < tb; i++) {
    if (id < N) {
      DC[id + i] = DA[id + i] + DB[id + i];
      }
  }
}

int main()
{ cudaFree(0);
  int *DA, *DB, *DC;
  int *HA = new int[N], *HB = new int[N], *HC = new int[N];
  int i; int size = N*sizeof(int);
  
  // reservamos espacio en la memoria global del device
  cudaError_t testerr;
  testerr = cudaMalloc((void**)&DA, size);
  if (testerr!= cudaSuccess) {
	printf("Error en cudaMalloc DA: %s\n", cudaGetErrorString(testerr));
	exit(0);
  }	
  testerr = cudaMalloc((void**)&DB, size);
  if (testerr!= cudaSuccess) {
	printf("Error en cudaMalloc DB: %s\n", cudaGetErrorString(testerr));
	exit(0);
  }		
  testerr = cudaMalloc((void**)&DC, size);
  if (testerr!= cudaSuccess) {
	printf("Error en cudaMalloc DC: %s\n", cudaGetErrorString(testerr));		
	exit(0);
  }
    
  // inicializamos HA y HB
  for (i=0; i<N; i++) {HA[i]=-i; HB[i] = 3*i;}
  
  // copiamos HA y HB del host a DA y DB en el device, respectivamente
  testerr = cudaMemcpy(DA, HA, size, cudaMemcpyHostToDevice);
  if (testerr != cudaSuccess) {
	printf("Error en cudaMemcpy del host al device: %s\n", cudaGetErrorString(testerr));		
	exit(0);
  }
  testerr = cudaMemcpy(DB, HB, size, cudaMemcpyHostToDevice);
  if (testerr != cudaSuccess) {
	printf("Error en cudaMemcpy del host al device: %s\n", cudaGetErrorString(testerr));		
	exit(0);
  }      
      

  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, 0);
  int maxThreads = devProp.maxThreadsPerBlock;
  int numBlocks = (N / (maxThreads * tb)) + 1;

  dim3 dimGrid(numBlocks);
  dim3 dimBlock(maxThreads);
  
  VecAdd <<<dimGrid, dimBlock>>>(DA, DB, DC);	// N o más hilos ejecutan el kernel en paralelo
  testerr = cudaGetLastError();
  if (testerr!= cudaSuccess) {
    printf("Error al ejecutar el kernel: %s\n", cudaGetErrorString(testerr));
	exit(0);
  }    
  
  // copiamos el resultado, que está en la memoria global del device, (DC) al host (a HC)
  testerr = cudaMemcpy(HC, DC, size, cudaMemcpyDeviceToHost);
  if (testerr != cudaSuccess) {
	printf("Error en cudaMemcpy del device al host: %s\n", cudaGetErrorString(testerr));		
	exit(0);  
  }
   
  // liberamos la memoria reservada en el device
  cudaFree(DA); cudaFree(DB); cudaFree(DC);  
  
  // una vez que tenemos los resultados en el host, comprobamos que son correctos
  for (i = 0; i < N; i++) // printf("%d + %d = %d\n",HA[i],HB[i],HC[i]);
    if (HC[i]!= (HA[i]+HB[i])) 
		{printf("error en componente %d\n", i); break;}
  
  return 0;
} 
