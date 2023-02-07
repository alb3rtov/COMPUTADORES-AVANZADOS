#include <stdio.h>

#define N 600 //En mi caso N debe ser mayor a 1024 para que de un error

__global__ void VecAdd(int* DA, int* DB, int* DC)
{
	int i = threadIdx.x;
  DC[i] = DA[i] + DB[i];
}

int main()
{ int HA[N], HB[N], HC[N];
  int *DA, *DB, *DC;
  int i; int size = N*sizeof(int);
  
  cudaError_t errSync;
  cudaError_t errAsync;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // reservamos espacio en la memoria global del device
  
  cudaMalloc((void**)&DA, size);
  
  errAsync = cudaDeviceSynchronize();
  if (errAsync != cudaSuccess)
    printf("Async malloc error: %s\n", cudaGetErrorString(errAsync));

  cudaMalloc((void**)&DB, size);
  
  errAsync = cudaDeviceSynchronize();
  if (errAsync != cudaSuccess)
    printf("Async malloc error: %s\n", cudaGetErrorString(errAsync));

  cudaMalloc((void**)&DC, size);

  errAsync = cudaDeviceSynchronize();
  if (errAsync != cudaSuccess)
    printf("Async malloc error: %s\n", cudaGetErrorString(errAsync));

  // inicializamos HA y HB
  for (i=0; i<N; i++) {HA[i]=-i; HB[i] = 3*i;}
  
  // copiamos HA y HB del host a DA y DB en el device, respectivamente
  
  cudaMemcpy(DA, HA, size, cudaMemcpyHostToDevice);
  
  errSync = cudaGetLastError();
  if (errSync != cudaSuccess)
    printf("Sync memcpy error: %s\n", cudaGetErrorString(errSync));
  

  cudaMemcpy(DB, HB, size, cudaMemcpyHostToDevice);
    
  errSync = cudaGetLastError();
  if (errSync != cudaSuccess)
    printf("Sync memcpy error: %s\n", cudaGetErrorString(errSync));

  // llamamos al kernel (1 bloque de N hilos)
  cudaEventRecord(start);
  VecAdd <<<1, N>>>(DA, DB, DC);	// N hilos ejecutan el kernel en paralelo
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Total sesion time (ms): %f\n", milliseconds);

  //cudaEventRecord(start);
  // copiamos el resultado, que está en la memoria global del device, (DC) al host (a HC)
  cudaMemcpy(HC, DC, size, cudaMemcpyDeviceToHost);
  
  errSync = cudaGetLastError();

  if (errSync != cudaSuccess)
    printf("Sync memcpy error: %s\n", cudaGetErrorString(errSync));

  errSync = cudaGetLastError();
  errAsync = cudaDeviceSynchronize();

  if (errSync != cudaSuccess)
    printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
  if (errAsync != cudaSuccess)
    printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));

  // liberamos la memoria reservada en el device
  cudaFree(DA); 
  cudaFree(DB); 
  cudaFree(DC);

  // una vez que tenemos los resultados en el host, comprobamos que son correctos
  // esta comprobación debe quitarse una vez que el programa es correcto (p. ej., para medir el tiempo de ejecución)
  for (i = 0; i < N; i++) // printf("%d + %d = %d\n",HA[i],HB[i],HC[i]);
    if (HC[i]!= (HA[i]+HB[i])) 
		{printf("error en componente %d\n", i); break;}

  return 0;
} 
