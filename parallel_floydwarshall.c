#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h> 
#include <time.h>
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>
#include <errno.h>

#define N 10                    // number of vertexes
#define MAX_WEIGHT 100          // max value of weight
#define ALPHA 0.25              // threshold for creating edges
#define INF (100*MAX_WEIGHT)    // infinite value


//#define DEBUG 
////////////////////////////////////////////////////////////
// init graph and data structures

void initGraph ( int G[N][N], int C[N][N], int D[N][N], int P[N][N], unsigned int seed ) {
  int i, j;
  //fprintf(stderr, "init start ... ");
  srand(seed);
  // init G and C
  for ( i=0; i < N; i++ ) {
    for ( j=0; j < N; j++ ) {
      G[i][j] = 0;
      C[i][j] = INF;
      D[i][j] = INF;
      P[i][j] = i;
    }
  }
  for ( i=0; i < N; i++ ) {
    int n_neighbour = (rand() % N) + 1; //random number between 1 .. N 
    j = 0;
    int count = 0;
    while ( count < n_neighbour )  {
      if ( j == i ) {
      	G[i][j] = 0;
      	C[i][j] = 0;
      	D[i][j] = 0;
      } else { 
        int alpha = rand();
      	if ( alpha < (ALPHA*INT_MAX) ) {
      	  // create edge
      	  int weight = (rand() % MAX_WEIGHT);
      	  G[i][j] = 1;
      	  C[i][j] = weight;
      	  D[i][j] = weight;
          count++;
      	}
      }
      j = (j+1) % N;
    }
    G[i][i] = 0;
    C[i][i] = 0;
    D[i][i] = 0;
  }
  //fprintf(stderr, "done !\n");
}

////////////////////////////////////////////////////////////
// Print ALL-Pair Shortest Paths

void printPath (int G[N][N], int P[N][N], int i, int j, FILE * fp) {
  if ( i != j ) {
    printPath (G, P, i, P[i][j], fp);
    G[P[i][j]][j] = 2;
  }
  fprintf (fp, " %d ", j);
}

void printAPSP ( int G[N][N], int C[N][N], int D[N][N], int P[N][N] ) {
  
  FILE * fp;
  int i, j;

  fp = fopen ("apsp-s.txt", "w+");

  for ( i=0; i<N; i++ ) {
    for ( j=0; j<N; j++ ) {

      fprintf(fp, "%d --> %d: ", i, j);
      printPath(G, P, i, j, fp);
      fprintf(fp, " \t\t new_cost=%d oldcost:%d\n", D[i][j], C[i][j]);

    }
  }

  fclose(fp);

}

////////////////////////////////////////////////////////////
// Floyd Warshall algorithm

void floydWarshall ( int G[N][N], int C[N][N], int P[N][N] ) {
  int i, j, k;
  //#pragma omp parallel 
  //{
    for ( k = 0; k < N; k++ ) 
      for ( i = 0; i < N; i++ ) 
        for ( j = 0; j < N; j++ ) 
          if ( C[i][j] > (C[i][k]+C[k][j]) ) {
            C[i][j] = C[i][k]+C[k][j];
            P[i][j] = P[k][j];
          }

}

////////////////////////////////////////////////////////////
// print matrix in text format

void printMat(int M[N][N], unsigned int seed, char * s) {

  FILE * fp;
  int i, j;

  time_t now;

  time(&now);
  
  fp = fopen (s, "w+");

  fprintf(fp, "%u\n", seed); 
  fprintf(fp, "%d\n", N); 

  for ( i=0; i<N; i++ ) {
    for ( j=0; j<N; j++ ) {
        if ( M[i][j] == INF ) fprintf(fp, "%4s ", "inf");
        else 
      	fprintf(fp, "% 4d ", M[i][j]);
    }
    fprintf(fp, "\n");
  }

  fprintf(fp, "#----------------------------------------\n");
  fprintf(fp, "#-- %s", ctime(&now)); 
  fprintf(fp, "#-- Matrix: %d x %d  Seed: %u\n", N, N, seed);

  fclose(fp);
}

////////////////////////////////////////////////////////////
int main(int argc, char * argv[]) {

  int (*G)[N]; // adjacent matrix  
  int (*C)[N]; // initial cost matrix  
  int (*D)[N]; // distant cost matrix used by floy-warshall procedure  
  int (*P)[N]; // predecessor matrix: P[i][j]=k --> k is a predecessor of j in the path i --> j
 
  unsigned int seed;

  struct timeval t[2];
  double dt;

  if ( argc > 1 ) {
    seed = strtoul(argv[1], 0L, 10);
  } else {
    seed = time(NULL)*getpid();
  }

  //fprintf(stderr, "Floyd-Warshall\n" );
  //fprintf(stderr, "N: %d\n",       N);
  //fprintf(stderr, "seed: %u\n", seed);

  if ( posix_memalign((void *)&G, 4096, N*N*sizeof(int)) != 0 ) {
    perror("ERROR: allocation of G FAILED:");
    exit(-1);
  }

  if ( posix_memalign((void *)&C, 4096, N*N*sizeof(int)) != 0 ) {
    perror("ERROR: allocation of G FAILED:");
    exit(-1);
  }

  if ( posix_memalign((void *)&D, 4096, N*N*sizeof(int)) != 0 ) {
    perror("ERROR: allocation of G FAILED:");
    exit(-1);
  }

  if ( posix_memalign((void *)&P, 4096, N*N*sizeof(int)) != 0 ) {
    perror("ERROR: allocation of G FAILED:");
    exit(-1);
  }

  initGraph(G, C, D, P, seed);

  printMat(C, seed, "matcost-s.in");

  gettimeofday (&t[0], NULL); 

  floydWarshall(G, D, P);

  gettimeofday (&t[1], NULL); 

  printMat(D, seed, "matcost-s.out");

#ifdef DEBUG
  printAPSP(G, C, D, P);
#endif

  //calcolo del tempo sottraendo t[1] a t[0]
  dt = (double)(t[1].tv_sec - t[0].tv_sec) + ((double)(t[1].tv_usec - t[0].tv_usec)*1.0e-6);

  fprintf(stderr, "Floyd-Warshall  N: %d  time: %.3f ms  seed: %u\n", N, dt*1.0e3, seed);
  
  exit(0);  
}
