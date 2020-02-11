#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <openacc.h>

// Variabili utilizzate per iterare il calcolo su matrici di dimensioni diverse
#define N_START 500
#define N_END 5000
#define N_STEP 500
//in questo caso calcola le matrici 500*500, 1000*1000, 1500*1500 ... 4500*4500, 5000*5000

#define MAX_WEIGHT 100.0 // Massimo peso degli archi
#define INFINITY (MAX_WEIGHT * 100.0) // Valore usato per un non-arco(se 2 nodi non si toccano)
#define ALPHA 0.8 // percentuale pienezza grafico (% degli archi)

/* #define PRINT_TERMINAL */

int initMatrice(int n, float *A, unsigned int seed);
int stampaTerminale(int n, float *A);

int main(int argc, char *argv[]) {
	int N = 0;
	unsigned int seed = 0;
	float *A = NULL;
	int i = 0, j = 0, k = 0;
	float temp = 0.0;

	// Variabili usate per calcolare il tempo effettivo
	struct timespec start, end;
	unsigned long int time_elapsed = 0;
	unsigned long int total_time_elapsed = 0;

	// MPI processes variables
	int rank = 0; //ogni processo ha il proprio rank
	int size = 0; //numero di processi
	int rowSize = 0; //Numero di righe di cui il processo si deve occupare
	int rowOffset = 0; //Offset di riga del blocco assegnato a ogni processo
	float *rowBlock = NULL; // Spazio di allocazione per un pezzo di matrice
	float *kRow = NULL; // k-esima riga condivisa in broadcast durante la k-esima iterazione
	int kRowProcessId = 0; // ID del processo che ha la k-esima riga durante l'iterazione k

	// Values used for dividing the matrix rows between a non-divisible processes number (MPI_Scatterv)
	int *blockSizes = NULL, *blockOffsets = NULL;
	int blockSize = 0, blockOffset = 0;

	// Seed opzionale come secondo argomento
	if (argc == 2) {
		seed = atoi(argv[1]);
	}

	// Inizio sezione parallela MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Setting the device number
	acc_set_device_num(rank % 16, acc_device_nvidia);

	if (rank == 0) {
		printf("size\tavg time (s)\t%d processes\n", size);
	}

	// Iteration with increasing graph sizes
	for (N = N_START; N <= N_END; N += N_STEP) {
		if (rank == 0) {
			if (argc != 2) {
				seed = time(NULL);
			}
			A = (float*) malloc(sizeof(float) * N * N);
			if (initMatrice(N, A, seed)) {
				fprintf(stderr, "Error in graph initialization\n");
			}

#ifdef PRINT_TERMINAL
			if (stampaTerminale(N, A)) {
				fprintf(stderr, "Error in terminal printing\n");
			}
#endif
		}

		// Da questo punto in poi, tutti i processi iniziano a calcolare FloydWarshall in parallelo
		MPI_Barrier(MPI_COMM_WORLD);

		clock_gettime(CLOCK_REALTIME, &start);

		// Ottiene le dimensioni e offset per i blocchi di ciascun processo
		rowSize = N / size;
		if (rank < N % size) { //se N non è multiplo di size qualche processo rimane fuori
			rowSize++;	//viene associata una riga di resto ai processi più piccoli(da 0 a resto di N%size)
			rowOffset = rank * rowSize;	//posizione di partenza di ciascun processo
		} else {
			rowOffset = rank * rowSize + (N % size); //posizione di partenza di ciascun processo
		}
		blockSize = rowSize * N; //blocco di cui si dovrà occupare ciascun processo(numero di righe*numero di colonne)
		blockOffset = rowOffset * N; //non so a cosa possa servire

		// Il processo principale deve ottenere tutte le dimensioni e offset per l'esecuzione di MPI_Scatterv
		if (rank == 0) {
			blockSizes = (int*) malloc(sizeof(int) * size);
			blockOffsets = (int*) malloc(sizeof(int) * size);
		}
		MPI_Gather(&blockSize, 1, MPI_INT, blockSizes, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Gather(&blockOffset, 1, MPI_INT, blockOffsets, 1, MPI_INT, 0, MPI_COMM_WORLD);

		// Divisione della matrice del grafico in blocchi tra tutti i processi in ordine di riga
		rowBlock = (float*) malloc(sizeof(float) * blockSize);
		MPI_Scatterv(A, blockSizes, blockOffsets, MPI_FLOAT, rowBlock, blockSize, MPI_FLOAT, 0, MPI_COMM_WORLD);

		// pulizia della zona di memoria non più utilizzata
		if (rank == 0) {
			free(A);
			A = NULL;
		}

		// Allocation of space for the shared k-row
		kRow = (float*) malloc(sizeof(float) * N);

		#pragma acc data copy(rowBlock[0:blockSize]) //copy:Alloca memoria sulla GPU e copia i dati dall'host alla GPU quando si accede alla regione e copia i dati all'host quando si esce dalla regione
		{
			for (k = 0; k < N; k++) {
				// The process that has the k-row copy it into kRow[] and broadcast it
				if (k >= rowOffset && k < rowOffset + rowSize) {
					#pragma acc parallel loop copyout(kRow[0:N])	//copyout:Alloca memoria sulla GPU e copia i dati sull'host quando si esce dalla regione
					for (j = 0; j < N; j++) {
						kRow[j] = rowBlock[(k - rowOffset) * N + j];//Per via della rappresentazione monoriga delle matrici 
					}
				}
				//Tutti i processi devono calcolare quale ha la riga k-esima
				//perchè in bcast serve la provenienza, quindi vogliono sapere chi gliela manda
				if (k < (N % size) * (N / size + 1)) {
					kRowProcessId = k / (N / size + 1);
				} else {
					kRowProcessId = (N % size) + (k - (N % size) * (N / size + 1)) / (N / size);
				}
				MPI_Bcast(kRow, N, MPI_INT, kRowProcessId, MPI_COMM_WORLD);

				// Algoritmo FloydWarshall per un blocco di righe
				#pragma acc parallel loop collapse(2) copyin(kRow[0:N]) //copyin: Alloca memoria sulla GPU e copia i dati dall'host alla GPU quando si accede regione
				for (i = 0; i < rowSize; i++) {
					for (j = 0; j < N; j++) {
						temp = kRow[j] + rowBlock[i * N + k];
						if (rowBlock[i * N + j] >= INFINITY || (temp < INFINITY && rowBlock[i * N + j] > temp)) {
							rowBlock[i * N + j] = temp;
						}
					}
				}
			}
		} // Fine della regione acc

		// liberazione memoria inutilizzata
		free(kRow);
		kRow = NULL;

		// Gather della matrice
		if (rank == 0) {
			A = (float*) malloc(sizeof(float) * N * N);
		}
		MPI_Gatherv(rowBlock, blockSize, MPI_FLOAT, A, blockSizes, blockOffsets, MPI_FLOAT, 0, MPI_COMM_WORLD);
		//gatherv perchè lasci un blockOffsets di spazio vuoto in quanto non occupano tutti lo stesso spazio

		// liberazione memoria inutilizzata
		free(rowBlock);
		rowBlock = NULL;
		if (rank == 0) {
			free(blockSizes);
			blockSizes = NULL;
			free(blockOffsets);
			blockOffsets = NULL;
		}

		clock_gettime(CLOCK_REALTIME, &end);

		if (rank == 0) {

#ifdef PRINT_TERMINAL
			if (stampaTerminale(N, A)) {
				fprintf(stderr, "Error in terminal printing\n");
			}
#endif

			free(A);
			A = NULL;
		}

		// Gathering the time elapsed
		time_elapsed =
			(long int) (end.tv_sec % 10000 * 1000000000 + end.tv_nsec) -
			(long int) (start.tv_sec % 10000 * 1000000000 + start.tv_nsec);
		MPI_Reduce(&time_elapsed, &total_time_elapsed, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
		
		// Output of execution times
		if (rank == 0) {
			printf("%d\t%5.3f\n", N, (double) total_time_elapsed / CLOCKS_PER_SEC / size / 1000);
		}
	} // End of N_START -> N_END

	// End of parallel section
	MPI_Finalize();

	return 0;
}

/**
 * Inizializzazione matrice con pesi pseudocasuali
 * con ALPHA percentuale di pienezza
 * 
 * @param n{int} number of vertices of the graph
 * @param A{float*} graph matrix (row-wise representation)
 * @param seed{unsigned int} seed for the pseudorandom number generator
 * @return {int}
 */
int initMatrice(int n, float *A, unsigned int seed) {
	if (n <= 0 || A == NULL) {
		return -1;
	}
	int i = 0, j = 0;
	int arc_number = ALPHA * (n - 1); // Number of links to be generated per vertex
	int count = 0;
	srand(seed);
	// Filling with INFINITY and 0.0 on the main diagonal
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			A[i * n + j] = INFINITY;
		}
		A[i * n + i] = 0.0;
	}
	// Adding random weights
	for (i = 0; i < n; i++) {
		count = 0;
		while (count < arc_number) {
			j = rand() % n;
			if (j != i && A[i * n + j] == INFINITY) {
				// Create edge
				A[i * n + j] = (float) rand() / RAND_MAX * MAX_WEIGHT;
				count++;
			}
		}
	}
	return 0;
}

/**
 * Prints a graph matrix to terminal
 * 
 * @param n{int} number of vertices of the graph
 * @param A{float*} graph matrix (row-wise representation)
 * @return {int}
 */
int stampaTerminale(int n, float *A) {
	if (n <= 0 || A == NULL) {
		return -1;
	}
	int i = 0, j = 0;
	float value = 0.0;
	// Avoid too verbose outputs
	if (n > (1 << 4)) {
		printf("Graph %d x %d, too big to display\n", n, n);
		return 0;
	}
	printf("@-");
	for (i = 0; i < n; i++) {
		printf("--------");
	}
	printf("@\n");
	for (i = 0; i < n; i++) {
		printf("| ");
		for (j = 0; j < n; j++) {
			value = A[i * n + j];
			if (value < INFINITY) {
				printf("%7.3f ", value);
			} else {
				printf("  ---   ");
			}
		}
		printf("|\n");
	}
	printf("@-");
	for (i = 0; i < n; i++) {
		printf("--------");
	}
	printf("@\n");
	return 0;
}
