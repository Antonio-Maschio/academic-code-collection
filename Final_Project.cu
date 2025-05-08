#include <stdio.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
#include <bits/stdc++.h>
#include <curand.h>
#include <curand_kernel.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/**
 * Ising Model Monte Carlo Simulation using CUDA
 * 
 * This implementation uses a checkerboard (black/white) decomposition approach for
 * efficient parallel simulation of the 2D Ising model on a square lattice.
 * The algorithm follows a Metropolis Monte Carlo scheme with periodic boundary conditions.
 */

// Error handling helper
static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

// Simulation parameters
const int LX = 1024;       // Lattice width
const int LY = 1024;       // Lattice height
const int ITERATIONS = 100 * 1024;  // Number of Monte Carlo sweeps
const int BLOCK_SIZE_X = 16;    // Thread block size X - tuned for modern GPUs
const int BLOCK_SIZE_Y = 16;    // Thread block size Y - tuned for modern GPUs
const float DEFAULT_BETA = 0.2f; // Default inverse temperature (J/kT)

// Constant memory for frequently accessed parameters
__device__ __constant__ float d_J = 1.0f;     // Coupling constant
__device__ __constant__ int d_LX = LX;        // Device copy of lattice width
__device__ __constant__ int d_LY = LY;        // Device copy of lattice height
__device__ __constant__ int d_SubL = LX / 2;  // Half the lattice size (for checkerboard)

// Global variable for temperature
__device__ float d_beta = DEFAULT_BETA;

// Host arrays
int *h_lattice;         // Full lattice
int *h_lattice_black;   // Black (even) sites
int *h_lattice_white;   // White (odd) sites

// Device arrays
int *d_lattice;         // Full lattice (for initialization)
int *d_lattice_black;   // Black sites on device
int *d_lattice_white;   // White sites on device

// Random Number Generator states for black and white sites
curandState *d_rng_states;

// Function prototypes
void initialize_lattice(int *lattice, bool random_init = false);
void decompose_lattice(int *lattice, int *even_sites, int *odd_sites, int size);
int calculate_magnetization(int *black_sites, int *white_sites, int size);
void save_lattice_vtk(const char* filename, int *black_sites, int *white_sites);

// CUDA kernel declarations
__global__ void init_rng(curandState *states, unsigned long long seed);
__global__ void update_black_sites(int *black_sites, int *white_sites, curandState *rng_states, float beta);
__global__ void update_white_sites(int *black_sites, int *white_sites, curandState *rng_states, float beta);
__global__ void set_temperature(float beta);

int main(int argc, char **argv) {
    float beta = DEFAULT_BETA;
    bool random_init = false;
    bool save_vtk = false;
    int n_save_files = 20;
    int savesteps = ITERATIONS/n_save_files;
    
    // Parse command line arguments if provided
    if (argc > 1) {
        beta = atof(argv[1]);
        if (argc > 2) {
            random_init = (atoi(argv[2]) != 0);
            if (argc > 3) {
                save_vtk = (atoi(argv[3]) != 0);
            }
        }
    }
    
    printf("Ising Model Simulation on %dx%d lattice\n", LX, LY);
    printf("Temperature: %f (beta: %f)\n", 1.0f/beta, beta);
    printf("Initialization: %s\n", random_init ? "Random" : "All Spins Up");
    
    // Allocate host memory
    size_t full_size = LX * LY;
    size_t half_size = full_size / 2;
    
    h_lattice = (int *)malloc(full_size * sizeof(int));
    h_lattice_black = (int *)malloc(half_size * sizeof(int));
    h_lattice_white = (int *)malloc(half_size * sizeof(int));
    
    // Initialize host lattice
    initialize_lattice(h_lattice, random_init);
    decompose_lattice(h_lattice, h_lattice_black, h_lattice_white, full_size);
    
    // Allocate device memory
    HANDLE_ERROR(cudaMalloc(&d_lattice, full_size * sizeof(int)));
    HANDLE_ERROR(cudaMalloc(&d_lattice_black, half_size * sizeof(int)));
    HANDLE_ERROR(cudaMalloc(&d_lattice_white, half_size * sizeof(int)));
    HANDLE_ERROR(cudaMalloc(&d_rng_states, half_size * sizeof(curandState)));
    
    // Copy data to device
    HANDLE_ERROR(cudaMemcpy(d_lattice, h_lattice, full_size * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_lattice_black, h_lattice_black, half_size * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_lattice_white, h_lattice_white, half_size * sizeof(int), cudaMemcpyHostToDevice));
    
    // Set up grid and blocks
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid((LX/2 + block.x - 1) / block.x, (LY + block.y - 1) / block.y);
    
    printf("Grid dimensions: %dx%d\n", grid.x, grid.y);
    printf("Block dimensions: %dx%d\n", block.x, block.y);
    
    // Verify that grid dimensions are within device limits
    cudaDeviceProp deviceProp;
    HANDLE_ERROR(cudaGetDeviceProperties(&deviceProp, 0));
    
    if (grid.x > deviceProp.maxGridSize[0] || grid.y > deviceProp.maxGridSize[1]) {
        printf("Error: Grid dimensions exceed device capabilities\n");
        exit(EXIT_FAILURE);
    }
    
    // Initialize RNG states (only once)
    init_rng<<<grid, block>>>(d_rng_states, time(NULL));
    HANDLE_ERROR(cudaDeviceSynchronize());
    
    // Set temperature
    set_temperature<<<1, 1>>>(beta);
    HANDLE_ERROR(cudaDeviceSynchronize());

    save_lattice_vtk("out/ising_step_0.vtk", h_lattice_black, h_lattice_white);

    // Performance measurement
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Main simulation loop
    for (int step = 1; step <= ITERATIONS; step++) {

        // Update black sites then white sites
        update_black_sites<<<grid, block>>>(d_lattice_black, d_lattice_white, d_rng_states, beta);
        HANDLE_ERROR(cudaDeviceSynchronize());
        
        update_white_sites<<<grid, block>>>(d_lattice_black, d_lattice_white, d_rng_states, beta);
        HANDLE_ERROR(cudaDeviceSynchronize());
        
        // Save step, turn off to have performance metrics
        if (save_vtk && step % savesteps  == 0) {
            // Copy data back to host
            HANDLE_ERROR(cudaMemcpy(h_lattice_black, d_lattice_black, half_size * sizeof(int), cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaMemcpy(h_lattice_white, d_lattice_white, half_size * sizeof(int), cudaMemcpyDeviceToHost));
            
            char filename[100];
            sprintf(filename, "out/ising_step_%06d.vtk", step);
            save_lattice_vtk(filename, h_lattice_black, h_lattice_white);
        }
    }
    
    // Copy final state back to host
    HANDLE_ERROR(cudaMemcpy(h_lattice_black, d_lattice_black, half_size * sizeof(int), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(h_lattice_white, d_lattice_white, half_size * sizeof(int), cudaMemcpyDeviceToHost));
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    
    // Calculate and report performance
    float seconds = duration / 1000000.0f;
    float mlups = (ITERATIONS * static_cast<float>(LX * LY) / seconds) / 1000000.0f;
    
    int magnetization = calculate_magnetization(h_lattice_black, h_lattice_white, half_size);
    
    printf("Simulation complete\n");
    printf("Elapsed time: %.4f seconds\n", seconds);
    printf("Performance: %.2f MLUPS (Million Lattice Updates Per Second)\n", mlups);
    printf("Final magnetization: %d (%.4f per site)\n", 
           magnetization, static_cast<float>(magnetization) / (LX * LY));
    
    // Free memory
    free(h_lattice);
    free(h_lattice_black);
    free(h_lattice_white);
    
    cudaFree(d_lattice);
    cudaFree(d_lattice_black);
    cudaFree(d_lattice_white);
    cudaFree(d_rng_states);
    
    return 0;
}

/**
 * Initialize the lattice with all spins up or random spins
 */
void initialize_lattice(int *lattice, bool random_init) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, 1);
    
    for (int i = 0; i < LX; i++) {
        for (int j = 0; j < LY; j++) {
            if (random_init) {
                // Random initialization: -1 or 1
                lattice[i + j * LX] = (dist(gen) * 2) - 1;
            } else {
                // All spins up
                lattice[i + j * LX] = 1;
            }
        }
    }
}

/**
 * Decompose the lattice into black (even) and white (odd) sites
 */
void decompose_lattice(int *lattice, int *black_sites, int *white_sites, int size) {
    int black_count = 0, white_count = 0;
    
    for (int i = 0; i < LX; ++i) {
        for (int j = 0; j < LY; ++j) {
            if ((i + j) % 2 == 0) {
                // Even sites (black)
                black_sites[black_count++] = lattice[j + i * LX];
            } else {
                // Odd sites (white)
                white_sites[white_count++] = lattice[j + i * LX];
            }
        }
    }
}

/**
 * Calculate the total magnetization of the lattice
 */
int calculate_magnetization(int *black_sites, int *white_sites, int half_size) {
    int sum = 0;
    
    for (int i = 0; i < half_size; i++) {
        sum += black_sites[i] + white_sites[i];
    }
    
    return sum;
}

/**
 * Initialize random number generator states
 */
__global__ void init_rng(curandState *states, unsigned long long seed) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x >= d_SubL || y >= d_LY)
        return;
        
    int id = x + y * d_SubL;
    
    // Initialize RNG with unique seed per thread
    curand_init(seed, id, 0, &states[id]);
}

/**
 * Set the temperature parameter (beta = 1/kT)
 */
__global__ void set_temperature(float beta) {
    // Only one thread needs to execute this
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_beta = beta;
    }
}

/**
 * Update the black (even) sites
 * Uses shared memory to reduce global memory accesses
 */
__global__ void update_black_sites(int *black_sites, int *white_sites, curandState *rng_states, float beta) {
    __shared__ int shared_white[BLOCK_SIZE_X+2][BLOCK_SIZE_Y+2];
    
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x >= d_SubL || y >= d_LY)
        return;
        
    int id = x + y * d_SubL;
    
    // Load local spin
    int spin = black_sites[id];
    
    // Local thread indexes for shared memory
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;
    
    // Load neighboring white spins into shared memory
    // This pattern handles the periodic boundary conditions
    
    // Current position neighbors
    int x_right = (x + 1) % d_SubL;
    int y_up = (y + 1) % d_LY;
    int y_down = (y - 1 + d_LY) % d_LY;
    
    shared_white[tx][ty] = white_sites[id];
    shared_white[tx+1][ty] = white_sites[x_right + y * d_SubL];
    shared_white[tx][ty+1] = white_sites[x + y_up * d_SubL];
    shared_white[tx][ty-1] = white_sites[x + y_down * d_SubL];
    
    // Make sure all threads have loaded data
    __syncthreads();
    
    // Calculate energy change for spin flip
    int deltaE = 2 * d_J * spin * (
        shared_white[tx][ty] +    // Current position
        shared_white[tx+1][ty] +  // Right
        shared_white[tx][ty+1] +  // Up
        shared_white[tx][ty-1]    // Down
    );
    
    // Get local RNG state and generate random number
    curandState local_state = rng_states[id];
    float random = curand_uniform(&local_state);
    
    // Metropolis update rule
    if (deltaE <= 0 || random < __expf(-beta * deltaE)) {
        spin *= -1;  // Flip the spin
    }
    
    // Save updated state
    black_sites[id] = spin;
    rng_states[id] = local_state;
}

/**
 * Update the white (odd) sites
 * Uses shared memory to reduce global memory accesses
 */
__global__ void update_white_sites(int *black_sites, int *white_sites, curandState *rng_states, float beta) {
    __shared__ int shared_black[BLOCK_SIZE_X+2][BLOCK_SIZE_Y+2];
    
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x >= d_SubL || y >= d_LY)
        return;
        
    int id = x + y * d_SubL;
    
    // Load local spin
    int spin = white_sites[id];
    
    // Local thread indexes for shared memory
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;
    
    // Load neighboring black spins into shared memory
    // This pattern handles the periodic boundary conditions
    
    // Current position neighbors
    int x_left = (x - 1 + d_SubL) % d_SubL;
    int y_up = (y + 1) % d_LY;
    int y_down = (y - 1 + d_LY) % d_LY;
    
    shared_black[tx][ty] = black_sites[id];
    shared_black[tx-1][ty] = black_sites[x_left + y * d_SubL];
    shared_black[tx][ty+1] = black_sites[x + y_up * d_SubL];
    shared_black[tx][ty-1] = black_sites[x + y_down * d_SubL];
    
    // Make sure all threads have loaded data
    __syncthreads();
    
    // Calculate energy change for spin flip
    int deltaE = 2 * d_J * spin * (
        shared_black[tx][ty] +    // Current position
        shared_black[tx-1][ty] +  // Left
        shared_black[tx][ty+1] +  // Up
        shared_black[tx][ty-1]    // Down
    );
    
    // Get local RNG state and generate random number
    curandState local_state = rng_states[id];
    float random = curand_uniform(&local_state);
    
    // Metropolis update rule
    if (deltaE <= 0 || random < __expf(-beta * deltaE)) {
        spin *= -1;  // Flip the spin
    }
    
    // Save updated state
    white_sites[id] = spin;
    rng_states[id] = local_state;
}

/**
 * Save the lattice state to a VTK file for visualization
 */
void save_lattice_vtk(const char* filename, int *black_sites, int *white_sites) {
    // Reconstruct the full lattice from black and white sites
    int *full_lattice = (int*)malloc(LX * LY * sizeof(int));
    int black_idx = 0, white_idx = 0;
    
    for (int i = 0; i < LX; i++) {
        for (int j = 0; j < LY; j++) {
            if ((i + j) % 2 == 0) {
                full_lattice[j + i * LX] = black_sites[black_idx++];
            } else {
                full_lattice[j + i * LX] = white_sites[white_idx++];
            }
        }
    }
    
    // Open the file
    std::ofstream file(filename, std::ios::out);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open " << filename << " for writing." << std::endl;
        free(full_lattice);
        return;
    }
    
    // Write VTK header
    file << "# vtk DataFile Version 3.0" << std::endl;
    file << "Ising Model Simulation" << std::endl;
    file << "ASCII" << std::endl;
    file << "DATASET STRUCTURED_POINTS" << std::endl;
    file << "DIMENSIONS " << LX << " " << LY << " 1" << std::endl;
    file << "ORIGIN 0 0 0" << std::endl;
    file << "SPACING 1 1 1" << std::endl;
    file << "POINT_DATA " << LX * LY << std::endl;
    file << "SCALARS spin float 1" << std::endl;
    file << "LOOKUP_TABLE default" << std::endl;
    
    // Write data
    for (int j = 0; j < LY; j++) {
        for (int i = 0; i < LX; i++) {
            file << full_lattice[i + j * LX] << std::endl;
        }
    }
    
    file.close();
    free(full_lattice);
    printf("Saved lattice state to %s\n", filename);
}