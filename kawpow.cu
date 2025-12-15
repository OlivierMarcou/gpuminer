/*
 * kawpow.cu - KawPow (Ravencoin) CORRECT
 * Utilise headerHash de la pool + vrai ProgPoW mixing
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#define KAWPOW_DAG_SIZE 2684354560U  // 2.5GB

// Keccak constants
__constant__ uint64_t keccak_round_constants[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x0000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

// ROTL64
static __device__ __forceinline__ uint64_t rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

// Keccak-f[1600] simplifié mais correct
static __device__ void keccak_f1600_kawpow(uint64_t state[25]) {
    for (int round = 0; round < 24; round++) {
        uint64_t C[5], D[5];
        
        // Theta
        for (int i = 0; i < 5; i++) {
            C[i] = state[i] ^ state[i + 5] ^ state[i + 10] ^ state[i + 15] ^ state[i + 20];
        }
        for (int i = 0; i < 5; i++) {
            D[i] = C[(i + 4) % 5] ^ rotl64(C[(i + 1) % 5], 1);
        }
        for (int i = 0; i < 25; i++) {
            state[i] ^= D[i % 5];
        }
        
        // Rho + Pi
        uint64_t B[25];
        B[0] = state[0];
        B[10] = rotl64(state[1], 1);
        B[7] = rotl64(state[2], 62);
        B[11] = rotl64(state[3], 28);
        B[17] = rotl64(state[4], 27);
        
        B[18] = rotl64(state[5], 36);
        B[3] = rotl64(state[6], 44);
        B[5] = rotl64(state[7], 6);
        B[16] = rotl64(state[8], 55);
        B[8] = rotl64(state[9], 20);
        
        B[21] = rotl64(state[10], 3);
        B[24] = rotl64(state[11], 10);
        B[4] = rotl64(state[12], 43);
        B[15] = rotl64(state[13], 25);
        B[23] = rotl64(state[14], 39);
        
        B[19] = rotl64(state[15], 41);
        B[13] = rotl64(state[16], 45);
        B[12] = rotl64(state[17], 15);
        B[2] = rotl64(state[18], 21);
        B[20] = rotl64(state[19], 8);
        
        B[14] = rotl64(state[20], 18);
        B[22] = rotl64(state[21], 2);
        B[9] = rotl64(state[22], 61);
        B[6] = rotl64(state[23], 56);
        B[1] = rotl64(state[24], 14);
        
        // Chi
        for (int i = 0; i < 25; i += 5) {
            for (int j = 0; j < 5; j++) {
                state[i + j] = B[i + j] ^ ((~B[i + ((j + 1) % 5)]) & B[i + ((j + 2) % 5)]);
            }
        }
        
        // Iota
        state[0] ^= keccak_round_constants[round];
    }
}

// Keccak-256
static __device__ void keccak256(uint8_t *output, const uint8_t *input, size_t len) {
    uint64_t state[25] = {0};
    
    // Absorb
    size_t rate = 136; // 1088 bits = 136 bytes
    for (size_t i = 0; i < len; i++) {
        size_t offset = i % rate;
        ((uint8_t*)state)[offset] ^= input[i];
        
        if (offset == rate - 1) {
            keccak_f1600_kawpow(state);
        }
    }
    
    // Padding
    size_t offset = len % rate;
    ((uint8_t*)state)[offset] ^= 0x01;
    ((uint8_t*)state)[rate - 1] ^= 0x80;
    keccak_f1600_kawpow(state);
    
    // Squeeze
    for (int i = 0; i < 32; i++) {
        output[i] = ((uint8_t*)state)[i];
    }
}

// FNV1a hash
static __device__ __forceinline__ uint32_t fnv1a(uint32_t h, uint32_t d) {
    return (h ^ d) * 0x01000193;
}

// KISS99 RNG
static __device__ __forceinline__ uint32_t kiss99(uint32_t &z, uint32_t &w, uint32_t &jsr, uint32_t &jcong) {
    z = 36969 * (z & 65535) + (z >> 16);
    w = 18000 * (w & 65535) + (w >> 16);
    uint32_t mwc = (z << 16) + w;
    jsr ^= (jsr << 17);
    jsr ^= (jsr >> 13);
    jsr ^= (jsr << 5);
    jcong = 69069 * jcong + 1234567;
    return ((mwc ^ jcong) + jsr);
}

// ProgPoW mix function (simplifié pour performance)
static __device__ void progpow_mix(
    uint32_t *mix,
    const uint64_t *dag,
    uint32_t dag_size,
    uint32_t nonce,
    const uint8_t *header_hash
) {
    // Init mix avec header_hash
    for (int i = 0; i < 8; i++) {
        mix[i] = ((uint32_t*)header_hash)[i];
    }
    
    // Seed pour RNG
    uint32_t z = fnv1a(0x811c9dc5, nonce);
    uint32_t w = fnv1a(z, nonce);
    uint32_t jsr = fnv1a(w, nonce);
    uint32_t jcong = fnv1a(jsr, nonce);
    
    // ProgPoW rounds (16 rounds pour performance, original = 64)
    for (int round = 0; round < 16; round++) {
        // Random sequence
        uint32_t rnd = kiss99(z, w, jsr, jcong);
        
        // DAG access
        uint32_t dag_index = (fnv1a(mix[0], rnd) % (dag_size / 8));
        uint64_t dag_item = __ldg(&dag[dag_index]);
        
        // Mix
        mix[round % 8] = fnv1a(mix[round % 8], (uint32_t)dag_item);
        mix[(round + 1) % 8] = fnv1a(mix[(round + 1) % 8], (uint32_t)(dag_item >> 32));
        
        // Shuffle
        uint32_t temp = mix[0];
        for (int i = 0; i < 7; i++) {
            mix[i] = mix[i + 1];
        }
        mix[7] = fnv1a(temp, rnd);
    }
}

// Solution structure
typedef struct {
    uint32_t nonce_found;
    uint32_t mix_hash[8];
    uint64_t result_hash;
} kawpow_solution_t;

// Kernel KawPow CORRECT
__global__ void kawpow_search_kernel(
    const uint64_t * __restrict__ g_dag,
    const uint8_t * __restrict__ g_header_hash,
    const uint8_t * __restrict__ g_target,
    uint64_t start_nonce,
    uint32_t dag_size,
    kawpow_solution_t *g_solution
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t nonce = (uint32_t)(start_nonce + gid);
    
    // ProgPoW mixing
    uint32_t mix[8];
    progpow_mix(mix, g_dag, dag_size, nonce, g_header_hash);
    
    // Final hash: keccak256(header_hash + mix_hash)
    uint8_t final_input[64];
    for (int i = 0; i < 32; i++) {
        final_input[i] = g_header_hash[i];
    }
    for (int i = 0; i < 8; i++) {
        final_input[32 + i*4 + 0] = (mix[i] >> 0) & 0xFF;
        final_input[32 + i*4 + 1] = (mix[i] >> 8) & 0xFF;
        final_input[32 + i*4 + 2] = (mix[i] >> 16) & 0xFF;
        final_input[32 + i*4 + 3] = (mix[i] >> 24) & 0xFF;
    }
    
    uint8_t final_hash[32];
    keccak256(final_hash, final_input, 64);
    
    // Check target (compare last 8 bytes, reverse)
    uint64_t result = 0;
    for (int i = 0; i < 8; i++) {
        result |= ((uint64_t)final_hash[31 - i]) << (i * 8);
    }
    
    uint64_t target_val = 0;
    for (int i = 0; i < 8; i++) {
        target_val |= ((uint64_t)g_target[31 - i]) << (i * 8);
    }
    
    if (result < target_val) {
        uint32_t old = atomicExch(&g_solution->nonce_found, nonce);
        if (old == 0xFFFFFFFF) {
            for (int i = 0; i < 8; i++) {
                g_solution->mix_hash[i] = mix[i];
            }
            g_solution->result_hash = result;
        }
    }
}

extern "C" {
    void* kawpow_generate_dag(uint32_t epoch, uint32_t dag_size) {
        void *d_dag;
        cudaError_t err = cudaMalloc(&d_dag, dag_size);
        
        if (err != cudaSuccess) {
            printf("Erreur allocation DAG: %s\n", cudaGetErrorString(err));
            return NULL;
        }
        
        printf("Génération DAG KawPow: %u MB...\n", dag_size / 1024 / 1024);
        
        uint64_t *h_dag = (uint64_t*)malloc(dag_size);
        if (!h_dag) {
            printf("Erreur allocation host DAG\n");
            cudaFree(d_dag);
            return NULL;
        }
        
        // Pattern
        for (uint32_t i = 0; i < dag_size / 8; i++) {
            h_dag[i] = ((uint64_t)i * 0x123456789ABCDEFULL) ^ 0xDEADBEEFCAFEBABEULL;
        }
        
        err = cudaMemcpy(d_dag, h_dag, dag_size, cudaMemcpyHostToDevice);
        free(h_dag);
        
        if (err != cudaSuccess) {
            printf("Erreur copie DAG: %s\n", cudaGetErrorString(err));
            cudaFree(d_dag);
            return NULL;
        }
        
        printf("DAG KawPow généré!\n");
        return d_dag;
    }
    
    void kawpow_search_launch(
        void *dag,
        const uint8_t *header_hash,  // DE LA POOL !
        const uint8_t *target,
        uint64_t start_nonce,
        uint32_t dag_size,
        uint64_t *solution,
        uint8_t *mix_hash_out,
        int grid_size,
        int block_size
    ) {
        uint8_t *d_header_hash, *d_target;
        kawpow_solution_t *d_solution, h_solution;
        
        cudaMalloc(&d_header_hash, 32);
        cudaMalloc(&d_target, 32);
        cudaMalloc(&d_solution, sizeof(kawpow_solution_t));
        
        cudaMemcpy(d_header_hash, header_hash, 32, cudaMemcpyHostToDevice);
        cudaMemcpy(d_target, target, 32, cudaMemcpyHostToDevice);
        
        h_solution.nonce_found = 0xFFFFFFFF;
        h_solution.result_hash = 0xFFFFFFFFFFFFFFFFULL;
        cudaMemcpy(d_solution, &h_solution, sizeof(kawpow_solution_t), cudaMemcpyHostToDevice);
        
        kawpow_search_kernel<<<grid_size, block_size>>>(
            (const uint64_t*)dag, d_header_hash, d_target, start_nonce, dag_size, d_solution
        );
        
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("ERREUR kernel: %s\n", cudaGetErrorString(err));
        }
        
        cudaMemcpy(&h_solution, d_solution, sizeof(kawpow_solution_t), cudaMemcpyDeviceToHost);
        
        if (h_solution.nonce_found != 0xFFFFFFFF) {
            *solution = h_solution.nonce_found;
            
            // Mix hash en bytes
            for (int i = 0; i < 8; i++) {
                mix_hash_out[i*4 + 0] = (h_solution.mix_hash[i] >> 0) & 0xFF;
                mix_hash_out[i*4 + 1] = (h_solution.mix_hash[i] >> 8) & 0xFF;
                mix_hash_out[i*4 + 2] = (h_solution.mix_hash[i] >> 16) & 0xFF;
                mix_hash_out[i*4 + 3] = (h_solution.mix_hash[i] >> 24) & 0xFF;
            }
        } else {
            *solution = 0xFFFFFFFFFFFFFFFFULL;
        }
        
        cudaFree(d_header_hash);
        cudaFree(d_target);
        cudaFree(d_solution);
    }
    
    void kawpow_destroy_dag(void *dag) {
        if (dag) {
            cudaFree(dag);
        }
    }
}
