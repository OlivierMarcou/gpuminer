/*
 * kawpow_debug.cu - Version DEBUG avec outputs détaillés
 * Pour comprendre pourquoi 0 shares
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#define KAWPOW_DAG_SIZE 2684354560U
#define PROGPOW_PERIOD 10
#define PROGPOW_LANES 16
#define PROGPOW_REGS 32
#define PROGPOW_CNT_DAG 64
#define PROGPOW_CNT_MATH 18

#define FNV_PRIME 0x01000193
#define FNV_OFFSET_BASIS 0x811c9dc5

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

static __device__ __forceinline__ uint64_t rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

static __device__ __forceinline__ uint32_t rotl32(uint32_t x, unsigned n) {
    return (x << n) | (x >> (32 - n));
}

static __device__ __forceinline__ uint32_t rotr32(uint32_t x, unsigned n) {
    return (x >> n) | (x << (32 - n));
}

// Keccak-f[1600]
static __device__ void keccak_f1600_kawpow(uint64_t state[25]) {
    for (int round = 0; round < 24; round++) {
        uint64_t C[5], D[5], B[25];
        
        for (int i = 0; i < 5; i++)
            C[i] = state[i] ^ state[i + 5] ^ state[i + 10] ^ state[i + 15] ^ state[i + 20];
        for (int i = 0; i < 5; i++)
            D[i] = C[(i + 4) % 5] ^ rotl64(C[(i + 1) % 5], 1);
        for (int i = 0; i < 25; i++)
            state[i] ^= D[i % 5];
        
        B[0] = state[0];
        B[10] = rotl64(state[1], 1); B[7] = rotl64(state[2], 62); B[11] = rotl64(state[3], 28); B[17] = rotl64(state[4], 27);
        B[18] = rotl64(state[5], 36); B[3] = rotl64(state[6], 44); B[5] = rotl64(state[7], 6); B[16] = rotl64(state[8], 55); B[8] = rotl64(state[9], 20);
        B[21] = rotl64(state[10], 3); B[24] = rotl64(state[11], 10); B[4] = rotl64(state[12], 43); B[15] = rotl64(state[13], 25); B[23] = rotl64(state[14], 39);
        B[19] = rotl64(state[15], 41); B[13] = rotl64(state[16], 45); B[12] = rotl64(state[17], 15); B[2] = rotl64(state[18], 21); B[20] = rotl64(state[19], 8);
        B[14] = rotl64(state[20], 18); B[22] = rotl64(state[21], 2); B[9] = rotl64(state[22], 61); B[6] = rotl64(state[23], 56); B[1] = rotl64(state[24], 14);
        
        for (int i = 0; i < 25; i += 5)
            for (int j = 0; j < 5; j++)
                state[i + j] = B[i + j] ^ ((~B[i + ((j + 1) % 5)]) & B[i + ((j + 2) % 5)]);
        
        state[0] ^= keccak_round_constants[round];
    }
}

static __device__ void keccak256(uint8_t *output, const uint8_t *input, size_t len) {
    uint64_t state[25] = {0};
    size_t rate = 136;
    
    for (size_t i = 0; i < len; i++) {
        ((uint8_t*)state)[i % rate] ^= input[i];
        if (i % rate == rate - 1) keccak_f1600_kawpow(state);
    }
    
    size_t offset = len % rate;
    ((uint8_t*)state)[offset] ^= 0x01;
    ((uint8_t*)state)[rate - 1] ^= 0x80;
    keccak_f1600_kawpow(state);
    
    for (int i = 0; i < 32; i++)
        output[i] = ((uint8_t*)state)[i];
}

static __device__ __forceinline__ uint32_t fnv1a(uint32_t h, uint32_t d) {
    return (h ^ d) * FNV_PRIME;
}

static __device__ __forceinline__ uint32_t kiss99(uint32_t &z, uint32_t &w, uint32_t &jsr, uint32_t &jcong) {
    z = 36969 * (z & 65535) + (z >> 16);
    w = 18000 * (w & 65535) + (w >> 16);
    jsr ^= (jsr << 17); jsr ^= (jsr >> 13); jsr ^= (jsr << 5);
    jcong = 69069 * jcong + 1234567;
    return ((z << 16) + w) ^ jcong ^ jsr;
}

static __device__ __forceinline__ uint32_t progpow_math(uint32_t a, uint32_t b, uint32_t r) {
    switch (r % 11) {
        case 0: return a + b;
        case 1: return a * b;
        case 2: return __umulhi(a, b);
        case 3: return min(a, b);
        case 4: return rotr32(a, b & 31);
        case 5: return rotl32(a, b & 31);
        case 6: return a & b;
        case 7: return a | b;
        case 8: return a ^ b;
        case 9: return __clz(a) + __clz(b);
        case 10: return __popc(a) + __popc(b);
    }
    return 0;
}

static __device__ void progpow_loop(
    uint32_t *mix,
    const uint64_t *dag,
    uint32_t dag_words,
    uint32_t loop_idx,
    uint32_t prog_seed
) {
    uint32_t z = fnv1a(FNV_OFFSET_BASIS, prog_seed);
    uint32_t w = fnv1a(z, loop_idx);
    uint32_t jsr = fnv1a(w, loop_idx);
    uint32_t jcong = fnv1a(jsr, loop_idx);
    
    uint32_t dag_addr = mix[0] % dag_words;
    uint64_t dag_item = __ldg(&dag[dag_addr]);
    
    for (int i = 0; i < PROGPOW_CNT_MATH; i++) {
        uint32_t rnd = kiss99(z, w, jsr, jcong);
        uint32_t src1 = rnd % PROGPOW_REGS;
        rnd = kiss99(z, w, jsr, jcong);
        uint32_t src2 = rnd % PROGPOW_REGS;
        rnd = kiss99(z, w, jsr, jcong);
        uint32_t dst = rnd % PROGPOW_REGS;
        rnd = kiss99(z, w, jsr, jcong);
        
        mix[dst] = progpow_math(mix[src1], mix[src2], rnd);
    }
    
    mix[0] = fnv1a(mix[0], (uint32_t)dag_item);
    mix[1] = fnv1a(mix[1], (uint32_t)(dag_item >> 32));
}

__global__ void kawpow_search_kernel_debug(
    const uint64_t * __restrict__ g_dag,
    const uint8_t * __restrict__ g_header_hash,
    const uint8_t * __restrict__ g_target,
    uint64_t start_nonce,
    uint32_t dag_size,
    uint32_t prog_seed,
    uint64_t *g_solution,
    uint32_t *g_mix_out,
    uint32_t *g_debug_info
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t nonce = start_nonce + gid;
    
    // Init mix
    uint32_t mix[PROGPOW_REGS];
    for (int i = 0; i < 8; i++) {
        mix[i] = ((uint32_t*)g_header_hash)[i];
    }
    for (int i = 8; i < PROGPOW_REGS; i++) {
        mix[i] = (uint32_t)nonce ^ i;
    }
    
    // DEBUG: Sauver état initial pour premier thread
    if (gid == 0) {
        for (int i = 0; i < 8; i++) {
            g_debug_info[i] = mix[i];
        }
        g_debug_info[8] = prog_seed;
        g_debug_info[9] = (uint32_t)nonce;
    }
    
    // ProgPoW: 64 rounds
    uint32_t dag_words = dag_size / 8;
    
    for (int i = 0; i < PROGPOW_CNT_DAG; i++) {
        progpow_loop(mix, g_dag, dag_words, i, prog_seed);
    }
    
    // Reduce mix
    uint32_t mix_hash[8];
    for (int i = 0; i < 8; i++) {
        mix_hash[i] = FNV_OFFSET_BASIS;
        for (int j = 0; j < 4; j++) {
            mix_hash[i] = fnv1a(mix_hash[i], mix[i * 4 + j]);
        }
    }
    
    // DEBUG: Sauver mix_hash pour premier thread
    if (gid == 0) {
        for (int i = 0; i < 8; i++) {
            g_debug_info[10 + i] = mix_hash[i];
        }
    }
    
    // Final hash: keccak256(header + mix)
    uint8_t final_input[64];
    for (int i = 0; i < 32; i++) {
        final_input[i] = g_header_hash[i];
    }
    for (int i = 0; i < 8; i++) {
        final_input[32 + i*4 + 0] = (mix_hash[i] >> 0) & 0xFF;
        final_input[32 + i*4 + 1] = (mix_hash[i] >> 8) & 0xFF;
        final_input[32 + i*4 + 2] = (mix_hash[i] >> 16) & 0xFF;
        final_input[32 + i*4 + 3] = (mix_hash[i] >> 24) & 0xFF;
    }
    
    uint8_t result[32];
    keccak256(result, final_input, 64);
    
    // DEBUG: Sauver result pour premier thread
    if (gid == 0) {
        for (int i = 0; i < 8; i++) {
            g_debug_info[18 + i] = ((uint32_t*)result)[i];
        }
    }
    
    // Compare target
    bool found = true;
    for (int i = 0; i < 32; i++) {
        if (result[i] > g_target[i]) {
            found = false;
            break;
        } else if (result[i] < g_target[i]) {
            break;
        }
    }
    
    if (found) {
        uint64_t old = atomicMin((unsigned long long*)g_solution, (unsigned long long)nonce);
        if (old == 0xFFFFFFFFFFFFFFFFULL) {
            for (int i = 0; i < 8; i++) {
                g_mix_out[i] = mix_hash[i];
            }
        }
    }
}

extern "C" {
    void* kawpow_generate_dag(const uint8_t *seed_hash, uint32_t dag_size) {
        void *d_dag;
        if (cudaMalloc(&d_dag, dag_size) != cudaSuccess) {
            printf("Erreur allocation DAG\n");
            return NULL;
        }
        
        printf("Génération DAG KawPow depuis seedHash: %u MB...\n", dag_size / 1024 / 1024);
        
        uint64_t *h_dag = (uint64_t*)malloc(dag_size);
        if (!h_dag) {
            cudaFree(d_dag);
            return NULL;
        }
        
        uint32_t num_items = dag_size / 64;
        
        uint64_t seed = 0;
        for (int i = 0; i < 8; i++) {
            seed ^= ((uint64_t)seed_hash[i]) << (i * 8);
        }
        
        for (uint32_t i = 0; i < num_items; i++) {
            uint64_t item_seed = seed ^ ((uint64_t)i * 0x9e3779b97f4a7c15ULL);
            
            for (int j = 0; j < 8; j++) {
                uint64_t val = item_seed;
                val = (val ^ (val >> 33)) * 0xff51afd7ed558ccdULL;
                val = (val ^ (val >> 33)) * 0xc4ceb9fe1a85ec53ULL;
                val = (val ^ (val >> 33)) ^ (j * 0x9e3779b97f4a7c15ULL);
                
                h_dag[i * 8 + j] = val;
            }
        }
        
        printf("DAG généré, copie vers GPU...\n");
        
        if (cudaMemcpy(d_dag, h_dag, dag_size, cudaMemcpyHostToDevice) != cudaSuccess) {
            free(h_dag);
            cudaFree(d_dag);
            return NULL;
        }
        
        free(h_dag);
        printf("DAG KawPow généré et chargé!\n");
        return d_dag;
    }
    
    void kawpow_search_launch_debug(
        void *dag,
        const uint8_t *header_hash,
        const uint8_t *target,
        uint64_t start_nonce,
        uint32_t dag_size,
        uint32_t prog_seed,
        uint64_t *solution,
        uint8_t *mix_hash_out,
        uint32_t *debug_info,
        int grid_size,
        int block_size
    ) {
        uint8_t *d_header, *d_target;
        uint64_t *d_solution;
        uint32_t *d_mix, *d_debug;
        
        cudaMalloc(&d_header, 32);
        cudaMalloc(&d_target, 32);
        cudaMalloc(&d_solution, 8);
        cudaMalloc(&d_mix, 32);
        cudaMalloc(&d_debug, 128); // 32 uint32_t
        
        cudaMemcpy(d_header, header_hash, 32, cudaMemcpyHostToDevice);
        cudaMemcpy(d_target, target, 32, cudaMemcpyHostToDevice);
        
        uint64_t init = 0xFFFFFFFFFFFFFFFFULL;
        cudaMemcpy(d_solution, &init, 8, cudaMemcpyHostToDevice);
        
        kawpow_search_kernel_debug<<<grid_size, block_size>>>(
            (uint64_t*)dag, d_header, d_target, start_nonce, dag_size, prog_seed, d_solution, d_mix, d_debug
        );
        
        cudaDeviceSynchronize();
        
        cudaMemcpy(solution, d_solution, 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(debug_info, d_debug, 128, cudaMemcpyDeviceToHost);
        
        if (*solution != 0xFFFFFFFFFFFFFFFFULL) {
            uint32_t mix_u32[8];
            cudaMemcpy(mix_u32, d_mix, 32, cudaMemcpyDeviceToHost);
            for (int i = 0; i < 8; i++) {
                mix_hash_out[i*4+0] = (mix_u32[i] >> 0) & 0xFF;
                mix_hash_out[i*4+1] = (mix_u32[i] >> 8) & 0xFF;
                mix_hash_out[i*4+2] = (mix_u32[i] >> 16) & 0xFF;
                mix_hash_out[i*4+3] = (mix_u32[i] >> 24) & 0xFF;
            }
        }
        
        cudaFree(d_header);
        cudaFree(d_target);
        cudaFree(d_solution);
        cudaFree(d_mix);
        cudaFree(d_debug);
    }
    
    void kawpow_search_launch(
        void *dag,
        const uint8_t *header_hash,
        const uint8_t *target,
        uint64_t start_nonce,
        uint32_t dag_size,
        uint32_t prog_seed,
        uint64_t *solution,
        uint8_t *mix_hash_out,
        int grid_size,
        int block_size
    ) {
        uint32_t debug_info[32];
        kawpow_search_launch_debug(dag, header_hash, target, start_nonce, dag_size, prog_seed,
                                   solution, mix_hash_out, debug_info, grid_size, block_size);
    }
    
    void kawpow_destroy_dag(void *dag) {
        if (dag) cudaFree(dag);
    }
}
