/*
 * ethash.cu - Implémentation Ethash complète avec DAG
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

// Constantes Ethash
#define ETHASH_DATASET_BYTES_INIT 1073741824U
#define ETHASH_DATASET_BYTES_GROWTH 8388608U
#define ETHASH_CACHE_BYTES_INIT 16777216U
#define ETHASH_CACHE_BYTES_GROWTH 131072U
#define ETHASH_MIX_BYTES 128
#define ETHASH_HASH_BYTES 64
#define ETHASH_DATASET_PARENTS 256
#define ETHASH_CACHE_ROUNDS 3
#define ETHASH_ACCESSES 64

#define FNV_PRIME 0x01000193
#define FNV_OFFSET_BASIS 0x811c9dc5

typedef union {
    uint32_t words[16];
    uint64_t double_words[8];
    uint8_t bytes[64];
} hash64_t;

typedef union {
    uint32_t words[32];
    uint64_t double_words[16];
    uint8_t bytes[128];
} hash128_t;

// Constantes Keccak
__device__ static const uint64_t keccakf_rndc[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

__device__ static const int keccakf_rotc[24] = {
    1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
    27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44
};

__device__ static const int keccakf_piln[24] = {
    10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1
};

// Versions CPU
static const uint64_t keccakf_rndc_host[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

static const int keccakf_rotc_host[24] = {
    1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
    27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44
};

static const int keccakf_piln_host[24] = {
    10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1
};

#define ROTL64(x, y) (((x) << (y)) | ((x) >> (64 - (y))))

__device__ __host__ void keccak_f1600(uint64_t state[25]) {
    uint64_t t, bc[5];
    
#ifdef __CUDA_ARCH__
    const uint64_t *rndc = keccakf_rndc;
    const int *rotc = keccakf_rotc;
    const int *piln = keccakf_piln;
#else
    const uint64_t *rndc = keccakf_rndc_host;
    const int *rotc = keccakf_rotc_host;
    const int *piln = keccakf_piln_host;
#endif
    
    #pragma unroll
    for (int round = 0; round < 24; round++) {
        // Theta
        #pragma unroll
        for (int i = 0; i < 5; i++)
            bc[i] = state[i] ^ state[i + 5] ^ state[i + 10] ^ state[i + 15] ^ state[i + 20];
        
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            t = bc[(i + 4) % 5] ^ ROTL64(bc[(i + 1) % 5], 1);
            #pragma unroll
            for (int j = 0; j < 25; j += 5)
                state[j + i] ^= t;
        }
        
        // Rho Pi
        t = state[1];
        #pragma unroll
        for (int i = 0; i < 24; i++) {
            int j = piln[i];
            bc[0] = state[j];
            state[j] = ROTL64(t, rotc[i]);
            t = bc[0];
        }
        
        // Chi
        #pragma unroll
        for (int j = 0; j < 25; j += 5) {
            #pragma unroll
            for (int i = 0; i < 5; i++)
                bc[i] = state[j + i];
            #pragma unroll
            for (int i = 0; i < 5; i++)
                state[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
        }
        
        // Iota
        state[0] ^= rndc[round];
    }
}

__device__ __host__ void keccak_256(hash64_t *ret, const uint8_t *data, size_t size) {
    uint64_t state[25] = {0};
    size_t pos = 0;
    
    while (pos + 136 <= size) {
        for (int i = 0; i < 17; i++)
            state[i] ^= ((uint64_t*)&data[pos])[i];
        keccak_f1600(state);
        pos += 136;
    }
    
    uint8_t temp[136] = {0};
    memcpy(temp, &data[pos], size - pos);
    temp[size - pos] = 1;
    temp[135] = 0x80;
    
    for (int i = 0; i < 17; i++)
        state[i] ^= ((uint64_t*)temp)[i];
    keccak_f1600(state);
    
    memcpy(ret->bytes, state, 64);
}

__device__ __host__ void keccak_512(hash64_t *ret, const uint8_t *data, size_t size) {
    uint64_t state[25] = {0};
    size_t pos = 0;
    
    while (pos + 72 <= size) {
        for (int i = 0; i < 9; i++)
            state[i] ^= ((uint64_t*)&data[pos])[i];
        keccak_f1600(state);
        pos += 72;
    }
    
    uint8_t temp[72] = {0};
    memcpy(temp, &data[pos], size - pos);
    temp[size - pos] = 1;
    temp[71] = 0x80;
    
    for (int i = 0; i < 9; i++)
        state[i] ^= ((uint64_t*)temp)[i];
    keccak_f1600(state);
    
    memcpy(ret->bytes, state, 64);
}

__device__ __host__ uint32_t fnv_hash(uint32_t x, uint32_t y) {
    return (x * FNV_PRIME) ^ y;
}

__device__ __host__ void fnv_hash_mix(hash128_t *mix, const hash64_t *data) {
    #pragma unroll
    for (int i = 0; i < 32; i++)
        mix->words[i] = fnv_hash(mix->words[i], data->words[i % 16]);
}

// Calcul item DAG
__device__ __host__ void calc_dataset_item(hash64_t *ret, uint32_t i, const hash64_t *cache, uint32_t cache_size) {
    uint32_t n = cache_size / sizeof(hash64_t);
    hash64_t mix = cache[i % n];
    mix.words[0] ^= i;
    keccak_512(&mix, mix.bytes, 64);
    
    #pragma unroll 1
    for (uint32_t j = 0; j < ETHASH_DATASET_PARENTS; j++) {
        uint32_t cache_index = fnv_hash(i ^ j, mix.words[j % 16]) % n;
        hash64_t temp;
        #pragma unroll
        for (int k = 0; k < 16; k++)
            temp.words[k] = mix.words[k] ^ cache[cache_index].words[k];
        mix = temp;
    }
    
    keccak_512(ret, mix.bytes, 64);
}

// Kernel Ethash OPTIMISÉ avec __ldg() et optimisations compilateur
__global__ void ethash_search_optimized(
    const uint64_t * __restrict__ g_dag,
    const uint8_t * __restrict__ g_header,
    uint64_t target,
    uint64_t start_nonce,
    uint32_t dag_size,
    uint64_t *g_solution
) {
    // Shared memory cache pour header (partagé entre tous threads)
    __shared__ uint64_t s_header[5];
    
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t tid = threadIdx.x;
    uint64_t nonce = start_nonce + gid;
    
    // Charger header en shared memory (une seule fois par bloc)
    if (tid < 4) {
        s_header[tid] = ((const uint64_t*)g_header)[tid];
    }
    if (tid == 0) {
        s_header[4] = nonce;
    }
    __syncthreads();
    
    // Utiliser header depuis shared memory
    s_header[4] = nonce;  // Chaque thread met son nonce
    
    // Initial hash
    hash64_t seed_hash;
    keccak_512(&seed_hash, (uint8_t*)s_header, 40);
    
    // Mix avec unroll complet
    hash128_t mix;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        mix.double_words[i] = seed_hash.double_words[i & 7];
    }
    
    uint32_t num_items = dag_size / 64;
    
    // OPTIMISATION: Loop unrolling + __ldg() pour lecture cache
    #pragma unroll 4
    for (int i = 0; i < ETHASH_ACCESSES; i++) {
        uint32_t index = fnv_hash(i ^ seed_hash.words[0], mix.words[i & 31]) % num_items;
        
        // OPTIMISATION: __ldg() = lecture via read-only cache L1
        hash64_t dag_item;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            dag_item.double_words[j] = __ldg(&g_dag[index * 8 + j]);
        }
        
        // FNV mix inline (optimisé)
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            mix.words[j] = mix.words[j] * FNV_PRIME ^ dag_item.words[j];
        }
    }
    
    // Compress mix (optimisé avec unroll)
    hash64_t cmix;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        cmix.words[i] = fnv_hash(
            fnv_hash(mix.words[i * 4], mix.words[i * 4 + 1]),
            fnv_hash(mix.words[i * 4 + 2], mix.words[i * 4 + 3])
        );
    }
    
    // Final hash
    uint8_t final_data[96];
    #pragma unroll
    for (int i = 0; i < 64; i++) final_data[i] = seed_hash.bytes[i];
    #pragma unroll  
    for (int i = 0; i < 32; i++) final_data[64 + i] = cmix.bytes[i];
    
    hash64_t result;
    keccak_256(&result, final_data, 96);
    
    // Check target (pas d'atomic si pas nécessaire)
    if (result.double_words[0] < target) {
        atomicMin((unsigned long long*)g_solution, (unsigned long long)nonce);
    }
}

// Ancien kernel non-optimisé (backup)
__global__ void ethash_search(
    uint64_t *g_dag,
    const uint8_t *g_header,
    uint64_t target,
    uint64_t start_nonce,
    uint32_t dag_size,
    uint64_t *g_solution
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t nonce = start_nonce + gid;
    
    // Préparer header + nonce
    uint8_t header_nonce[40];
    memcpy(header_nonce, g_header, 32);
    memcpy(&header_nonce[32], &nonce, 8);
    
    // Initial hash
    hash64_t seed_hash;
    keccak_512(&seed_hash, header_nonce, 40);
    
    // Mix
    hash128_t mix;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        mix.double_words[i] = seed_hash.double_words[i % 8];
    }
    
    uint32_t num_items = dag_size / 64;
    
    #pragma unroll 1
    for (int i = 0; i < ETHASH_ACCESSES; i++) {
        uint32_t index = fnv_hash(i ^ seed_hash.words[0], mix.words[i % 32]) % num_items;
        
        hash64_t dag_item;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            dag_item.double_words[j] = g_dag[index * 8 + j];
        }
        
        fnv_hash_mix(&mix, &dag_item);
    }
    
    // Compress mix
    hash64_t cmix;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        cmix.words[i] = fnv_hash(
            fnv_hash(mix.words[i * 4], mix.words[i * 4 + 1]),
            fnv_hash(mix.words[i * 4 + 2], mix.words[i * 4 + 3])
        );
    }
    
    // Final hash
    uint8_t final_data[64 + 32];
    memcpy(final_data, seed_hash.bytes, 64);
    memcpy(final_data + 64, cmix.bytes, 32);
    
    hash64_t result;
    keccak_256(&result, final_data, 96);
    
    // Check target
    if (result.double_words[0] < target) {
        atomicMin((unsigned long long*)g_solution, (unsigned long long)nonce);
    }
}

// Génération cache CPU
void generate_cache(hash64_t *cache, uint32_t cache_size, const uint8_t *seed) {
    uint32_t n = cache_size / sizeof(hash64_t);
    
    // Initial hash
    hash64_t temp;
    keccak_512(&temp, seed, 32);
    cache[0] = temp;
    
    for (uint32_t i = 1; i < n; i++) {
        keccak_512(&cache[i], cache[i-1].bytes, 64);
    }
    
    // RandMemoHash
    for (uint32_t round = 0; round < ETHASH_CACHE_ROUNDS; round++) {
        for (uint32_t i = 0; i < n; i++) {
            uint32_t v = cache[i].words[0] % n;
            hash64_t temp;
            for (int j = 0; j < 16; j++)
                temp.words[j] = cache[(i-1+n)%n].words[j] ^ cache[v].words[j];
            keccak_512(&cache[i], temp.bytes, 64);
        }
    }
}

// Génération DAG complet sur GPU
__global__ void generate_dag_kernel(
    uint64_t *g_dag,
    const hash64_t *g_cache,
    uint32_t cache_size,
    uint32_t start,
    uint32_t count
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    
    uint32_t item_index = start + gid;
    hash64_t item;
    calc_dataset_item(&item, item_index, g_cache, cache_size);
    
    for (int i = 0; i < 8; i++) {
        g_dag[item_index * 8 + i] = item.double_words[i];
    }
}

// Wrapper C
extern "C" {
    void* ethash_create_dag(uint32_t epoch, uint32_t *dag_size_out) {
        uint32_t cache_size = ETHASH_CACHE_BYTES_INIT + ETHASH_CACHE_BYTES_GROWTH * epoch;
        uint32_t dag_size = ETHASH_DATASET_BYTES_INIT + ETHASH_DATASET_BYTES_GROWTH * epoch;
        
        *dag_size_out = dag_size;
        
        // Seed
        uint8_t seed[32] = {0};
        for (uint32_t i = 0; i < epoch; i++) {
            hash64_t temp;
            keccak_256(&temp, seed, 32);
            memcpy(seed, temp.bytes, 32);
        }
        
        // Cache CPU
        hash64_t *cache = (hash64_t*)malloc(cache_size);
        generate_cache(cache, cache_size, seed);
        
        // Upload cache GPU
        hash64_t *d_cache;
        cudaMalloc(&d_cache, cache_size);
        cudaMemcpy(d_cache, cache, cache_size, cudaMemcpyHostToDevice);
        
        // Allouer DAG GPU
        uint64_t *d_dag;
        cudaMalloc(&d_dag, dag_size);
        
        // Générer DAG
        uint32_t num_items = dag_size / 64;
        uint32_t threads = 256;
        uint32_t blocks = (num_items + threads - 1) / threads;
        
        printf("Génération DAG: %u MB...\n", dag_size / 1024 / 1024);
        generate_dag_kernel<<<blocks, threads>>>(d_dag, d_cache, cache_size, 0, num_items);
        cudaDeviceSynchronize();
        
        cudaFree(d_cache);
        free(cache);
        
        printf("DAG généré!\n");
        return d_dag;
    }
    
    // Wrapper pour compatibilité avec ancien code
    void* ethash_generate_dag(uint32_t epoch, uint32_t dag_size) {
        uint32_t actual_dag_size;
        void *dag = ethash_create_dag(epoch, &actual_dag_size);
        return dag;
    }
    
    void ethash_search_launch(
        void *dag,
        const uint8_t *header,
        uint64_t target,
        uint64_t start_nonce,
        uint32_t dag_size,
        uint64_t *solution,
        int grid_size,
        int block_size
    ) {
        uint8_t *d_header;
        uint64_t *d_solution;
        
        cudaMalloc(&d_header, 32);
        cudaMalloc(&d_solution, 8);
        
        cudaMemcpy(d_header, header, 32, cudaMemcpyHostToDevice);
        uint64_t max_nonce = 0xFFFFFFFFFFFFFFFFULL;
        cudaMemcpy(d_solution, &max_nonce, 8, cudaMemcpyHostToDevice);
        
        // OPTIMISATION: Augmenter grid/block pour plus de threads
        int optimized_grid = grid_size * 4;    // 4x plus de blocs
        int optimized_block = 256;              // Taille optimale pour cache L1
        
        // Lancer kernel OPTIMISÉ avec __ldg() et shared memory
        ethash_search_optimized<<<optimized_grid, optimized_block>>>(
            (const uint64_t*)dag, d_header, target, start_nonce, dag_size, d_solution
        );
        
        cudaDeviceSynchronize();
        cudaMemcpy(solution, d_solution, 8, cudaMemcpyDeviceToHost);
        
        cudaFree(d_header);
        cudaFree(d_solution);
    }
    
    // Version non-optimisée (backup)
    void ethash_search_launch_legacy(
        void *dag,
        const uint8_t *header,
        uint64_t target,
        uint64_t start_nonce,
        uint32_t dag_size,
        uint64_t *solution,
        int grid_size,
        int block_size
    ) {
        uint8_t *d_header;
        uint64_t *d_solution;
        
        cudaMalloc(&d_header, 32);
        cudaMalloc(&d_solution, 8);
        
        cudaMemcpy(d_header, header, 32, cudaMemcpyHostToDevice);
        uint64_t max_nonce = 0xFFFFFFFFFFFFFFFFULL;
        cudaMemcpy(d_solution, &max_nonce, 8, cudaMemcpyHostToDevice);
        
        ethash_search<<<grid_size, block_size>>>(
            (uint64_t*)dag, d_header, target, start_nonce, dag_size, d_solution
        );
        
        cudaDeviceSynchronize();
        cudaMemcpy(solution, d_solution, 8, cudaMemcpyDeviceToHost);
        
        cudaFree(d_header);
        cudaFree(d_solution);
    }
    
    void ethash_destroy_dag(void *dag) {
        cudaFree(dag);
    }
}
