/*
 * sha256.cu - Implémentation SHA256 CUDA
 * Kernel GPU optimisé pour le minage
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

// Constantes SHA256
__constant__ uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// Macros SHA256
#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define SHR(x, n) ((x) >> (n))
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define EP1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define SIG0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ SHR(x, 3))
#define SIG1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ SHR(x, 10))

/*
 * Kernel SHA256 - Version optimisée pour minage Bitcoin/SHA256
 */
__global__ void sha256_mining_kernel(
    uint32_t *block_header,   // 80 bytes header
    uint32_t target,          // Target difficulty
    uint32_t start_nonce,
    uint32_t *found_nonce,
    uint32_t *found_hash
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t nonce = start_nonce + idx;
    
    // Variables SHA256
    uint32_t a, b, c, d, e, f, g, h;
    uint32_t t1, t2;
    uint32_t W[64];
    uint32_t hash[8];
    
    // Copier le header dans la mémoire locale
    uint32_t local_header[20];
    #pragma unroll
    for (int i = 0; i < 19; i++) {
        local_header[i] = block_header[i];
    }
    local_header[19] = nonce; // Injecter le nonce
    
    // Padding SHA256 (le header fait 80 bytes = 640 bits)
    uint32_t data[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        if (i < 5) {
            data[i] = __byte_perm(local_header[i*4], local_header[i*4+1], 0x0123);
            data[i] = __byte_perm(data[i], local_header[i*4+2], 0x0145);
            data[i] = __byte_perm(data[i], local_header[i*4+3], 0x0167);
        } else if (i == 5) {
            data[i] = 0x80000000; // Padding
        } else if (i == 15) {
            data[i] = 640; // Longueur en bits
        } else {
            data[i] = 0;
        }
    }
    
    // Premier SHA256
    // Initialisation
    a = 0x6a09e667;
    b = 0xbb67ae85;
    c = 0x3c6ef372;
    d = 0xa54ff53a;
    e = 0x510e527f;
    f = 0x9b05688c;
    g = 0x1f83d9ab;
    h = 0x5be0cd19;
    
    // Expansion
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        W[i] = data[i];
    }
    
    #pragma unroll
    for (int i = 16; i < 64; i++) {
        W[i] = SIG1(W[i-2]) + W[i-7] + SIG0(W[i-15]) + W[i-16];
    }
    
    // Compression
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        t1 = h + EP1(e) + CH(e, f, g) + K[i] + W[i];
        t2 = EP0(a) + MAJ(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    
    // Premier hash
    hash[0] = a + 0x6a09e667;
    hash[1] = b + 0xbb67ae85;
    hash[2] = c + 0x3c6ef372;
    hash[3] = d + 0xa54ff53a;
    hash[4] = e + 0x510e527f;
    hash[5] = f + 0x9b05688c;
    hash[6] = g + 0x1f83d9ab;
    hash[7] = h + 0x5be0cd19;
    
    // Second SHA256 (double hash pour Bitcoin)
    // Préparer les données
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        data[i] = hash[i];
    }
    data[8] = 0x80000000; // Padding
    #pragma unroll
    for (int i = 9; i < 15; i++) {
        data[i] = 0;
    }
    data[15] = 256; // 256 bits
    
    // Réinitialiser
    a = 0x6a09e667;
    b = 0xbb67ae85;
    c = 0x3c6ef372;
    d = 0xa54ff53a;
    e = 0x510e527f;
    f = 0x9b05688c;
    g = 0x1f83d9ab;
    h = 0x5be0cd19;
    
    // Expansion
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        W[i] = data[i];
    }
    
    #pragma unroll
    for (int i = 16; i < 64; i++) {
        W[i] = SIG1(W[i-2]) + W[i-7] + SIG0(W[i-15]) + W[i-16];
    }
    
    // Compression
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        t1 = h + EP1(e) + CH(e, f, g) + K[i] + W[i];
        t2 = EP0(a) + MAJ(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    
    // Hash final
    hash[0] = a + 0x6a09e667;
    hash[1] = b + 0xbb67ae85;
    hash[2] = c + 0x3c6ef372;
    hash[3] = d + 0xa54ff53a;
    hash[4] = e + 0x510e527f;
    hash[5] = f + 0x9b05688c;
    hash[6] = g + 0x1f83d9ab;
    hash[7] = h + 0x5be0cd19;
    
    // Vérifier si le hash est inférieur au target
    // Bitcoin utilise little-endian, donc on compare hash[7] d'abord
    if (hash[7] == 0 && hash[6] < target) {
        // Solution potentielle trouvée
        int old = atomicCAS(found_nonce, 0xFFFFFFFF, nonce);
        if (old == 0xFFFFFFFF) {
            // Premier à trouver cette solution
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                found_hash[i] = hash[i];
            }
        }
    }
}

/*
 * Wrapper CPU pour lancer le kernel SHA256
 */
extern "C" void launch_sha256_kernel(
    uint32_t *d_header,
    uint32_t target,
    uint32_t start_nonce,
    uint32_t *d_found_nonce,
    uint32_t *d_found_hash,
    int grid_size,
    int block_size
) {
    sha256_mining_kernel<<<grid_size, block_size>>>(
        d_header,
        target,
        start_nonce,
        d_found_nonce,
        d_found_hash
    );
    
    cudaDeviceSynchronize();
}

/*
 * Implémentation CPU SHA256 (pour vérification)
 */
void sha256_cpu(const uint8_t *data, size_t len, uint8_t *hash) {
    // Implémentation standard SHA256 en CPU
    // (Pour debug et vérification des résultats GPU)
    
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    
    // TODO: Implémenter version CPU complète
    // Pour l'instant, version simplifiée
    
    memcpy(hash, state, 32);
}

/*
 * Vérifier qu'un hash est valide pour un target donné
 */
int verify_hash(uint32_t *hash, uint32_t target) {
    // Bitcoin compare en little-endian
    if (hash[7] != 0) return 0;
    if (hash[6] >= target) return 0;
    return 1;
}

/*
 * Calculer la difficulté depuis le target
 */
double target_to_difficulty(uint32_t target) {
    if (target == 0) return 0.0;
    return (double)0x00000000FFFF0000ULL / (double)target;
}

/*
 * Convertir difficulté en target
 */
uint32_t difficulty_to_target(double difficulty) {
    if (difficulty == 0.0) return 0xFFFFFFFF;
    return (uint32_t)((double)0x00000000FFFF0000ULL / difficulty);
}
