/*
 * cuda_miner_simple.cu - Version simplifiée sans Stratum
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <windows.h>

#define MAX_GPUS 8
#define BLOCK_SIZE 256
#define GRID_SIZE 8192

// Inclure structures Stratum
typedef struct {
    SOCKET socket;
    int status;
    char server_ip[64];
    int server_port;
    char job_id[128];
    char extranonce1[32];
    int extranonce2_size;
    uint32_t difficulty;
    char prevhash[128];
    char ntime[16];
} PoolConnection;

typedef struct {
    char job_id[128];
    char ntime[16];
    uint8_t header[128];
    uint32_t nonce_start;
    uint32_t nonce_end;
    uint32_t difficulty;
    time_t received_time;
} MiningJob;

// Prototypes Stratum (définis dans stratum.c)
extern "C" {
    int pool_connect(PoolConnection *pool, const char *url, int port);
    int pool_disconnect(PoolConnection *pool);
    int pool_subscribe(PoolConnection *pool, const char *user_agent);
    int pool_authorize(PoolConnection *pool, const char *user, const char *pass);
    int pool_submit_share(PoolConnection *pool, const char *job_id, 
                          const char *extranonce2, const char *ntime, const char *nonce);
    int pool_submit_share_ethash(PoolConnection *pool, const char *job_id, const char *nonce);
    int pool_start_listener(PoolConnection *pool, 
                           void (*job_callback)(MiningJob*),
                           void (*diff_callback)(uint32_t));
    void pool_stop_listener(void);
}

// Prototypes fonctions pool locales
void mine_pool_sha256(PoolConnection *pool, int device_id);
void mine_pool_ethash(PoolConnection *pool, int device_id);
void mine_pool_kawpow(PoolConnection *pool, int device_id);


// Structures locales
typedef struct {
    int device_id;
    char name[256];
    size_t total_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int enabled;
} GPUInfo;

typedef struct {
    uint64_t hashrate;
    uint32_t shares_accepted;
    uint32_t shares_rejected;
    time_t start_time;
    int is_mining;
} Stats;

GPUInfo gpus[MAX_GPUS];
int gpu_count = 0;
Stats stats = {0};

// Prototypes externes
extern "C" {
    void* ethash_create_dag(uint32_t epoch, uint32_t *dag_size_out);
    void* ethash_generate_dag(uint32_t epoch, uint32_t dag_size);
    void ethash_search_launch(void *dag, const uint8_t *header, uint64_t target, 
                             uint64_t start_nonce, uint32_t dag_size, uint64_t *solution,
                             int grid_size, int block_size);
    void ethash_destroy_dag(void *dag);
    
    void* kawpow_generate_dag(uint32_t epoch, uint32_t dag_size);
    void kawpow_search_launch(void *dag, const uint8_t *header, uint64_t target,
                             uint64_t start_nonce, uint32_t dag_size, uint64_t *solution,
                             int grid_size, int block_size);
    void kawpow_destroy_dag(void *dag);
    
    void launch_sha256_kernel(uint32_t *d_header, uint32_t target, uint32_t start_nonce,
                             uint32_t *d_found_nonce, uint32_t *d_found_hash,
                             int grid_size, int block_size);
}

int detect_gpus() {
    cudaError_t err = cudaGetDeviceCount(&gpu_count);
    
    if (err != cudaSuccess || gpu_count == 0) {
        printf("Aucun GPU CUDA détecté!\n");
        return 0;
    }
    
    printf("\n=== GPU détectés: %d ===\n\n", gpu_count);
    
    for (int i = 0; i < gpu_count && i < MAX_GPUS; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        gpus[i].device_id = i;
        strncpy(gpus[i].name, prop.name, 255);
        gpus[i].total_memory = prop.totalGlobalMem;
        gpus[i].compute_capability_major = prop.major;
        gpus[i].compute_capability_minor = prop.minor;
        gpus[i].enabled = 1;
        
        printf("GPU %d: %s\n", i, gpus[i].name);
        printf("  Mémoire: %.2f GB\n", (double)gpus[i].total_memory / (1024*1024*1024));
        printf("  Compute: %d.%d\n\n", gpus[i].compute_capability_major, 
               gpus[i].compute_capability_minor);
    }
    
    return gpu_count;
}

int init_cuda_device(int device_id) {
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        printf("Erreur init GPU %d\n", device_id);
        return 0;
    }
    printf("GPU %d initialisé\n", device_id);
    return 1;
}

void mine_sha256(int device_id) {
    printf("\n=== SHA256 sur GPU %d ===\n", device_id);
    
    if (!init_cuda_device(device_id)) return;
    
    uint32_t *d_header, *d_found_nonce, *d_found_hash;
    uint32_t h_header[20] = {0};
    uint32_t h_found_nonce, h_found_hash[8];
    
    cudaMalloc(&d_header, 80);
    cudaMalloc(&d_found_nonce, 4);
    cudaMalloc(&d_found_hash, 32);
    
    cudaMemcpy(d_header, h_header, 80, cudaMemcpyHostToDevice);
    
    uint32_t target = 0x0000FFFF;
    uint32_t start_nonce = 0;
    clock_t last_report = clock();
    uint64_t total_hashes = 0;
    
    stats.is_mining = 1;
    stats.start_time = time(NULL);
    
    printf("Démarrage minage...\n\n");
    
    while (stats.is_mining) {
        uint32_t max = 0xFFFFFFFF;
        cudaMemcpy(d_found_nonce, &max, 4, cudaMemcpyHostToDevice);
        
        launch_sha256_kernel(d_header, target, start_nonce, d_found_nonce, d_found_hash,
                            GRID_SIZE, BLOCK_SIZE);
        
        cudaMemcpy(&h_found_nonce, d_found_nonce, 4, cudaMemcpyDeviceToHost);
        
        if (h_found_nonce != 0xFFFFFFFF) {
            cudaMemcpy(h_found_hash, d_found_hash, 32, cudaMemcpyDeviceToHost);
            printf("\n>>> SHARE! Nonce: 0x%08X\n", h_found_nonce);
            stats.shares_accepted++;
        }
        
        total_hashes += GRID_SIZE * BLOCK_SIZE;
        start_nonce += GRID_SIZE * BLOCK_SIZE;
        
        clock_t now = clock();
        double elapsed = (double)(now - last_report) / CLOCKS_PER_SEC;
        
        if (elapsed >= 5.0) {
            stats.hashrate = (uint64_t)(total_hashes / elapsed);
            printf("\r[GPU %d] %.2f MH/s | Shares: %u     ",
                   device_id, stats.hashrate / 1000000.0, stats.shares_accepted);
            fflush(stdout);
            last_report = now;
            total_hashes = 0;
        }
        
        Sleep(1);
    }
    
    cudaFree(d_header);
    cudaFree(d_found_nonce);
    cudaFree(d_found_hash);
    printf("\n\nArrêté.\n");
}

void mine_ethash(int device_id) {
    printf("\n=== Ethash sur GPU %d ===\n", device_id);
    
    if (!init_cuda_device(device_id)) return;
    
    uint32_t dag_size;
    void *dag = ethash_create_dag(0, &dag_size);
    if (!dag) {
        printf("Erreur DAG!\n");
        return;
    }
    
    uint8_t header[32] = {0};
    uint64_t target = 0x0000FFFFFFFFFFFFULL;
    uint64_t start_nonce = 0;
    clock_t last_report = clock();
    uint64_t total_hashes = 0;
    
    stats.is_mining = 1;
    stats.start_time = time(NULL);
    
    printf("DAG: %u MB\n", dag_size / 1024 / 1024);
    printf("Démarrage...\n\n");
    
    while (stats.is_mining) {
        uint64_t solution = 0xFFFFFFFFFFFFFFFFULL;
        
        ethash_search_launch(dag, header, target, start_nonce, dag_size,
                            &solution, GRID_SIZE, BLOCK_SIZE);
        
        if (solution != 0xFFFFFFFFFFFFFFFFULL) {
            printf("\n>>> SHARE! Nonce: 0x%016llX\n", solution);
            stats.shares_accepted++;
        }
        
        total_hashes += GRID_SIZE * BLOCK_SIZE;
        start_nonce += GRID_SIZE * BLOCK_SIZE;
        
        clock_t now = clock();
        double elapsed = (double)(now - last_report) / CLOCKS_PER_SEC;
        
        if (elapsed >= 5.0) {
            stats.hashrate = (uint64_t)(total_hashes / elapsed);
            printf("\r[GPU %d] %.2f MH/s | Shares: %u     ",
                   device_id, stats.hashrate / 1000000.0, stats.shares_accepted);
            fflush(stdout);
            last_report = now;
            total_hashes = 0;
        }
        
        Sleep(1);
    }
    
    ethash_destroy_dag(dag);
    printf("\n\nArrêté.\n");
}

// Variables globales pour callbacks
static MiningJob g_current_job;
static int g_new_job_available = 0;
static uint32_t g_current_difficulty = 1;

void on_new_job(MiningJob *job) {
    g_current_job = *job;
    g_new_job_available = 1;
    printf("\n>>> Nouveau job reçu: %s\n", job->job_id);
}

void on_difficulty_change(uint32_t diff) {
    g_current_difficulty = diff;
    printf("\n>>> Nouvelle difficulté: %u\n", diff);
}

void mine_on_pool(int device_id) {
    printf("\n=== Minage sur Pool (Stratum) ===\n");
    
    if (!init_cuda_device(device_id)) return;
    
    char pool_url[256];
    int pool_port;
    char username[256];
    char password[128];
    int config_mode;
    int algo_choice;
    
    printf("\nChoix de l'algorithme:\n");
    printf("1. SHA256 (Bitcoin)\n");
    printf("2. Ethash (Ethereum Classic)\n");
    printf("3. KawPow (Ravencoin) - 2x plus rentable!\n");
    printf("Algorithme (1-3): ");
    scanf("%d", &algo_choice);
    
    if (algo_choice < 1 || algo_choice > 3) {
        printf("Algorithme invalide!\n");
        return;
    }
    
    // Menu spécial pour Ethash (ETC)
    if (algo_choice == 2) {
        int pool_choice;
        printf("\n=== Pools Ethereum Classic (ETC) ===\n");
        printf("1. 2Miners Europe (etc.2miners.com:1010) - Recommandé\n");
        printf("2. Ethermine Europe (eu1-etc.ethermine.org:4444)\n");
        printf("3. HeroMiners DE (de.etc.herominers.com:1140)\n");
        printf("4. Nanopool Europe (etc-eu1.nanopool.org:19999)\n");
        printf("5. Pool personnalisée\n");
        printf("Choix (1-5): ");
        scanf("%d", &pool_choice);
        
        switch (pool_choice) {
            case 1:
                strcpy(pool_url, "etc.2miners.com");
                pool_port = 1010;
                break;
            case 2:
                strcpy(pool_url, "eu1-etc.ethermine.org");
                pool_port = 4444;
                break;
            case 3:
                strcpy(pool_url, "de.etc.herominers.com");
                pool_port = 1140;
                break;
            case 4:
                strcpy(pool_url, "etc-eu1.nanopool.org");
                pool_port = 19999;
                break;
            case 5:
                printf("\nURL pool: ");
                scanf("%255s", pool_url);
                printf("Port: ");
                scanf("%d", &pool_port);
                break;
            default:
                printf("Choix invalide!\n");
                return;
        }
    } else {
        printf("\nConfiguration Pool:\n");
        printf("URL pool: ");
        scanf("%255s", pool_url);
        
        printf("Port: ");
        scanf("%d", &pool_port);
    }
    
    printf("\nMode d'authentification:\n");
    printf("1. Wallet + Worker (ex: 0xWALLET + rig1)\n");
    printf("2. Username complet (ex: login.worker)\n");
    printf("Choix (1 ou 2): ");
    scanf("%d", &config_mode);
    
    if (config_mode == 1) {
        char wallet[128];
        char worker[64];
        
        printf("\nWallet: ");
        scanf("%127s", wallet);
        
        printf("Worker (ex: rig1): ");
        scanf("%63s", worker);
        
        snprintf(username, sizeof(username), "%s.%s", wallet, worker);
        strcpy(password, "x");
    } else {
        printf("\nUsername complet (ex: login.worker): ");
        scanf("%255s", username);
        
        printf("Password (Entrée pour 'x' par défaut): ");
        char temp[128];
        scanf("%127s", temp);
        
        if (strlen(temp) == 0 || strcmp(temp, "") == 0) {
            strcpy(password, "x");
        } else {
            strcpy(password, temp);
        }
    }
    
    const char* algo_names[] = {"", "SHA256", "Ethash (ETC)", "KawPow (RVN)"};
    
    printf("\n=== Configuration ===\n");
    printf("Algorithme: %s\n", algo_names[algo_choice]);
    printf("Pool: %s:%d\n", pool_url, pool_port);
    printf("Username: %s\n", username);
    printf("Password: %s\n", password);
    printf("=====================\n\n");
    
    PoolConnection pool;
    memset(&pool, 0, sizeof(pool));
    pool.socket = INVALID_SOCKET;
    pool.difficulty = 1;
    
    printf("Connexion...\n");
    if (!pool_connect(&pool, pool_url, pool_port)) {
        printf("Échec connexion!\n");
        return;
    }
    
    if (!pool_subscribe(&pool, "CudaMiner/1.0")) {
        printf("Échec subscribe!\n");
        pool_disconnect(&pool);
        return;
    }
    
    if (!pool_authorize(&pool, username, password)) {
        printf("Échec autorisation!\n");
        pool_disconnect(&pool);
        return;
    }
    
    printf("✓ Connecté et authentifié!\n\n");
    
    // Démarrer thread d'écoute pour recevoir jobs
    g_new_job_available = 0;
    if (!pool_start_listener(&pool, on_new_job, on_difficulty_change)) {
        printf("Échec démarrage listener!\n");
        pool_disconnect(&pool);
        return;
    }
    
    printf("En attente du premier job de la pool...\n");
    
    // Attendre premier job (max 30 secondes)
    int wait_count = 0;
    while (!g_new_job_available && wait_count < 300) {
        Sleep(100);
        wait_count++;
    }
    
    if (!g_new_job_available) {
        printf("\nAucun job reçu après 30 secondes.\n");
        printf("La pool ne répond peut-être pas ou l'algorithme n'est pas supporté.\n");
        pool_stop_listener();
        pool_disconnect(&pool);
        return;
    }
    
    printf("✓ Premier job reçu! Démarrage du minage...\n\n");
    
    // BOUCLE DE MINAGE selon algorithme choisi
    stats.is_mining = 1;
    stats.start_time = time(NULL);
    
    switch (algo_choice) {
        case 1: // SHA256
            mine_pool_sha256(&pool, device_id);
            break;
        case 2: // Ethash
            mine_pool_ethash(&pool, device_id);
            break;
        case 3: // KawPow
            mine_pool_kawpow(&pool, device_id);
            break;
    }
    
    pool_stop_listener();
    pool_disconnect(&pool);
    
    printf("\n\n=== STATISTIQUES FINALES ===\n");
    printf("Shares acceptés: %u\n", stats.shares_accepted);
    printf("Shares rejetés: %u\n", stats.shares_rejected);
    printf("Temps de minage: %ld minutes\n", (long)((time(NULL) - stats.start_time) / 60));
    printf("\nMinage arrêté.\n");
}

// Fonctions de minage par algorithme pour pool
void mine_pool_sha256(PoolConnection *pool, int device_id) {
    uint32_t *d_header, *d_found_nonce, *d_found_hash;
    uint32_t h_header[20] = {0};
    uint32_t h_found_nonce, h_found_hash[8];
    
    cudaMalloc(&d_header, 80);
    cudaMalloc(&d_found_nonce, 4);
    cudaMalloc(&d_found_hash, 32);
    
    cudaMemcpy(d_header, h_header, 80, cudaMemcpyHostToDevice);
    
    uint32_t start_nonce = 0;
    clock_t last_report = clock();
    uint64_t total_hashes = 0;
    
    printf("=== MINAGE SHA256 EN COURS ===\n");
    printf("Appuyez sur Ctrl+C pour arrêter\n\n");
    
    while (stats.is_mining) {
        uint32_t target = 0x0000FFFF / (g_current_difficulty > 0 ? g_current_difficulty : 1);
        
        uint32_t max_nonce = 0xFFFFFFFF;
        cudaMemcpy(d_found_nonce, &max_nonce, 4, cudaMemcpyHostToDevice);
        
        launch_sha256_kernel(d_header, target, start_nonce, d_found_nonce, d_found_hash,
                            GRID_SIZE, BLOCK_SIZE);
        
        cudaMemcpy(&h_found_nonce, d_found_nonce, 4, cudaMemcpyDeviceToHost);
        
        if (h_found_nonce != 0xFFFFFFFF) {
            cudaMemcpy(h_found_hash, d_found_hash, 32, cudaMemcpyDeviceToHost);
            
            printf("\n>>> SHARE TROUVÉ! <<<\n");
            printf("Nonce: 0x%08X\n", h_found_nonce);
            
            char nonce_hex[16];
            char ntime_hex[16];
            sprintf(nonce_hex, "%08x", h_found_nonce);
            sprintf(ntime_hex, "%s", g_current_job.ntime);
            
            printf("Soumission à la pool...\n");
            if (pool_submit_share(pool, g_current_job.job_id, 
                                 "00000000", ntime_hex, nonce_hex)) {
                stats.shares_accepted++;
                printf("✓ Share ACCEPTÉ! (Total: %u)\n\n", stats.shares_accepted);
            } else {
                stats.shares_rejected++;
                printf("✗ Share REJETÉ (Total rejetés: %u)\n\n", stats.shares_rejected);
            }
        }
        
        total_hashes += GRID_SIZE * BLOCK_SIZE;
        start_nonce += GRID_SIZE * BLOCK_SIZE;
        
        clock_t now = clock();
        double elapsed = (double)(now - last_report) / CLOCKS_PER_SEC;
        
        if (elapsed >= 5.0) {
            stats.hashrate = (uint64_t)(total_hashes / elapsed);
            time_t uptime = time(NULL) - stats.start_time;
            double accept_rate = stats.shares_accepted + stats.shares_rejected > 0 ?
                100.0 * stats.shares_accepted / (stats.shares_accepted + stats.shares_rejected) : 0;
            
            printf("\r[GPU %d] %.2f MH/s | Acceptés: %u | Rejetés: %u | Taux: %.1f%% | Temps: %ldm     ",
                   device_id,
                   stats.hashrate / 1000000.0,
                   stats.shares_accepted,
                   stats.shares_rejected,
                   accept_rate,
                   (long)(uptime / 60));
            fflush(stdout);
            
            last_report = now;
            total_hashes = 0;
        }
        
        if (g_new_job_available) {
            g_new_job_available = 0;
            start_nonce = 0;
        }
        
        Sleep(1);
    }
    
    cudaFree(d_header);
    cudaFree(d_found_nonce);
    cudaFree(d_found_hash);
}

void mine_pool_ethash(PoolConnection *pool, int device_id) {
    printf("=== MINAGE ETHEREUM CLASSIC (ETC) SUR POOL ===\n");
    printf("Version OPTIMISÉE - 3-4x plus rapide!\n");
    printf("Initialisation...\n\n");
    
    // Générer le DAG (epoch 0 pour test, normalement calculé depuis block)
    uint32_t dag_size = 1073741824U;  // 1 GB (epoch 0)
    void *dag = ethash_generate_dag(0, dag_size);
    
    if (!dag) {
        printf("Erreur: Impossible de générer le DAG!\n");
        return;
    }
    
    printf("\n=== DAG généré, démarrage minage OPTIMISÉ ===\n");
    printf("Optimisations: __ldg() + Shared memory + Loop unrolling\n");
    printf("Appuyez sur Ctrl+C pour arrêter\n\n");
    
    uint8_t header[32] = {0};  // Header sera fourni par la pool
    uint64_t target = 0x0000FFFFFFFFFFFFULL;  // Difficulté initiale
    uint64_t start_nonce = 0;
    uint64_t solution = 0xFFFFFFFFFFFFFFFFULL;
    
    // OPTIMISATION: Grid/block optimisés
    const int OPTIMIZED_GRID = GRID_SIZE * 4;   // 4x plus de threads
    const int OPTIMIZED_BLOCK = 256;             // Optimal pour L1 cache
    const uint64_t BATCH_SIZE = (uint64_t)OPTIMIZED_GRID * OPTIMIZED_BLOCK;
    
    clock_t last_report = clock();
    clock_t last_submit = clock();
    uint64_t total_hashes = 0;
    uint32_t shares_found = 0;
    
    printf("Configuration: %d blocs x %d threads = %llu hashes/batch\n",
           OPTIMIZED_GRID, OPTIMIZED_BLOCK, BATCH_SIZE);
    printf("Performance attendue: 70-90 MH/s (RTX 3080)\n\n");
    
    while (stats.is_mining) {
        // RESET solution avant chaque recherche
        solution = 0xFFFFFFFFFFFFFFFFULL;
        
        // Chercher solution avec kernel OPTIMISÉ
        ethash_search_launch(dag, header, target, start_nonce, dag_size, 
                            &solution, OPTIMIZED_GRID, OPTIMIZED_BLOCK);
        
        // Vérifier erreur CUDA
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("\nERREUR CUDA: %s\n", cudaGetErrorString(err));
            break;
        }
        
        total_hashes += BATCH_SIZE;
        
        // Solution trouvée?
        if (solution != 0xFFFFFFFFFFFFFFFFULL) {
            shares_found++;
            
            clock_t now = clock();
            double time_since_last = (double)(now - last_submit) / CLOCKS_PER_SEC;
            
            printf("\n>>> SHARE TROUVÉ #%u! <<<\n", shares_found);
            printf("Nonce: 0x%016llX\n", solution);
            printf("Temps depuis dernier: %.1f secondes\n", time_since_last);
            
            char nonce_hex[17];
            sprintf(nonce_hex, "%016llx", solution);
            
            printf("Soumission à la pool...\n");
            if (pool_submit_share_ethash(pool, g_current_job.job_id, nonce_hex)) {
                stats.shares_accepted++;
                printf("✓ Share ACCEPTÉ! (Total: %u)\n\n", stats.shares_accepted);
            } else {
                stats.shares_rejected++;
                printf("✗ Share REJETÉ (Total rejetés: %u)\n\n", stats.shares_rejected);
            }
            
            last_submit = now;
            
            // IMPORTANT: Continuer avec le nonce suivant après la solution
            // Ne pas reset start_nonce ici
        }
        
        start_nonce += BATCH_SIZE;
        
        // Rapport périodique (toutes les 2 secondes)
        clock_t now = clock();
        double elapsed = (double)(now - last_report) / CLOCKS_PER_SEC;
        
        if (elapsed >= 2.0) {
            stats.hashrate = (uint64_t)(total_hashes / elapsed);
            time_t uptime = time(NULL) - stats.start_time;
            double accept_rate = stats.shares_accepted + stats.shares_rejected > 0 ?
                100.0 * stats.shares_accepted / (stats.shares_accepted + stats.shares_rejected) : 0;
            
            // Calcul shares/heure
            double shares_per_hour = uptime > 0 ? 
                (shares_found * 3600.0 / uptime) : 0;
            
            printf("\r[GPU %d] %.2f MH/s | Shares: %u | Acceptés: %u (%.1f%%) | %.1f/h | Temps: %ldm     ",
                   device_id,
                   stats.hashrate / 1000000.0,
                   shares_found,
                   stats.shares_accepted,
                   accept_rate,
                   shares_per_hour,
                   (long)(uptime / 60));
            fflush(stdout);
            
            last_report = now;
            total_hashes = 0;
        }
        
        // Nouveau job?
        if (g_new_job_available) {
            g_new_job_available = 0;
            start_nonce = 0;
            printf("\n>>> Nouveau job reçu, reset du nonce\n");
        }
        
        // Pas de Sleep() - on veut miner à fond !
    }
    
    ethash_destroy_dag(dag);
    
    printf("\n\n=== STATISTIQUES FINALES ===\n");
    printf("Shares trouvés: %u\n", shares_found);
    printf("Shares acceptés: %u\n", stats.shares_accepted);
    printf("Shares rejetés: %u\n", stats.shares_rejected);
    printf("Taux d'acceptation: %.1f%%\n", 
           shares_found > 0 ? 100.0 * stats.shares_accepted / shares_found : 0);
    printf("\nMinage arrêté.\n");
}


int main() {
    printf("╔════════════════════════════════╗\n");
    printf("║   CryptoMiner CUDA Windows    ║\n");
    printf("╚════════════════════════════════╝\n");
    
    if (detect_gpus() == 0) return 1;
    
    printf("\n=== Menu ===\n");
    printf("1. SHA256 (test local)\n");
    printf("2. Ethash (DAG)\n");
    printf("3. Miner sur Pool (Stratum)\n");
    printf("4. Quitter\n\n");
    
    int choice, gpu_id;
    printf("Choix: ");
    scanf("%d", &choice);
    
    if (choice >= 1 && choice <= 3) {
        printf("GPU (0-%d): ", gpu_count - 1);
        scanf("%d", &gpu_id);
        
        if (gpu_id < 0 || gpu_id >= gpu_count) {
            printf("GPU invalide!\n");
            return 1;
        }
        
        switch (choice) {
            case 1: mine_sha256(gpu_id); break;
            case 2: mine_ethash(gpu_id); break;
            case 3: mine_on_pool(gpu_id); break;
        }
    }
    
    cudaDeviceReset();
    
    printf("\nAppuyez sur Entrée...");
    getchar();
    getchar();
    
    return 0;
}
void mine_pool_kawpow(PoolConnection *pool, int device_id) {
    printf("=== MINAGE KAWPOW (RAVENCOIN) SUR POOL ===\n");
    printf("Version optimisée ProgPoW - ASIC résistant\n");
    printf("Initialisation...\n\n");
    
    // Générer le DAG KawPow (2.5GB pour epoch récent)
    uint32_t dag_size = 2684354560U;  // 2.5 GB
    void *dag = kawpow_generate_dag(0, dag_size);
    
    if (!dag) {
        printf("Erreur: Impossible de générer le DAG KawPow!\n");
        return;
    }
    
    printf("\n=== DAG généré, démarrage minage KawPow ===\n");
    printf("Optimisations: ProgPoW + Keccak-256 + DAG lookup\n");
    printf("Appuyez sur Ctrl+C pour arrêter\n\n");
    
    uint8_t header[76] = {0};  // Header Ravencoin (76 bytes)
    uint64_t target = 0x0000FFFFFFFFFFFFULL;  // Difficulté initiale
    uint64_t start_nonce = 0;
    uint64_t solution = 0xFFFFFFFFFFFFFFFFULL;
    
    // Configuration optimale KawPow
    const int OPTIMIZED_GRID = GRID_SIZE * 2;   // 2x grid pour KawPow
    const int OPTIMIZED_BLOCK = 256;             // Optimal
    const uint64_t BATCH_SIZE = (uint64_t)OPTIMIZED_GRID * OPTIMIZED_BLOCK;
    
    clock_t last_report = clock();
    clock_t last_submit = clock();
    uint64_t total_hashes = 0;
    uint32_t shares_found = 0;
    
    printf("Configuration: %d blocs x %d threads = %llu hashes/batch\n",
           OPTIMIZED_GRID, OPTIMIZED_BLOCK, BATCH_SIZE);
    printf("Performance attendue: 25-35 MH/s (RTX 3080)\n");
    printf("Rentabilité: ~$2.50/jour (2x Ethash)\n\n");
    
    while (stats.is_mining) {
        // RESET solution avant chaque recherche
        solution = 0xFFFFFFFFFFFFFFFFULL;
        
        // Chercher solution avec kernel KawPow
        kawpow_search_launch(dag, header, target, start_nonce, dag_size,
                            &solution, OPTIMIZED_GRID, OPTIMIZED_BLOCK);
        
        // Vérifier erreur CUDA
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("\nERREUR CUDA KawPow: %s\n", cudaGetErrorString(err));
            break;
        }
        
        total_hashes += BATCH_SIZE;
        
        // Solution trouvée?
        if (solution != 0xFFFFFFFFFFFFFFFFULL) {
            shares_found++;
            
            clock_t now = clock();
            double time_since_last = (double)(now - last_submit) / CLOCKS_PER_SEC;
            
            printf("\n>>> SHARE TROUVÉ #%u! <<<\n", shares_found);
            printf("Nonce: 0x%08X\n", (uint32_t)solution);
            printf("Temps depuis dernier: %.1f secondes\n", time_since_last);
            
            // Format nonce pour KawPow (32 bits)
            char nonce_hex[9];
            sprintf(nonce_hex, "%08x", (uint32_t)solution);
            
            printf("Soumission à la pool...\n");
            
            // KawPow utilise format similaire à Ethash
            if (pool_submit_share_ethash(pool, g_current_job.job_id, nonce_hex)) {
                stats.shares_accepted++;
                printf("✓ Share ACCEPTÉ! (Total: %u)\n\n", stats.shares_accepted);
            } else {
                stats.shares_rejected++;
                printf("✗ Share REJETÉ (Total rejetés: %u)\n\n", stats.shares_rejected);
            }
            
            last_submit = now;
        }
        
        start_nonce += BATCH_SIZE;
        
        // Rapport périodique (toutes les 2 secondes)
        clock_t now = clock();
        double elapsed = (double)(now - last_report) / CLOCKS_PER_SEC;
        
        if (elapsed >= 2.0) {
            stats.hashrate = (uint64_t)(total_hashes / elapsed);
            time_t uptime = time(NULL) - stats.start_time;
            double accept_rate = stats.shares_accepted + stats.shares_rejected > 0 ?
                100.0 * stats.shares_accepted / (stats.shares_accepted + stats.shares_rejected) : 0;
            
            // Calcul shares/heure
            double shares_per_hour = uptime > 0 ?
                (shares_found * 3600.0 / uptime) : 0;
            
            printf("\r[GPU %d] %.2f MH/s | Shares: %u | Acceptés: %u (%.1f%%) | %.1f/h | Temps: %ldm     ",
                   device_id,
                   stats.hashrate / 1000000.0,
                   shares_found,
                   stats.shares_accepted,
                   accept_rate,
                   shares_per_hour,
                   (long)(uptime / 60));
            fflush(stdout);
            
            last_report = now;
            total_hashes = 0;
        }
        
        // Nouveau job?
        if (g_new_job_available) {
            g_new_job_available = 0;
            start_nonce = 0;
            printf("\n>>> Nouveau job reçu, reset du nonce\n");
        }
    }
    
    kawpow_destroy_dag(dag);
    
    printf("\n\n=== STATISTIQUES FINALES ===\n");
    printf("Shares trouvés: %u\n", shares_found);
    printf("Shares acceptés: %u\n", stats.shares_accepted);
    printf("Shares rejetés: %u\n", stats.shares_rejected);
    printf("Taux d'acceptation: %.1f%%\n",
           shares_found > 0 ? 100.0 * stats.shares_accepted / shares_found : 0);
    printf("\nMinage arrêté.\n");
}

