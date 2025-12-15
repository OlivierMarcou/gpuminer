/*
 * stratum.c - Client Stratum complet avec threading
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <process.h>
#include "cJSON.h"

#pragma comment(lib, "ws2_32.lib")

// Structures locales (copiées de miner.h pour autonomie)

// Forward declaration
typedef struct MiningJob MiningJob;

typedef enum {
    POOL_DISCONNECTED = 0,
    POOL_CONNECTING = 1,
    POOL_CONNECTED = 2,
    POOL_AUTHORIZED = 3
} PoolStatus;

typedef struct {
    SOCKET socket;
    PoolStatus status;
    char server_ip[64];
    int server_port;
    char username[256];
    char job_id[128];
    char extranonce1[32];
    int extranonce2_size;
    uint32_t difficulty;
    char prevhash[128];
    char coinb1[256];
    char coinb2[256];
    char merkle_branches[16][128];
    int merkle_count;
    char version[16];
    char nbits[16];
    char ntime[16];
    void (*job_callback)(MiningJob*);
    void (*diff_callback)(uint32_t);
} PoolConnection;

typedef struct MiningJob {
    char job_id[128];
    char ntime[16];
    uint8_t header[128];
    uint32_t nonce_start;
    uint32_t nonce_end;
    uint32_t difficulty;
    time_t received_time;
    
    // KawPow specific
    uint8_t header_hash[32];    // headerHash from pool
    uint8_t seed_hash[32];      // seedHash for DAG
    uint8_t target[32];         // target difficulty
    uint32_t height;            // block height
} MiningJob;

static WSADATA wsa_data;
static int wsa_initialized = 0;
static HANDLE listen_thread = NULL;
static volatile int keep_listening = 0;

typedef struct {
    PoolConnection *pool;
    void (*job_callback)(MiningJob *job);
    void (*diff_callback)(uint32_t diff);
} ListenThreadData;

// Prototypes de fonctions
void pool_parse_notify(PoolConnection *pool, const char *json, MiningJob *job);
static void parse_set_difficulty(PoolConnection *pool, const char *json);

static int init_winsock() {
    if (wsa_initialized) return 1;
    
    if (WSAStartup(MAKEWORD(2, 2), &wsa_data) != 0) {
        printf("Erreur WSAStartup: %d\n", WSAGetLastError());
        return 0;
    }
    
    wsa_initialized = 1;
    return 1;
}

int pool_connect(PoolConnection *pool, const char *url, int port) {
    if (!init_winsock()) return 0;
    
    printf("Connexion à %s:%d...\n", url, port);
    
    struct addrinfo hints, *result;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    
    char port_str[16];
    sprintf(port_str, "%d", port);
    
    if (getaddrinfo(url, port_str, &hints, &result) != 0) {
        printf("Erreur résolution DNS: %d\n", WSAGetLastError());
        return 0;
    }
    
    pool->socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (pool->socket == INVALID_SOCKET) {
        printf("Erreur création socket: %d\n", WSAGetLastError());
        freeaddrinfo(result);
        return 0;
    }
    
    DWORD timeout = 30000;
    setsockopt(pool->socket, SOL_SOCKET, SO_RCVTIMEO, (char*)&timeout, sizeof(timeout));
    setsockopt(pool->socket, SOL_SOCKET, SO_SNDTIMEO, (char*)&timeout, sizeof(timeout));
    
    if (connect(pool->socket, result->ai_addr, (int)result->ai_addrlen) < 0) {
        printf("Erreur connexion: %d\n", WSAGetLastError());
        closesocket(pool->socket);
        freeaddrinfo(result);
        return 0;
    }
    
    freeaddrinfo(result);
    
    strncpy(pool->server_ip, url, sizeof(pool->server_ip) - 1);
    pool->server_port = port;
    pool->status = POOL_CONNECTED;
    
    printf("Connecté!\n");
    return 1;
}

int pool_disconnect(PoolConnection *pool) {
    if (pool->socket != INVALID_SOCKET) {
        closesocket(pool->socket);
        pool->socket = INVALID_SOCKET;
    }
    pool->status = POOL_DISCONNECTED;
    return 1;
}

static int send_json(SOCKET sock, const char *json) {
    char buffer[4096];
    int len = snprintf(buffer, sizeof(buffer), "%s\n", json);
    
    int sent = send(sock, buffer, len, 0);
    if (sent < 0) {
        printf("Erreur envoi: %d\n", WSAGetLastError());
        return 0;
    }
    
    printf(">>> %s\n", json);
    return 1;
}

int pool_receive_message(PoolConnection *pool, char *buffer, int max_len) {
    int received = recv(pool->socket, buffer, max_len - 1, 0);
    
    if (received < 0) {
        if (WSAGetLastError() == WSAETIMEDOUT) {
            return 0;
        }
        printf("Erreur réception: %d\n", WSAGetLastError());
        return -1;
    }
    
    if (received == 0) {
        printf("Connexion fermée par la pool\n");
        return -1;
    }
    
    buffer[received] = '\0';
    return received;
}

int pool_subscribe(PoolConnection *pool, const char *user_agent) {
    char json[512];
    snprintf(json, sizeof(json),
        "{\"id\":1,\"method\":\"mining.subscribe\",\"params\":[\"%s\"]}",
        user_agent);
    
    if (!send_json(pool->socket, json)) return 0;
    
    char buffer[4096];
    int len = pool_receive_message(pool, buffer, sizeof(buffer));
    if (len <= 0) return 0;
    
    printf("<<< %s\n", buffer);
    
    cJSON *root = cJSON_Parse(buffer);
    if (!root) return 0;
    
    cJSON *result = cJSON_GetObjectItem(root, "result");
    if (result && cJSON_IsArray(result)) {
        cJSON *subscriptions = cJSON_GetArrayItem(result, 0);
        
        cJSON *extranonce1 = cJSON_GetArrayItem(result, 1);
        if (cJSON_IsString(extranonce1)) {
            strncpy(pool->extranonce1, cJSON_GetStringValue(extranonce1), 
                   sizeof(pool->extranonce1) - 1);
        }
        
        cJSON *extranonce2_size = cJSON_GetArrayItem(result, 2);
        if (cJSON_IsNumber(extranonce2_size)) {
            pool->extranonce2_size = (int)cJSON_GetNumberValue(extranonce2_size);
        }
        
        printf("Extranonce1: %s\n", pool->extranonce1);
        printf("Extranonce2 size: %d\n", pool->extranonce2_size);
    }
    
    cJSON_Delete(root);
    return 1;
}

int pool_authorize(PoolConnection *pool, const char *user, const char *pass) {
    // Stocker le username pour utilisation dans mining.submit
    strncpy(pool->username, user, sizeof(pool->username) - 1);
    pool->username[sizeof(pool->username) - 1] = '\0';
    
    char json[512];
    snprintf(json, sizeof(json),
        "{\"id\":2,\"method\":\"mining.authorize\",\"params\":[\"%s\",\"%s\"]}",
        user, pass);
    
    if (!send_json(pool->socket, json)) return 0;
    
    char buffer[4096];
    int len = pool_receive_message(pool, buffer, sizeof(buffer));
    if (len <= 0) return 0;
    
    printf("<<< %s\n", buffer);
    
    cJSON *root = cJSON_Parse(buffer);
    if (!root) return 0;
    
    cJSON *result = cJSON_GetObjectItem(root, "result");
    int authorized = cJSON_IsTrue(result);
    
    cJSON_Delete(root);
    
    if (authorized) {
        pool->status = POOL_AUTHORIZED;
        printf("Authentification réussie!\n");
        return 1;
    }
    
    printf("Authentification échouée!\n");
    return 0;
}

int pool_submit_share(PoolConnection *pool, const char *job_id, 
                      const char *extranonce2, const char *ntime,
                      const char *nonce) {
    static int submit_id = 100;
    char json[1024];
    
    snprintf(json, sizeof(json),
        "{\"id\":%d,\"method\":\"mining.submit\",\"params\":[\"%s\",\"%s\",\"%s\",\"%s\",\"%s\"]}",
        submit_id++,
        pool->username,
        job_id,
        extranonce2,
        ntime,
        nonce);
    
    if (!send_json(pool->socket, json)) return 0;
    
    char buffer[4096];
    int len = pool_receive_message(pool, buffer, sizeof(buffer));
    if (len <= 0) return 0;
    
    int accepted = 0;
    int found_result = 0;
    
    // Traiter chaque ligne séparément (peut contenir réponse + nouveau job)
    char *line_start = buffer;
    char *line_end;
    char line[4096];
    
    while ((line_end = strchr(line_start, '\n')) != NULL) {
        int line_len = line_end - line_start;
        if (line_len >= sizeof(line)) line_len = sizeof(line) - 1;
        memcpy(line, line_start, line_len);
        line[line_len] = '\0';
        
        if (line_len > 2) {
            printf("<<< %s\n", line);
            
            // Vérifier si c'est la réponse au submit
            if (strstr(line, "\"result\"") && !found_result) {
                cJSON *root = cJSON_Parse(line);
                if (root) {
                    cJSON *result = cJSON_GetObjectItem(root, "result");
                    cJSON *error = cJSON_GetObjectItem(root, "error");
                    
                    if (cJSON_IsTrue(result)) {
                        printf("✓ Share accepté!\n");
                        accepted = 1;
                    } else if (error && !cJSON_IsNull(error)) {
                        cJSON *err_msg = cJSON_GetArrayItem(error, 1);
                        printf("✗ Share rejeté: %s\n", 
                               cJSON_IsString(err_msg) ? cJSON_GetStringValue(err_msg) : "unknown");
                    }
                    
                    cJSON_Delete(root);
                    found_result = 1;
                }
            }
            // Vérifier si c'est un nouveau job
            else if (strstr(line, "mining.notify")) {
                MiningJob job;
                memset(&job, 0, sizeof(job));
                pool_parse_notify(pool, line, &job);
                
                if (pool->job_callback) {
                    pool->job_callback(&job);
                }
            }
            // Vérifier si c'est un changement de difficulté
            else if (strstr(line, "mining.set_difficulty")) {
                parse_set_difficulty(pool, line);
                
                if (pool->diff_callback) {
                    pool->diff_callback(pool->difficulty);
                }
            }
            else if (strstr(line, "mining.set_target")) {
                printf("INFO: mining.set_target reçu (ignoré)\n");
            }
        }
        
        line_start = line_end + 1;
    }
    
    // Traiter la dernière ligne si pas de \n final
    if (*line_start != '\0' && !found_result) {
        int line_len = strlen(line_start);
        if (line_len < sizeof(line) && line_len > 2) {
            strcpy(line, line_start);
            printf("<<< %s\n", line);
            
            if (strstr(line, "\"result\"")) {
                cJSON *root = cJSON_Parse(line);
                if (root) {
                    cJSON *result = cJSON_GetObjectItem(root, "result");
                    cJSON *error = cJSON_GetObjectItem(root, "error");
                    
                    if (cJSON_IsTrue(result)) {
                        printf("✓ Share accepté!\n");
                        accepted = 1;
                    } else if (error && !cJSON_IsNull(error)) {
                        cJSON *err_msg = cJSON_GetArrayItem(error, 1);
                        printf("✗ Share rejeté: %s\n", 
                               cJSON_IsString(err_msg) ? cJSON_GetStringValue(err_msg) : "unknown");
                    }
                    
                    cJSON_Delete(root);
                }
            }
        }
    }
    
    return accepted;
}

// Fonction spécifique pour Ethash/Etchash (format 3 paramètres seulement)
int pool_submit_share_ethash(PoolConnection *pool, const char *job_id, const char *nonce) {
    static int submit_id = 100;
    char json[1024];
    
    // Format Ethash: seulement 3 paramètres [wallet, job_id, nonce]
    // Le nonce doit être préfixé avec 0x
    snprintf(json, sizeof(json),
        "{\"id\":%d,\"method\":\"mining.submit\",\"params\":[\"%s\",\"%s\",\"0x%s\"]}",
        submit_id++,
        pool->username,
        job_id,
        nonce);
    
    printf(">>> %s\n", json);
    
    if (!send_json(pool->socket, json)) {
        printf("Erreur envoi share!\n");
        return 0;
    }
    
    char buffer[4096];
    int len = pool_receive_message(pool, buffer, sizeof(buffer));
    if (len <= 0) {
        printf("Pas de réponse pool\n");
        return 0;
    }
    
    int accepted = 0;
    int found_result = 0;
    
    // Traiter chaque ligne séparément
    char *line_start = buffer;
    char *line_end;
    char line[4096];
    
    while ((line_end = strchr(line_start, '\n')) != NULL) {
        int line_len = line_end - line_start;
        if (line_len >= sizeof(line)) line_len = sizeof(line) - 1;
        memcpy(line, line_start, line_len);
        line[line_len] = '\0';
        
        if (line_len > 2) {
            printf("<<< %s\n", line);
            
            if (strstr(line, "\"result\"") && !found_result) {
                cJSON *root = cJSON_Parse(line);
                if (root) {
                    cJSON *result = cJSON_GetObjectItem(root, "result");
                    cJSON *error = cJSON_GetObjectItem(root, "error");
                    
                    if (cJSON_IsTrue(result)) {
                        accepted = 1;
                    } else if (error && !cJSON_IsNull(error)) {
                        cJSON *message = cJSON_GetObjectItem(error, "message");
                        if (message && cJSON_IsString(message)) {
                            printf("Erreur pool: %s\n", cJSON_GetStringValue(message));
                        }
                    }
                    
                    cJSON_Delete(root);
                    found_result = 1;
                }
            }
            else if (strstr(line, "mining.notify")) {
                MiningJob job;
                memset(&job, 0, sizeof(job));
                pool_parse_notify(pool, line, &job);
                
                if (pool->job_callback) {
                    pool->job_callback(&job);
                }
            }
        }
        
        line_start = line_end + 1;
    }
    
    // Traiter dernière ligne si pas de \n final
    if (*line_start != '\0' && !found_result) {
        int line_len = strlen(line_start);
        if (line_len < sizeof(line) && line_len > 2) {
            strcpy(line, line_start);
            printf("<<< %s\n", line);
            
            if (strstr(line, "\"result\"")) {
                cJSON *root = cJSON_Parse(line);
                if (root) {
                    cJSON *result = cJSON_GetObjectItem(root, "result");
                    if (cJSON_IsTrue(result)) {
                        accepted = 1;
                    }
                    cJSON_Delete(root);
                }
            }
        }
    }
    
    return accepted;
}
// Fonction KawPow (5 paramètres)
int pool_submit_share_kawpow(PoolConnection *pool, const char *job_id, const char *nonce,
                             const char *header_hash, const char *mix_hash) {
    static int submit_id = 100;
    char json[2048];
    
    // Format KawPow: 5 paramètres avec 0x prefix
    snprintf(json, sizeof(json),
        "{\"id\":%d,\"method\":\"mining.submit\",\"params\":[\"%s\",\"%s\",\"0x%s\",\"0x%s\",\"0x%s\"]}",
        submit_id++,
        pool->username,
        job_id,
        nonce,
        header_hash,
        mix_hash);
    
    printf(">>> %s\n", json);
    
    if (!send_json(pool->socket, json)) {
        printf("Erreur envoi share!\n");
        return 0;
    }
    
    char buffer[4096];
    int len = pool_receive_message(pool, buffer, sizeof(buffer));
    if (len <= 0) {
        printf("Pas de réponse pool\n");
        return 0;
    }
    
    int accepted = 0;
    int found_result = 0;
    char *line_start = buffer;
    char *line_end;
    char line[4096];
    
    while ((line_end = strchr(line_start, '\n')) != NULL) {
        int line_len = line_end - line_start;
        if (line_len >= sizeof(line)) line_len = sizeof(line) - 1;
        memcpy(line, line_start, line_len);
        line[line_len] = '\0';
        
        if (line_len > 2) {
            printf("<<< %s\n", line);
            
            if (strstr(line, "\"result\"") && !found_result) {
                cJSON *root = cJSON_Parse(line);
                if (root) {
                    cJSON *result = cJSON_GetObjectItem(root, "result");
                    cJSON *error = cJSON_GetObjectItem(root, "error");
                    
                    if (cJSON_IsTrue(result)) {
                        accepted = 1;
                    } else if (error && !cJSON_IsNull(error)) {
                        cJSON *message = cJSON_GetObjectItem(error, "message");
                        if (message && cJSON_IsString(message)) {
                            printf("Erreur pool: %s\n", cJSON_GetStringValue(message));
                        }
                    }
                    
                    cJSON_Delete(root);
                    found_result = 1;
                }
            }
        }
        
        line_start = line_end + 1;
    }
    
    return accepted;
}


void pool_parse_notify(PoolConnection *pool, const char *json, MiningJob *job) {
    cJSON *root = cJSON_Parse(json);
    if (!root) return;
    
    cJSON *params = cJSON_GetObjectItem(root, "params");
    if (!params || !cJSON_IsArray(params)) {
        cJSON_Delete(root);
        return;
    }
    
    cJSON *job_id = cJSON_GetArrayItem(params, 0);
    if (cJSON_IsString(job_id)) {
        const char *job_id_str = cJSON_GetStringValue(job_id);
        printf("DEBUG: Parsing job_id from params[0]: '%s'\n", job_id_str);
        strncpy(job->job_id, job_id_str, sizeof(job->job_id) - 1);
        job->job_id[sizeof(job->job_id) - 1] = '\0';
        printf("DEBUG: Stored in job->job_id: '%s'\n", job->job_id);
    }
    
    // Détecter format: Zcash vs Bitcoin vs KawPow
    int array_size = cJSON_GetArraySize(params);
    int is_zcash = (array_size >= 9);  // Zcash a 10 params
    int is_kawpow = (array_size == 7); // KawPow a 7 params
    
    if (is_kawpow) {
        // Format KawPow/Ravencoin
        // params: [job_id, headerHash, seedHash, target, clean, height, ntime]
        printf("DEBUG: Format KawPow détecté (%d params)\n", array_size);
        
        // headerHash (32 bytes hex) - params[1]
        cJSON *header_hash = cJSON_GetArrayItem(params, 1);
        if (cJSON_IsString(header_hash)) {
            const char *hh_str = cJSON_GetStringValue(header_hash);
            // Convertir hex string en bytes
            for (int i = 0; i < 32 && i*2 < (int)strlen(hh_str); i++) {
                unsigned int temp;
                sscanf(&hh_str[i*2], "%2x", &temp);
                job->header_hash[i] = (uint8_t)temp;
            }
            printf("DEBUG: headerHash: %s\n", hh_str);
        }
        
        // seedHash (32 bytes hex) - params[2]
        cJSON *seed_hash = cJSON_GetArrayItem(params, 2);
        if (cJSON_IsString(seed_hash)) {
            const char *sh_str = cJSON_GetStringValue(seed_hash);
            for (int i = 0; i < 32 && i*2 < (int)strlen(sh_str); i++) {
                unsigned int temp;
                sscanf(&sh_str[i*2], "%2x", &temp);
                job->seed_hash[i] = (uint8_t)temp;
            }
            printf("DEBUG: seedHash: %s\n", sh_str);
        }
        
        // target (32 bytes hex) - params[3]
        cJSON *target = cJSON_GetArrayItem(params, 3);
        if (cJSON_IsString(target)) {
            const char *t_str = cJSON_GetStringValue(target);
            for (int i = 0; i < 32 && i*2 < (int)strlen(t_str); i++) {
                unsigned int temp;
                sscanf(&t_str[i*2], "%2x", &temp);
                job->target[i] = (uint8_t)temp;
            }
            printf("DEBUG: target: %s\n", t_str);
        }
        
        // height - params[5]
        cJSON *height_json = cJSON_GetArrayItem(params, 5);
        if (cJSON_IsNumber(height_json)) {
            job->height = (uint32_t)cJSON_GetNumberValue(height_json);
            printf("DEBUG: height: %u\n", job->height);
        }

        
        // ntime - params[6]
        cJSON *ntime = cJSON_GetArrayItem(params, 6);
        if (cJSON_IsString(ntime)) {
            const char *ntime_str = cJSON_GetStringValue(ntime);
            strncpy(pool->ntime, ntime_str, sizeof(pool->ntime) - 1);
            pool->ntime[sizeof(pool->ntime) - 1] = '\0';
            strncpy(job->ntime, ntime_str, sizeof(job->ntime) - 1);
            job->ntime[sizeof(job->ntime) - 1] = '\0';
            printf("DEBUG: ntime: %s\n", ntime_str);
        }
        
    } else if (is_zcash) {
        // Format Zcash/Equihash
        // params: [job_id, version, prevhash, merkleroot, reserved, nbits, ntime, clean, algo, personal]
        printf("DEBUG: Format Zcash détecté (%d params)\n", array_size);
        
        cJSON *prevhash = cJSON_GetArrayItem(params, 2);
        if (cJSON_IsString(prevhash)) {
            strncpy(pool->prevhash, cJSON_GetStringValue(prevhash), sizeof(pool->prevhash) - 1);
        }
        
        cJSON *nbits = cJSON_GetArrayItem(params, 5);
        if (cJSON_IsString(nbits)) {
            strncpy(pool->nbits, cJSON_GetStringValue(nbits), sizeof(pool->nbits) - 1);
        }
        
        cJSON *ntime = cJSON_GetArrayItem(params, 6);
        if (cJSON_IsString(ntime)) {
            const char *ntime_str = cJSON_GetStringValue(ntime);
            strncpy(pool->ntime, ntime_str, sizeof(pool->ntime) - 1);
            pool->ntime[sizeof(pool->ntime) - 1] = '\0';
            printf("DEBUG: ntime from params[6]: '%s' (length: %d)\n", ntime_str, (int)strlen(ntime_str));
        }
    } else {
        // Format Bitcoin
        // params: [job_id, prevhash, coinb1, coinb2, merkle, version, nbits, ntime]
        printf("DEBUG: Format Bitcoin détecté (%d params)\n", array_size);
        
        cJSON *prevhash = cJSON_GetArrayItem(params, 1);
        if (cJSON_IsString(prevhash)) {
            strncpy(pool->prevhash, cJSON_GetStringValue(prevhash), sizeof(pool->prevhash) - 1);
        }
        
        cJSON *coinb1 = cJSON_GetArrayItem(params, 2);
        if (cJSON_IsString(coinb1)) {
            strncpy(pool->coinb1, cJSON_GetStringValue(coinb1), sizeof(pool->coinb1) - 1);
        }
        
        cJSON *coinb2 = cJSON_GetArrayItem(params, 3);
        if (cJSON_IsString(coinb2)) {
            strncpy(pool->coinb2, cJSON_GetStringValue(coinb2), sizeof(pool->coinb2) - 1);
        }
        
        cJSON *merkle_branches = cJSON_GetArrayItem(params, 4);
        if (cJSON_IsArray(merkle_branches)) {
            pool->merkle_count = cJSON_GetArraySize(merkle_branches);
            for (int i = 0; i < pool->merkle_count && i < 16; i++) {
                cJSON *branch = cJSON_GetArrayItem(merkle_branches, i);
                if (cJSON_IsString(branch)) {
                    strncpy(pool->merkle_branches[i], cJSON_GetStringValue(branch), 127);
                }
            }
        }
        
        cJSON *version = cJSON_GetArrayItem(params, 5);
        if (cJSON_IsString(version)) {
            strncpy(pool->version, cJSON_GetStringValue(version), sizeof(pool->version) - 1);
        }
        
        cJSON *nbits = cJSON_GetArrayItem(params, 6);
        if (cJSON_IsString(nbits)) {
            strncpy(pool->nbits, cJSON_GetStringValue(nbits), sizeof(pool->nbits) - 1);
        }
        
        cJSON *ntime = cJSON_GetArrayItem(params, 7);
        if (cJSON_IsString(ntime)) {
            const char *ntime_str = cJSON_GetStringValue(ntime);
            strncpy(pool->ntime, ntime_str, sizeof(pool->ntime) - 1);
            pool->ntime[sizeof(pool->ntime) - 1] = '\0';
            printf("DEBUG: ntime from params[7]: '%s' (length: %d)\n", ntime_str, (int)strlen(ntime_str));
        }
    }
    
    job->received_time = time(NULL);
    job->nonce_start = 0;
    job->nonce_end = 0xFFFFFFFF;
    
    // Copier le ntime depuis pool vers job
    strncpy(job->ntime, pool->ntime, sizeof(job->ntime) - 1);
    job->ntime[sizeof(job->ntime) - 1] = '\0';
    printf("DEBUG: Copié ntime '%s' dans job->ntime\n", job->ntime);
    
    printf("Nouveau job: %s\n", job->job_id);
    
    cJSON_Delete(root);
}

static void parse_set_difficulty(PoolConnection *pool, const char *json) {
    cJSON *root = cJSON_Parse(json);
    if (!root) return;
    
    cJSON *params = cJSON_GetObjectItem(root, "params");
    if (params && cJSON_IsArray(params)) {
        cJSON *diff = cJSON_GetArrayItem(params, 0);
        if (cJSON_IsNumber(diff)) {
            pool->difficulty = (uint32_t)cJSON_GetNumberValue(diff);
            printf("Nouvelle difficulté: %u\n", pool->difficulty);
        }
    }
    
    cJSON_Delete(root);
}

static unsigned __stdcall listen_thread_func(void *data) {
    printf("DEBUG: Thread listener DÉMARRÉ!\n");
    fflush(stdout);
    
    ListenThreadData *thread_data = (ListenThreadData*)data;
    PoolConnection *pool = thread_data->pool;
    
    printf("DEBUG: Pool status = %d\n", pool->status);
    printf("DEBUG: keep_listening = %d\n", keep_listening);
    fflush(stdout);
    
    char buffer[8192];
    char line[8192];
    
    printf("DEBUG: Entrée dans boucle d'écoute...\n");
    fflush(stdout);
    
    while (keep_listening && pool->status == POOL_AUTHORIZED) {
        int len = pool_receive_message(pool, buffer, sizeof(buffer));
        
        if (len < 0) {
            printf("Connexion perdue\n");
            pool->status = POOL_DISCONNECTED;
            break;
        }
        
        if (len == 0) continue;
        
        // Traiter chaque ligne (message JSON) séparément
        char *line_start = buffer;
        char *line_end;
        
        while ((line_end = strchr(line_start, '\n')) != NULL) {
            // Copier la ligne
            int line_len = line_end - line_start;
            if (line_len >= sizeof(line)) line_len = sizeof(line) - 1;
            memcpy(line, line_start, line_len);
            line[line_len] = '\0';
            
            // Traiter cette ligne
            if (line_len > 2) {  // Au moins "{}"
                printf("<<< %s\n", line);
                
                if (strstr(line, "mining.notify")) {
                    MiningJob job;
                    memset(&job, 0, sizeof(job));
                    pool_parse_notify(pool, line, &job);
                    
                    if (thread_data->job_callback) {
                        thread_data->job_callback(&job);
                    }
                }
                else if (strstr(line, "mining.set_difficulty")) {
                    parse_set_difficulty(pool, line);
                    
                    if (thread_data->diff_callback) {
                        thread_data->diff_callback(pool->difficulty);
                    }
                }
                else if (strstr(line, "mining.set_target")) {
                    // Ignorer pour l'instant - target sera calculé depuis difficulty
                    printf("INFO: mining.set_target reçu (ignoré)\n");
                }
                else if (strstr(line, "\"result\"")) {
                    // Réponse à un submit - déjà géré
                }
            }
            
            line_start = line_end + 1;
        }
        
        // Traiter la dernière ligne si pas de \n final
        if (*line_start != '\0') {
            int line_len = strlen(line_start);
            if (line_len < sizeof(line)) {
                strcpy(line, line_start);
                
                if (line_len > 2) {
                    printf("<<< %s\n", line);
                    
                    if (strstr(line, "mining.notify")) {
                        MiningJob job;
                        memset(&job, 0, sizeof(job));
                        pool_parse_notify(pool, line, &job);
                        
                        if (thread_data->job_callback) {
                            thread_data->job_callback(&job);
                        }
                    }
                    else if (strstr(line, "mining.set_difficulty")) {
                        parse_set_difficulty(pool, line);
                        
                        if (thread_data->diff_callback) {
                            thread_data->diff_callback(pool->difficulty);
                        }
                    }
                    else if (strstr(line, "mining.set_target")) {
                        printf("INFO: mining.set_target reçu (ignoré)\n");
                    }
                }
            }
        }
        
        Sleep(100);
    }
    
    return 0;
}

int pool_start_listener(PoolConnection *pool, 
                       void (*job_callback)(MiningJob*),
                       void (*diff_callback)(uint32_t)) {
    if (listen_thread) return 0;
    
    // Stocker les callbacks dans pool pour utilisation dans pool_submit_share
    pool->job_callback = job_callback;
    pool->diff_callback = diff_callback;
    
    ListenThreadData *data = (ListenThreadData*)malloc(sizeof(ListenThreadData));
    data->pool = pool;
    data->job_callback = job_callback;
    data->diff_callback = diff_callback;
    
    keep_listening = 1;
    printf("DEBUG: Création thread...\n");
    fflush(stdout);
    
    listen_thread = (HANDLE)_beginthreadex(NULL, 0, listen_thread_func, data, 0, NULL);
    
    printf("DEBUG: Thread handle = %p\n", listen_thread);
    fflush(stdout);
    
    if (!listen_thread) {
        printf("ERREUR: Échec création thread!\n");
        fflush(stdout);
        free(data);
        return 0;
    }
    
    printf("Thread d'écoute démarré\n");
    fflush(stdout);
    return 1;
}

void pool_stop_listener() {
    if (listen_thread) {
        keep_listening = 0;
        WaitForSingleObject(listen_thread, 5000);
        CloseHandle(listen_thread);
        listen_thread = NULL;
    }
}

void cleanup_winsock() {
    if (wsa_initialized) {
        WSACleanup();
        wsa_initialized = 0;
    }
}