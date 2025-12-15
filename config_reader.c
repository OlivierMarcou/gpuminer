/*
 * config_reader.c - Lecture fichier pool_config.ini
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE 512
#define MAX_SECTION 64
#define MAX_KEY 64
#define MAX_VALUE 256

typedef struct {
    char pool_url[256];
    int pool_port;
    char wallet[256];
    char worker[64];
    char username[256];
    char password[128];
    int auth_mode;  // 1 = wallet+worker, 2 = username
} PoolConfig;

// Trim whitespace
static void trim(char *str) {
    char *start = str;
    char *end;
    
    // Trim leading
    while (*start == ' ' || *start == '\t') start++;
    
    // Trim trailing
    end = start + strlen(start) - 1;
    while (end > start && (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r')) {
        *end = '\0';
        end--;
    }
    
    // Shift
    if (start != str) {
        memmove(str, start, strlen(start) + 1);
    }
}

// Parse une ligne key=value
static int parse_key_value(const char *line, char *key, char *value) {
    const char *eq = strchr(line, '=');
    if (!eq) return 0;
    
    // Copier key
    size_t key_len = eq - line;
    if (key_len >= MAX_KEY) key_len = MAX_KEY - 1;
    strncpy(key, line, key_len);
    key[key_len] = '\0';
    trim(key);
    
    // Copier value
    strncpy(value, eq + 1, MAX_VALUE - 1);
    value[MAX_VALUE - 1] = '\0';
    trim(value);
    
    return 1;
}

// Lire une section du fichier ini
int read_pool_config(const char *filename, const char *section, PoolConfig *config) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        printf("ATTENTION: Fichier %s introuvable!\n", filename);
        printf("Créez le fichier ou utilisez configuration manuelle.\n");
        return 0;
    }
    
    char line[MAX_LINE];
    char current_section[MAX_SECTION] = "";
    int in_section = 0;
    
    // Init config
    memset(config, 0, sizeof(PoolConfig));
    config->auth_mode = 1;  // Par défaut: wallet+worker
    strcpy(config->password, "x");
    
    while (fgets(line, sizeof(line), f)) {
        trim(line);
        
        // Skip empty lines and comments
        if (line[0] == '\0' || line[0] == '#' || line[0] == ';') {
            continue;
        }
        
        // Section header [SECTION]
        if (line[0] == '[') {
            char *end = strchr(line, ']');
            if (end) {
                size_t len = end - line - 1;
                if (len < MAX_SECTION) {
                    strncpy(current_section, line + 1, len);
                    current_section[len] = '\0';
                    trim(current_section);
                    
                    in_section = (strcmp(current_section, section) == 0);
                }
            }
            continue;
        }
        
        // Si on est dans la bonne section, parser les key=value
        if (in_section) {
            char key[MAX_KEY];
            char value[MAX_VALUE];
            
            if (parse_key_value(line, key, value)) {
                if (strcmp(key, "pool_url") == 0) {
                    strncpy(config->pool_url, value, sizeof(config->pool_url) - 1);
                }
                else if (strcmp(key, "pool_port") == 0) {
                    config->pool_port = atoi(value);
                }
                else if (strcmp(key, "wallet") == 0) {
                    strncpy(config->wallet, value, sizeof(config->wallet) - 1);
                }
                else if (strcmp(key, "worker") == 0) {
                    strncpy(config->worker, value, sizeof(config->worker) - 1);
                }
                else if (strcmp(key, "username") == 0) {
                    strncpy(config->username, value, sizeof(config->username) - 1);
                }
                else if (strcmp(key, "password") == 0) {
                    strncpy(config->password, value, sizeof(config->password) - 1);
                }
                else if (strcmp(key, "auth_mode") == 0) {
                    config->auth_mode = atoi(value);
                }
            }
        }
    }
    
    fclose(f);
    
    // Vérifier si on a trouvé des données
    if (config->pool_url[0] == '\0') {
        printf("Section [%s] introuvable dans %s\n", section, filename);
        return 0;
    }
    
    return 1;
}

// Lister les sections disponibles
void list_pool_configs(const char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        printf("Fichier %s introuvable\n", filename);
        return;
    }
    
    char line[MAX_LINE];
    int count = 1;
    
    printf("\nConfigurations disponibles:\n");
    
    while (fgets(line, sizeof(line), f)) {
        trim(line);
        
        if (line[0] == '[') {
            char *end = strchr(line, ']');
            if (end) {
                *end = '\0';
                printf("%d. %s\n", count++, line + 1);
            }
        }
    }
    
    fclose(f);
}

// Fonction pour obtenir le nom de la section selon index
int get_section_name(const char *filename, int index, char *section_out) {
    FILE *f = fopen(filename, "r");
    if (!f) return 0;
    
    char line[MAX_LINE];
    int count = 0;
    
    while (fgets(line, sizeof(line), f)) {
        trim(line);
        
        if (line[0] == '[') {
            char *end = strchr(line, ']');
            if (end) {
                *end = '\0';
                count++;
                
                if (count == index) {
                    strcpy(section_out, line + 1);
                    fclose(f);
                    return 1;
                }
            }
        }
    }
    
    fclose(f);
    return 0;
}
