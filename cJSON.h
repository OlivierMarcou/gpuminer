/*
 * cJSON.h - Bibliothèque JSON légère
 */

#ifndef CJSON_H
#define CJSON_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cJSON {
    struct cJSON *next;
    struct cJSON *prev;
    struct cJSON *child;
    int type;
    char *valuestring;
    int valueint;
    double valuedouble;
    char *string;
} cJSON;

#define cJSON_False 0
#define cJSON_True 1
#define cJSON_NULL 2
#define cJSON_Number 3
#define cJSON_String 4
#define cJSON_Array 5
#define cJSON_Object 6

cJSON *cJSON_Parse(const char *value);
void cJSON_Delete(cJSON *c);
cJSON *cJSON_GetObjectItem(const cJSON *object, const char *string);
int cJSON_GetArraySize(const cJSON *array);
cJSON *cJSON_GetArrayItem(const cJSON *array, int item);
char *cJSON_GetStringValue(cJSON *item);
double cJSON_GetNumberValue(cJSON *item);
int cJSON_IsString(const cJSON *item);
int cJSON_IsNumber(const cJSON *item);
int cJSON_IsArray(const cJSON *item);
int cJSON_IsBool(const cJSON *item);
int cJSON_IsTrue(const cJSON *item);
int cJSON_IsNull(const cJSON *item);

#ifdef __cplusplus
}
#endif

#endif
