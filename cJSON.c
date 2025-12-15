/*
 * cJSON.c - Implémentation simplifiée pour Stratum
 */

#include "cJSON.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>

static const char *skip(const char *in) {
    while (in && *in && (unsigned char)*in <= 32) in++;
    return in;
}

static cJSON *cJSON_New_Item(void) {
    cJSON *node = (cJSON*)malloc(sizeof(cJSON));
    if (node) memset(node, 0, sizeof(cJSON));
    return node;
}

void cJSON_Delete(cJSON *c) {
    cJSON *next;
    while (c) {
        next = c->next;
        if (c->child) cJSON_Delete(c->child);
        if (c->valuestring) free(c->valuestring);
        if (c->string) free(c->string);
        free(c);
        c = next;
    }
}

static const char *parse_string(cJSON *item, const char *str) {
    const char *ptr = str + 1;
    char *ptr2;
    char *out;
    int len = 0;
    
    if (*str != '\"') return 0;
    
    while (*ptr != '\"' && *ptr && ++len) {
        if (*ptr++ == '\\') ptr++;
    }
    
    out = (char*)malloc(len + 1);
    if (!out) return 0;
    
    ptr = str + 1;
    ptr2 = out;
    
    while (*ptr != '\"' && *ptr) {
        if (*ptr != '\\') *ptr2++ = *ptr++;
        else {
            ptr++;
            switch (*ptr) {
                case 'b': *ptr2++ = '\b'; break;
                case 'f': *ptr2++ = '\f'; break;
                case 'n': *ptr2++ = '\n'; break;
                case 'r': *ptr2++ = '\r'; break;
                case 't': *ptr2++ = '\t'; break;
                default: *ptr2++ = *ptr; break;
            }
            ptr++;
        }
    }
    *ptr2 = 0;
    
    item->type = cJSON_String;
    item->valuestring = out;
    
    if (*ptr == '\"') ptr++;
    return ptr;
}

static const char *parse_number(cJSON *item, const char *num) {
    double n = 0, sign = 1, scale = 0;
    int subscale = 0, signsubscale = 1;
    
    if (*num == '-') sign = -1, num++;
    if (*num == '0') num++;
    if (*num >= '1' && *num <= '9') {
        do n = (n * 10.0) + (*num++ - '0');
        while (*num >= '0' && *num <= '9');
    }
    if (*num == '.' && num[1] >= '0' && num[1] <= '9') {
        num++;
        do n = (n * 10.0) + (*num++ - '0'), scale--;
        while (*num >= '0' && *num <= '9');
    }
    if (*num == 'e' || *num == 'E') {
        num++;
        if (*num == '+') num++;
        else if (*num == '-') signsubscale = -1, num++;
        while (*num >= '0' && *num <= '9')
            subscale = (subscale * 10) + (*num++ - '0');
    }
    
    // pow(10.0, exp) manually
    double multiplier = 1.0;
    int exponent = (int)(scale + subscale * signsubscale);
    if (exponent > 0) {
        for (int i = 0; i < exponent; i++) multiplier *= 10.0;
    } else if (exponent < 0) {
        for (int i = 0; i > exponent; i--) multiplier /= 10.0;
    }
    
    n = sign * n * multiplier;
    
    item->valuedouble = n;
    item->valueint = (int)n;
    item->type = cJSON_Number;
    
    return num;
}

static const char *parse_array(cJSON *item, const char *value);
static const char *parse_object(cJSON *item, const char *value);

static const char *parse_value(cJSON *item, const char *value) {
    if (!value) return 0;
    if (!strncmp(value, "null", 4)) { item->type = cJSON_NULL; return value + 4; }
    if (!strncmp(value, "false", 5)) { item->type = cJSON_False; return value + 5; }
    if (!strncmp(value, "true", 4)) { item->type = cJSON_True; return value + 4; }
    if (*value == '\"') return parse_string(item, value);
    if (*value == '-' || (*value >= '0' && *value <= '9')) return parse_number(item, value);
    if (*value == '[') return parse_array(item, value);
    if (*value == '{') return parse_object(item, value);
    return 0;
}

static const char *parse_array(cJSON *item, const char *value) {
    cJSON *child;
    if (*value != '[') return 0;
    
    item->type = cJSON_Array;
    value = skip(value + 1);
    if (*value == ']') return value + 1;
    
    item->child = child = cJSON_New_Item();
    if (!item->child) return 0;
    value = skip(parse_value(child, skip(value)));
    if (!value) return 0;
    
    while (*value == ',') {
        cJSON *new_item = cJSON_New_Item();
        if (!new_item) return 0;
        child->next = new_item;
        new_item->prev = child;
        child = new_item;
        value = skip(parse_value(child, skip(value + 1)));
        if (!value) return 0;
    }
    
    if (*value == ']') return value + 1;
    return 0;
}

static const char *parse_object(cJSON *item, const char *value) {
    cJSON *child;
    if (*value != '{') return 0;
    
    item->type = cJSON_Object;
    value = skip(value + 1);
    if (*value == '}') return value + 1;
    
    item->child = child = cJSON_New_Item();
    if (!item->child) return 0;
    value = skip(parse_string(child, skip(value)));
    if (!value) return 0;
    child->string = child->valuestring;
    child->valuestring = 0;
    if (*value != ':') return 0;
    value = skip(parse_value(child, skip(value + 1)));
    if (!value) return 0;
    
    while (*value == ',') {
        cJSON *new_item = cJSON_New_Item();
        if (!new_item) return 0;
        child->next = new_item;
        new_item->prev = child;
        child = new_item;
        value = skip(parse_string(child, skip(value + 1)));
        if (!value) return 0;
        child->string = child->valuestring;
        child->valuestring = 0;
        if (*value != ':') return 0;
        value = skip(parse_value(child, skip(value + 1)));
        if (!value) return 0;
    }
    
    if (*value == '}') return value + 1;
    return 0;
}

cJSON *cJSON_Parse(const char *value) {
    cJSON *c = cJSON_New_Item();
    if (!c) return 0;
    if (!parse_value(c, skip(value))) {
        cJSON_Delete(c);
        return 0;
    }
    return c;
}

cJSON *cJSON_GetObjectItem(const cJSON *object, const char *string) {
    cJSON *c = object->child;
    while (c && strcmp(c->string, string)) c = c->next;
    return c;
}

int cJSON_GetArraySize(const cJSON *array) {
    cJSON *c = array->child;
    int i = 0;
    while (c) i++, c = c->next;
    return i;
}

cJSON *cJSON_GetArrayItem(const cJSON *array, int item) {
    cJSON *c = array->child;
    while (c && item > 0) item--, c = c->next;
    return c;
}

char *cJSON_GetStringValue(cJSON *item) {
    return item ? item->valuestring : NULL;
}

double cJSON_GetNumberValue(cJSON *item) {
    return item ? item->valuedouble : 0.0;
}

int cJSON_IsString(const cJSON *item) {
    return item && item->type == cJSON_String;
}

int cJSON_IsNumber(const cJSON *item) {
    return item && item->type == cJSON_Number;
}

int cJSON_IsArray(const cJSON *item) {
    return item && item->type == cJSON_Array;
}

int cJSON_IsBool(const cJSON *item) {
    return item && (item->type == cJSON_True || item->type == cJSON_False);
}

int cJSON_IsTrue(const cJSON *item) {
    return item && item->type == cJSON_True;
}

int cJSON_IsNull(const cJSON *item) {
    return item && item->type == cJSON_NULL;
}
