# FIX: Ntime Vide dans Soumission

## 🐛 Problème

Le ntime était **VIDE** dans la soumission:

```json
>>> {"id":100,"method":"mining.submit","params":["omarcou.worker4","7965632d65366539-806","00000000","","00000000"]}
                                                                                                      ^^
```

Mais le parsing fonctionnait: `DEBUG: ntime from params[6]: '6475021e'`

## 🔍 Cause

**Structure MiningJob n'avait PAS de champ ntime:**
```c
typedef struct {
    char job_id[128];
    // ❌ PAS de ntime !
} MiningJob;
```

Le ntime était dans `pool->ntime` mais pas copié dans `g_current_job`.

## ✅ Solution

1. **Ajouté `char ntime[16]` à `MiningJob`**
2. **Copié `pool->ntime` → `job->ntime` dans `pool_parse_notify()`**
3. **Utilisé `g_current_job.ntime` au lieu de `pool->ntime`**

## 🎯 Résultat Attendu

```
DEBUG: Copié ntime '6475021e' dans job->ntime
  Ntime: 6475021e
>>> {"params":["omarcou.worker4","7965632d65366539-806","00000000","6475021e","00000000"]}
                                                                     ^^^^^^^^ NON VIDE!
```

## 🧪 Test

```cmd
del *.obj *.exe
build_cuda.bat
cuda_miner.exe
```

Chercher: `Ntime: 6475021e` dans les logs
