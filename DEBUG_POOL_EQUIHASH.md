# Debugging Pool Equihash 192,7

## Problèmes Identifiés

### 1. Job_ID Incorrect (PRIORITAIRE)

**Symptôme:**
```
>>> {"id":100,"method":"mining.submit","params":["europe.mining-dutch.nl","00b1c71c71c71c70...","00000000","","00000000"]}
<<< {"id":100,"result":null,"error":[20,"Invalid job",null]}
```

**Cause:**
Le job_id soumis est "00b1c71c71c71c70..." (le target) au lieu de "7965632d64636635-370" (le vrai job_id).

**Corrections Appliquées:**

1. **cuda_miner.cu:** Utiliser `g_current_job.job_id` correctement
2. **stratum.c:** Ajout de debug pour voir ce qui est parsé:
```c
printf("DEBUG: Parsing job_id from params[0]: '%s'\n", job_id_str);
printf("DEBUG: Stored in job->job_id: '%s'\n", job->job_id);
```

**Test après compilation:**
Chercher dans la sortie:
```
DEBUG: Parsing job_id from params[0]: '7965632d64636635-370'
DEBUG: Stored in job->job_id: '7965632d64636635-370'
```

Si ça affiche le bon job_id, le problème est ailleurs.

### 2. Trop de Fausses Solutions (132 solutions!)

**Symptôme:**
```
>>> 132 SOLUTION(S) Equihash 192,7!
```

**Cause:**
Le kernel `equihash_192_7_kernel()` est **trop simplifié** et trouve beaucoup de faux positifs. Il vérifie seulement:
```c
if ((bucket_id & 0xFFFF) < 32 && hash[3] < 32) {
    // "Solution" trouvée
}
```

Ce n'est PAS une vraie vérification Equihash!

**Correction Temporaire:**
Limiter à 1 solution soumise par itération:
```c
uint32_t max_submit = (solution_count > 1) ? 1 : solution_count;
```

**Correction Permanente Nécessaire:**
Implémenter un vrai algorithme Wagner avec vérification complète des collisions.

### 3. Nonce et Paramètres de Soumission

**Problème:**
Le nonce soumis était `solutions[i * 128]` qui est juste l'index, pas un vrai nonce.

**Correction:**
```c
uint32_t nonce_value = start_nonce + solutions[i * 128];
char nonce_hex[16];
sprintf(nonce_hex, "%08x", nonce_value);
```

## Workflow de Debugging

### Étape 1: Compiler et Tester

```cmd
del *.obj *.exe
build_cuda.bat
cuda_miner.exe
```

### Étape 2: Vérifier Messages Debug

Chercher dans la sortie:
```
DEBUG: Parsing job_id from params[0]: '...'
DEBUG: Stored in job->job_id: '...'
Nouveau job: ...
>>> Nouveau job reçu: ...
```

**Si les 4 lignes montrent le même job_id:** ✅ Parsing correct

**Si elles montrent des valeurs différentes:** ❌ Bug à identifier

### Étape 3: Vérifier Soumission

```
Soumission solution 1 à la pool...
  Job ID: 7965632d64636635-370  ← Doit être le BON job_id
  Nonce: 1a2b3c4d                ← Doit être un nonce valide
>>> {"id":100,"method":"mining.submit","params":[...]}
<<< {"id":100,"result":...}
```

**Si result: true:** ✅ Share accepté
**Si error: [20,"Invalid job",null]:** ❌ Mauvais job_id
**Si error: [21,"Invalid nonce",null]:** ❌ Mauvais nonce
**Si error: [22,"Duplicate share",null]:** ⚠️ Déjà soumis

## Format Stratum Equihash vs Bitcoin

### Bitcoin (mining.notify):
```json
{
  "params": [
    "job_id",      // 0
    "prevhash",    // 1
    "coinb1",      // 2
    "coinb2",      // 3
    ["merkle"],    // 4
    "version",     // 5
    "nbits",       // 6
    "ntime"        // 7
  ]
}
```

### Zcash/Equihash (mining.notify):
```json
{
  "params": [
    "job_id",      // 0 ← Même position!
    "version",     // 1
    "prevhash",    // 2
    "merkleroot",  // 3
    "reserved",    // 4
    "nbits",       // 5
    "ntime",       // 6
    "clean_jobs",  // 7
    "algo",        // 8 (ex: "192_7")
    "personal"     // 9 (ex: "ZcashPoW")
  ]
}
```

**Important:** Le job_id est à la **même position** (params[0]) dans les deux formats!

## Message mining.set_target

Reçu avant mining.notify:
```json
{"id":null,"method":"mining.set_target","params":["00b1c71c71c71c70..."]}
```

**Ce n'est PAS un job!** C'est juste le target/difficulté.

Le code actuel **ignore** ce message (pas de handler).

## Solutions Valides Equihash 192,7

Une **vraie** solution Equihash 192,7 doit:

1. Contenir **128 indices** (2^7)
2. Chaque indice: 0 à 2^24 (16,777,216)
3. Les indices doivent former un arbre de collisions valide
4. XOR de tous les indices = 0
5. Chaque collision respecte le bit-length (24 bits / 8 = 3 bits par round)

**Le kernel actuel ne vérifie RIEN de cela!**

## Corrections Nécessaires

### Court Terme (Pour tester pool)
- ✅ Corriger soumission job_id
- ✅ Limiter à 1 solution/itération
- ✅ Ajouter debug parsing

### Moyen Terme (Pour minage réel)
- ⚠️ Implémenter vérification basique solutions
- ⚠️ Utiliser vraies données job (prevhash, ntime, etc)
- ⚠️ Construire header Zcash correct

### Long Terme (Pour production)
- ❌ Implémenter Wagner algorithm complet
- ❌ Optimiser kernels GPU
- ❌ Support des différentes variantes Equihash

## Test avec Pool

### Configuration Recommandée

**Pour tester le parsing/connexion:**
```
Pool: europe.mining-dutch.nl:6660
Username: votre_username.worker
Password: d=0.045
```

**Pool test alternative:**
```
Pool: eu1-zcash.flypool.org:3333
Wallet: t1... (adresse Zcash)
Worker: test
```

### Résultats Attendus

**Si parsing job_id est corrigé:**
```
>>> {"id":100,"method":"mining.submit","params":["username.worker","7965632d64636635-370",...]}
<<< {"id":100,"result":true,"error":null}
✓ Share ACCEPTÉ!
```

**Ou:**
```
<<< {"id":100,"result":false,"error":[23,"Low difficulty share",null]}
✗ Share REJETÉ (difficulté insuffisante)
```

Même rejeté pour "low difficulty", c'est mieux que "invalid job"!

## Compilation

```cmd
del *.obj *.exe 2>nul
build_cuda.bat
```

Vérifier que:
- stratum.obj compile (avec nouveau debug)
- cuda_miner.obj compile (avec corrections)
- Linkage réussit

## Prochaines Étapes

1. **Tester avec debug activé** → Voir job_id réel
2. **Si job_id correct mais toujours rejeté** → Vérifier format submit
3. **Si shares acceptés** → Implémenter vraie vérification solutions
4. **Implémenter Wagner algorithm** → Vraies solutions Equihash

## Notes Importantes

- Les 132 "solutions" sont des **faux positifs**
- Le kernel actuel est un **placeholder** pour tester la pool
- **Ne pas miner en production** avec ce kernel
- Une vraie implémentation Equihash 192,7 nécessite plusieurs jours de développement

## Référence Format Soumission

**Correct:**
```json
{
  "id": 100,
  "method": "mining.submit",
  "params": [
    "username.worker",           // Worker name
    "7965632d64636635-370",      // Job ID (du mining.notify)
    "00000000",                   // Extranonce2 (ou solution partielle)
    "dae3021e",                   // Ntime (du job)
    "1a2b3c4d"                    // Nonce trouvé
  ]
}
```

**Notes:**
- username.worker = ce qu'on a autorisé
- job_id = params[0] du dernier mining.notify
- ntime = params[6] du mining.notify
- nonce = valeur trouvée par le minage

## Contact Pool

Si les shares sont toujours rejetés après corrections:
1. Vérifier logs pool (si accès dashboard)
2. Tester avec pool alternative
3. Comparer avec logs d'un vrai mineur (ex: lolMiner)
4. Demander support pool (forum/Discord)

## Conclusion

**Statut actuel:**
- ✅ Connexion pool fonctionne
- ✅ Parsing messages fonctionne (probablement)
- ⚠️ Soumission shares en cours de debug
- ❌ Solutions trouvées sont fausses

**Objectif immédiat:**
Corriger le job_id pour que shares soient acceptés, même si solutions sont fausses.

**Objectif final:**
Implémenter vrai Equihash 192,7 avec Wagner algorithm.
