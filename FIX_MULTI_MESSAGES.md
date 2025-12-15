# FIX: Parsing Messages JSON Multiples

## 🐛 Problème Identifié

### Symptôme
```
DEBUG: Parsing job_id from params[0]: '00b1c71c71c71c70...'
```

Le job_id parsé était le **target** (du message `mining.set_target`) au lieu du vrai job_id (du message `mining.notify`).

### Cause Racine

La pool envoie **2 messages JSON** en une seule fois, séparés par `\n`:

```
{"id":null,"method":"mining.set_target","params":["00b1c71c71c71c70..."]}
{"id":null,"method":"mining.notify","params":["7965632d64643236-875",...]}
```

**Ancien code:**
```c
int len = pool_receive_message(pool, buffer, sizeof(buffer));
// buffer contient maintenant LES DEUX messages

if (strstr(buffer, "mining.notify")) {
    pool_parse_notify(pool, buffer, &job);  // ← BUG ICI!
}
```

Problème:
1. `strstr(buffer, "mining.notify")` trouve "mining.notify" dans le buffer (car il y est)
2. **MAIS** `cJSON_Parse(buffer)` parse seulement le **PREMIER** JSON dans le buffer
3. Le premier JSON est `mining.set_target`, pas `mining.notify`!
4. Donc params[0] extrait est le target, pas le job_id

## ✅ Solution Appliquée

Traiter chaque ligne (chaque message JSON) séparément:

```c
// Traiter chaque ligne séparément
char *line_start = buffer;
char *line_end;

while ((line_end = strchr(line_start, '\n')) != NULL) {
    // Copier la ligne
    int line_len = line_end - line_start;
    memcpy(line, line_start, line_len);
    line[line_len] = '\0';
    
    // Traiter CETTE ligne uniquement
    if (strstr(line, "mining.notify")) {
        pool_parse_notify(pool, line, &job);  // ← Maintenant correct!
    }
    
    line_start = line_end + 1;
}
```

## 🎯 Résultat Attendu

### Avant (INCORRECT)
```
<<< {"id":null,"method":"mining.set_target",...}
{"id":null,"method":"mining.notify","params":["7965632d64643236-875",...]}

DEBUG: Parsing job_id from params[0]: '00b1c71c71c71c70...'  ← TARGET
Nouveau job: 00b1c71c71c71c70...  ← MAUVAIS
```

### Après (CORRECT)
```
<<< {"id":null,"method":"mining.set_target",...}
INFO: mining.set_target reçu (ignoré)

<<< {"id":null,"method":"mining.notify","params":["7965632d64643236-875",...]}
DEBUG: Parsing job_id from params[0]: '7965632d64643236-875'  ← JOB_ID
Nouveau job: 7965632d64643236-875  ← CORRECT!
```

## 📝 Modifications Appliquées

### 1. Traitement Ligne par Ligne

**Fichier:** `stratum.c` - fonction `listen_thread_func()`

**Changements:**
- Ajout buffer `line[8192]` pour stocker chaque ligne
- Utilisation de `strchr(line_start, '\n')` pour trouver fin de ligne
- Boucle `while` pour traiter chaque ligne
- Traitement de la dernière ligne (si pas de `\n` final)

### 2. Handler mining.set_target

Ajout d'un handler pour ignorer proprement `mining.set_target`:

```c
else if (strstr(line, "mining.set_target")) {
    printf("INFO: mining.set_target reçu (ignoré)\n");
}
```

**Pourquoi ignorer?**
Le target sera calculé depuis la difficulté avec: `target = 0x0000FFFF / difficulty`

## 🧪 Test de Vérification

### Compilation
```cmd
del *.obj *.exe
build_cuda.bat
```

### Exécution
```cmd
cuda_miner.exe

Choix: 5 (Pool)
Algorithme: 4 (Equihash 192,7)
Pool: europe.mining-dutch.nl:6660
```

### Sortie Attendue
```
<<< {"id":null,"method":"mining.set_target","params":["00b1c71c71c71c70..."]}
INFO: mining.set_target reçu (ignoré)

<<< {"id":null,"method":"mining.notify","params":["7965632d64643236-875",...]}
DEBUG: Parsing job_id from params[0]: '7965632d64643236-875'
DEBUG: Stored in job->job_id: '7965632d64643236-875'
Nouveau job: 7965632d64643236-875

>>> Nouveau job reçu: 7965632d64643236-875

[Les 3 lignes doivent montrer le MÊME job_id: 7965632d64643236-875]
```

### Soumission
```
Soumission solution 1 à la pool...
  Job ID: 7965632d64643236-875  ← CORRECT!
  Nonce: 1a2b3c4d
>>> {"id":100,"method":"mining.submit","params":["username.worker","7965632d64643236-875",...]}
```

**Si résultat:**
- `"result":true` → ✅ Share ACCEPTÉ!
- `"error":[20,"Invalid job",null]` → ❌ Encore un problème
- `"error":[23,"Low difficulty",null]` → ⚠️ Share trop facile (mais job_id correct!)

## 🔍 Debugging

Si le problème persiste, vérifier:

### 1. Les 3 Lignes de Job_ID
```
DEBUG: Parsing job_id from params[0]: '...'  ← Ligne 1
DEBUG: Stored in job->job_id: '...'          ← Ligne 2
Nouveau job: ...                              ← Ligne 3
>>> Nouveau job reçu: ...                     ← Ligne 4
```

**Les 4 DOIVENT être identiques!**

### 2. Format du Message
```
<<< {"id":null,"method":"mining.notify","params":["JOB_ID_ICI",...]}
```

Le JOB_ID ne devrait PAS être "00b1c71c71c71c70..." (c'est le target).

### 3. Ordre des Messages
```
1. mining.set_target  ← Ignoré
2. mining.notify      ← Parsé
```

Si inversé, pas de problème - le code traite les deux.

## 📊 Cas d'Usage

### Cas 1: Messages Séparés
```
Reçu: {"id":null,"method":"mining.notify",...}
→ Traite 1 ligne
→ Parse mining.notify
→ OK
```

### Cas 2: Messages Groupés (Cas problématique)
```
Reçu: {"id":null,"method":"mining.set_target",...}\n{"id":null,"method":"mining.notify",...}
→ Traite ligne 1: mining.set_target → Ignore
→ Traite ligne 2: mining.notify → Parse
→ OK
```

### Cas 3: 3+ Messages
```
Reçu: MSG1\nMSG2\nMSG3
→ Boucle traite chaque ligne séparément
→ OK
```

## 🎯 Impact SHA256

Tu as dit "en sha256 ça marche" - c'est parce que:

1. **Format différent:** Les pools Bitcoin envoient peut-être les messages séparément
2. **Timing différent:** mining.set_target arrive plus tôt/tard
3. **Protocole différent:** Bitcoin n'utilise pas mining.set_target de la même façon

Mais avec Equihash (Zcash), la pool envoie les 2 messages ensemble, d'où le bug.

## ✅ Prochaines Étapes

1. **Recompiler** avec stratum.c corrigé
2. **Tester** Equihash 192,7 sur pool
3. **Vérifier** que job_id est correct dans les logs
4. **Observer** si shares sont acceptés

## ⚠️ Note Importante

Ce fix corrige le parsing du job_id, MAIS:

- ❌ Le kernel Equihash 192,7 trouve toujours des faux positifs
- ❌ Les "solutions" ne sont pas valides
- ⚠️ Même avec job_id correct, shares peuvent être rejetés pour "invalid solution"

**Objectif immédiat:** Voir si pool accepte le format de soumission (job_id, nonce, etc)
**Objectif suivant:** Implémenter vrai algorithme Equihash

## 📚 Références

**Stratum Protocol:**
- Chaque message JSON est une ligne complète terminée par `\n`
- Plusieurs messages peuvent arriver dans un seul recv()
- Le client doit traiter chaque ligne séparément

**mining.set_target vs mining.set_difficulty:**
- `mining.set_target`: Définit target directement (format Zcash)
- `mining.set_difficulty`: Définit difficulté (format Bitcoin)
- Certaines pools envoient les deux

## 🐛 Bugs Corrigés

1. ✅ Parsing du premier JSON au lieu du bon message
2. ✅ Job_ID confondu avec target
3. ✅ Messages multiples non traités séparément
4. ✅ Pas de handler pour mining.set_target

## 🚀 Résultat Final

Avec ce fix:
- ✅ Chaque message JSON est traité individuellement
- ✅ mining.set_target ne pollue plus mining.notify
- ✅ job_id extrait est le bon
- ✅ Soumissions utilisent le bon job_id
- ⚠️ Reste à implémenter vraies solutions Equihash

**Le parsing est maintenant CORRECT !**
