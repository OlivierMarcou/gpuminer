# 🚀 FEUILLE DE ROUTE KAWPOW - 10 JOURS

## 🎯 OBJECTIF FINAL:
**Mineur KawPow 100% fonctionnel avec shares acceptés par la pool**

---

## 📅 PLAN JOUR PAR JOUR:

### **JOUR 3 (Aujourd'hui) - Diagnostic & Analyse** 🔍

**Objectifs:**
1. ✅ Comprendre pourquoi 0 shares (11.98 MH/s mais 0 résultats)
2. ✅ Ajouter debug détaillé au kernel
3. ✅ Lire spec ProgPoW officielle (EIP-1057)
4. ✅ Identifier différences avec mon implémentation

**Actions:**
- [x] Version debug avec outputs intermédiaires
- [ ] Compiler et lancer version debug
- [ ] Analyser outputs: mix_hash, result hash, comparaison
- [ ] Lire spec ProgPoW complète
- [ ] Noter toutes les différences potentielles

**Livrables:**
- kawpow_debug.cu avec outputs détaillés
- Liste des différences spec vs code
- Hypothèses sur bugs

---

### **JOUR 4 - KISS99 RNG Correct** 🎲

**Problème identifié:**
KISS99 doit générer EXACTEMENT la même séquence que la spec pour que les operations soient dans le bon ordre.

**Objectifs:**
1. Implémenter KISS99 selon spec EXACTE
2. Créer test vectors pour KISS99
3. Vérifier que la séquence est correcte

**Actions:**
- [ ] Étudier implémentation KISS99 de référence (kawpowminer)
- [ ] Implémenter avec valeurs initiales correctes
- [ ] Tester avec seed connus → vérifier séquence
- [ ] Comparer avec valeurs attendues

**Test:**
```c
KISS99 avec seed=0x12345678
→ Doit générer: val[0]=0xABCD1234, val[1]=0xEF567890, ...
→ Comparer avec ma version
```

**Critère succès:**
Séquence KISS99 identique à 100% avec référence

---

### **JOUR 5 - Math Operations & Register Selection** 🧮

**Problème identifié:**
L'ordre des operations et le choix des registres doivent être EXACTEMENT selon KISS99.

**Objectifs:**
1. Vérifier que les 11 math operations sont correctes
2. S'assurer que src1, src2, dst sont bien choisis selon RNG
3. Vérifier l'ordre d'exécution

**Actions:**
- [ ] Pour chaque round, logger: src1, src2, dst, op_type
- [ ] Comparer avec implémentation de référence
- [ ] Corriger si différences
- [ ] Vérifier que mix[dst] = op(mix[src1], mix[src2])

**Test:**
```
Round 0:
  RNG → src1=5, src2=12, dst=7, op=ADD
  mix[7] = mix[5] + mix[12]
  
Round 1:
  RNG → src1=3, src2=8, dst=15, op=XOR
  mix[15] = mix[3] ^ mix[8]
  
... comparer avec référence
```

**Critère succès:**
Toutes les operations dans le bon ordre avec bons registres

---

### **JOUR 6 - DAG Accesses & Merge** 💾

**Problème identifié:**
- Adresse DAG calculée peut être incorrecte
- Merge des données DAG peut être faux

**Objectifs:**
1. Vérifier calcul adresse DAG
2. Vérifier merge avec FNV1a
3. S'assurer que DAG items sont lus correctement

**Actions:**
- [ ] Logger adresse DAG pour chaque round
- [ ] Logger DAG item lu
- [ ] Vérifier merge: mix[0/1] = fnv1a(mix[0/1], dag_data)
- [ ] Comparer avec référence

**Test:**
```
Round 0:
  dag_addr = mix[0] % dag_size = 0x12345
  dag_item = dag[0x12345] = 0xABCDEF...
  mix[0] = fnv1a(mix[0], dag_item_low)
  mix[1] = fnv1a(mix[1], dag_item_high)
```

**Critère succès:**
DAG accesses identiques à référence

---

### **JOUR 7 - Mix Reduction** 🔄

**Problème identifié:**
La réduction de 32 registres → 8 mots peut être incorrecte.

**Objectifs:**
1. Vérifier algorithme de réduction
2. S'assurer que FNV1a est appliqué correctement

**Actions:**
- [ ] Vérifier: mix_hash[i] = FNV_INIT
- [ ] Pour j=0..3: mix_hash[i] = fnv1a(mix_hash[i], mix[i*4+j])
- [ ] Logger mix_hash[0..7]
- [ ] Comparer avec référence

**Test:**
```
Avant réduction:
  mix[0..31] = {...}
  
Après réduction:
  mix_hash[0] = fnv1a(fnv1a(fnv1a(fnv1a(0x811c9dc5, mix[0]), mix[1]), mix[2]), mix[3])
  mix_hash[1] = fnv1a(..., mix[4..7])
  ...
```

**Critère succès:**
mix_hash identique à référence

---

### **JOUR 8 - Keccak Final & Byte Order** 🔐

**Problème identifié:**
- Keccak-256 peut avoir des erreurs subtiles
- Byte order (endianness) peut être incorrect

**Objectifs:**
1. Vérifier Keccak-256 avec test vectors officiels
2. Vérifier byte order dans final_input
3. Vérifier byte order dans result

**Actions:**
- [ ] Tester Keccak-256 seul avec vectors connus
- [ ] Vérifier construction de final_input[64]
- [ ] Vérifier que header_hash est copié correctement
- [ ] Vérifier que mix_hash est dans le bon ordre
- [ ] Comparer result avec référence

**Test:**
```
Keccak256("test") = 0x9c22ff5f21f0b81b113e63f7db6da94fedef11b2119b4088b89664fb9a3cb658
→ Vérifier que ma version donne pareil
```

**Critère succès:**
Keccak-256 100% correct + byte order correct

---

### **JOUR 9 - DAG Generation Correct** 🏗️

**Problème identifié:**
Mon DAG generation est peut-être trop simple.

**Objectifs:**
1. Implémenter vrai algo DAG Ethash
2. Adapter pour KawPow si nécessaire
3. Vérifier avec seedHash connus

**Actions:**
- [ ] Étudier algo DAG Ethash (calcdag)
- [ ] Implémenter: cache → DAG items
- [ ] Vérifier avec epoch connus
- [ ] Comparer DAG[0], DAG[1000], DAG[1000000] avec référence

**Référence:**
```
Ethash DAG generation:
1. Generate cache (16MB) depuis seed
2. Generate DAG items depuis cache
3. Chaque item = mix de cache items avec Keccak
```

**Critère succès:**
DAG items identiques à Ethash/KawPow de référence

---

### **JOUR 10 - Integration & Tests Finaux** 🎯

**Objectifs:**
1. Assembler tous les composants corrigés
2. Tester sur pool testnet
3. Tester sur pool mainnet
4. Optimiser performance

**Actions:**
- [ ] Compiler version finale
- [ ] Tester: doit trouver shares
- [ ] Vérifier shares acceptés par pool
- [ ] Si rejetés: debug avec logs pool
- [ ] Optimiser: shared memory, registres, etc.
- [ ] Atteindre 12+ MH/s stable

**Critères succès:**
- ✅ Shares trouvés
- ✅ Shares acceptés par pool
- ✅ Hashrate 12+ MH/s
- ✅ Stable pendant 1h

---

## 📚 RESSOURCES NÉCESSAIRES:

### **Spec Officielle:**
- EIP-1057: ProgPoW specification
- Ethash spec (pour DAG)
- KawPow differences vs ProgPoW

### **Implémentations de Référence:**
- kawpowminer (open source)
- ethminer (Ethash reference)
- ProgPoW test vectors

### **Test Vectors:**
- KISS99 test vectors
- ProgPoW test vectors (known inputs → outputs)
- Keccak-256 test vectors
- DAG test vectors

---

## 🔍 MÉTHODOLOGIE:

### **Pour chaque composant:**

1. **Isoler** le composant (ex: KISS99)
2. **Tester** avec inputs connus
3. **Comparer** avec output attendu
4. **Corriger** si différent
5. **Valider** avec multiples tests
6. **Intégrer** dans le code complet

### **Debug process:**

```
Pour chaque hash calculé:
1. Logger tous les états intermédiaires
2. Comparer avec référence (même input)
3. Identifier EXACTEMENT où ça diffère
4. Corriger cette étape
5. Re-tester
6. Répéter jusqu'à 100% identique
```

---

## 🎯 POINTS DE VALIDATION:

### **Jour 4:**
☐ KISS99 génère la même séquence que référence

### **Jour 5:**
☐ Math operations dans le bon ordre
☐ Registres src/dst corrects

### **Jour 6:**
☐ DAG accesses corrects
☐ Merge FNV1a correct

### **Jour 7:**
☐ Mix reduction identique

### **Jour 8:**
☐ Keccak-256 validé
☐ Byte order correct

### **Jour 9:**
☐ DAG items corrects

### **Jour 10:**
☐ **SHARES ACCEPTÉS !** ✅

---

## 💪 ENGAGEMENT:

**Je vais:**
1. Travailler méthodiquement, composant par composant
2. Tester chaque étape avec rigueur
3. Comparer avec implémentations de référence
4. Ne pas "deviner" - vérifier avec spec
5. Logger et debugger jusqu'à ce que ce soit 100% correct

**Résultat attendu:**
**Mineur KawPow fonctionnel avec shares acceptés dans 10 jours !** 🎉

---

## 📝 JOURNAL DE PROGRESSION:

### **Jour 3 - AUJOURD'HUI:**
- [x] Créé version debug
- [x] Créé feuille de route
- [ ] Compiler et tester version debug
- [ ] Analyser outputs
- [ ] Lire spec ProgPoW

### **Jour 4 - À VENIR:**
- [ ] KISS99 correct
- [ ] Test vectors KISS99
- [ ] Validation séquence

### **...**

---

**ON VA Y ARRIVER ! 💪🔥**

**Jour par jour, composant par composant, jusqu'à la victoire !** 🎯
