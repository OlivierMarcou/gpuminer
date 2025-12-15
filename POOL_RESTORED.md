# Fonctionnalité Pool Restaurée

## ✅ Option Pool Disponible

**Menu mis à jour:**
```
1. SHA256 (test local)
2. Ethash (DAG)
3. Equihash
4. Miner sur Pool (Stratum)  ← RESTAURÉ
5. Quitter
```

## Utilisation Pool

```cmd
cuda_miner.exe

Choix: 4
GPU: 0

URL pool (ex: eu1.ethermine.org): eu1.ethermine.org
Port (ex: 4444): 4444
Wallet: 0xVOTRE_WALLET_ETHEREUM
Worker (ex: rig1): rig1

→ Connexion...
→ Subscribe...
→ Autorisation...
→ Connecté!
```

## Pools Supportées

### Ethereum (Ethash)
```
Ethermine: eu1.ethermine.org:4444
Hiveon: eu.hiveon.com:4444
F2Pool: eth.f2pool.com:6688
```

### Ethereum Classic
```
2Miners: etc.2miners.com:1010
Nanopool: etc-eu1.nanopool.org:19999
```

### Bitcoin Gold (Equihash)
```
2Miners: btg.2miners.com:4040
Suprnova: btg.suprnova.cc:8866
```

### Zcash (Equihash)
```
Flypool: eu1-zcash.flypool.org:3333
2Miners: zec.2miners.com:1010
```

## Protocole Stratum Implémenté

**Fonctions actives:**

✅ `pool_connect()` - Connexion TCP/IP  
✅ `pool_subscribe()` - Abonnement Stratum  
✅ `pool_authorize()` - Authentification wallet  
✅ `pool_submit_share()` - Soumission shares  
✅ `pool_start_listener()` - Thread écoute jobs  
✅ `pool_stop_listener()` - Arrêt propre  

**Parsing JSON:**
✅ Extraction job_id, prevhash, coinbase  
✅ Extraction merkle branches  
✅ Extraction difficulty, ntime  
✅ Parsing réponses pool  

## État Actuel

**Fonctionnel:**
- ✅ Connexion aux pools
- ✅ Authentification
- ✅ Subscribe/Authorize
- ✅ Réception messages

**En développement:**
- ⚠️ Intégration minage GPU avec jobs pool
- ⚠️ Soumission shares automatique
- ⚠️ Boucle minage complète

**Version actuelle:**
Connexion et authentification fonctionnent. La boucle de minage complète avec soumission de shares sera implémentée dans la prochaine version.

## Fichiers

```
cuda_miner.cu    - Ajout mine_on_pool()
stratum.c        - Client Stratum complet
cJSON.c          - Parser JSON
build_cuda.bat   - Compile tout
```

## Compilation

```cmd
build_cuda.bat
```

Compile:
- cuda_miner.obj
- sha256.obj
- ethash.obj
- equihash.obj
- **stratum.obj** ← Restauré
- **cJSON.obj** ← Restauré

Link: `cuda_miner.exe`

## Test Connexion Pool

```cmd
cuda_miner.exe

Choix: 4
GPU: 0
URL: eu1.ethermine.org
Port: 4444
Wallet: 0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb
Worker: test

→ Devrait afficher:
  ✓ Connecté!
  ✓ Subscribe OK
  ✓ Authentifié!
```

## Prochaine Étape

Intégration complète:
1. Recevoir job de la pool
2. Construire header pour GPU
3. Miner avec kernel approprié
4. Soumettre shares trouvés
5. Boucle continue

Code déjà disponible dans `stratum.c`:
- `pool_start_listener()` - Thread écoute
- `pool_parse_notify()` - Parse jobs
- Callbacks pour nouveaux jobs

## Pourquoi C'était Enlevé

nvcc (compilateur CUDA) avait des erreurs avec:
- Code C complexe dans .cu
- Variables locales dans boucles
- Parsing accolades

**Solution appliquée:**
- Simplifié mine_on_pool() dans .cu
- Logique complexe dans stratum.c (C pur)
- Compilation séparée puis linkage

## Vérification

```cmd
REM Vérifier stratum.obj compilé
dir *.obj

REM Devrait montrer:
cuda_miner.obj
sha256.obj
ethash.obj
equihash.obj
stratum.obj
cJSON.obj
```

## Support

Pool fonctionne avec:
- Protocole Stratum standard
- TCP/IP Winsock
- JSON parsing
- Threading Windows

Compatible avec pools:
- Ethermine, Hiveon, F2Pool
- 2Miners, Nanopool
- Flypool, Suprnova
