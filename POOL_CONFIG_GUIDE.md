# Configuration Pool - Guide Complet

## Modes d'Authentification

Le mineur supporte **2 modes** de configuration pool:

### Mode 1: Wallet + Worker
Pour pools nécessitant une adresse wallet (Ethermine, Hiveon, F2Pool, etc.)

### Mode 2: Username + Password
Pour pools avec système de compte (2Miners avec compte, NiceHash, MiningPoolHub, etc.)

## Exemples par Pool

### 🔹 Ethermine (Ethereum)
```
Mode: 1 (Wallet + Worker)
URL: eu1.ethermine.org
Port: 4444
Wallet: 0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb
Worker: rig1
→ Username généré: 0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb.rig1
→ Password: x (par défaut)
```

### 🔹 Hiveon (Ethereum)
```
Mode: 1 (Wallet + Worker)
URL: eu.hiveon.com
Port: 4444
Wallet: 0xYOUR_ETH_WALLET
Worker: worker1
→ Username: wallet.worker1
→ Password: x
```

### 🔹 2Miners avec Compte
```
Mode: 2 (Username complet)
URL: eth.2miners.com
Port: 2020
Username: username.worker1
Password: votre_password
→ Utilise directement username + password
```

### 🔹 2Miners sans Compte (Wallet)
```
Mode: 1 (Wallet + Worker)
URL: eth.2miners.com
Port: 2020
Wallet: 0xYOUR_ETH_WALLET
Worker: rig1
→ Username: wallet.rig1
→ Password: x
```

### 🔹 NiceHash
```
Mode: 2 (Username complet)
URL: daggerhashimoto.eu.nicehash.com
Port: 3353
Username: 3JqPBxd8TKkjL3rKYLz62YbU9xxxx.worker
Password: x (ou votre password si configuré)
```

### 🔹 MiningPoolHub
```
Mode: 2 (Username complet)
URL: us-east.ethash-hub.miningpoolhub.com
Port: 20535
Username: username.workername
Password: votre_password_mph
```

### 🔹 Nanopool (Ethereum Classic)
```
Mode: 1 (Wallet + Worker)
URL: etc-eu1.nanopool.org
Port: 19999
Wallet: 0xYOUR_ETC_WALLET
Worker: worker1
→ Username: wallet.worker1/your@email.com
→ Password: x
```

### 🔹 Flexpool
```
Mode: 1 (Wallet + Worker)
URL: eth.flexpool.io
Port: 4444
Wallet: 0xYOUR_ETH_WALLET
Worker: rig1
→ Username: wallet.rig1
→ Password: x
```

### 🔹 Sparkpool
```
Mode: 1 (Wallet + Worker)
URL: eu.sparkpool.com
Port: 3333
Wallet: 0xYOUR_ETH_WALLET
Worker: rig1
→ Username: wallet.rig1
→ Password: x
```

### 🔹 F2Pool
```
Mode: 2 (Username complet)
URL: eth.f2pool.com
Port: 6688
Username: username.worker
Password: anything (F2Pool ignore le password)
```

### 🔹 Suprnova (Bitcoin Gold)
```
Mode: 2 (Username complet)
URL: btg.suprnova.cc
Port: 8866
Username: votre_login.worker1
Password: votre_password_suprnova
```

### 🔹 Flypool (Zcash)
```
Mode: 1 (Wallet + Worker)
URL: eu1-zcash.flypool.org
Port: 3333
Wallet: t1abcdefghijklmnopqrstuvwxyz123456789
Worker: rig1
→ Username: wallet.rig1
→ Password: x
```

## Flux de Configuration

```
cuda_miner.exe

Choix: 4 (Miner sur Pool)
GPU: 0

URL pool: eu1.ethermine.org
Port: 4444

Mode d'authentification:
1. Wallet + Worker
2. Username complet
Choix: 1

Wallet: 0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb
Worker: rig1

=== Configuration ===
Pool: eu1.ethermine.org:4444
Username: 0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb.rig1
Password: x
=====================

Connexion...
✓ Connecté!
✓ Subscribe OK
✓ Authentifié!
```

## Cas d'Usage par Type de Pool

### Pools Wallet-Only (Mode 1)
Pas besoin de créer compte, juste une adresse wallet:
- Ethermine
- Hiveon
- Flexpool
- Sparkpool
- Nanopool
- Flypool

**Avantages:**
- Anonyme
- Pas d'inscription
- Payements directs au wallet

### Pools avec Compte (Mode 2)
Nécessitent inscription sur le site:
- MiningPoolHub
- Suprnova
- NiceHash (avec compte)
- 2Miners (optionnel)

**Avantages:**
- Dashboard web détaillé
- Statistiques avancées
- Notifications
- Seuils de paiement personnalisables

### Pools Hybrides (Mode 1 ou 2)
Supportent les deux modes:
- 2Miners
- F2Pool

## Password Optionnel

**Quand le password est ignoré:**
- Ethermine: ignore, utilise toujours "x"
- Hiveon: ignore
- F2Pool: ignore
- La plupart des pools wallet-only

**Quand le password est requis:**
- MiningPoolHub: **obligatoire**
- Suprnova: **obligatoire**
- NiceHash avec compte: recommandé
- Pools avec authentification forte

**Quand le password est optionnel:**
- 2Miners avec compte: utilise si fourni
- Certains pools stratum personnalisés

## Format Username Spécial

### Nanopool - Email dans Username
```
Username: wallet.worker/email@example.com
Password: x
```

### NiceHash - Format BTC Address
```
Username: 3JqPBxd8TKkjL3rKYLz62YbU9xxxx.worker
Password: x
```

### MiningPoolHub - Format hub_ACCOUNTNAME
```
Username: hub_accountname.worker
Password: password_mph
```

## Vérification Configuration

**Pool retourne erreur "unauthorized":**
- ❌ Mauvais username/wallet
- ❌ Mauvais password (si requis)
- ❌ Worker non autorisé
- ❌ Compte inactif/suspendu

**Pool accepte puis déconnecte:**
- ❌ Version stratum incompatible
- ❌ Algorithme non supporté
- ❌ Région bloquée

**Pool accepte et reste connecté:**
- ✅ Configuration correcte
- ✅ Prêt à miner

## Exemples Complets

### Exemple 1: Débutant Ethereum
```
Pool: Ethermine (simple, fiable)
Mode: 1
URL: eu1.ethermine.org
Port: 4444
Wallet: créer sur MetaMask
Worker: rig1
Password: x
```

### Exemple 2: Mineur Pro avec Stats
```
Pool: MiningPoolHub
Mode: 2
URL: us-east.ethash-hub.miningpoolhub.com
Port: 20535
Username: compte_mph.worker1
Password: password_mph_dashboard
```

### Exemple 3: Multi-Algo Switching
```
Pool: NiceHash
Mode: 2
URL: daggerhashimoto.eu.nicehash.com
Port: 3353
Username: BTC_address.worker
Password: x
```

## Recommandations

**Pour Ethereum:**
1. Ethermine (débutant)
2. Flexpool (frais bas)
3. Hiveon (0% frais)

**Pour Ethereum Classic:**
1. 2Miners
2. Nanopool

**Pour Bitcoin Gold:**
1. 2Miners
2. Suprnova

**Pour Multi-Algo:**
1. NiceHash (simple)
2. MiningPoolHub (avancé)

## Support

Si erreur "unauthorized" persiste:
1. Vérifier format wallet (0x pour ETH)
2. Vérifier compte actif (pools avec login)
3. Tester avec Mode 2 si Mode 1 échoue
4. Vérifier password si pool avec compte
5. Consulter doc pool spécifique

## Liste Complète Ports Courants

**Ethereum:**
- 4444 (Ethermine, Hiveon, Flexpool)
- 3333 (Sparkpool)
- 6688 (F2Pool)

**Ethereum Classic:**
- 1010 (2Miners)
- 19999 (Nanopool)

**Bitcoin Gold:**
- 4040 (2Miners)
- 8866 (Suprnova)

**Zcash:**
- 3333 (Flypool)
- 1010 (2Miners)

**NiceHash:**
- 3353 (Dagger/Ethash)
