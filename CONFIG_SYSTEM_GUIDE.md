# 📁 SYSTÈME DE CONFIGURATION - pool_config.ini

## 🎯 **POURQUOI CE FICHIER ?**

**AVANT:**
```
À chaque lancement, tu devais retaper:
- URL pool: europe.mining-dutch.nl
- Port: 9985
- Username: omarcou.workerK
- Password: d=4000
```
**= Fatiguant et source d'erreurs ! 😓**

**APRÈS:**
```
1. Édite pool_config.ini UNE FOIS
2. Lance cuda_miner.exe
3. Choisis "1. Config rapide"
4. Choisis ton profil
5. Mine directement !
```
**= Simple et rapide ! 🚀**

---

## 📋 **STRUCTURE DU FICHIER:**

### **Format:**
```ini
[NOM_PROFIL]
pool_url=adresse.pool.com
pool_port=9999
wallet=TON_WALLET
worker=rig1
password=x
algo=ethash
```

### **Exemple Complet:**
```ini
[KAWPOW_MINING_DUTCH]
pool_url=europe.mining-dutch.nl
pool_port=9985
username=omarcou.workerK
password=d=4000
algo=kawpow
auth_mode=2

[ETHASH_2MINERS]
pool_url=etc.2miners.com
pool_port=1010
wallet=0x1234567890ABCDEF...
worker=rig1
password=x
algo=ethash
```

---

## ⚙️ **PARAMÈTRES DISPONIBLES:**

### **Obligatoires:**
- `pool_url` - Adresse de la pool (sans http://)
- `pool_port` - Port de la pool
- `algo` - ethash ou kawpow

### **Mode Authentification 1 (Wallet + Worker):**
```ini
wallet=TON_WALLET_ICI
worker=rig1
password=x
```
**Usage:** Pools standards (2Miners, Ethermine, Flypool, etc.)

### **Mode Authentification 2 (Username complet):**
```ini
username=login.worker
password=mot_de_passe
auth_mode=2
```
**Usage:** Mining-Dutch et pools spéciales

---

## 🔧 **CONFIGURATION:**

### **Étape 1: Éditer pool_config.ini**

**Pour ETHASH (Ethereum Classic):**
```ini
[MON_PROFIL_ETC]
pool_url=etc.2miners.com
pool_port=1010
wallet=0xTON_WALLET_ETC_ICI
worker=rig1
password=x
algo=ethash
```

**Remplace:** `0xTON_WALLET_ETC_ICI` par ton VRAI wallet ETC !

---

**Pour KAWPOW (Ravencoin):**
```ini
[MON_PROFIL_RVN]
pool_url=rvn.2miners.com
pool_port=6060
wallet=TON_WALLET_RVN_ICI
worker=rig1
password=x
algo=kawpow
```

**Remplace:** `TON_WALLET_RVN_ICI` par ton VRAI wallet RVN !

---

### **Étape 2: Sauvegarder**

Sauvegarde `pool_config.ini` dans le **MÊME dossier** que `cuda_miner.exe` !

---

### **Étape 3: Utiliser**

```cmd
cuda_miner.exe
3 (Miner sur Pool)
GPU 0
3 (KawPow)
1 (Config rapide) ← NOUVEAU !
```

**Le programme liste les profils:**
```
Configurations disponibles:
1. ETHASH_2MINERS
2. KAWPOW_2MINERS
3. KAWPOW_MINING_DUTCH
4. MON_PROFIL_RVN

Choix (numéro): 4
```

**Et charge automatiquement la config !** ✅

---

## 🌟 **PROFILS PRÉ-CONFIGURÉS:**

Le fichier `pool_config.ini` fourni contient **6 profils prêts à l'emploi** :

### **ETHASH (Ethereum Classic):**
1. **ETHASH_2MINERS** - 2Miners Europe (recommandé)
2. **ETHASH_ETHERMINE** - Ethermine Europe

### **KAWPOW (Ravencoin):**
1. **KAWPOW_2MINERS** - 2Miners RVN (recommandé)
2. **KAWPOW_MINING_DUTCH** - Mining-Dutch (ton profil actuel)
3. **KAWPOW_FLYPOOL** - Flypool RVN
4. **KAWPOW_HEROMINERS** - HeroMiners RVN

**Il suffit de remplacer les wallets par les tiens !**

---

## ✏️ **ÉDITER LE FICHIER:**

### **Windows:**
```cmd
notepad pool_config.ini
```

### **Ou:**
- Clic droit → Modifier
- N'importe quel éditeur de texte

### **⚠️ IMPORTANT:**
- **NE PAS** utiliser Word !
- **NE PAS** ajouter d'espaces avant/après les `=`
- **SAUVEGARDER** en UTF-8 ou ANSI

---

## 📊 **EXEMPLE COMPLET:**

### **Ton Profil Mining-Dutch (déjà configuré):**
```ini
[KAWPOW_MINING_DUTCH]
pool_url=europe.mining-dutch.nl
pool_port=9985
username=omarcou.workerK
password=d=4000
algo=kawpow
auth_mode=2
```

**Ce profil est PRÊT à l'emploi !** ✅

### **Ajouter ton wallet 2Miners:**
```ini
[MON_RVN_2MINERS]
pool_url=rvn.2miners.com
pool_port=6060
wallet=RNrW8vxxx...ton_wallet...xxxxx
worker=gtx1660
password=x
algo=kawpow
```

---

## 🎯 **AVANTAGES:**

### **1. Plus rapide** ⚡
Une fois configuré, lance en 10 secondes !

### **2. Pas d'erreurs** ✅
Plus de typos dans l'URL ou le port

### **3. Multiples profils** 🔄
Bascule facilement entre pools

### **4. Partageable** 📤
Envoie ton config à un ami

### **5. Historique** 📜
Garde trace de toutes tes pools

---

## 🔄 **USAGE QUOTIDIEN:**

### **Scénario 1: Utiliser config existante**
```cmd
cuda_miner.exe
3 → 0 → 3 → 1 → 3 (KAWPOW_MINING_DUTCH)
```
**= 5 clics ! ⚡**

### **Scénario 2: Config manuelle (comme avant)**
```cmd
cuda_miner.exe
3 → 0 → 3 → 2 (Config manuelle)
[Tape URL, port, etc.]
```
**= Toujours possible si tu préfères !**

---

## 🛠️ **COMPILATION:**

**Le fichier config_reader.c est maintenant compilé automatiquement !**

```cmd
build_simple.bat
```

**Compile:**
- sha256.cu ✅
- ethash.cu ✅
- kawpow.cu ✅
- stratum.c ✅
- cJSON.c ✅
- **config_reader.c** ✅ NOUVEAU !

---

## 📁 **FICHIERS NÉCESSAIRES:**

```
ton_dossier/
├── cuda_miner.exe
├── pool_config.ini       ← Édite ce fichier !
├── kawpow.cu
├── ethash.cu
├── sha256.cu
├── stratum.c
├── cJSON.c
├── config_reader.c       ← Nouveau
├── build_simple.bat
└── ...
```

---

## 🐛 **DÉPANNAGE:**

### **"Fichier pool_config.ini introuvable"**

**Cause:** Fichier pas dans le bon dossier

**Solution:**
```cmd
REM Vérifie que pool_config.ini est dans le même dossier que cuda_miner.exe
dir pool_config.ini
```

### **"Section [XXX] introuvable"**

**Cause:** Nom de profil incorrect ou mal tapé

**Solution:**
```cmd
REM Ouvre pool_config.ini et vérifie le nom exact de la section
notepad pool_config.ini
```

### **"Erreur lecture configuration"**

**Cause:** Fichier mal formaté

**Solution:**
- Vérifie pas d'espaces avant/après `=`
- Vérifie les crochets `[SECTION]`
- Pas de caractères bizarres

---

## 💡 **ASTUCES:**

### **Créer plusieurs profils pour la même pool:**
```ini
[RVN_2MINERS_GTX1660]
pool_url=rvn.2miners.com
pool_port=6060
wallet=RNxxx...
worker=gtx1660
password=x
algo=kawpow

[RVN_2MINERS_RTX3080]
pool_url=rvn.2miners.com
pool_port=6060
wallet=RNxxx...
worker=rtx3080
password=x
algo=kawpow
```

### **Tester différentes pools facilement:**
Configure 4-5 pools, teste-les toutes, garde la meilleure !

### **Backup:**
```cmd
copy pool_config.ini pool_config.ini.backup
```

---

## 🎉 **RÉSUMÉ:**

**1 FOIS:**
- ✅ Édite `pool_config.ini`
- ✅ Remplace les wallets par les tiens
- ✅ Sauvegarde

**À CHAQUE LANCEMENT:**
- ✅ Lance `cuda_miner.exe`
- ✅ Choisis "Config rapide"
- ✅ Choisis ton profil
- ✅ **C'EST TOUT ! ⚡**

---

**ÉDITE pool_config.ini MAINTENANT ET SIMPLIFIE-TOI LA VIE !** 🚀
