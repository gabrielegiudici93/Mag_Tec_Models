# Strategie per Migliorare i Risultati del Modello

## Problemi Attuali
- **RMSE**: 0.32-0.59 N (10-20% di errore)
- **Offset Accuracy**: 0.29-0.31 (peggio del random 20%)
- **Stretch Accuracy**: 0.07 (molto peggio del random 33%)
- **Dati sbilanciati**: Pochi dati per forze basse (0-1N)

## Strategie di Miglioramento (Senza Cambiare il Modello)

### 1. **Feature Engineering** ⭐ (Alto Impatto)
Attualmente usiamo solo i valori raw (45 features). Possiamo aggiungere:

#### A. Statistiche per Sensore
- Mean, std, max, min per ogni sensore (15 sensori × 4 stats = 60 features)
- Magnitudine del campo magnetico: `sqrt(Bx² + By² + Bz²)` per ogni sensore (15 features)

#### B. Features Spaziali
- Differenze tra sensori vicini (gradienti)
- Media dei sensori centrali (indice 6, 7, 8)
- Differenze tra sensori periferici e centrali

#### C. Features Temporali (se disponibili)
- Posizione nella sequenza (normalizzata)
- Velocità di cambiamento (derivata)
- Accelerazione (seconda derivata)

### 2. **Preprocessing Migliorato**
- **Filtro passa-basso**: Ridurre rumore ad alta frequenza
- **Smoothing temporale**: Media mobile per sequenze
- **Outlier removal più aggressivo**: Z-threshold più basso (2.5 invece di 3.0)

### 3. **Data Augmentation**
- **Aggiungere rumore controllato**: Gaussian noise con std piccolo
- **Interpolazione per forze basse**: Generare più esempi sintetici per 0-1N
- **Time warping**: Variare leggermente la velocità temporale

### 4. **Hyperparameter Tuning**
- Grid search per ottimizzare:
  - `n_estimators`: 200 → 500-1000
  - `max_depth`: 30 → 20-40
  - `min_samples_split`: 5 → 2-10
  - `min_samples_leaf`: 2 → 1-5

### 5. **Migliorare la Raccolta Dati**
- **Più dati per forze basse**: Raccogliere più sequenze partendo da 0N
- **Migliorare calibrazione**: Verificare offset del sensore di forza
- **Aumentare numero di sequenze**: Da 50 a 100+ per offset

### 6. **Feature Selection**
- Rimuovere features ridondanti o poco informative
- Usare feature importance per selezionare le migliori

### 7. **Ensemble di Modelli** (senza cambiare il tipo)
- Trainare più RandomForest con parametri diversi
- Media delle predizioni (voting)

## Priorità di Implementazione

1. **Feature Engineering** (più facile, alto impatto)
2. **Hyperparameter Tuning** (facile, medio impatto)
3. **Preprocessing Migliorato** (medio, medio impatto)
4. **Data Augmentation** (medio, basso impatto)
5. **Migliorare Raccolta Dati** (difficile, alto impatto ma richiede tempo)

