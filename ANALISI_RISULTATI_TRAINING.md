# Analisi Risultati Training - Single Point Models

## Risultati Attuali

```
Model           Sequences    Train Seq    Test Seq     Samples      Train Samples   Test Samples    RMSE       Offset Acc  
--------------------------------------------------------------------------------
000pct          149          104          45           11406        8294            3112            0.3970     0.3094      
010pct          149          104          45           8445         6025            2420            0.3155     0.2963      
020pct          149          104          45           8955         6372            2583            0.6062     0.2989      
combined        447          312          135          28806        20462           8344            0.3579     1.0000      
```

## Problemi Identificati

### 1. **RMSE Alto (0.3-0.6 N)**
- **Problema**: RMSE di 0.3-0.6 N è molto alto rispetto al range di 0-3N (10-20% di errore)
- **Causa probabile**:
  - Features magnetiche non sufficientemente informative
  - Mancanza di normalizzazione (ora risolto)
  - Offset troppo vicini tra loro (solo 2.5-5mm di differenza)
  - Dati di bassa qualità o rumore elevato

### 2. **Offset Accuracy Bassa (0.29-0.31)**
- **Problema**: Accuracy del 30% è peggio del random (20% per 5 classi)
- **Causa probabile**:
  - Features magnetiche non cambiano abbastanza tra offset vicini (2.5-5mm)
  - Mancanza di normalizzazione (ora risolto)
  - Modello troppo semplice (RandomForest con 200 alberi)
  - Potrebbe servire feature engineering (differenze, gradienti, etc.)

### 3. **Stretch Accuracy Molto Bassa (0.07)**
- **Problema**: Accuracy del 7% è molto peggio del random (33% per 3 classi)
- **Causa probabile**:
  - Features magnetiche non cambiano abbastanza con lo stretch
  - Dati non bilanciati tra stretch levels
  - Modello troppo semplice

### 4. **Offset Accuracy = 1.0 per Combined (BUG)**
- **Problema**: Accuracy perfetta è sospetta e probabilmente un bug
- **Causa probabile**:
  - Tutti gli offset sono "unknown" nel test set
  - Bug nella logica di calcolo dell'accuracy
  - Confronto tra array di lunghezza diversa

## Miglioramenti Implementati

### 1. **Normalizzazione delle Features**
- **Prima**: Features magnetiche con range -10689 a 9458, std=1641
- **Dopo**: Features normalizzate con mean=0, std=1 (StandardScaler)
- **Beneficio**: Migliore convergenza del modello, features su scala comparabile

### 2. **Valore Assoluto di Fz**
- **Prima**: Fz negativo (-3.3 a -1.0 N)
- **Dopo**: Fz positivo (magnitudine della forza)
- **Beneficio**: Più corretto per la regressione

### 3. **Aumento n_estimators e Limitazione max_depth**
- **Prima**: n_estimators=200, max_depth=None
- **Dopo**: n_estimators=400, max_depth=30, min_samples_split=5
- **Beneficio**: Più alberi = migliore accuratezza, max_depth limitato previene overfitting

### 4. **Debug Migliorato**
- Stampa distribuzione delle classi
- Stampa confusion matrix shape
- Migliore gestione degli errori

## Raccomandazioni per Migliorare i Risultati

### 1. **Feature Engineering**
- Aggiungere differenze tra sensori vicini
- Aggiungere gradienti spaziali
- Aggiungere statistiche temporali (media, std, max, min)
- Aggiungere features derivate (velocità, accelerazione)

### 2. **Aumentare Complessità del Modello**
- Usare XGBoost o LightGBM invece di RandomForest
- Aumentare n_estimators a 500-1000
- Usare grid search per ottimizzare hyperparameters

### 3. **Migliorare Qualità dei Dati**
- Verificare che gli offset siano correttamente estratti
- Rimuovere sequenze con rumore elevato
- Aumentare il numero di sequenze per offset

### 4. **Verificare Bug Offset Accuracy = 1.0**
- Verificare che gli offset siano correttamente estratti nel combined model
- Verificare che ci siano più classi nel test set
- Aggiungere più debug per capire cosa succede

### 5. **Usare Modelli Più Avanzati**
- Neural Networks (MLP, CNN)
- Ensemble di modelli diversi
- Transfer learning da modelli pre-addestrati

## Prossimi Passi

1. **Rieseguire il training** con le modifiche implementate
2. **Verificare i risultati** e confrontare con i precedenti
3. **Analizzare i dati** per capire perché le features non sono sufficientemente informative
4. **Implementare feature engineering** se necessario
5. **Ottimizzare hyperparameters** con grid search

