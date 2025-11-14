# ECB Vasicek Calibration

Calibra il modello Vasicek su dati di curva dei tassi ECB (spot rates) da file locale.

## Prerequisiti

- `ecb_data.csv` con:
  - Indice: `TIME_PERIOD` (date)
  - Colonne: `ecb_0`, `ecb_3m`, `ecb_6m`, `ecb_1y`, `ecb_2y`, ...

## Installazione

```bash
pip install -r requirements.txt
