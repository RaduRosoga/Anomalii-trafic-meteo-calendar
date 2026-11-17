# Anomalii trafic–meteo–calendar · U14

Roșoga Radu-Ștefan · MEDE · **Livrabile CP0–CP3**. CP4–CP6 nu fac parte din această publicare.

| Checkpoint | De deschis mai întâi | Dovezi |
|---|---|---|
| CP0 | [Prezentare](CP0/CP0_U14.pdf) | Concept și acces verificat |
| CP1 v2 | [Prezentare](CP1/CP1_U14.pdf) | Data card, SOA cu 10 articole, acoperire și exemplu de aliniere |
| CP2 | [Prezentare](CP2/CP2_U14.pdf) | Matrice completă comprimată, dicționar, statistici, PCA/loadings |
| CP3 | [Prezentare](CP3/CP3_U14.pdf) | Protocol, relații, comparația PCA–RBF, semințe și subperioade |

## Verificare rapidă

Instalează `pip install -r requirements.txt`, apoi deschide `verificare_CP0_CP3.ipynb`. Matricea derivată și rezultatele sunt incluse; nu sunt necesare bazele brute pentru această inspecție. Notebookul și CSV-urile sunt lizibile și direct în GitHub.

## Recalculare din date brute

Python 3.12+. Creează `data/` în rădăcină și adaugă fișierele indicate în CP1/datacard_U14.pdf: METR-LA.csv, open-meteo-34.06N118.24W91m.csv și cele două CSV-uri calendar în subfolderul American Holidays 1722 - 2122. Surse: https://zenodo.org/records/5146275 ; https://open-meteo.com/en/docs/historical-weather-api ; https://www.kaggle.com/datasets/jeremygerdes/us-federal-pay-and-leave-holidays-2004-to-2100-csv . Hash-urile exporturilor exacte sunt în CP1/acoperire.json. Bazele brute voluminoase nu sunt publicate aici.

Rulează `python src/run_cp0_cp3.py`. Se încarcă toate sursele, se aliniază meteo orar la sloturi de 5 minute, se construiește matricea și se repetă PCA/RBF pe surse separate și combinate. Prezentările și documentele sunt versiunile verificate; modificarea datelor cere actualizarea cifrelor narative.

## Protocol și limite

18.628 observații senzor–zi, 181 senzori, 105 zile și 41 trăsături active. Meteo UTC este convertit în America/Los_Angeles, apoi H:00 se repetă la H:00–H:55 fără interpolare. Cantitatea de ploaie se însumează o dată pe oră. Fusul traficului este o ipoteză documentată; DST și ultima zi meteo incompletă sunt excluse. Scalare și imputare numai pe train.

Weather data by [Open-Meteo.com](https://open-meteo.com/), CC BY 4.0; conversie de fus, repetare și agregare. Drepturile surselor exacte și proveniența sunt documentate în data card; licența codului unei biblioteci nu este licența datelor.
