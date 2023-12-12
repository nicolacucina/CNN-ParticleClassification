# Specie
I file il cui nome inizia per _e_ si riferiscono ad elettroni, mentre quelli il cui nome inizia per _p_ si riferiscono a protoni

# Energia
I file sono divisi per energia totale rilasciata nel calorimetro (N.B. non è necessariamente uguale all'energia iniziale della particella). La suddivisione è in bin logaritmici, cioè tra il bin di energia i-esimo e il bin (i+1)-esimo c'è un rapporto costante. L'unico bin diverso è il primo _0-100_ in cui ho messo tutti gli eventi che hanno depositato meno di 100 GeV.

# Contenuto File
Ogni file contiene un _tree_ chiamato "showers" che a sua volta è composto da 3 _branch_:
- id: numero progressivo che identifica l'evento a livello globale (cioè per tutte le energie di quella specie particellare)
- E0: vera energia della particella iniziale
- dep: array con la frazione di deposito energetico medio in un certo bin di `t` e `r` (coordinata longitudinale e trasversale dello sciame)
Un tree è praticamente una tabella e ogni branch una colonna. Ogni riga del tree corrisponde ad un evento.

# Caratteristiche dep
Ogni array ha 400 elementi, ma rappresenta una matrice 20x20 srotolata. Con il metodo `reshape(20,20)` puoi riottenere la forma originale associata al binnaggio in `t` e `r`.
I valori di deposito sono rapportati al deposito totale dell'evento in questione per cui dovrebbero sempre essere compresi tra 0 e 1

# Lettura dati
Usando la libreria python `uproot` puoi facilmente importare il `tree` nel tuo progetto sottoforma di `numpy.array` o `pandas.DataFrame`:
```
import uproot as up
import numpy as np

e_file = up.open("path/enebins/e_100-126_eventTables.root")
dep = ef["showers"]["dep"].array(library="np")

# shower del primo evento di elettroni con energia depositata tra 100 e 126 GeV
dep0 = dep[0].reshape(20,20) 

```
