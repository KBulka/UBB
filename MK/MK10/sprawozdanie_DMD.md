# Sprawozdanie: Analiza DMD (Dynamic Mode Decomposition)

## Wariant 11

---

## 1. Cel i metoda

Zadanie polegało na wyznaczeniu macierzy przekształcenia **A** dla liniowego układu dynamicznego:

$$X' = A \cdot X$$

gdzie **X** to macierz stanów początkowych, a **X'** to macierz stanów następnych.

Użyta metoda: **DMD (Dynamic Mode Decomposition)** przez pseudoodwrotność Moore'a-Penrose'a:

$$A = X' \cdot X^+$$

gdzie $X^+ = (X^T X)^{-1} X^T$ jest obliczana za pomocą rozkładu SVD dla stabilności numerycznej.

---

## 2. Dane wejściowe

| Parametr | Wartość |
|----------|---------|
| Liczba próbek | 20 |
| Liczba cech/wymiarów | 37 |
| Macierz X | (20, 37) |
| Macierz X' | (20, 37) |
| Zakres wartości | $[1.05 \times 10^2, 1.59 \times 10^{39}]$ |

Dane pochodzą z plików CSV: `War11_X.csv` i `War11_Xprime.csv` z separatorem średnika i przecinkiem jako separatorem dziesiętnym.

---

## 3. Metoda obliczeniowa

### Krok 1: Rozkład SVD
Macierz **X** rozkładamy: $X = U \Sigma V^T$

### Krok 2: Pseudoodwrotność
$$X^+ = V \Sigma^{-1} U^T$$

Wartości singularne mniejsze niż próg ($10^{-10} \times \max(\Sigma)$) są ignorowane dla stabilności.

### Krok 3: Obliczenie A
$$A = X' \cdot V \cdot \Sigma^{-1} \cdot U^T$$

---

## 4. Wyniki

### Macierz A
Wymiary: **(20, 20)**

Wynikowa macierz A zawiera elementy w zakresie od $-2.47 \times 10^{-2}$ do $2.61 \times 10^{-2}$.

### Jakość dopasowania

Weryfikacja warunku $X' \approx A \cdot X$:

| Metryka | Wartość |
|---------|---------|
| Błąd bezwzględny (max) | $4.55 \times 10^{3}$ |
| Błąd bezwzględny (średni) | $2.30 \times 10^{2}$ |
| Błąd względny (max) | **0.85%** |
| Błąd względny (średni) | **0.28%** |
| **Błąd Frobeniusa** | **0.32%** |

Bardzo mały błąd względny wskazuje na doskonałe dopasowanie modelu DMD do danych.

---

## 5. Analiza stabilności

### Wartości własne macierzy A

Macierz A ma 20 wartości własnych. Maksymalna amplituda:

$$\max|\lambda| = 1.053$$

**Wnioski:**
- Prawie wszystkie wartości własne leżą wewnątrz okręgu jednostkowego (|λ| < 1)
- Jedna wartość własna przewyższa $|λ| > 1$
- Układ jest **marginalnie niestabilny** (ma charakter okresowy/oscylacyjny)
- Amplituda drgań wzrasta bardzo powoli (współczynnik wzmocnienia ≈ 1.053 na krok)

---

## 6. Interpretacja wyników

Przeprowadzona analiza DMD pokazuje, że:

1. **Liniowość modelu**: Błąd Frobeniusa 0.32% potwierdza, że dane mają charakter liniowy i dobrze opisane są modelem $X' = AX$.

2. **Dynamika układu**: Macierz A ma wymiar 20×20 i reprezentuje transformację liniową z przestrzeni 20-wymiarowej.

3. **Stabilność**: Układ wykazuje powolny wzrost amplitudy z każdym krokiem (marginalnie niestabilny), co wskazuje na lekko chaotyczną lub rosnącą dynamikę.

4. **Praktyczne zastosowanie**: Macierz A może być używana do:
   - Predykcji przyszłych stanów: $X_{n+1} = A \cdot X_n$
   - Analizy modalnej (rozkład na mody dominujące)
   - Identyfikacji systemów dynamicznych

---

## 7. Podsumowanie

**Macierz A wyznaczona metodą DMD:** 
- Wymiary: 20×20
- Metoda: Pseudoodwrotność Moore'a-Penrose'a (SVD)
- Błąd modelowania: 0.32% (norma Frobeniusa)
- Status stabilności: Marginalnie niestabilny (max |λ| ≈ 1.053)
- Plik wynikowy: `Macierz_A.csv`

Analiza wykazała wysoką jakość dopasowania i poprawność metody DMD dla tego zestawu danych.

---

*Sprawozdanie przygotowane w ramach analizy dynamicznych stanów układu (Wariant 11)*
