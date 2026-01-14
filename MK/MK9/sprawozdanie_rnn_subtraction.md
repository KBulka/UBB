# Sprawozdanie: RNN do odejmowania binarnego

## Wariant 11: Różnica dwóch liczb 15-bitowych

---

## 1. O co chodzi w zadaniu

Zadanie polegało na nauczeniu sieci neuronowej wykonywania odejmowania dwóch liczb binarnych. Sieć dostaje na wejściu dwa bity (po jednym z każdej liczby) i musi wypluć bit wyniku. Haczyk polega na tym, że musi "pamiętać" pożyczkę z poprzedniego kroku - dokładnie tak jak my robimy odejmowanie pisemne.

**Parametry wariantu 11:**
- Liczby 15-bitowe (maksymalna wartość: 16383)
- 2000 próbek treningowych
- Operacja: odejmowanie (A - B, gdzie A ≥ B)

---

## 2. Jak to działa

### Dane wejściowe

Generujemy losowe pary liczb i zamieniamy je na bity. Ważne: bity są odwrócone (najpierw najmniej znaczący), bo tak łatwiej się liczy od lewej do prawej.

Przykład:
```
x1:   100111000000000   14369
x2: - 011010000000000    2908
      ---------------   -----
t:  = 001101000000000   11461
```

### Architektura sieci

Sieć ma prostą strukturę:

```
Wejście (2 bity) → Warstwa liniowa → Stany rekurencyjne → Warstwa liniowa → Sigmoid → Wyjście (1 bit)
```

Kluczowe elementy:
- **3 neurony ukryte** - przechowują informację o pożyczce
- **Funkcja tanh** - nieliniowość w warstwie rekurencyjnej  
- **Sigmoid na wyjściu** - daje prawdopodobieństwo że bit = 1

Wzór na aktualizację stanu:
$$S_{k+1} = \tanh(X_k \cdot W_{in} + S_k \cdot W_{rec} + b)$$

### Uczenie sieci

Używamy algorytmu BPTT (Backpropagation Through Time) - czyli propagujemy błąd wstecz przez wszystkie 15 kroków czasowych.

Optymalizator: **RMSProp + Nesterov momentum**
- Learning rate: 0.05
- Momentum: 0.80
- Minibatch: 100 próbek
- Epoki: 5

---

## 3. Wyniki

### Przebieg treningu

Strata spada z ~0.69 (losowe zgadywanie) do praktycznie 0 po około 50 iteracjach.

### Testy

Sieć działa idealnie na nowych danych:

```
Przykład 1:
x1:   101100011000000   6291
x2: - 000010010000000   1156
t:  = 101010001000000   5135  ← cel
y:  = 101010001000000         ← predykcja ✓

Przykład 2:
x1:   011011100100000   9582
x2: - 110000100000000   2147
t:  = 101010000100000   7435  ← cel
y:  = 101010000100000         ← predykcja ✓
```

**Dokładność: 100%** - sieć nauczyła się prawidłowo obsługiwać mechanizm pożyczki.

---

## 4. Co sieć musiała "zrozumieć"

Żeby poprawnie odejmować, sieć musiała nauczyć się:

1. **Kiedy jest pożyczka** - gdy odejmujemy większy bit od mniejszego
2. **Jak ją przechować** - w stanach ukrytych między krokami
3. **Jak ją uwzględnić** - w następnym kroku czasowym

To pokazuje, że nawet prosta sieć RNN z 3 neuronami może nauczyć się złożonej logiki sekwencyjnej.

---

## 5. Podsumowanie

| Co | Jak |
|----|-----|
| Zadanie | Odejmowanie binarne 15-bit |
| Architektura | RNN z 3 stanami ukrytymi |
| Trening | 5 epok, RMSProp + Nesterov |
| Wynik | 100% dokładności |

Główny wniosek: RNN są dobre do zadań gdzie trzeba "pamiętać" coś z poprzednich kroków. W przypadku odejmowania - to właśnie ta pożyczka, którą sieć nauczyła się przechowywać w swoich stanach ukrytych.

Potencjalne rozszerzenia:
- Obsługa liczb ujemnych (uzupełnienie do dwóch)
- Dłuższe liczby (32-bit, 64-bit)
- Inne operacje (mnożenie)
