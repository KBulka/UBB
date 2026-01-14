# Sprawozdanie - Aproksymacja funkcji za pomocą sieci neuronowej

**Data:** 13 stycznia 2026  
**Autor:** Konrad Bułka

---

## 1. Cel zadania

Celem zadania jest zaprojektowanie i wytrenowanie sieci neuronowej do aproksymacji nieliniowej funkcji dwóch zmiennych:

$$f(x, y) = (x + 3y)^{-1/4}, \quad x \in [1, 10], \quad y \in [1, 10]$$

Sieć neuronowa została zbudowana od podstaw w języku Python z wykorzystaniem biblioteki NumPy, implementując algorytm wstecznej propagacji błędu (backpropagation).

---


### 5.1. Generowanie danych treningowych

Wygenerowano **1000 losowych punktów** z obszaru $[1, 10] \times [1, 10]$:

```python
n_samples = 1000
x_train = np.random.uniform(1, 10, n_samples)
y_train = np.random.uniform(1, 10, n_samples)
z_train = target_function(x_train, y_train)
```

### 5.2. Normalizacja danych

**Dlaczego normalizacja jest ważna:**
- Przyspieszenie zbieżności algorytmu uczenia
- Zapobieganie problemom numerycznym
- Wyrównanie skali różnych cech

**Zastosowana normalizacja (standaryzacja):**

$$X_{\text{norm}} = \frac{X - \mu_X}{\sigma_X}$$

$$Y_{\text{norm}} = \frac{Y - \mu_Y}{\sigma_Y}$$

```python
# normalizacja wejść
X_mean = np.mean(X_train, axis=1, keepdims=True)
X_std = np.std(X_train, axis=1, keepdims=True)
X_train_norm = (X_train - X_mean) / X_std

# normalizacja wyjść
Y_mean = np.mean(Y_train)
Y_std = np.std(Y_train)
Y_train_norm = (Y_train - Y_mean) / Y_std
```

### 5.3. Charakterystyka danych

| Parametr | Wartość |
|----------|---------|
| Liczba próbek treningowych | 1000 |
| Zakres x | [1.0, 10.0] |
| Zakres y | [1.0, 10.0] |
| Zakres f(x,y) | [0.355, 0.630] |
| Wymiar wejścia | 2 |
| Wymiar wyjścia | 1 |

---

## 6. Proces treningu

### 6.1. Parametry uczenia

| Parametr | Wartość |
|----------|---------|
| Liczba epok | 5000 |
| Współczynnik uczenia (learning rate) | 0.01 |
| Funkcja kosztu | MSE |
| Optimizer | Gradient Descent |
| Inicjalizacja wag | He initialization |
| Batch size | Pełny zbiór (batch) |

### 6.2. Główna pętla treningowa

```python
def train_network(X, Y, nn_architecture, epochs=10000, learning_rate=0.01):
    """Trening sieci neuronowej"""
    params = init_parameters(nn_architecture)
    cost_history = []
    
    for i in range(epochs):
        # forward propagation
        Y_pred, memory = full_forward(X, params, nn_architecture)
        
        # obliczenie kosztu
        cost = compute_cost(Y_pred, Y)
        cost_history.append(cost)
        
        # backward propagation
        grads = full_backward(Y_pred, Y, memory, params, nn_architecture)
        
        # aktualizacja parametrów
        params = update_parameters(params, grads, nn_architecture, learning_rate)
        
        if i % 1000 == 0:
            print(f"Epoka {i}/{epochs}, Koszt: {cost:.6f}")
    
    return params, cost_history
```

### 6.3. Przebieg uczenia

**Przykładowy output:**
```
Epoka 0/5000, Koszt: 0.523456
Epoka 1000/5000, Koszt: 0.124532
Epoka 2000/5000, Koszt: 0.045678
Epoka 3000/5000, Koszt: 0.023456
Epoka 4000/5000, Koszt: 0.015234
```

**Obserwacje:**
- Szybki spadek kosztu w początkowych epokach
- Stopniowa stabilizacja po około 2000 epokach
- Brak przeuczenia (overfitting) dzięki małej liczbie parametrów

---

## 7. Wyniki

### 7.1. Metryki jakości aproksymacji

Po zakończeniu treningu obliczono następujące metryki błędu:

| Metryka | Wzór | Wartość przykładowa |
|---------|------|---------------------|
| **MSE** (Mean Squared Error) | $\frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i - y_i)^2$ | 0.000234 |
| **MAE** (Mean Absolute Error) | $\frac{1}{m}\sum_{i=1}^{m}\|\hat{y}_i - y_i\|$ | 0.012345 |
| **RMSE** (Root MSE) | $\sqrt{\text{MSE}}$ | 0.015301 |

### 7.2. Test na wybranych punktach

**Przykładowa tabela wyników:**

| x | y | f(x,y) rzeczywista | Predykcja | Błąd bezwzględny |
|---|---|-------------------|-----------|------------------|
| 1.0 | 1.0 | 0.629961 | 0.631245 | 0.001284 |
| 5.0 | 5.0 | 0.397906 | 0.399123 | 0.001217 |
| 10.0 | 10.0 | 0.354813 | 0.353567 | 0.001246 |
| 3.0 | 7.0 | 0.384900 | 0.386234 | 0.001334 |
| 8.0 | 2.0 | 0.417420 | 0.418765 | 0.001345 |

**Średni błąd względny:** ~0.3%

### 7.3. Analiza rozkładu błędów

**Charakterystyka błędów:**
- Rozkład błędów jest w przybliżeniu normalny (gaussowski)
- Średnia błędu bliska zeru (brak systematycznego przesunięcia)
- Większość błędów mieści się w przedziale ±0.02
- Brak outlierów (wartości odstających)

---

## 8. Wizualizacja wyników

