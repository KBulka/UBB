## 1. Cel zadania

Celem ćwiczenia jest implementacja od podstaw rekurencyjnej sieci neuronowej (RNN) w języku Python i NumPy, ze szczególnym uwzględnieniem:

1. **Implementacji minimalnej liniowej RNN** do obliczania średniej wartości w sekwencji
2. **Analizy algorytmu wstecznej propagacji przez czas (BPTT)** - Backpropagation Through Time
3. **Badania zjawiska zanikających i eksplodujących gradientów**
4. **Optymalizacji algorytmem RProp** (Resilient Backpropagation)
5. **Wizualizacji powierzchni błędu** i trajektorii uczenia

---

## 2. Specyfikacja wariantu 11

### 2.1. Dane wejściowe

- **Liczba sekwencji**: 30
- **Długość każdej sekwencji**: 20 kroków czasowych
- **Sposób generowania**: Rozkład jednolity losowy `rand()`, zaokrąglany do jednej z wartości: **{0, 0.2, 0.4, 0.6, 0.8, 1.0}**

### 2.2. Cel (target)

Dla każdej sekwencji $X = [x_1, x_2, ..., x_{20}]$, celem jest **średnia wartość** liczb w sekwencji:

$$t = \frac{1}{T} \sum_{k=1}^{T} x_k$$

gdzie $T = 20$ (długość sekwencji).

### 2.3. Generowanie danych

```python
nb_of_samples = 30
sequence_len = 20

X = np.zeros((nb_of_samples, sequence_len))
for row_idx in range(nb_of_samples):
    random_values = np.random.rand(sequence_len)
    # Zaokrąglij do {0, 0.2, 0.4, 0.6, 0.8, 1.0}
    X[row_idx,:] = np.round(random_values * 5) / 5

# Oblicz średnie dla każdej sekwencji
t = np.mean(X, axis=1)
```

**Przykładowa sekwencja:**
```
X[0] = [0.8, 0.4, 1.0, 0.2, 0.6, 0.4, 0.8, 1.0, ...]
t[0] = 0.54 (średnia)
```

---

## 3. Podstawy teoretyczne

### 3.1. Rekurencyjna sieć neuronowa (RNN)

RNN to typ sieci neuronowej przeznaczonej do przetwarzania **danych sekwencyjnych**. Kluczową cechą RNN jest **pamięć** - stan ukryty $S_k$ w kroku czasowym $k$ zależy od stanu poprzedniego $S_{k-1}$.

**Architektura:**
```
   X₁    X₂    X₃   ...   X_T
    ↓     ↓     ↓          ↓
  ┌───┐ ┌───┐ ┌───┐     ┌───┐
  │ S₁│→│ S₂│→│ S₃│→...→│ S_T│ → y
  └───┘ └───┘ └───┘     └───┘
```

### 3.2. Liniowa RNN

W naszym przypadku RNN jest **liniowa** (bez funkcji aktywacji):

$$S_k = S_{k-1} \cdot W_{rec} + X_k \cdot W_x$$

gdzie:
- $S_k$ - stan ukryty w kroku czasowym $k$
- $S_{k-1}$ - stan ukryty z poprzedniego kroku
- $X_k$ - wejście w kroku czasowym $k$
- $W_{rec}$ - **waga rekurencyjna** (połączenie między stanami)
- $W_x$ - **waga wejściowa** (połączenie wejście → stan)

**Stan początkowy:** $S_0 = 0$

**Predykcja:** $\hat{y} = S_T$ (stan końcowy po przetworzeniu całej sekwencji)

### 3.3. Funkcja kosztu

Stosujemy **Mean Squared Error (MSE)**:

$$\mathcal{L} = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - t^{(i)})^2$$

gdzie:
- $m$ - liczba sekwencji (30)
- $\hat{y}^{(i)}$ - predykcja dla sekwencji $i$
- $t^{(i)}$ - rzeczywista średnia dla sekwencji $i$

---

## 4. Forward Propagation

### 4.1. Algorytm

Dla każdej sekwencji, rozwijamy sieć w czasie:

```
S₀ = 0
S₁ = S₀ · W_rec + X₁ · W_x
S₂ = S₁ · W_rec + X₂ · W_x
S₃ = S₂ · W_rec + X₃ · W_x
...
S_T = S_{T-1} · W_rec + X_T · W_x
```

**Predykcja końcowa:** $\hat{y} = S_T$

### 4.2. Implementacja

```python
def update_state(xk, sk, wx, wRec):
    """
    Oblicz stan k z poprzedniego stanu (sk) i bieżącego wejścia (xk).
    """
    return xk * wx + sk * wRec

def forward_states(X, wx, wRec):
    """
    Rozwiń sieć i oblicz wszystkie stany dla wszystkich sekwencji.
    """
    # S ma wymiar (nb_samples, sequence_len + 1)
    # Kolumna S[:, 0] to stan początkowy (0)
    # Kolumna S[:, -1] to stan końcowy (predykcja)
    S = np.zeros((X.shape[0], X.shape[1] + 1))
    
    for k in range(0, X.shape[1]):
        S[:, k+1] = update_state(X[:, k], S[:, k], wx, wRec)
    
    return S
```

### 4.3. Przykład dla T=3

Dla sekwencji `[0.6, 0.4, 0.8]` z wagami `W_x = 0.5`, `W_rec = 0.8`:

```
S₀ = 0
S₁ = 0 × 0.8 + 0.6 × 0.5 = 0.30
S₂ = 0.30 × 0.8 + 0.4 × 0.5 = 0.44
S₃ = 0.44 × 0.8 + 0.8 × 0.5 = 0.752
```

Predykcja: $\hat{y} = 0.752$  
Rzeczywista średnia: $(0.6 + 0.4 + 0.8)/3 = 0.6$

---

## 5. Backward Propagation Through Time (BPTT)

### 5.1. Koncepcja

BPTT to algorytm obliczania gradientów w RNN. Rozwijamy sieć w czasie i propagujemy błąd **wstecz** od końca sekwencji ($k=T$) do początku ($k=0$).

### 5.2. Wzory matematyczne

**1. Gradient dla wyjścia:**

$$\frac{\partial \mathcal{L}}{\partial \hat{y}} = \hat{y} - t$$

Dla MSE: $\frac{\partial \mathcal{L}}{\partial \hat{y}} = \frac{\partial}{\partial \hat{y}} \left[\frac{1}{2}(\hat{y} - t)^2\right] = \hat{y} - t$

**2. Gradienty w czasie (propagacja wsteczna):**

$$\frac{\partial \mathcal{L}}{\partial S_k} = \frac{\partial \mathcal{L}}{\partial S_{k+1}} \cdot W_{rec}$$

Gradient "płynie" wstecz przez wagę rekurencyjną!

**3. Gradienty parametrów (akumulacja):**

$$\frac{\partial \mathcal{L}}{\partial W_x} = \sum_{k=1}^{T} \frac{\partial \mathcal{L}}{\partial S_k} \cdot X_{k-1}$$

$$\frac{\partial \mathcal{L}}{\partial W_{rec}} = \sum_{k=1}^{T} \frac{\partial \mathcal{L}}{\partial S_k} \cdot S_{k-1}$$

### 5.3. Implementacja

```python
def output_gradient(y, t):
    """Gradient MSE względem wyjścia."""
    return 2. * (y - t)

def backward_gradient(X, S, grad_out, wRec):
    """
    Propaguj gradient wstecz przez sieć.
    Zwraca gradienty parametrów i gradienty w czasie.
    """
    # Inicjalizacja gradientów w czasie
    grad_over_time = np.zeros((X.shape[0], X.shape[1] + 1))
    grad_over_time[:, -1] = grad_out
    
    # Akumulatory gradientów parametrów
    wx_grad = 0
    wRec_grad = 0
    
    # Propagacja wsteczna przez czas
    for k in range(X.shape[1], 0, -1):
        # Akumuluj gradienty parametrów
        wx_grad += np.sum(np.mean(grad_over_time[:, k] * X[:, k-1], axis=0))
        wRec_grad += np.sum(np.mean(grad_over_time[:, k] * S[:, k-1]), axis=0)
        
        # Propaguj gradient do poprzedniego kroku
        grad_over_time[:, k-1] = grad_over_time[:, k] * wRec
    
    return (wx_grad, wRec_grad), grad_over_time
```

### 5.4. Problem zanikających/eksplodujących gradientów

Gradient propagowany wstecz jest **mnożony** przez $W_{rec}$ w każdym kroku:

$$\frac{\partial \mathcal{L}}{\partial S_0} = \frac{\partial \mathcal{L}}{\partial S_T} \cdot W_{rec}^T$$

**Zanikające gradienty** (gdy $|W_{rec}| < 1$):
- Gradient maleje wykładniczo: $W_{rec}^T$
- Trudność uczenia się długoterminowych zależności
- Początkowe kroki czasowe mają znikomy wpływ na uczenie

**Eksplodujące gradienty** (gdy $|W_{rec}| > 1$):
- Gradient rośnie wykładniczo
- Niestabilność numeryczna
- Ogromne aktualizacje wag

**Przykład dla T=20:**
- Jeśli $W_{rec} = 0.5$: gradient × $0.5^{20} \approx 10^{-6}$ (zanika!)
- Jeśli $W_{rec} = 1.5$: gradient × $1.5^{20} \approx 3325$ (eksploduje!)

---

## 6. Optymalizacja RProp

### 6.1. Motywacja

Standardowy gradient descent ma problemy z RNN:
- Wrażliwość na skalę gradientów
- Potrzeba starannego doboru learning rate
- Problemy z zanikającymi/eksplodującymi gradientami

**RProp** (Resilient Backpropagation) rozwiązuje te problemy!

### 6.2. Idea algorytmu

RProp używa **tylko znaku gradientu**, ignorując jego wielkość:

1. Każdy parametr ma swoją **własną wartość kroku** (Δ)
2. Jeśli gradient **nie zmienił znaku**: zwiększ Δ (przyspiesz)
3. Jeśli gradient **zmienił znak**: zmniejsz Δ (przekroczyliśmy minimum)
4. Aktualizuj wagi: $w := w - \text{sign}(\nabla w) \cdot \Delta$

### 6.3. Hiperparametry

- $\eta^+ = 1.2$ - współczynnik zwiększania Δ
- $\eta^- = 0.5$ - współczynnik zmniejszania Δ
- $\Delta_0 = 0.001$ - początkowa wartość kroku

### 6.4. Algorytm

```python
def update_rprop(X, t, W, W_prev_sign, W_delta, eta_p, eta_n):
    """
    Aktualizacja RProp dla jednej iteracji.
    """
    # Forward i backward pass
    S = forward_states(X, W[0], W[1])
    grad_out = output_gradient(S[:, -1], t)
    W_grads, _ = backward_gradient(X, S, grad_out, W[1])
    
    # Znak nowego gradientu
    W_sign = np.sign(W_grads)
    
    # Aktualizacja Δ dla każdego parametru
    for i in range(len(W)):
        if W_sign[i] == W_prev_sign[i]:
            # Gradient nie zmienił znaku → zwiększ krok
            W_delta[i] *= eta_p
        else:
            # Gradient zmienił znak → zmniejsz krok
            W_delta[i] *= eta_n
    
    return W_delta, W_sign
```

### 6.5. Główna pętla uczenia

```python
# Parametry początkowe
W = [-1.5, 2.0]  # [W_x, W_rec]
W_delta = [0.001, 0.001]
W_sign = [0, 0]
eta_p = 1.2
eta_n = 0.5

# 500 iteracji
for i in range(500):
    # Aktualizacja RProp
    W_delta, W_sign = update_rprop(X, t, W, W_sign, W_delta, eta_p, eta_n)
    
    # Aktualizacja wag
    for j in range(len(W)):
        W[j] -= W_sign[j] * W_delta[j]
```

