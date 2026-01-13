

## 1. Cel zadania

Przeprowadzenie badań funkcji aktywacji ReLU (Rectified Linear Unit) z użyciem języka Python. Zadanie obejmuje:
- Obliczenie gradientu funkcji
- Wizualizację funkcji wraz z gradientem na jednym wykresie
- Analizę zastosowań funkcji ReLU w praktycznych zagadnieniach

---

## 2. Definicja funkcji ReLU

### 2.1. Wzór matematyczny

Funkcja ReLU jest zdefiniowana jako:

$$f(x) = \max(0, x) = \begin{cases} 
x & \text{dla } x > 0 \\
0 & \text{dla } x \leq 0
\end{cases}$$

### 2.2. Gradient funkcji ReLU

Gradient (pochodna) funkcji ReLU:

$$f'(x) = \begin{cases} 
1 & \text{dla } x > 0 \\
0 & \text{dla } x < 0 \\
\text{nieokreślony} & \text{dla } x = 0
\end{cases}$$

W praktyce przyjmuje się, że w punkcie $x = 0$ gradient wynosi 0.

### 2.3. Implementacja w Python

```python
import numpy as np

def relu(x):
    """Funkcja ReLU: f(x) = max(0, x)"""
    return np.maximum(0, x)

def relu_gradient(x):
    """Gradient funkcji ReLU"""
    return np.where(x > 0, 1, 0)
```

---

## 3. Przeprowadzone obliczenia

### 3.1. Parametry badania

- Zakres wartości: $x \in [-5, 5]$
- Liczba punktów: 1000
- Testowe punkty: $x \in \{-3, -1, 0, 1, 3\}$

### 3.2. Wartości funkcji i gradientu w wybranych punktach

| x    | ReLU(x) | Gradient |
|------|---------|----------|
| -3.0 | 0.0     | 0.0      |
| -1.0 | 0.0     | 0.0      |
|  0.0 | 0.0     | 0.0      |
|  1.0 | 1.0     | 1.0      |
|  3.0 | 3.0     | 1.0      |

### 3.3. Zakresy wartości

- Zakres $x$: $[-5.0, 5.0]$
- Zakres $\text{ReLU}(x)$: $[0.0, 5.0]$
- Zakres gradientu: $[0.0, 1.0]$

---

## 4. Wizualizacja wyników

### 4.1. Wykres funkcji ReLU i jej gradientu

Wykres przedstawia funkcję ReLU (niebieska linia ciągła) oraz jej gradient (czerwona linia przerywana) w przedziale $x \in [-5, 5]$.

**Obserwacje z wykresu:**
- Dla $x < 0$: funkcja ReLU zwraca 0, gradient wynosi 0
- Dla $x > 0$: funkcja ReLU jest funkcją liniową $f(x) = x$, gradient wynosi 1
- W punkcie $x = 0$ występuje załamanie funkcji

### 4.2. Porównanie z innymi funkcjami aktywacji

Porównano funkcję ReLU z funkcjami sigmoid i tanh:

**Sigmoid:**
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**Tanh:**
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

---

## 5. Analiza właściwości funkcji ReLU

### 5.1. Główne cechy

1. **Prostota obliczeniowa**
   - Bardzo prosta implementacja
   - Niska złożoność obliczeniowa
   - Brak operacji wykładniczych

2. **Gradient**
   - Stały gradient = 1 dla $x > 0$
   - Gradient = 0 dla $x < 0$
   - Brak problemu zanikającego gradientu dla wartości dodatnich

3. **Nieliniowość**
   - Mimo prostoty wprowadza nieliniowość do modelu
   - Umożliwia aproksymację złożonych funkcji

4. **Sparse activation**
   - Tylko część neuronów jest aktywna (gdy $x > 0$)
   - Efektywniejsze wykorzystanie zasobów

---

## 6. Zastosowania funkcji ReLU

### 6.1. Sieci neuronowe głębokie (Deep Learning)

- **Warstwy ukryte w sieciach konwolucyjnych (CNN)**
  - Najpopularniejszy wybór dla warstw ukrytych
  - Stosowana w architekturach: ResNet, VGG, AlexNet, Inception
  
- **Przetwarzanie języka naturalnego (NLP)**
  - Warstwy w modelach typu Transformer
  - Sieci rekurencyjne (RNN, LSTM - rzadziej)

### 6.2. Computer Vision

1. **Klasyfikacja obrazów**
   - ResNet (Residual Networks)
   - VGG (Visual Geometry Group)
   - AlexNet
   - DenseNet

2. **Detekcja obiektów**
   - YOLO (You Only Look Once)
   - Faster R-CNN
   - SSD (Single Shot Detector)

3. **Segmentacja semantyczna**
   - U-Net
   - Mask R-CNN
   - DeepLab

### 6.3. Zalety w stosunku do sigmoid i tanh

| Cecha | ReLU | Sigmoid | Tanh |
|-------|------|---------|------|
| Zanikający gradient | Nie (dla x > 0) | Tak | Tak |
| Koszt obliczeniowy | Niski | Wysoki | Wysoki |
| Zakres wartości | $[0, \infty)$ | $(0, 1)$ | $(-1, 1)$ |
| Sparse activation | Tak | Nie | Nie |
| Centrowanie wokół 0 | Nie | Nie | Tak |

**Główne zalety ReLU:**
- Brak problemu zanikającego gradientu dla wartości dodatnich
- Szybsze uczenie dzięki sparse activation
- Znacznie niższy koszt obliczeniowy (brak operacji wykładniczych)
- Lepsza propagacja gradientów w głębokich sieciach

---

## 7. Ograniczenia i warianty ReLU

### 7.1. Problem "umierających neuronów" (Dying ReLU)

- Neurony otrzymujące ujemne wartości wejściowe zwracają gradient = 0
- Taki neuron może przestać się uczyć ("umrzeć")
- Problem występuje szczególnie przy dużych wartościach learning rate

### 7.2. Brak centrowania wokół zera

- Wartości wyjściowe są zawsze $\geq 0$
- Może prowadzić do wolniejszej zbieżności w niektórych przypadkach

### 7.3. Warianty ReLU rozwiązujące te problemy

1. **Leaky ReLU**
   $$f(x) = \begin{cases} x & \text{dla } x > 0 \\ \alpha x & \text{dla } x \leq 0 \end{cases}$$
   gdzie $\alpha$ jest małą stałą (np. 0.01)

2. **PReLU (Parametric ReLU)**
   - Podobna do Leaky ReLU, ale $\alpha$ jest uczonym parametrem

3. **ELU (Exponential Linear Unit)**
   $$f(x) = \begin{cases} x & \text{dla } x > 0 \\ \alpha(e^x - 1) & \text{dla } x \leq 0 \end{cases}$$

4. **GELU (Gaussian Error Linear Unit)**
   - Stosowana w nowszych architekturach (BERT, GPT)

---

## 8. Wnioski

1. **Funkcja ReLU** jest obecnie najpopularniejszą funkcją aktywacji w głębokich sieciach neuronowych ze względu na swoją **prostotę**, **efektywność obliczeniową** i skuteczność w rozwiązywaniu problemu zanikającego gradientu.

2. **Gradient funkcji ReLU** ma bardzo prostą postać:
   - 1 dla wartości dodatnich
   - 0 dla wartości ujemnych
   
   To zapobiega zanikaniu gradientu podczas propagacji wstecznej w głębokich sieciach.

3. **Zastosowania praktyczne** ReLU obejmują praktycznie wszystkie nowoczesne architektury sieci neuronowych, szczególnie w:
   - Computer Vision (CNN)
   - Klasyfikacji obrazów
   - Detekcji obiektów
   - Sieciach głębokich

4. Mimo pewnych ograniczeń (dying ReLU, brak centrowania), ReLU pozostaje **domyślnym wyborem** dla większości zastosowań dzięki swojej prostocie i skuteczności.

5. W przypadkach, gdy standardowa ReLU nie sprawdza się, można wykorzystać jej warianty (Leaky ReLU, PReLU, ELU), które rozwiązują specyficzne problemy zachowując zalety oryginalnej funkcji.

---

## 9. Bibliografia

1. Nair, V., & Hinton, G. E. (2010). Rectified linear units improve restricted boltzmann machines. *ICML*.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep residual learning for image recognition. *CVPR*.
3. Glorot, X., Bordes, A., & Bengio, Y. (2011). Deep sparse rectifier neural networks. *AISTATS*.
4. Maas, A. L., Hannun, A. Y., & Ng, A. Y. (2013). Rectifier nonlinearities improve neural network acoustic models. *ICML*.

---

**Koniec sprawozdania**
