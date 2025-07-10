[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Lj3YlzJp)

# Proyecto Final 2025-1: Brain Ager

## **CS2013 Programación III** · Informe Final

[Video presentación](https://drive.google.com/drive/folders/16q8kut1xyEES6GOGlfScgpX4DKD4m5Fp?usp=sharing)

### **Descripción**

Recreación del minijuego de matemáticas de Brain-Age usando redes neuronales en C++ para la identificación de dígitos.

### Contenidos

1. [Datos generales](#datos-generales)
2. [Requisitos e instalación](#requisitos-e-instalación)
3. [Investigación teórica](#1-investigación-teórica)
4. [Diseño e implementación](#2-diseño-e-implementación)
5. [Análisis del rendimiento](#3-análisis-del-rendimiento)
6. [Trabajo en equipo](#4-trabajo-en-equipo)
7. [Conclusiones](#5-conclusiones)
8. [Bibliografía](#6-bibliografía)
9. [Licencia](#licencia)

---

### Datos generales

- **Tema**: Red Neuronal intérprete de imágenes
- **Grupo**: `turinmachin`
- **Integrantes**:

  - Grayson Tejada, José Daniel – 202410372 ()
  - Lopez Del Carpio, Joaquin Adrian – 202410220 ()
  - Figueroa Winkelried, Diego Alonso – 202410533 ()
  - Valladolid Jimenes, Gonzalo Andrés – 202410186 ()
  - Walde Verano, Matías Sebastian – 202410626 ()

---

### Requisitos e instalación

1. **Compilador**: GCC 14+
2. **Dependencias**:

   - CMake 3.29+
   - SDL3
   - SDL3_ttf
   - Catch2 v3 (incluido en el repositorio, no requiere instalación)

3. **Instalación**:

```bash
git clone https://github.com/CS1103/projecto-final-turinmachin.git
cd projecto-final-turinmachin

# Descargar SDL y SDL_ttf
git clone https://github.com/libsdl-org/SDL.git vendored\SDL
git clone https://github.com/libsdl-org/SDL_ttf.git vendored\SDL_ttf
cd .\vendored\SDL_ttf
.\external\Get-GitModules.ps1

cmake -S . -B build
cd build

cmake --build .
./brain_ager
```

---

### 1. Investigación teórica

- **Objetivo**: Explorar fundamentos y arquitecturas de redes neuronales.
- **Contenido de ejemplo**:

#### 1.1 Historia y evolución de las redes neuronales

- **1943**: McCulloch y Pitts proponen el primer modelo de neurona artificial, capaz de realizar operaciones lógicas básicas.
- **1958**: Rosenblatt introduce el *Perceptrón*, una red de una sola capa capaz de clasificar datos linealmente separables.
- **1986**: Se populariza el uso del algoritmo de *backpropagation*, que permite entrenar redes con múltiples capas (MLP).
- **Década de 2010**: Surgen redes profundas (*deep learning*) con muchas capas ocultas, impulsadas por el aumento en capacidad computacional y la disponibilidad de grandes volúmenes de datos.
- **Actualidad**: Las redes neuronales están presentes en múltiples campos como visión por computadora, procesamiento de lenguaje natural, videojuegos, medicina, finanzas, entre otros.

---

#### 1.2 Principales arquitecturas

##### MLP (Perceptrón Multicapa)
- Compuesta por capas densamente conectadas.
- Cada neurona aplica una transformación lineal seguida por una función de activación.
- Útil en tareas de clasificación y regresión con datos tabulares o estructurados.

##### CNN (Convolutional Neural Network)
- Especializadas en el procesamiento de datos con estructura espacial, como imágenes.
- Usan capas convolucionales que detectan características locales (bordes, formas).
- Incorporan *pooling* para reducir dimensiones y mejorar la eficiencia.
- Ampliamente usadas en reconocimiento facial, visión autónoma y clasificación de imágenes.

##### RNN (Recurrent Neural Network)
- Diseñadas para manejar datos secuenciales (texto, audio, series temporales).
- Cada salida depende no solo del input actual, sino también de estados anteriores.
- Variantes modernas como LSTM y GRU resuelven problemas de memoria a largo plazo.
- Aplicadas en traducción automática, generación de texto y análisis de sentimientos.

---

#### 1.3 Algoritmos de entrenamiento

##### Backpropagation
- Algoritmo base para ajustar los pesos en redes neuronales.
- Calcula el gradiente del error respecto a cada peso usando la regla de la cadena.
- Permite que la red aprenda minimizando una función de pérdida mediante optimización.

##### Funciones de pérdida (loss functions)
- Miden qué tan lejos están las predicciones del valor real.
- Ejemplos:
  - **MSE (Error Cuadrático Medio)** para regresión.
  - **Cross-Entropy** para clasificación.

##### Optimizadores
- Usan los gradientes calculados para actualizar los pesos eficientemente.
- Optimizadores comunes:
  - **GD (Gradient Descent)**: Optimizador simple basado en gradiente.
  - **Adam**: adapta el learning rate individualmente para cada parámetro.
  - **RMSprop**: combina ideas de SGD y adaptación dinámica de la tasa de aprendizaje.

---

### 2. Diseño e implementación

#### 2.1 Arquitectura de la solución

- **Patrones de diseño**: Factory para ecuaciones del juego, Registry para capas.
- **Estructura de carpetas**:

  ```
  projecto-final-turinmachin
  ├── docs/
  ├── doxy.conf
  ├── CMakeLists.txt
  ├── config.h.in
  ├── default.nix
  ├── shell.nix
  ├── flake.nix
  ├── flake.lock
  ├── deps/
  │   └── catch/
  ├── include
  │   ├── common/
  │   ├── game/
  │   │   ├── math/
  │   │   └── sdl/
  │   ├── trainer/
  │   └── utec/
  │       ├── algebra/
  │       ├── nn/
  │       └── utils/
  ├── LICENSE
  ├── README.md
  ├── share/
  ├── src/
  │   ├── common/
  │   ├── game/
  │   │   ├── math/
  │   │   └── sdl/
  │   ├── trainer/
  │   └── trainer_kan/
  └── tests/
  ```

#### 2.2 Manual de uso y casos de prueba

- **Cómo ejecutar**: `./build/brain_ager`
- **Casos de prueba**:

  - Test unitario de capa Dense, al verificar que una capa Dense y una capa de activación Sigmoid produce una salida del tamaño correcto y con valores entre 0 y 1, validando la propagación hacia adelante.
  - Test de función de activación ReLU, para aproximar funciones lineales $(f(x) = 2x)$, no lineales $(f(x) = x^2)$ y periódicas $(f(x) = sin(x))$, evaluando que la salida predicha se acerca suficientemente al valor esperado.
  - Se entrena una red con múltiples épocas sobre el dataset XOR. Se valida que la función de pérdida disminuye y que los resultados se ajustan a los valores esperados ($<0.2$ para $0$ y $>0.8$ para $1$).
  - Verifica que una red neuronal sin capas devuelve exactamente el mismo tensor que recibe como entrada.

---


### 3. Análisis del rendimiento

- **Métricas de ejemplo**:

  - Iteraciones: 500 épocas.
  - Tiempo total de entrenamiento: 1m30s.
  - Precisión final: 85.42%.

- **Ventajas/Desventajas**:

  - Código ligero y dependencias mínimas.
  - Sin paralelización, rendimiento limitado.

- **Mejoras futuras**:

  - Uso de paralelización para mejor rendimiento.
  - Uso de CUDA para mejorar potencia de cómputo.
  - Mejorar implemetación de Kolmogorov Arnold Networks (KAN) para volverrlo viable.

---

### 4. Trabajo en equipo

| Tarea                     | Miembro                            | Rol                       |
| ------------------------- | ---------------------------------- | ------------------------- |
| Investigación teórica     | Lopez Del Carpio, Joaquin Adrian   | Documentar bases teóricas |
| Diseño de la arquitectura | Grayson Tejada, José Daniel        | UML y esquemas de clases  |
| Implementación del modelo | Figueroa Winkelried, Diego Alonso  | Código C++ de la NN       |
| Pruebas y benchmarking    | Walde Verano, Matías Sebastian     | Generación de métricas    |
| Documentación y demo      | Valladolid Jimenes, Gonzalo Andrés | Tutorial y video demo     |

---

### 5. Conclusiones

- **Logros**: Implementar NN desde cero, validar en dataset de ejemplo.
    - Se implementó desde cero una red neuronal en C++.
    - El sistema reconoce números escritos a mano y resuelve operaciones matemáticas, como en Brain Age.
    - Se integró la lógica de predicción con una interfaz visual basada en SDL3.
- **Evaluación**:
    - El modelo demuestra buena precisión en pruebas simples.
    - Cumple los objetivos académicos con un diseño ligero y funcional.
- **Aprendizajes**:
    - Comprensión práctica de backpropagation.
    - Uso de funciones de activación (ReLU, Sigmoid) y optimización con MSE y BSE.
- **Recomendaciones**:
    - Ampliar el entrenamiento con datasets reales para hacer pruebas completas.

---

### 6. Bibliografía

[KAN: Kolmogorov-Arnold Networks](https://arxiv.org/pdf/2404.19756?)

[KAN 2.0: Kolmogorov-Arnold Networks Meet Science](https://arxiv.org/pdf/2408.10205)
---

### Licencia

Este proyecto usa la licencia **GPL 3**. Ver [LICENSE](LICENSE) para detalles.

---
