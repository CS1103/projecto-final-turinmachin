[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Lj3YlzJp)

# Proyecto Final 2025-1: Brain Ager

## **CS2013 Programación III** · Informe Final

### **Descripción**

Recreación del minijuego de matemáticas de Brain-Age usando redes neuronales en C++ para la identificación de dígitos.

### Contenidos

1. [Datos generales](#datos-generales)
2. [Requisitos e instalación](#requisitos-e-instalación)
3. [Investigación teórica](#1-investigación-teórica)
4. [Diseño e implementación](#2-diseño-e-implementación)
5. [Ejecución](#3-ejecución)
6. [Análisis del rendimiento](#4-análisis-del-rendimiento)
7. [Trabajo en equipo](#5-trabajo-en-equipo)
8. [Conclusiones](#6-conclusiones)
9. [Bibliografía](#7-bibliografía)
10. [Licencia](#licencia)

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
   git clone https://github.com/CS1103/projecto-final-turinmachin
   cd projecto-final-turinmachin
   cmake -S . -B build
   cd build
   cmake --build .
   ./brain_ager
   ```

---

### 1. Investigación teórica

- **Objetivo**: Explorar fundamentos y arquitecturas de redes neuronales.
- **Contenido de ejemplo**:

  1. Historia y evolución de las NNs.
  2. Principales arquitecturas: MLP, CNN, RNN.
  3. Algoritmos de entrenamiento: backpropagation, optimizadores.

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

### 3. Ejecución

> **Demo de ejemplo**: Video/demo alojado en `docs/demo.mp4`.
> Pasos:
>
> 1. Preparar datos de entrenamiento (formato CSV).
> 2. Ejecutar comando de entrenamiento.
> 3. Evaluar resultados con script de validación.

---

### 4. Análisis del rendimiento

- **Métricas de ejemplo**:

  - Iteraciones: 1000 épocas.
  - Tiempo total de entrenamiento: 2m30s.
  - Precisión final: 92.5%.

- **Ventajas/Desventajas**:

  - - Código ligero y dependencias mínimas.
  - – Sin paralelización, rendimiento limitado.

- **Mejoras futuras**:

  - Uso de BLAS para multiplicaciones (Justificación).
  - Paralelizar entrenamiento por lotes (Justificación).

---

### 5. Trabajo en equipo

| Tarea                     | Miembro  | Rol                       |
| ------------------------- | -------- | ------------------------- |
| Investigación teórica     | Alumno A | Documentar bases teóricas |
| Diseño de la arquitectura | Alumno B | UML y esquemas de clases  |
| Implementación del modelo | Alumno C | Código C++ de la NN       |
| Pruebas y benchmarking    | Alumno D | Generación de métricas    |
| Documentación y demo      | Alumno E | Tutorial y video demo     |

> _Actualizar con tareas y nombres reales._

---

### 6. Conclusiones

- **Logros**: Implementar NN desde cero, validar en dataset de ejemplo.
- **Evaluación**: Calidad y rendimiento adecuados para propósito académico.
- **Aprendizajes**: Profundización en backpropagation y optimización.
- **Recomendaciones**: Escalar a datasets más grandes y optimizar memoria.

---

### 7. Bibliografía

> _Actualizar con bibliografía utilizada, al menos 4 referencias bibliográficas y usando formato IEEE de referencias bibliográficas._

---

### Licencia

Este proyecto usa la licencia **MIT**. Ver [LICENSE](LICENSE) para detalles.

---
