[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Lj3YlzJp)

# Proyecto Final 2025-1: Brain Ager

## **CS2013 Programación III** · Informe Final

### **Descripción**

Recreación del juego Brain-Ager usando redes neuronales en C++ para la identificación de dígitos.

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
   - pkg-config 0.29+
   - SDL3
   - SDL3_ttf
   - catch2_3

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

- **Patrones de diseño**: ejemplo: Factory para capas, Strategy para optimizadores.
- **Estructura de carpetas (ejemplo)**:

  ```
  proyecto-final/
  ├── src/
  │   ├── layers/
  │   ├── optimizers/
  │   └── main.cpp
  ├── tests/
  └── docs/
  ```

#### 2.2 Manual de uso y casos de prueba

- **Cómo ejecutar**: `./build/neural_net_demo input.csv output.csv`
- **Casos de prueba**:

  - Test unitario de capa densa.
  - Test de función de activación ReLU.
  - Test de convergencia en dataset de ejemplo.

> _Personalizar rutas, comandos y casos reales._

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
