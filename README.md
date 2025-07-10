[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Lj3YlzJp)
# Proyecto Final 2025-1: AI Neural Network
## **CS2013 Programación III** · Informe Final

### **Descripción**

Implementación de una red neuronal multicapa en C++20 utilizando tensores como estructura de datos base. El sistema es capaz de entrenar modelos simples para clasificación (por ejemplo, dígitos manuscritos) aplicando retropropagación y optimización con SGD o Adam.

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

* **Tema**: Red neuronal feedforward entrenada con descenso del gradiente.
* **Grupo**: `black arch mishatron 9000`
* **Integrantes**:

  * Mia Daniela Vargas Maycock – 202410507 (Responsable de investigación teórica)
  * Miguel Adrian Espinoza Arnero – 202320031 (Desarrollo de la arquitectura)
  * Arnaud Jean-Alain Pierre Bellicha – 209900003 (Implementación del modelo)
  * Miguel Rodriguez – 209900004 (Pruebas y benchmarking)
  * Sebastian Hernan Reategui Bellido – 202410048 (Documentación y demo)

---

### Requisitos e instalación

1. **Compilador**: GCC 11 o superior
2. **Dependencias**:
   * CMake 3.18+
3. **Instalación**:

   ```bash
   git clone https://github.com/USUARIO/proyecto-final-2025.git
   cd proyecto-final-2025
   mkdir build && cd build
   cmake ..
   make
   ```

### 1. Investigación teórica

* **Objetivo**: Explorar fundamentos y arquitecturas de redes neuronales.
* **Marco teórico**:
1. Fundamentos de las redes neuronales:
Las redes neuronales artificiales (ANNs) son modelos computacionales inspirados en el funcionamiento del cerebro humano, compuestos por capas de nodos   (neuronas artificiales) que transforman datos de entrada mediante operaciones matemáticas y funciones de activación. Cada conexión entre nodos tiene un peso ajustable, lo que permite a la red aprender relaciones complejas en los datos. Según Goodfellow et al., las ANNs son especialmente potentes para tareas de  clasificación, regresión y procesamiento de señales cuando se entrenan correctamente mediante algoritmos de optimización como backpropagation [1].

2. Estructura y funcionamiento:
Una red neuronal multicapa (MLP) está formada por una capa de entrada, una o más capas ocultas, y una capa de salida. Cada capa aplica una transformación lineal seguida por una función de activación no lineal, como ReLU o Sigmoid. El entrenamiento de una red consiste en ajustar los pesos para minimizar una función de pérdida (por ejemplo, MSE o BCE), comparando la salida predicha con los valores reales del dataset [2]. Este proceso se realiza mediante el algoritmo de retropropagación, que calcula el gradiente de la pérdida respecto a los pesos y lo propaga en sentido inverso.

3. Funciones de activación:
Las funciones de activación introducen no linealidades en el modelo, lo que permite a la red aprender representaciones complejas. Una de las más utilizadas es la ReLU (Rectified Linear Unit), que mejora la convergencia y reduce el problema del desvanecimiento del gradiente [5]. Por otro lado, la función Sigmoid es útil en tareas de clasificación binaria, ya que comprime la salida en un rango entre 0 y 1.

4. Algoritmos de optimización:
Los optimizadores son fundamentales para el entrenamiento efectivo de redes neuronales. El algoritmo de descenso por gradiente estocástico (SGD) es simple y eficiente, pero puede converger lentamente o atascarse en mínimos locales. Por ello, se han desarrollado variantes como Adam, que combina el momentum y la adaptación de la tasa de aprendizaje para mejorar la estabilidad y velocidad de convergencia [3]. La elección del optimizador puede afectar drásticamente el rendimiento del modelo [4].

5. Ingeniería de software y buenas prácticas:
En la práctica, el diseño modular del código y el uso de patrones como interfaces (ILayer, ILoss, IOptimizer) permiten construir sistemas extensibles y fáciles de depurar utilizando una arquitectura limpia y desacoplada facilita el entrenamiento, la evaluación y la experimentación con distintos modelos y configuraciones. En este proyecto, se aplicó este enfoque para permitir la adición sencilla de nuevas capas, funciones de activación y optimizadores.

---

### 2. Diseño e implementación
#### 2.1 Arquitectura de la solución
* **Patrones de diseño**:
1. Strategy – para funciones de pérdida y optimización
Ubicación:
ILoss<T,2> → MSELoss<T>, BCELoss<T>
IOptimizer<T> → SGD<T>, Adam<T>
Descripción:
El patrón Strategy permite cambiar dinámicamente el algoritmo de pérdida o de optimización sin modificar la estructura de la red.

2. Template Method – en NeuralNetwork::train()
Ubicación:
train() define el flujo general de entrenamiento: forward, cálculo de pérdida, backward, actualización.
Descripción:
El método train() define un algoritmo general (la plantilla) mientras que las subclases (ILayer, ILoss, IOptimizer) definen los pasos específicos.

3. Polimorfismo (Interface-Based Design) – como forma de Abstract Factory ligera
Ubicación:
ILayer<T> es la interfaz común para Dense, ReLU, Sigmoid
ILoss<T,2>, IOptimizer<T> también siguen esta lógica
Descripción:
Aunque no implementas una factoría como tal, el uso de punteros a interfaces (std::unique_ptr<ILayer<T>>) permite tratar distintas capas de forma homogénea y dinámica.

* **Estructura de carpetas**:

  ```
  proyecto-final/
  ├── src/
  │   ├── tensor.h              
  │   ├── nn_interfaces.h      
  │   ├── nn_dense.h            
  │   ├── nn_activation.h       
  │   ├── nn_loss.h             
  │   ├── nn_optimizer.h        
  │   ├── neural_network.h      
  │   └── main.cpp  
  ```

#### 2.2 Manual de uso y casos de prueba

* **Cómo ejecutar**: `./build/neural_net_demo`
* **Casos de prueba**:

  * Test unitario de capa densa.
  * Test de función de activación ReLU.
  * Test de entrenamiento con MSE y BCE
  * Test de optimizadores (SGD y Adam)

### 3. Ejecución

> **Demo de ejemplo**: Video/demo alojado en [link_de_drive](https://drive.google.com/file/d/1CaXJZbkH4r2Y_2o4j4QQauVI1HqoMW22/view?usp=sharing).
> Pasos:
>
> 1. Ejecutar el archivo main.cpp, el cual genera un conjunto de datos sintético (pares entrada-objetivo) para entrenar la red neuronal.
> 2. Se construye manualmente la red neuronal añadiendo capas Dense, funciones de activación (ReLU, Sigmoid) y se entrena con NeuralNetwork::train().
> 3. Se imprime por consola la pérdida final (loss) y las predicciones (predict()) para evaluar visualmente el rendimiento del modelo.

---

### 4. Análisis del rendimiento

* **Métricas de ejemplo**:
  * Iteraciones: 100 Épocas.
  * Tiempo total de entrenamiento: ~2m.
  * Pérdida inicial (loss): 124.76
  * Pérdida final (loss): 1.50
  * Tendencia del error: decreciente y estable a partir de la época 80
    
* **Ventajas/Desventajas**:
  * Código ligero y sin dependencias externas
  * Implementación modular, clara y reutilizable
  * Buen desempeño en tareas simples de regresión
  * No se realiza entrenamiento con GPU
  * No hay paralelización del entrenamiento por lotes
  * No se incluye validación automática de error ni early stopping
    
* **Mejoras futuras**:
  * Uso de bibliotecas BLAS o Eigen para acelerar el matrix_product
  * Implementar procesamiento por lotes paralelos (batch training con hilos)
  * Añadir métricas más robustas (MAE, RMSE, Accuracy, etc.) y validación cruzada

---

### 5. Trabajo en equipo

| Tarea                     | Miembro            | Rol                       |
| ------------------------- | ------------------ | ------------------------- |
| Investigación teórica     | Daniela Vargas     | Documentar bases teóricas |
| Diseño de la arquitectura | Miguel Espinoza    | UML y esquemas de clases  |
| Implementación del modelo | Arnaud Pierre      | Código C++ de la NN       |
| Pruebas y benchmarking    | Miguel Rodriguez   | Generación de métricas    |
| Documentación y demo      | Sebastian Reategui | Tutorial y video demo     |


---

### 6. Conclusiones

* **Logros**:  Se implementó desde cero una red neuronal feedforward utilizando únicamente C++20 y estructuras propias como Tensor<T, Rank>. Se logró entrenar correctamente el modelo en un dataset sintético, observando una clara reducción en la función de pérdida y resultados coherentes en las predicciones.
* **Evaluación**:  El modelo mostró un comportamiento consistente y estable en términos de entrenamiento. La arquitectura modular y orientada a interfaces permitió una buena organización del código y facilita futuras extensiones.
* **Aprendizajes**: El proyecto permitió profundizar en conceptos fundamentales como backpropagation, inicialización de pesos, funciones de activación, optimización por descenso de gradiente (SGD y Adam) y diseño orientado a objetos usando plantillas en C++.
* **Recomendaciones**: Se sugiere escalar la solución para trabajar con datasets reales como MNIST, incorporar validación automática durante el entrenamiento, y optimizar el rendimiento usando técnicas como paralelización o integración con bibliotecas numéricas (e.g., BLAS).

---

### 7. Bibliografía
[1] I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning, MIT Press, 2016. [Online]. Available: https://www.deeplearningbook.org/
[2] M. Nielsen, Neural Networks and Deep Learning, Determination Press, 2015. [Online]. Available: http://neuralnetworksanddeeplearning.com/
[3] D. P. Kingma and J. Ba, “Adam: A Method for Stochastic Optimization,” International Conference on Learning Representations (ICLR), 2015. [Online]. Available: https://arxiv.org/abs/1412.6980
[4] S. Ruder, “An Overview of Gradient Descent Optimization Algorithms,” 2016. [Online]. Available: https://arxiv.org/abs/1609.04747
[5] K. He, X. Zhang, S. Ren, and J. Sun, “Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification,” in Proc. IEEE Int. Conf. Computer Vision (ICCV), 2015, pp. 1026–1034. [Online]. Available: https://arxiv.org/abs/1502.01852

---

### Licencia

Este proyecto usa la licencia **MIT**. Ver [LICENSE](LICENSE) para detalles.

---
