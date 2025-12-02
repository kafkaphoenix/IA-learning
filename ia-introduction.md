# Módulo 1 - Machine Learning (Aprendizaje Automático)

Inteligencia Artificial > Machine Learning (Árboles de Decisión, Redes Neuronales) > Deep Learning (Redes Neuronales) > Modelos Generativos (Basados en técnicas de Deep Learning, relacionado con datos no estructurados como imágenes, audio y texto)

Machine learning (ML) es una rama de la inteligencia artificial que se centra en el desarrollo de algoritmos y técnicas que permiten a las computadoras aprender y mejorar su rendimiento en tareas específicas a partir de datos, sin ser explícitamente programadas para cada tarea.

## Big Data y su Relación con la IA

El Big Data se refiere a conjuntos de datos extremadamente grandes y complejos que requieren herramientas y técnicas avanzadas para su procesamiento y análisis. La relación entre Big Data e Inteligencia Artificial (IA) es fundamental, ya que la IA, especialmente el Machine Learning, depende en gran medida de grandes volúmenes de datos para entrenar modelos precisos y efectivos.

## Componentes del aprendizaje automático

1. **Datos**: La base del aprendizaje automático son los datos. Cuantos más datos de calidad se tengan, mejor podrá aprender el modelo.

2. **Algoritmos**: Son las reglas y procedimientos que el modelo utiliza para aprender de los datos. Existen diversos tipos de algoritmos, como los de clasificación, regresión y clustering.

3. **Modelos**: Un modelo es el resultado del proceso de entrenamiento del algoritmo con los datos. Es la representación matemática que puede hacer predicciones o tomar decisiones basadas en nuevos datos. Por ejemplo, árboles de decisión, máquinas de vectores de soporte y redes neuronales son tipos comunes de modelos.

4. **Entrenamiento**: Es el proceso mediante el cual el modelo aprende a partir de los datos. Durante el entrenamiento, el modelo ajusta sus parámetros para minimizar los errores en sus predicciones.

5. **Evaluación**: Después del entrenamiento, es crucial evaluar el rendimiento del modelo utilizando datos de prueba que no se hayan utilizado durante el entrenamiento. Esto ayuda a medir la precisión y la capacidad de generalización del modelo.

6. **Predicción**: Una vez que el modelo ha sido entrenado y evaluado, puede utilizarse para hacer predicciones sobre nuevos datos.

Todo es posible con la computación.

## Retos actuales

### Conjunto de datos

1. **Calidad de los Datos**: La calidad de los datos es crucial para el rendimiento del modelo. Datos incompletos, ruidosos o sesgados pueden llevar a modelos inexactos o injustos.

2. **Sobreajuste (Overfitting)**: Ocurre cuando un modelo aprende demasiado bien los datos de entrenamiento, incluyendo el ruido y las anomalías, lo que resulta en un mal rendimiento en datos nuevos.

3. **Escalabilidad / Volumen de los datos**: A medida que los conjuntos de datos crecen, es necesario desarrollar algoritmos y técnicas que puedan escalar eficientemente para manejar grandes volúmenes de datos.

4. **Ética y Sesgo**: Es fundamental abordar cuestiones éticas relacionadas con el uso de datos y garantizar que los modelos no perpetúen sesgos existentes en los datos de entrenamiento.

5. **Privacidad**: Proteger la privacidad de los datos utilizados para entrenar modelos es un desafío importante, especialmente en aplicaciones sensibles como la salud y las finanzas.

### Problemas con los modelos

1. **Black Box**: Muchos modelos de IA, especialmente los basados en deep learning, son considerados "cajas negras" debido a la dificultad para entender cómo llegan a sus decisiones.

2. **Degradación del Modelo**: Con el tiempo, los modelos pueden volverse menos precisos a medida que cambian los datos del mundo real, lo que requiere actualizaciones y reentrenamientos periódicos.

3. **Requerimientos Computacionales**: Algunos modelos de IA, especialmente los grandes modelos de deep learning, requieren recursos computacionales significativos para su entrenamiento y despliegue.

4. **Automatización y Monitorización**: Implementar sistemas automatizados para la monitorización continua del rendimiento del modelo y su actualización es un desafío técnico y operativo.

## Herramientas y Bibliotecas Populares
1. **Scikit-learn**: Biblioteca de Python para machine learning que ofrece herramientas simples y eficientes para análisis predictivo y minería de datos.
2. **TensorFlow**: Plataforma de código abierto desarrollada por Google para construir y entrenar modelos de deep learning.
3. **Keras**: Biblioteca de alto nivel para deep learning que funciona sobre TensorFlow, facilitando la creación y el entrenamiento de redes neuronales.
4. **PyTorch**: Biblioteca de deep learning desarrollada por Facebook, conocida por su flexibilidad y facilidad de uso, especialmente en investigación. Hoy en día ha reemplazado a TensorFlow como la librería más popular.
5. **XGBoost**: Implementación optimizada de gradient boosting para tareas de clasificación y regresión, ampliamente utilizada en competiciones de machine learning.

## Machine Learning

Separación de Machine Learning y Deep Learning. Ya que Deep Learning es una subárea de Machine Learning que se enfoca en el uso de redes neuronales profundas para modelar y resolver problemas complejos. Mientras que Machine Learning abarca una variedad más amplia de técnicas y algoritmos, incluyendo métodos estadísticos y de aprendizaje supervisado y no supervisado.

## Aplicaciones de Machine Learning

1. **Reconocimiento de Imágenes**: Utilizado en aplicaciones como la identificación de objetos en fotografías y videos. Modelo YOLO (You Only Look Once) es un ejemplo popular, o RetinaNet.

2. **Procesamiento del Lenguaje Natural (NLP)**: Aplicado en chatbots, traducción automática y análisis de sentimientos.

3. **Sistemas de Recomendación**: Utilizado por plataformas como Netflix y Amazon para sugerir productos o contenido basado en el comportamiento del usuario.

4. **Detección de Fraude**: Empleado en la industria financiera para identificar transacciones sospechosas.

5. **Predicción de Mantenimiento**: Utilizado en la industria manufacturera para predecir fallos en equipos y optimizar el mantenimiento preventivo.

6. **Diagnóstico Médico**: Aplicado en la detección temprana de enfermedades a partir de imágenes médicas y datos clínicos.

7. **Conducción Autónoma**: Utilizado en vehículos autónomos para interpretar el entorno y tomar decisiones de conducción.

8. **Análisis de Sentimientos**: Empleado para analizar opiniones y sentimientos en redes sociales y reseñas de productos.

9. **Optimización de Procesos**: Utilizado en diversas industrias para mejorar la eficiencia operativa y reducir costos mediante el análisis de datos y la automatización de decisiones.

10. **Juegos y Simulaciones**: Aplicado en el desarrollo de inteligencia artificial para videojuegos y simulaciones complejas.

11. **Generación de Contenido**: Utilizado para crear texto, imágenes, música y otros tipos de contenido de manera automática.

## Tipos de datos

1. **Datos Estructurados**: Datos organizados en formatos tabulares, como bases de datos relacionales. Ejemplos incluyen hojas de cálculo y tablas SQL.

2. **Datos No Estructurados**: Datos que no tienen una estructura predefinida, como texto libre, imágenes, audio y video. Ejemplos incluyen correos electrónicos, publicaciones en redes sociales y archivos multimedia.

## Tipos de Aprendizaje Automático / Machine Learning

### 1. **Aprendizaje Supervisado**
En este tipo de aprendizaje, el modelo se entrena con datos etiquetados, es decir, cada entrada tiene una salida correspondiente conocida. El objetivo es que el modelo aprenda a predecir la salida correcta para nuevas entradas. Distinguimos dos tipos principales de problemas en el aprendizaje supervisado: clasificación (cuando la salida es una categoría) y regresión (cuando la salida es un valor continuo). Por ejemplo, predecir si un correo electrónico es spam o no es un problema de clasificación, mientras que predecir el precio de una casa basado en sus características es un problema de regresión.

### 2. **Aprendizaje No Supervisado**
Aquí, el modelo se entrena con datos no etiquetados. El objetivo es que el modelo identifique patrones o estructuras ocultas en los datos, como agrupamientos o asociaciones. Distinguimos tres tipos principales de problemas en el aprendizaje no supervisado: clustering (agrupamiento de datos similares), reducción de dimensionalidad (simplificación de datos manteniendo su esencia) y reglas de asociación (descubrimiento de relaciones entre variables). Por ejemplo, el análisis de segmentos de clientes en marketing es un problema de clustering, mientras que la reducción de dimensionalidad se utiliza a menudo en la visualización de datos complejos y un ejemplo de reglas de asociación es el análisis de la cesta de la compra en supermercados (relaciones entre productos comprados juntos, para ponerlos juntos en las tiendas).

### 3. **Aprendizaje Semi-Supervisado**
Combina elementos del aprendizaje supervisado y no supervisado. En este enfoque, el modelo se entrena con una pequeña cantidad de datos etiquetados junto con una gran cantidad de datos no etiquetados. Esto es útil cuando la obtención de datos etiquetados es costosa o laboriosa, pero hay abundancia de datos no etiquetados disponibles. Por ejemplo, en el diagnóstico médico de enfermedades mentales, puede haber pocos casos etiquetados, pero muchos registros médicos no etiquetados que podemos intentar clasificar (pseudolabeling).
Y al disponer ahora de todos los datos con etiquetas, podemos volver a entrenar el modelo de forma supervisada para mejorar su precisión.

### 4. **Aprendizaje por Refuerzo**
En el aprendizaje por refuerzo, un agente aprende a tomar decisiones mediante la interacción con un entorno, recibiendo recompensas o penalizaciones según sus acciones. El objetivo es maximizar la recompensa acumulada a lo largo del tiempo. Este enfoque es común en aplicaciones como juegos y robótica.

### 5. **Deep Learning**
El deep learning es una rama del machine learning que emplea redes neuronales profundas (con tres o más capas) para resolver problemas complejos, especialmente con datos no estructurados (más eficiente) como imágenes, audio y texto (clasificar especies de pájaros por ejemplo, fotos o audios). Se denomina deep porque utiliza múltiples capas ocultas para aprender representaciones jerárquicas de los datos. Estas redes cuentan con una capa de entrada, varias capas ocultas y una capa de salida, aunque su funcionamiento interno aún no se comprende completamente.

#### Diferencias entre Machine Learning y Deep Learning

| Característica               | Machine Learning                           | Deep Learning                              |
|------------------------------|--------------------------------------------|--------------------------------------------|
| Tipo de Datos                | Funciona bien con datos estructurados      | Excelente para datos no estructurados         |
| Arquitectura                 | Utiliza algoritmos tradicionales           | Utiliza redes neuronales profundas               |
| Requerimientos Computacionales| Menores requerimientos. Entrenamientos más rápidos y puede usar una CPU                     | Altos requerimientos computacionales. Entrenamientos más lentos y generalmente requieren GPU o TPU/NPU         |
| Interpretabilidad            | Más interpretable                          | Menos interpretable (caja negra)                      |
| Cantidad de Datos            | Requiere menos datos para entrenar         | Requiere grandes volúmenes de datos          |
| Aplicaciones Comunes         | Predicción, clasificación, regresión       | Reconocimiento de imágenes, NLP (Procesamiento de Lenguaje Natural), generación de contenido |

### 6. **Modelos generativos**

#### 6.1 **Redes Bayesianas**
Modelos probabilísticos que representan relaciones entre variables mediante grafos dirigidos acíclicos. Se utilizan para inferencia y toma de decisiones bajo incertidumbre. Por ejemplo, en diagnóstico médico para evaluar la probabilidad de enfermedades basadas en síntomas.

#### 6.2 **Modelos de difusión**
Algoritmos generativos que crean datos nuevos al modelar el proceso de difusión inversa. Se utilizan en generación de imágenes, audio y texto. Un ejemplo es DALL-E, que genera imágenes a partir de descripciones textuales. Para ello usa un proceso de difusión que transforma ruido aleatorio en imágenes coherentes. También se usan en síntesis de voz y generación de música. Y para la mejora de imágenes, como la eliminación de ruido y la superresolución.

#### 6.3 **Redes Generativas Antagónicas (GANs)**
Consisten en dos redes neuronales que compiten entre sí: un generador que crea datos falsos y un discriminador que evalúa su autenticidad. Se utilizan en generación de imágenes, videos y música. Un ejemplo es la generación de rostros humanos realistas. También se aplican en mejora de imágenes, transferencia de estilo y creación de contenido multimedia.

#### 6.4 **Autoencoders Variacionales (VAEs)**
Son modelos generativos que aprenden a codificar datos en un espacio latente y luego decodificarlos para generar nuevos datos similares. Se utilizan en generación de imágenes, compresión de datos y síntesis de voz. Un ejemplo es la generación de imágenes de alta calidad a partir de representaciones latentes. También se aplican en reducción de dimensionalidad y detección de anomalías.



## Medición de resultados en Machine Learning

1. **Supervised Learning. Regresión**
   - Error Cuadrático Medio (MSE)
   - Error Absoluto Medio (MAE)
   - R² (Coeficiente de Determinación)

2. **Supervised Learning. Clasificación**
   - Exactitud (Accuracy)
   - Precisión, Recall y F1-Score
   - Matriz de Confusión
   - AUC-ROC (Área bajo la curva ROC)

3. **Unsupervised Learning**
    - Índice de Silueta
    - Inercia (para clustering)
    - Pérdida de reconstrucción (para autoencoders)

4. **Reinforcement Learning**
   - Dispersión de la política aprendida
   - Interquartil Range (IQR) de las recompensas obtenidas
   - Conditional Value at Risk (CVaR) para evaluar riesgos en entornos estocásticos

5. **Deep Learning**
   Se puede hacer cualquier otra de las subtareas, así que según el problema y el tipo de dato:
   - Métricas de regresión o clasificación
   - Modelos de lenguaje: métricas como Perplejidad (Perplexity) y BLEU Score para evaluar la coherencia y calidad del texto generado.
   - Criterio Humano

6. **Modelos Generativos**
    Según el tipo de modelo generativo y la tarea específica, se pueden utilizar diversas métricas para evaluar su rendimiento:
    - Frechet Inception Distance (FID): Mide la calidad de las imágenes generadas comparándolas con imágenes reales.
    - Inception Score (IS): Evalúa la calidad y diversidad de las imágenes generadas.
    - Modelos de lenguaje: métricas como Perplejidad (Perplexity) y BLEU Score para evaluar la coherencia y calidad del texto generado.


### Ejemplos de casos de uso

1. **Empresa de marketing digital**: Empresa de marketing quiere entrar en un nuevo mercado y para ello compra una base de datos con información demográfica y de comportamiento de usuarios en ese mercado, no etiquetados. Usa aprendizaje no supervisado (clustering) para segmentar a los usuarios en grupos con características similares y luego diseña campañas específicas para cada segmento.
Para medir el desempeño, utiliza el índice de silueta para evaluar la calidad del clustering y analiza las tasas de conversión de las campañas para cada segmento.

2. **Fábrica de automóviles**: Fábrica quiere mejorar detección de fallos en máquinas, ha ocurrido unas pocas veces en un conjunto de máquinas, pero ha tenido un gran impacto en la producción. Usa aprendizaje semi-supervisado para entrenar un modelo que detecte patrones asociados a fallos (clasificación), utilizando los pocos datos etiquetados de fallos y una gran cantidad de datos no etiquetados de funcionamiento normal.
Usando los datos etiquetados y pseudolabeling, evalúa el nuevo modelo con precisión, recall y F1-score para medir su capacidad de detectar fallos correctamente.
Impacto positivo en la reducción de tiempos de inactividad y costos de mantenimiento.

3. **Periódico online**: Periódico online quiere reducir el tiempo de revisión de artículos antes de su publicación. Su objetivo es tener una herramienta que pueda, al mismo tiempo sumar y completar texto que previamente haya escrito un periodista. Tienen una gran base de artículo previamente digitalizados. Dato no estructurado, lenguaje natural (NLP). Usa deep learning con un modelo generativo de lenguaje (como GPT) para entrenar un modelo que pueda sugerir mejoras y completar textos.
Evalúa el modelo utilizando métricas de calidad sino supervisadas como Perplejidad (Perplexity) y BLEU Score, además de realizar pruebas con periodistas para obtener feedback cualitativo sobre la utilidad de las sugerencias generadas.

## Estrategias de preparación de datos
1. **Limpieza de Datos**: Eliminar o corregir datos erróneos, duplicados o inconsistentes para mejorar la calidad del conjunto de datos.
2. **Normalización y Estandarización**: Escalar las características numéricas para que tengan una distribución similar, lo que puede mejorar el rendimiento de algunos algoritmos de machine learning.
3. **Manejo de Valores Faltantes**: Imputar o eliminar registros con valores faltantes para evitar sesgos en el modelo.
4. **Codificación de Variables Categóricas**: Convertir variables categóricas en formatos numéricos mediante técnicas como one-hot encoding o label encoding.
5. **División del Conjunto de Datos**: Separar los datos en conjuntos de entrenamiento, validación y prueba para evaluar el rendimiento del modelo de manera efectiva.