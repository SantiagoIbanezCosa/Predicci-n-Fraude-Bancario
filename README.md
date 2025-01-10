Fraud Detection Model Comparison
Este proyecto compara varios modelos de aprendizaje automático para la detección de fraudes utilizando un conjunto de datos de transacciones bancarias. Se aplican diferentes modelos de regresión y clasificación para predecir la cantidad de transacciones y detectar actividades fraudulentas. Los modelos utilizados son:

Regresión Lineal
Regresión Logística
Bosque Aleatorio (Random Forest)
XGBoost
Descripción
Este proyecto utiliza el conjunto de datos fraudtrain.csv y fraudtest.csv para entrenar y evaluar varios modelos de Machine Learning. Los modelos son evaluados mediante varias métricas de rendimiento:

Accuracy: Proporción de predicciones correctas.
MSE (Mean Squared Error): Error cuadrático medio.
MAE (Mean Absolute Error): Error absoluto medio.
R2: Coeficiente de determinación.
Además, se visualizan los resultados de las métricas de rendimiento mediante gráficos de barras para facilitar la comparación entre los modelos.

Requisitos
Para ejecutar este proyecto, asegúrate de tener instaladas las siguientes bibliotecas de Python:

pandas
matplotlib
seaborn
scikit-learn
xgboost
Puedes instalar las dependencias necesarias utilizando pip:

bash
Copiar código
pip install pandas matplotlib seaborn scikit-learn xgboost
Archivos
fraudtrain.csv: Conjunto de datos de entrenamiento.
fraudtest.csv: Conjunto de datos de prueba.
script.py: El código principal para la carga de datos, preprocesamiento, entrenamiento y evaluación de los modelos.
Obtención de los Datos
Dado que no se pueden subir los conjuntos de datos completos debido a su tamaño, puedes obtener los archivos necesarios para este proyecto desde el siguiente enlace en Kaggle:

Fraud Detection Dataset en Kaggle
Una vez que descargues los archivos fraudtrain.csv y fraudtest.csv, colócalos en el directorio raíz del proyecto.

Uso
Clona el repositorio:

bash
Copiar código
git clone https://github.com/tu_usuario/fraud-detection.git
cd fraud-detection
Asegúrate de tener los archivos fraudtrain.csv y fraudtest.csv en el directorio raíz.

Ejecuta el script para entrenar y evaluar los modelos:

bash
Copiar código
python script.py
Los resultados se mostrarán en la consola, y los gráficos de barras comparando las métricas de rendimiento se mostrarán automáticamente.

Resultados
El proyecto generará gráficos comparativos que te permitirán evaluar el rendimiento de los modelos en función de las métricas seleccionadas (Accuracy, MSE, MAE y R2). Cada uno de los modelos se entrena y evalúa en función de los datos de entrenamiento y prueba, y se presentarán los resultados de manera clara en gráficos separados.

Ejemplo de Gráficos
Accuracy: Compara la precisión de los modelos.
MSE (Error Cuadrático Medio): Evalúa el error entre las predicciones y los valores reales.
MAE (Error Absoluto Medio): Mide la diferencia entre las predicciones y los valores reales.
R2: Mide qué tan bien se ajustan las predicciones a los datos reales.
Contribuciones
Las contribuciones son bienvenidas. Si deseas mejorar este proyecto, por favor abre un pull request o crea un issue para discutir cambios.
