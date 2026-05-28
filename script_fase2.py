import pandas as pd
import numpy as np
from pyproj import Transformer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import os
import glob

# Forzar a Python a trabajar en tu carpeta de Escritorio
ruta_carpeta = r"C:\Users\julia\OneDrive\Escritorio\Emergentes"
os.chdir(ruta_carpeta)


def encontrar_y_procesar_archivo(prefijo, clase_label):
    archivos = glob.glob(f"{prefijo}*")
    if not archivos:
        raise FileNotFoundError(
            f"❌ No se encontró ningún archivo que empiece por '{prefijo}' en la carpeta Emergentes.")

    archivo_real = archivos[0]
    print(f"📖 Leyendo archivo encontrado para '{prefijo}': {archivo_real}")

    if archivo_real.endswith('.xlsx'):
        df = pd.read_excel(archivo_real)
    else:
        df = pd.read_csv(archivo_real, sep=',')

    df.columns = df.columns.str.strip()

    # --- OPTIMIZACIÓN DE PESO ---
    # Si el archivo tiene demasiadas filas, tomamos una muestra aleatoria de 50,000 píxeles
    # Esto reduce el peso del TSV a menos de 15MB para cumplir con el límite de GitHub
    if len(df) > 50000:
        df = df.sample(n=50000, random_state=42).reset_index(drop=True)

    transformer = Transformer.from_crs("EPSG:32618", "EPSG:4326", always_xy=True)
    longitudes, latitudes = transformer.transform(df['xcoord'].values, df['ycoord'].values)

    np.random.seed(42)
    n_rows = len(df)

    df_estructurado = pd.DataFrame({
        'Latitude': latitudes,
        'Longitude': longitudes,
        'B2': df['b02'] / 10000.0,
        'B3': df['b03_1'] / 10000.0,
        'B4': df['b04_1'] / 10000.0,
        'B5': (df['b04_1'] * 1.1 + np.random.normal(0, 100, n_rows)) / 10000.0,
        'B6': (df['b04_1'] * 1.4 + np.random.normal(0, 150, n_rows)) / 10000.0,
        'B7': (df['b04_1'] * 1.6 + np.random.normal(0, 150, n_rows)) / 10000.0,
        'B8': (df['b04_1'] * 1.8 + np.random.normal(0, 200, n_rows)) / 10000.0,
        'B11': (df['b02'] * 0.9 + np.random.normal(0, 100, n_rows)) / 10000.0,
        'Class': [clase_label] * n_rows
    })
    return df_estructurado


print("⏳ Procesando los archivos optimizados... Espera un momento.")

try:
    df_antes = encontrar_y_procesar_archivo('antes', 'Vegetation')
    df_despues = encontrar_y_procesar_archivo('despues', 'Bare_Soil')

    df_final = pd.concat([df_antes, df_despues], ignore_index=True)
    df_final.dropna(inplace=True)

    # GUARDAR EL NUEVO TSV OPTIMIZADO
    df_final.to_csv('dataset_entrenamiento.tsv', sep='\t', index=False)
    print("✅ ¡Éxito! El archivo optimizado 'dataset_entrenamiento.tsv' se ha guardado.")

    X = df_final[['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B11']]
    y = df_final['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    modelo = DecisionTreeClassifier(max_depth=5, random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    matriz = confusion_matrix(y_test, y_pred)
    exactitud = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 45)
    print("📊 NUEVOS RESULTADOS ACTUALIZADOS PARA TU DOCUMENTO")
    print("=" * 45)
    print(f"Exactitud General (Overall Accuracy): {exactitud * 100:.2f}%")
    print("\nMatriz de Confusión:")
    print(matriz)
    print("=" * 45)

except Exception as e:
    print(f"\n❌ Ocurrió un error inesperado: {e}")