"""
Actividad 4 - Aprendizaje No Supervisado
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ---- cargo los datos ----
datos = pd.read_csv("dataset_transporte.csv")
print("Datos cargados correctamente")
print(datos.head())

# ---- selecciono las columnas que voy a usar ----
# en no supervisado no uso la columna objetivo (tiempo_real_min)
columnas = ['hora_salida', 'pasajeros', 'distancia_km', 'lluvia', 'temp_celsius']
X = datos[columnas]

# ---- escalo los datos porque kmeans es sensible a las unidades ----
escalador = StandardScaler()
X_escalado = escalador.fit_transform(X)
print("\nDatos escalados correctamente")

# ---- busco el numero de clusters ideal con el metodo del codo ----
print("\nCalculando numero ideal de clusters...")
inercias = []
for k in range(2, 8):
    modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
    modelo.fit(X_escalado)
    inercias.append(modelo.inertia_)
    print(f"  k={k} -> inercia: {modelo.inertia_:.1f}")

# grafico del codo
plt.figure(figsize=(6, 4))
plt.plot(range(2, 8), inercias, 'bo-')
plt.title("Metodo del Codo")
plt.xlabel("Numero de clusters")
plt.ylabel("Inercia")
plt.tight_layout()
plt.savefig("grafico_codo.png")
plt.show()
print("Grafico del codo guardado")

# ---- entreno el modelo kmeans con k=3 ----
k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
datos['cluster'] = kmeans.fit_predict(X_escalado)

# calculo silhouette para ver que tan buenos son los clusters
puntaje = silhouette_score(X_escalado, datos['cluster'])
print(f"\nPuntaje Silhouette: {puntaje:.3f}  (entre 0 y 1, mas alto es mejor)")

# ---- muestro como quedo cada cluster ----
print("\nResumen de cada cluster:")
for c in range(k):
    grupo = datos[datos['cluster'] == c]
    print(f"\n  Cluster {c} - {len(grupo)} viajes")
    print(f"    Hora promedio:      {grupo['hora_salida'].mean():.1f}")
    print(f"    Pasajeros promedio: {grupo['pasajeros'].mean():.0f}")
    print(f"    Distancia promedio: {grupo['distancia_km'].mean():.1f} km")
    print(f"    Tiempo promedio:    {grupo['tiempo_real_min'].mean():.1f} min")

# ---- uso PCA para reducir a 2 dimensiones y poder graficar ----
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_escalado)

varianza = pca.explained_variance_ratio_
print(f"\nPCA - varianza explicada: {varianza[0]*100:.1f}% y {varianza[1]*100:.1f}%")

# ---- grafico final con los clusters ----
colores = ['blue', 'orange', 'green']
etiquetas = ['Cluster 0', 'Cluster 1', 'Cluster 2']

plt.figure(figsize=(8, 5))
for c in range(k):
    puntos = X_pca[datos['cluster'] == c]
    plt.scatter(puntos[:, 0], puntos[:, 1],
                c=colores[c], label=etiquetas[c], alpha=0.7)

plt.title("Clusters de viajes - proyeccion PCA")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.legend()
plt.tight_layout()
plt.savefig("grafico_clusters.png")
plt.show()
print("Grafico de clusters guardado")

# ---- resumen final ----
print("\n=============================")
print("        RESUMEN FINAL")
print("=============================")
print(f"Total de viajes analizados: {len(datos)}")
print(f"Numero de clusters:         {k}")
print(f"Puntaje Silhouette:         {puntaje:.3f}")
print(f"Varianza explicada PCA:     {sum(varianza)*100:.1f}%")
print("Archivos generados: grafico_codo.png, grafico_clusters.png")