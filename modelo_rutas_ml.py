import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  SISTEMA INTELIGENTE DE RUTAS SUPERVISADO")
print("=" * 60)

# -----------------------------------------------------------------------
# 1. CARGA Y EXPLORACIÓN DE DATOS
# -----------------------------------------------------------------------
print("\n[1] CARGA Y EXPLORACIÓN DE DATOS")
print("-" * 40)

df = pd.read_csv("dataset_transporte.csv")
print(f"  Filas: {df.shape[0]} | Columnas: {df.shape[1]}")
print(f"\n  Columnas: {list(df.columns)}")
print(f"\n  Primeras 5 filas:\n{df.head().to_string(index=False)}")
print(f"\n  Estadísticas descriptivas:\n{df.describe().round(2).to_string()}")
print(f"\n  Valores nulos:\n{df.isnull().sum()}")

# -----------------------------------------------------------------------
# 2. PREPROCESAMIENTO
# -----------------------------------------------------------------------
print("\n\n[2] PREPROCESAMIENTO")
print("-" * 40)

le = LabelEncoder()
df['origen_enc'] = le.fit_transform(df['origen'])
df['destino_enc'] = le.fit_transform(df['destino'])

# Codificación de hora como categorías (madrugada, mañana, tarde, noche)
def categorizar_hora(h):
    if 5 <= h <= 9:   return 0  # mañana pico
    elif 10 <= h <= 14: return 1  # mediodía
    elif 15 <= h <= 19: return 2  # tarde pico
    else:               return 3  # noche/madrugada

df['hora_cat'] = df['hora_salida'].apply(categorizar_hora)

# Features finales
features = ['origen_enc', 'destino_enc', 'hora_cat', 'dia_semana',
            'pasajeros', 'lluvia', 'temp_celsius', 'distancia_km']
target = 'tiempo_real_min'

X = df[features]
y = df[target]

print(f"  Features usadas: {features}")
print(f"  Variable objetivo: {target}")
print(f"  Rango objetivo: {y.min()} - {y.max()} minutos")

# División entrenamiento/prueba (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n  Entrenamiento: {X_train.shape[0]} muestras")
print(f"  Prueba:        {X_test.shape[0]} muestras")

# -----------------------------------------------------------------------
# 3. ENTRENAMIENTO DE MODELOS
# -----------------------------------------------------------------------
print("\n\n[3] ENTRENAMIENTO DE MODELOS")
print("-" * 40)

modelos = {
    "Regresión Lineal":   LinearRegression(),
    "Árbol de Decisión":  DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest":      RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
}

resultados = {}

for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    cv   = cross_val_score(modelo, X, y, cv=5, scoring='r2').mean()

    resultados[nombre] = {
        "modelo": modelo,
        "y_pred": y_pred,
        "MAE":    round(mae, 3),
        "RMSE":   round(rmse, 3),
        "R2":     round(r2, 3),
        "CV_R2":  round(cv, 3)
    }

    print(f"\n  [{nombre}]")
    print(f"    MAE  (error absoluto medio):   {mae:.3f} min")
    print(f"    RMSE (raíz error cuadrático):  {rmse:.3f} min")
    print(f"    R²   (coef. determinación):    {r2:.3f}")
    print(f"    R²   (validación cruzada 5-k): {cv:.3f}")

# -----------------------------------------------------------------------
# 4. COMPARACIÓN Y MEJOR MODELO
# -----------------------------------------------------------------------
print("\n\n[4] COMPARACIÓN DE MODELOS")
print("-" * 40)

df_res = pd.DataFrame({
    n: {k: v for k, v in r.items() if k != "modelo" and k != "y_pred"}
    for n, r in resultados.items()
}).T
print(df_res.to_string())

mejor = max(resultados, key=lambda n: resultados[n]["R2"])
print(f"\n  ★ Mejor modelo: {mejor}  (R² = {resultados[mejor]['R2']})")

# -----------------------------------------------------------------------
# 5. IMPORTANCIA DE FEATURES (Random Forest)
# -----------------------------------------------------------------------
rf_model = resultados["Random Forest"]["modelo"]
importancias = pd.Series(
    rf_model.feature_importances_, index=features
).sort_values(ascending=False)

print("\n\n[5] IMPORTANCIA DE VARIABLES (Random Forest)")
print("-" * 40)
for feat, imp in importancias.items():
    bar = "█" * int(imp * 40)
    print(f"  {feat:<20} {imp:.4f}  {bar}")

# -----------------------------------------------------------------------
# 6. PREDICCIONES DE EJEMPLO (sistema de rutas)
# -----------------------------------------------------------------------
print("\n\n[6] PREDICCIONES DE EJEMPLO")
print("-" * 40)

# Mapa de distancias (del grafo original)
dist_map = {
    ("A","B"): 4.0, ("A","C"): 6.5, ("A","D"): 5.5, ("A","E"): 8.5,
    ("B","C"): 3.0, ("B","D"): 4.5, ("B","E"): 6.0,
    ("C","D"): 2.0, ("C","E"): 3.0, ("D","E"): 1.5
}
# Codificación manual
enc_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

consultas_demo = [
    {"origen": "A", "destino": "E", "hora": 8,  "dia": 1, "pasajeros": 180, "lluvia": 1, "temp": 21},
    {"origen": "A", "destino": "D", "hora": 12, "dia": 3, "pasajeros": 60,  "lluvia": 0, "temp": 28},
    {"origen": "B", "destino": "E", "hora": 17, "dia": 5, "pasajeros": 190, "lluvia": 0, "temp": 24},
    {"origen": "A", "destino": "E", "hora": 7,  "dia": 7, "pasajeros": 100, "lluvia": 0, "temp": 24},
]

print(f"  {'Ruta':<10} {'Hora':<6} {'Pasajeros':<12} {'Lluvia':<8} {'Pred. (min)':<12}")
print("  " + "-" * 50)
for q in consultas_demo:
    key = (q["origen"], q["destino"])
    dist = dist_map.get(key, dist_map.get((q["destino"], q["origen"]), 5.0))
    fila = pd.DataFrame([{
        "origen_enc":  enc_map[q["origen"]],
        "destino_enc": enc_map[q["destino"]],
        "hora_cat":    categorizar_hora(q["hora"]),
        "dia_semana":  q["dia"],
        "pasajeros":   q["pasajeros"],
        "lluvia":      q["lluvia"],
        "temp_celsius":q["temp"],
        "distancia_km":dist
    }])
    pred = rf_model.predict(fila)[0]
    ruta = f"{q['origen']}→{q['destino']}"
    print(f"  {ruta:<10} {q['hora']:<6} {q['pasajeros']:<12} {'Sí' if q['lluvia'] else 'No':<8} {pred:.1f}")

# -----------------------------------------------------------------------
# 7. VISUALIZACIONES
# -----------------------------------------------------------------------
print("\n\n[7] GENERANDO VISUALIZACIONES...")

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Sistema Inteligente de Rutas – Aprendizaje Supervisado", fontsize=14, fontweight='bold')

# 7a. Comparación de métricas
nombres = list(resultados.keys())
r2_vals = [resultados[n]["R2"] for n in nombres]
colores = ['#4C72B0', '#DD8452', '#55A868']
axes[0,0].bar(nombres, r2_vals, color=colores, edgecolor='black', linewidth=0.5)
axes[0,0].set_title("Comparación R² por Modelo")
axes[0,0].set_ylabel("R²")
axes[0,0].set_ylim(0, 1.1)
for i, v in enumerate(r2_vals):
    axes[0,0].text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')

# 7b. Real vs Predicho (mejor modelo)
y_pred_best = resultados[mejor]["y_pred"]
axes[0,1].scatter(y_test, y_pred_best, alpha=0.7, color='#4C72B0', edgecolor='white', s=50)
lims = [min(y_test.min(), y_pred_best.min()) - 1, max(y_test.max(), y_pred_best.max()) + 1]
axes[0,1].plot(lims, lims, 'r--', linewidth=1.5, label='Predicción perfecta')
axes[0,1].set_xlabel("Tiempo Real (min)")
axes[0,1].set_ylabel("Tiempo Predicho (min)")
axes[0,1].set_title(f"Real vs Predicho – {mejor}")
axes[0,1].legend()

# 7c. Importancia de features
importancias.plot(kind='barh', ax=axes[1,0], color='#55A868', edgecolor='black', linewidth=0.5)
axes[1,0].set_title("Importancia de Variables (Random Forest)")
axes[1,0].set_xlabel("Importancia")
axes[1,0].invert_yaxis()

# 7d. Distribución de tiempos por hora
hora_labels = {0: 'Mañana\n(5-9h)', 1: 'Mediodía\n(10-14h)', 2: 'Tarde\n(15-19h)', 3: 'Noche\n(20-4h)'}
df['hora_label'] = df['hora_cat'].map(hora_labels)
grupos = [df[df['hora_cat'] == k]['tiempo_real_min'].values for k in [0,1,2,3]]
axes[1,1].boxplot(grupos, labels=['Mañana\n(5-9h)', 'Mediodía\n(10-14h)', 'Tarde\n(15-19h)', 'Noche\n(20-4h)'],
                  patch_artist=True)
axes[1,1].set_title("Distribución de Tiempos por Franja Horaria")
axes[1,1].set_ylabel("Tiempo (min)")

plt.tight_layout()
plt.savefig("resultados_ml.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Gráfico guardado: resultados_ml.png")

# -----------------------------------------------------------------------
# 8. RESUMEN FINAL
# -----------------------------------------------------------------------
print("\n\n[8] RESUMEN FINAL")
print("=" * 60)
print(f"  Dataset:          100 registros, {len(features)} features")
print(f"  Entrenamiento:    80 muestras  |  Prueba: 20 muestras")
print(f"  Mejor modelo:     {mejor}")
print(f"  R²:               {resultados[mejor]['R2']}")
print(f"  MAE:              {resultados[mejor]['MAE']} minutos")
print(f"  RMSE:             {resultados[mejor]['RMSE']} minutos")
print("=" * 60)
print("\n  ✓ Archivos generados:")
print("    - resultados_ml.png  (visualizaciones)")
print("\n  ✓ Ejecución completada exitosamente.")
