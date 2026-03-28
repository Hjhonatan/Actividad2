# Sistema Inteligente de Rutas 
Base de conocimiento con reglas lógicas para encontrar la mejor ruta entre estaciones de transporte masivo.

---

## ¿Qué hace?

El programa recibe dos estaciones (origen y destino) y devuelve la ruta más rápida aplicando tres reglas lógicas que descartan caminos inválidos o innecesarios.

```
De A hasta E → A → B → D → E (13 minutos)
De A hasta D → A → B → D     (11 minutos)
De B hasta E → B → D → E     (8 minutos)
```

---

## Requisitos

- Python 
- No requiere librerías externas

---

## Ejecución

```bash
Actividad2.py
```

---

## Estructura del código

| Sección | Qué hace |
|---|---|
| `hechos` | Conexiones entre estaciones y sus tiempos |
| `reglas` | 3 reglas lógicas que filtran rutas malas |
| `construir_grafo` | Convierte los hechos en un mapa navegable |
| `buscar_ruta` | Motor de inferencia — aplica las reglas y encuentra la ruta óptima |

---

## Reglas lógicas

- **R1** — Solo moverse si existe conexión directa entre estaciones
- **R2** — No pasar dos veces por la misma estación
- **R3** — Abandonar el camino si ya es más lento que el mejor encontrado

---

## Personalización

Para agregar estaciones editar la lista `hechos` en el archivo:

```python
hechos = [
    ("A", "B", 5),  # De A a B tarda 5 minutos
    ("B", "C", 4),  # De B a C tarda 4 minutos
    ...
]
```

Para consultar otras rutas editar la lista `consultas`:

```python
consultas = [
    ("A", "E"),
    ("C", "E"),
]
```
