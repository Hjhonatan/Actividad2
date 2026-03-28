"""
SISTEMA INTELIGENTE DE RUTAS 
"""

hechos = [
    ("A", "B", 5),
    ("B", "C", 4),
    ("A", "C", 12),   
    ("B", "D", 6),
    ("C", "D", 3),
    ("C", "E", 7),
    ("D", "E", 2),
]

def regla_conexion_existe(grafo, origen, destino):
    return destino in grafo.get(origen, {})

def regla_no_repetir(visitados, estacion):
    return estacion not in visitados

def regla_mejor_ruta(costo_actual, mejor_costo):
    return costo_actual < mejor_costo



def construir_grafo(hechos):
    grafo = {}
    for origen, destino, tiempo in hechos:
        grafo.setdefault(origen, {})[destino] = tiempo
        grafo.setdefault(destino, {})[origen] = tiempo  
    return grafo

def buscar_ruta(grafo, origen, destino):
 
    mejor = {"ruta": None, "costo": float("inf")}

    def explorar(actual, visitados, ruta, costo):
        # 
        if actual == destino:
            if costo < mejor["costo"]:
                mejor["ruta"] = list(ruta)
                mejor["costo"] = costo
            return

        for vecino, tiempo in grafo.get(actual, {}).items():
            if (regla_conexion_existe(grafo, actual, vecino)   
            and regla_no_repetir(visitados, vecino)            
            and regla_mejor_ruta(costo + tiempo, mejor["costo"])): 
                explorar(vecino, visitados | {vecino}, ruta + [vecino], costo + tiempo)

    explorar(origen, {origen}, [origen], 0)
    return mejor["ruta"], mejor["costo"]


grafo = construir_grafo(hechos)

print("=== SISTEMA DE RUTAS ===\n")
consultas = [("A", "E"), ("A", "D"), ("B", "E")]

for origen, destino in consultas:
    ruta, costo = buscar_ruta(grafo, origen, destino)
    print(f"De {origen} → {destino}")
    print(f"  Ruta:   {' → '.join(ruta)}")
    print(f"  Tiempo: {costo} minutos\n")