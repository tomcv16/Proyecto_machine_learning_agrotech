"""Etapa del cultivo de tomate a partir de UNA foto.

Uso:
    python analizar_foto.py foto.jpg
    python analizar_foto.py carpeta_con_fotos
    python analizar_foto.py foto.jpg --sin-imagen --imgsz 640
"""
import argparse
import json
import time
import unicodedata
from pathlib import Path
import re

import cv2
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent

# ---------------- CONFIGURACIÓN ----------------
MODELO = BASE / "tomato_3classes_2000images_21_4_2023.pt"
IMGSZ = 1280               # bajar a 640 si va lento (puede perder frutos pequeños)
CONF_VERDE = 0.40          # confianza mínima de frutos inmaduros
CONF_LISTO = 0.30          # confianza mínima de frutos naranjas/rojos (el filtro útil es el color)
IOU = 0.5

# Filtro de color para frutos naranjas/rojos (OpenCV HSV)
SAT_MIN = 90
VAL_MIN = 110
FRAC_COLOR = 0.25

# Reglas de etapa (frutos detectados en UNA foto) - provisionales
LISTOS_COSECHA = 3         # >= 3 frutos naranjas/rojos -> maduración y cosecha
LISTOS_INICIO = 1          # 1-2 listos solo cuentan si hay suficientes verdes
VERDES_INICIO = 3
VERDES_DESARROLLO = 6      # >= 6 verdes (sin listos) -> desarrollo de los frutos

# Detección de floración con el CNN del repo (apagado por defecto, sin validar)
USAR_CNN_FLORACION = False
MODELO_CNN = BASE / "fine_tuned_tomato_classifier.keras"
CNN_LABELS = ["Flowering", "Semi-Ripe", "Raw", "Ripe"]
CNN_UMBRAL = 0.70

# Imagen anotada
ANCHO_ANOTADA = 1600
COLORES = {"inmaduro": (255, 0, 255), "semimaduro": (0, 140, 255), "maduro": (0, 0, 230)}
# ------------------------------------------------

ETAPAS = {
    "vegetativo": "Crecimiento vegetativo",
    "floracion_formacion": "Floración y formación del fruto",
    "desarrollo": "Desarrollo de los frutos",
    "maduracion_cosecha": "Maduración y cosecha",
    "indeterminada": "Sin determinar (verificar en campo)",
}
_cnn = None


def categoria(nombre):
    n = nombre.lower()
    if "unripe" in n or "raw" in n:
        return "inmaduro"
    if "semi" in n:
        return "semimaduro"
    return "maduro"


def fraccion_color_fruto(crop):
    """Proporción de píxeles rojos/naranjas/amarillos vivos en el recorte."""
    if crop.size == 0:
        return 0.0
    h, s, v = cv2.split(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV))
    tono = (h <= 35) | (h >= 165)
    return float((tono & (s >= SAT_MIN) & (v >= VAL_MIN)).mean())


def hay_floracion(img):
    """CNN del repo sobre la foto completa. Solo se usa si no hay frutos."""
    global _cnn
    if _cnn is None:
        from tensorflow import keras
        _cnn = keras.models.load_model(str(MODELO_CNN))
    x = cv2.resize(img, (224, 224)) / 255.0
    p = _cnn.predict(x[None, ...], verbose=0)[0]
    return CNN_LABELS[int(p.argmax())] == "Flowering" and float(p.max()) >= CNN_UMBRAL


def decidir_etapa(inm, semi, mad, floracion):
    """Devuelve (código de etapa, confianza, recomendación)."""
    listos = semi + mad
    total = inm + listos

    if listos >= LISTOS_COSECHA:
        extra = " Cosechar primero los ya maduros." if mad > 0 else ""
        return ("maduracion_cosecha", "alta",
                "Hay frutos con color naranja o rojo: cosechar los que ya viran; "
                f"maduran fuera de la planta.{extra}")
    if listos >= LISTOS_INICIO and inm >= VERDES_INICIO:
        return ("maduracion_cosecha", "media",
                "Inicio de maduración: cosecha selectiva de los frutos que viran y revisar en 3 a 5 días.")
    if listos >= LISTOS_INICIO:
        return ("indeterminada", "baja",
                "Pocos frutos y alguno con color: puede ser un falso positivo. Confirmar con el historial o en campo.")
    if inm >= VERDES_DESARROLLO:
        return ("desarrollo", "alta",
                "Frutos verdes en crecimiento. Mantener riego y fertilización; aún no hay frutos para cosecha.")
    if inm >= VERDES_INICIO:
        return ("floracion_formacion", "media",
                "Pocos frutos verdes: cuajado de frutos. Mantener riego y fertilización y vigilar la polinización.")
    if total >= 1:
        return ("indeterminada", "baja",
                "Muy pocos frutos detectados: puede ser inicio de fruto o fin de ciclo. Confirmar con el historial.")
    if floracion:
        return ("floracion_formacion", "media",
                "Flores detectadas sin frutos aún. Vigilar polinización y cuajado.")
    return ("vegetativo", "media",
            "Sin frutos visibles. Mantener manejo del cultivo y monitoreo.")


def sin_tildes(t):
    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")


def dibujar(img, aceptadas, conteo, etapa_nombre):
    esc = ANCHO_ANOTADA / img.shape[1]
    vista = cv2.resize(img, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA)

    for x1, y1, x2, y2, cat in sorted(aceptadas, key=lambda t: (t[2] - t[0]) * (t[3] - t[1]), reverse=True):
        cv2.rectangle(vista, (int(x1 * esc), int(y1 * esc)), (int(x2 * esc), int(y2 * esc)),
                      COLORES[cat], 2, cv2.LINE_AA)

    filas = [(None, "Etapa: " + sin_tildes(etapa_nombre)),
             ("inmaduro", f"Inmaduro (verde): {conteo['inmaduro']}"),
             ("semimaduro", f"Semimaduro (cosechable): {conteo['semimaduro']}"),
             ("maduro", f"Maduro: {conteo['maduro']}")]
    overlay = vista.copy()
    cv2.rectangle(overlay, (8, 8), (440, 8 + 24 * len(filas) + 10), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.65, vista, 0.35, 0, vista)
    for i, (cat, txt) in enumerate(filas):
        y = 20 + i * 24
        if cat:
            cv2.rectangle(vista, (16, y), (30, y + 14), COLORES[cat], -1)
        cv2.putText(vista, txt, (38 if cat else 16, y + 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return vista


def analizar_foto(ruta, model, guardar_imagen=True, carpeta_salida=BASE / "salida", imgsz=IMGSZ):
    """Analiza UNA foto y devuelve un diccionario listo para convertir a JSON."""
    ruta = Path(ruta)
    t0 = time.time()
    img = cv2.imread(str(ruta))
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {ruta}")

    r = model.predict(img, imgsz=imgsz, conf=min(CONF_VERDE, CONF_LISTO), iou=IOU, verbose=False)[0]

    conteo = {"inmaduro": 0, "semimaduro": 0, "maduro": 0}
    aceptadas, descartadas = [], 0
    for box in r.boxes:
        x1, y1, x2, y2 = [max(int(v), 0) for v in box.xyxy[0].tolist()]
        conf = float(box.conf[0])
        cat = categoria(model.names[int(box.cls[0])])
        if cat == "inmaduro":
            if conf < CONF_VERDE:
                continue
        elif conf < CONF_LISTO or fraccion_color_fruto(img[y1:y2, x1:x2]) < FRAC_COLOR:
            descartadas += 1          # huecos de tierra, caras, mangueras, etc.
            continue
        conteo[cat] += 1
        aceptadas.append((x1, y1, x2, y2, cat))

    total = sum(conteo.values())
    floracion = USAR_CNN_FLORACION and total == 0 and hay_floracion(img)
    etapa, confianza, recomendacion = decidir_etapa(
        conteo["inmaduro"], conteo["semimaduro"], conteo["maduro"], floracion)

    imagen_anotada = None
    if guardar_imagen:
        carpeta_salida.mkdir(parents=True, exist_ok=True)
        imagen_anotada = str(carpeta_salida / f"{ruta.stem}_anotada.jpg")
        cv2.imwrite(imagen_anotada, dibujar(img, aceptadas, conteo, ETAPAS[etapa]),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

    return {
        "archivo": ruta.name,
        "etapa": etapa,
        "etapa_nombre": ETAPAS[etapa],
        "confianza": confianza,
        "requiere_revision": confianza == "baja",
        "conteo": {**conteo, "total": total, "descartadas": descartadas},
        "recomendacion": recomendacion,
        "floracion_evaluada": USAR_CNN_FLORACION,
        "imagen_anotada": imagen_anotada,
        "tiempo_s": round(time.time() - t0, 2),
    }


_FECHA = re.compile(r"(\d{4}-\d{2}-\d{2})\s+(\d{2})[;:\-](\d{2})[;:\-](\d{2})")


def clave_orden(p):
    m = _FECHA.search(p.name)
    return ("".join(m.groups()) if m else "9999", p.name)


def reunir_fotos(entradas):
    exts = {".jpg", ".jpeg", ".png"}
    fotos = []
    for e in entradas:
        p = Path(e)
        if p.is_dir():
            fotos += [x for x in p.iterdir() if x.suffix.lower() in exts]
        elif p.exists():
            fotos.append(p)
        else:
            print(f"[no existe] {p}")
    return sorted(fotos, key=clave_orden)


def main():
    ap = argparse.ArgumentParser(description="Etapa del cultivo de tomate a partir de una foto")
    ap.add_argument("entradas", nargs="+", help="foto(s) o carpeta(s)")
    ap.add_argument("--sin-imagen", action="store_true", help="no guardar la imagen anotada")
    ap.add_argument("--imgsz", type=int, default=IMGSZ, help="tamaño de entrada de YOLO")
    args = ap.parse_args()

    fotos = reunir_fotos(args.entradas)
    if not fotos:
        raise SystemExit("No se encontraron imágenes.")

    model = YOLO(str(MODELO))
    resultados = [analizar_foto(f, model, not args.sin_imagen, imgsz=args.imgsz) for f in fotos]

    if len(resultados) == 1:
        print(json.dumps(resultados[0], ensure_ascii=False, indent=2))
    else:
        for x in resultados:
            c = x["conteo"]
            print(f"{x['archivo'][:30]:30} | {x['etapa_nombre']:32} | {x['confianza']:5} | "
                  f"I:{c['inmaduro']:<3} S:{c['semimaduro']:<3} M:{c['maduro']:<3} | {x['tiempo_s']}s")


if __name__ == "__main__":
    main()