"""Benchmark CPU del modelo de tomates: latencia, estres y prueba sostenida.

Uso dentro del contenedor:
    python benchmark.py latencia --imgsz 640,960,1280
    python benchmark.py estres --pasos 1,2,4 --duracion 60
    python benchmark.py sostenido --duracion 600
"""
import argparse
import csv
import gc
import json
import os
import platform
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import psutil
import torch
import ultralytics
from ultralytics import YOLO

import analizar_foto as af

NHILOS = int(os.environ.get("OMP_NUM_THREADS", "4"))
torch.set_num_threads(NHILOS)
cv2.setNumThreads(NHILOS)


# ---------------------------------------------------------------- monitor
class Monitor(threading.Thread):
    """Muestrea CPU y memoria cada 0.5 s: del proceso (psutil) y del contenedor (cgroup)."""

    def __init__(self, intervalo=0.5):
        super().__init__(daemon=True)
        self.intervalo = intervalo
        self.parar = threading.Event()
        self.muestras = []
        self.etiqueta = "inicio"
        self.proc = psutil.Process()
        self.t0 = time.time()

    @staticmethod
    def _leer(ruta):
        try:
            return Path(ruta).read_text().strip()
        except Exception:
            return None

    def _cgroup(self):
        """(cpu_seg, mem_bytes, estrangulado_seg) del contenedor, o None."""
        stat = self._leer("/sys/fs/cgroup/cpu.stat")                    # cgroup v2
        if stat:
            d = {k: int(v) for k, v in (l.split() for l in stat.splitlines())}
            mem = self._leer("/sys/fs/cgroup/memory.current")
            return d.get("usage_usec", 0) / 1e6, int(mem or 0), d.get("throttled_usec", 0) / 1e6
        uso = self._leer("/sys/fs/cgroup/cpuacct/cpuacct.usage")        # cgroup v1
        if uso:
            mem = self._leer("/sys/fs/cgroup/memory/memory.usage_in_bytes")
            thr = self._leer("/sys/fs/cgroup/cpu/cpu.stat") or ""
            t = {k: int(v) for k, v in (l.split() for l in thr.splitlines() if len(l.split()) == 2)}
            return int(uso) / 1e9, int(mem or 0), t.get("throttled_time", 0) / 1e9
        return None

    def run(self):
        self.proc.cpu_percent(None)
        prev, t_prev = self._cgroup(), time.time()
        while not self.parar.wait(self.intervalo):
            ahora = time.time()
            fila = {
                "ts": ahora,
                "t": round(ahora - self.t0, 2),
                "etiqueta": self.etiqueta,
                "cpu_proc_pct": self.proc.cpu_percent(None),      # 400 = 4 nucleos al tope
                "rss_mb": round(self.proc.memory_info().rss / 2**20, 1),
            }
            cg = self._cgroup()
            if cg and prev:
                fila["cpu_cont_pct"] = round((cg[0] - prev[0]) / (ahora - t_prev) * 100, 1)
                fila["mem_cont_mb"] = round(cg[1] / 2**20, 1)     # incluye cache de archivos
                fila["throttled_s"] = round(cg[2] - prev[2], 3)   # tiempo frenado por el limite de CPU
            prev, t_prev = cg, ahora
            self.muestras.append(fila)


def limites_contenedor():
    cpu = Monitor._leer("/sys/fs/cgroup/cpu.max")
    mem = Monitor._leer("/sys/fs/cgroup/memory.max")
    try:
        c, p = cpu.split()
        cpu_lim = "sin limite" if c == "max" else round(int(c) / int(p), 2)
    except Exception:
        cpu_lim = "desconocido"
    try:
        mem_lim = "sin limite" if mem == "max" else round(int(mem) / 2**20)
    except Exception:
        mem_lim = "desconocido"
    return cpu_lim, mem_lim


# ---------------------------------------------------------------- analisis
def cargar_modelo():
    return YOLO(str(af.MODELO))


def procesar(modelo, ruta, imgsz):
    """Replica la logica de analizar_foto.py midiendo cada etapa (en ms)."""
    t0 = time.perf_counter()
    img = cv2.imread(str(ruta))
    t1 = time.perf_counter()
    r = modelo.predict(img, imgsz=imgsz, conf=min(af.CONF_VERDE, af.CONF_LISTO),
                       iou=af.IOU, device="cpu", verbose=False)[0]
    t2 = time.perf_counter()

    inm = semi = mad = 0
    for box in r.boxes:
        x1, y1, x2, y2 = [max(int(v), 0) for v in box.xyxy[0].tolist()]
        conf = float(box.conf[0])
        cat = af.categoria(modelo.names[int(box.cls[0])])
        if cat == "inmaduro":
            if conf < af.CONF_VERDE:
                continue
            inm += 1
        else:
            if conf < af.CONF_LISTO or af.fraccion_color_fruto(img[y1:y2, x1:x2]) < af.FRAC_COLOR:
                continue
            if cat == "semimaduro":
                semi += 1
            else:
                mad += 1
    af.decidir_etapa(inm, semi, mad, False)
    t3 = time.perf_counter()

    return {
        "lectura_ms": round((t1 - t0) * 1e3, 1),
        "preproceso_ms": round(r.speed["preprocess"], 1),
        "inferencia_ms": round(r.speed["inference"], 1),
        "nms_ms": round(r.speed["postprocess"], 1),
        "filtros_ms": round((t3 - t2) * 1e3, 1),
        "total_ms": round((t3 - t0) * 1e3, 1),
        "detecciones": inm + semi + mad,
    }


def modo_latencia(fotos, args, mon):
    modelo = cargar_modelo()
    fases = {}
    for imgsz in args.imgsz:
        etiqueta = f"imgsz={imgsz}"
        mon.etiqueta = etiqueta + "_calentamiento"
        for _ in range(args.calentamiento):
            procesar(modelo, fotos[0], imgsz)
        mon.etiqueta = etiqueta
        filas, t0 = [], time.perf_counter()
        for _ in range(args.repeticiones):
            for f in fotos:
                r = procesar(modelo, f, imgsz)
                r["fase"] = etiqueta
                filas.append(r)
        fases[etiqueta] = (filas, time.perf_counter() - t0)
        print(f"  {etiqueta}: {len(filas)} fotos procesadas")
    return fases


def ejecutar_carga(n, duracion, imgsz, fotos, etiqueta, mon):
    """n trabajadores en paralelo, cada uno con su propia copia del modelo."""
    mon.etiqueta = etiqueta + "_carga_modelos"
    modelos = [cargar_modelo() for _ in range(n)]
    for m in modelos:
        procesar(m, fotos[0], imgsz)                 # calentamiento
    mon.etiqueta = etiqueta

    filas, errores = [], []
    fin = time.time() + duracion

    def trabajador(modelo, desplazamiento):
        i = desplazamiento
        while time.time() < fin:
            try:
                r = procesar(modelo, fotos[i % len(fotos)], imgsz)
                r["fase"], r["ts"] = etiqueta, time.time()
                filas.append(r)
            except Exception as e:
                errores.append(repr(e))
                time.sleep(0.1)
            i += 1

    hilos = [threading.Thread(target=trabajador, args=(m, k * 7)) for k, m in enumerate(modelos)]
    t0 = time.perf_counter()
    for h in hilos:
        h.start()
    for h in hilos:
        h.join()
    dur = time.perf_counter() - t0
    del modelos
    gc.collect()
    print(f"  {etiqueta}: {len(filas)} fotos en {dur:.0f}s, errores: {len(errores)}")
    return filas, dur, errores


# ---------------------------------------------------------------- resumenes
def estadistica(v):
    v = sorted(v)
    if not v:
        return {}

    def p(q):
        k = (len(v) - 1) * q / 100
        f, c = int(k), min(int(k) + 1, len(v) - 1)
        return v[f] + (v[c] - v[f]) * (k - f)

    return {"media": round(sum(v) / len(v), 1), "p50": round(p(50), 1),
            "p95": round(p(95), 1), "max": round(v[-1], 1)}


def resumir_fase(filas, muestras, etiqueta, dur):
    ms = [m for m in muestras if m["etiqueta"] == etiqueta]
    cpu = [m["cpu_proc_pct"] for m in ms] or [0]
    rss = [m["rss_mb"] for m in ms] or [0]
    res = {
        "fotos": len(filas),
        "duracion_s": round(dur, 1),
        "fotos_por_s": round(len(filas) / dur, 3) if dur else 0,
        "cpu_proc_pct_media": round(sum(cpu) / len(cpu)),
        "cpu_proc_pct_max": round(max(cpu)),
        "rss_mb_max": round(max(rss)),
    }
    for k in ("total_ms", "lectura_ms", "preproceso_ms", "inferencia_ms", "nms_ms", "filtros_ms"):
        res[k] = estadistica([f[k] for f in filas])
    cont = [m for m in ms if "cpu_cont_pct" in m]
    if cont:
        res["cpu_cont_pct_media"] = round(sum(m["cpu_cont_pct"] for m in cont) / len(cont))
        res["mem_cont_mb_max"] = round(max(m["mem_cont_mb"] for m in cont))
        res["estrangulado_s"] = round(sum(m["throttled_s"] for m in cont), 2)
    return res


def ventanas(filas, muestras, etiqueta, tam=60):
    ms = [m for m in muestras if m["etiqueta"] == etiqueta]
    if not filas or not ms:
        return []
    t_ini = min(m["ts"] for m in ms)
    n = int(max(f["ts"] for f in filas) - t_ini) // tam + 1
    out = []
    for w in range(n):
        a, b = t_ini + w * tam, t_ini + (w + 1) * tam
        fw = [f["total_ms"] for f in filas if a <= f["ts"] < b]
        mw = [m["rss_mb"] for m in ms if a <= m["ts"] < b]
        if fw and mw:
            out.append({"minuto": w + 1, "fotos": len(fw),
                        "total_p50_ms": estadistica(fw)["p50"],
                        "rss_mb": round(sum(mw) / len(mw))})
    return out


def guardar_csv(ruta, filas):
    if not filas:
        return
    campos = []
    for f in filas:
        for k in f:
            if k not in campos:
                campos.append(k)
    with open(ruta, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)


def grafico(muestras, ruta):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    t = [m["t"] for m in muestras]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    a1.plot(t, [m["cpu_proc_pct"] for m in muestras], color="#d32f2f")
    a1.axhline(NHILOS * 100, ls="--", color="gray")
    a1.set_ylabel("CPU del proceso (% de un nucleo)")
    a2.plot(t, [m["rss_mb"] for m in muestras], color="#1976d2")
    a2.set_ylabel("Memoria RSS (MB)")
    a2.set_xlabel("Tiempo (s)")
    prev = None
    for m in muestras:
        if m["etiqueta"] != prev:
            for a in (a1, a2):
                a.axvline(m["t"], color="black", lw=0.5, alpha=0.4)
            a1.text(m["t"], a1.get_ylim()[1] * 0.97, m["etiqueta"], rotation=90, va="top", fontsize=7)
            prev = m["etiqueta"]
    plt.tight_layout()
    plt.savefig(ruta, dpi=140)


def imprimir(resumen, args):
    print(f"\n{'fase':18}{'fotos':>6}{'foto/s':>8}{'tot p50':>9}{'tot p95':>9}{'infer p50':>10}"
          f"{'CPU med':>9}{'CPU max':>9}{'RSS max':>9}{'frenado':>9}")
    for et, r in resumen.items():
        print(f"{et:18}{r['fotos']:>6}{r['fotos_por_s']:>8}{r['total_ms'].get('p50', 0):>9}"
              f"{r['total_ms'].get('p95', 0):>9}{r['inferencia_ms'].get('p50', 0):>10}"
              f"{r['cpu_proc_pct_media']:>9}{r['cpu_proc_pct_max']:>9}{r['rss_mb_max']:>9}"
              f"{r.get('estrangulado_s', '-'):>9}")
    print("\n(ms; CPU en % de un nucleo, 400 = 4 nucleos; frenado = segundos limitados por la cuota de CPU)")

    print(f"\n=== ESTIMACION PARA RASPBERRY PI 5 (factor x{args.factor_pi}, presupuesto {args.presupuesto_s}s/foto, "
          f"RAM {args.ram_mb} MB) ===")
    for et, r in resumen.items():
        est = r["total_ms"].get("p95", 0) * args.factor_pi / 1000
        ok = "OK" if est <= args.presupuesto_s else "REVISAR"
        print(f"  {et:18} p95 PC {r['total_ms'].get('p95', 0)/1000:6.2f}s -> Pi ~{est:6.2f}s   [{ok}]")
    pico = max(r["rss_mb_max"] for r in resumen.values())
    print(f"  Memoria pico (RSS): {pico} MB de {args.ram_mb} MB   [{'OK' if pico <= args.ram_mb else 'REVISAR'}]")


def info_sistema(modelo):
    cpu_lim, mem_lim = limites_contenedor()
    return {
        "python": platform.python_version(),
        "arquitectura": platform.machine(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_disponible": torch.cuda.is_available(),
        "hilos_torch": torch.get_num_threads(),
        "nucleos_visibles": psutil.cpu_count(logical=True),
        "limite_cpu_contenedor": cpu_lim,
        "limite_ram_contenedor_mb": mem_lim,
        "modelo_mb": round(Path(af.MODELO).stat().st_size / 2**20, 1),
        "modelo_parametros_M": round(sum(p.numel() for p in modelo.model.parameters()) / 1e6, 2),
        "clases": modelo.names,
    }


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Benchmark CPU del modelo de tomates")
    ap.add_argument("modo", choices=["latencia", "estres", "sostenido"])
    ap.add_argument("--fotos", default="imagenes")
    ap.add_argument("--imgsz", default="1280", help="lista separada por comas; estres y sostenido usan solo el primero")
    ap.add_argument("--calentamiento", type=int, default=3)
    ap.add_argument("--repeticiones", type=int, default=3)
    ap.add_argument("--pasos", default="1,2,4", help="trabajadores simultaneos en la prueba de estres")
    ap.add_argument("--duracion", type=int, default=None, help="segundos por paso (estres 60, sostenido 600)")
    ap.add_argument("--factor-pi", type=float, default=3.0, help="cuantas veces mas lenta se estima la Pi 5")
    ap.add_argument("--presupuesto-s", type=float, default=10.0, help="maximo aceptable por foto en la Pi")
    ap.add_argument("--ram-mb", type=int, default=4096, help="RAM disponible para la app en la Pi")
    args = ap.parse_args()
    args.imgsz = [int(x) for x in args.imgsz.split(",")]
    dur = args.duracion or (600 if args.modo == "sostenido" else 60)

    fotos = af.reunir_fotos([args.fotos])
    if not fotos:
        raise SystemExit(f"No hay imagenes en {args.fotos}")

    m0 = cargar_modelo()
    info = info_sistema(m0)
    del m0
    print(json.dumps(info, ensure_ascii=False, indent=2, default=str))
    print(f"\n{len(fotos)} fotos | modo: {args.modo}\n")

    mon = Monitor()
    mon.start()
    fases, errores = {}, []
    if args.modo == "latencia":
        fases = modo_latencia(fotos, args, mon)
    elif args.modo == "estres":
        for n in [int(x) for x in args.pasos.split(",")]:
            f, d, e = ejecutar_carga(n, dur, args.imgsz[0], fotos, f"workers={n}", mon)
            fases[f"workers={n}"] = (f, d)
            errores += e
    else:
        f, d, e = ejecutar_carga(1, dur, args.imgsz[0], fotos, "sostenido", mon)
        fases["sostenido"] = (f, d)
        errores += e
    mon.parar.set()
    mon.join()

    resumen = {et: resumir_fase(f, mon.muestras, et, d) for et, (f, d) in fases.items()}
    salida = {"info": info, "args": vars(args), "errores": errores[:20], "fases": resumen}
    if args.modo == "sostenido":
        v = ventanas(fases["sostenido"][0], mon.muestras, "sostenido")
        salida["ventanas_por_minuto"] = v
        if len(v) >= 2:
            salida["deriva"] = {
                "latencia_pct": round((v[-1]["total_p50_ms"] / v[0]["total_p50_ms"] - 1) * 100, 1),
                "rss_mb": v[-1]["rss_mb"] - v[0]["rss_mb"],
            }

    carpeta = Path("resultados") / f"bench_{args.modo}_{datetime.now():%Y%m%d_%H%M%S}"
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / "resumen.json").write_text(json.dumps(salida, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    guardar_csv(carpeta / "muestras_recursos.csv", mon.muestras)
    guardar_csv(carpeta / "latencias.csv", [f for fl, _ in fases.values() for f in fl])
    grafico(mon.muestras, carpeta / "grafico.png")

    imprimir(resumen, args)
    if "ventanas_por_minuto" in salida:
        print("\nminuto | fotos | total p50 (ms) | RSS (MB)")
        for w in salida["ventanas_por_minuto"]:
            print(f"{w['minuto']:>6} | {w['fotos']:>5} | {w['total_p50_ms']:>14} | {w['rss_mb']:>8}")
        if "deriva" in salida:
            print(f"Deriva: latencia {salida['deriva']['latencia_pct']:+}% | memoria {salida['deriva']['rss_mb']:+} MB")
    if errores:
        print(f"\nATENCION: {len(errores)} errores. Ejemplo: {errores[0]}")
    print(f"\nResultados guardados en {carpeta}")


if __name__ == "__main__":
    main()