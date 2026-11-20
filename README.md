# Proyecto AGROTECH: detección de madurez del tomate con machine learning

Este repositorio contiene un modelo YOLO que analiza fotos de un cultivo de tomate. En cada foto detecta los frutos, los clasifica como **inmaduro**, **semimaduro** o **maduro**, determina la etapa del cultivo y genera una recomendación de cosecha. El sistema está pensado para ejecutarse en una **Raspberry Pi 5**, que recibe las fotos enviadas por los nodos.

El repositorio tiene dos carpetas:

| Carpeta | Para qué sirve |
|---|---|
| `Modelo_machine_learning_agrotech` | Analiza las fotos una por una y genera las imágenes con los tomates etiquetados |
| `Prueba_rendimiento_modelo` | Mide con Docker el tiempo, la CPU y la memoria que consume el modelo, simulando la Raspberry Pi 5 |

---

## Modelo entrenado (obligatorio en ambas carpetas)

Las dos carpetas necesitan el modelo ya entrenado **`tomato_3classes_2000images_21_4_2023.pt`**, que está en la raíz del repositorio.

**Copia este archivo dentro de cada carpeta**, junto a `analizar_foto.py`. El script busca el modelo en su misma carpeta y, si no lo encuentra, no funciona.

- El nombre debe ser exactamente `tomato_3classes_2000images_21_4_2023.pt`. Si cambias el nombre, actualiza la variable `MODELO` al inicio de `analizar_foto.py`.
- Verifica que el archivo pese varios MB. Si pesa unos pocos KB, no se descargó el modelo real: descárgalo de nuevo con el botón **Download raw file** de GitHub.

---

## Estructura de carpetas

Además de los archivos del repositorio, debes **crear la carpeta `imagenes`** en cada carpeta y poner allí las fotos del cultivo (`.jpg`, `.jpeg` o `.png`). Las demás carpetas se crean solas al ejecutar.

```
Proyecto_machine_learning_agrotech/
│
├── Modelo_machine_learning_agrotech/
│   ├── analizar_foto.py
│   ├── requirements.txt
│   ├── tomato_3classes_2000images_21_4_2023.pt   ← copiar el modelo aquí
│   ├── imagenes/                                 ← CREAR: fotos a analizar
│   ├── .venv/                                    ← se crea al hacer el entorno virtual
│   └── salida/                                   ← se crea sola: imágenes etiquetadas
│
├── Prueba_rendimiento_modelo/
│   ├── analizar_foto.py
│   ├── benchmark.py
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── .dockerignore                             ← CREAR si no está (ver abajo)
│   ├── requirements.txt
│   ├── tomato_3classes_2000images_21_4_2023.pt   ← copiar el modelo aquí
│   ├── imagenes/                                 ← CREAR: fotos para la prueba
│   └── resultados_bench/                         ← se crea sola: resultados de las pruebas
│
├── tomato_3classes_2000images_21_4_2023.pt       ← modelo original
└── README.md
```

**Nombre recomendado de las fotos:** `ID_AAAA-MM-DD HH;MM;SS_RGB.jpg`, por ejemplo `17_2024-11-13 11;38;11_RGB.JPG`. Con este formato los resultados salen ordenados por fecha. Si el nombre no tiene fecha, la foto se analiza igual.

---

## 1. Análisis por foto (`Modelo_machine_learning_agrotech`)

### Requisitos

- Python 3.10 o superior
- Visual Studio Code (recomendado)

### Instalación

Abre una terminal dentro de la carpeta `Modelo_machine_learning_agrotech` y ejecuta:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows (PowerShell)
# source .venv/bin/activate         # Linux / Raspberry Pi

pip install -r requirements.txt
```

Si PowerShell no permite activar el entorno, ejecuta primero:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

### Uso

```powershell
# Todas las fotos de la carpeta imagenes
python analizar_foto.py imagenes

# Una sola foto (entre comillas si el nombre tiene espacios o ;)
python analizar_foto.py "imagenes\17_2024-11-13 11;38;11_RGB.JPG"
```

### Resultado

- **Una foto:** la terminal muestra un JSON con la etapa del cultivo, el conteo de frutos por clase, la recomendación y el tiempo de análisis.
- **Varias fotos:** la terminal muestra una línea por foto con la etapa y el conteo.
- **Carpeta `salida/`:** guarda cada foto con los frutos etiquetados:
  - Fucsia: inmaduro (verde)
  - Naranja: semimaduro (listo para cosechar)
  - Rojo: maduro

**Etapas que puede devolver:** Crecimiento vegetativo · Floración y formación del fruto · Desarrollo de los frutos · Maduración y cosecha · Sin determinar (verificar en campo).

---

## 2. Prueba de rendimiento con Docker (`Prueba_rendimiento_modelo`)

Esta prueba ejecuta el modelo dentro de un contenedor Docker configurado para parecerse a la Raspberry Pi 5:

- Solo CPU: la tarjeta gráfica queda desactivada.
- 4 núcleos.
- 4 GB de RAM.

No necesita entorno virtual, porque Docker instala todo dentro del contenedor.

### Instalar Docker Desktop

1. Descarga e instala **Docker Desktop** desde <https://www.docker.com/products/docker-desktop/>.
2. Durante la instalación deja marcada la opción **Use WSL 2** (Windows Subsystem for Linux). Docker la necesita en Windows porque los contenedores funcionan con Linux.
3. Si Windows pide instalar o actualizar WSL, abre PowerShell como administrador y ejecuta:
   ```powershell
   wsl --install
   ```
   Luego reinicia el equipo.
4. Abre **Docker Desktop** y espera a que indique que está en ejecución (*Engine running*). Debe estar abierto cada vez que corras las pruebas.
5. Comprueba la instalación:
   ```powershell
   docker --version
   docker compose version
   ```

> En Linux o en la Raspberry Pi no se usa Docker Desktop: se instala Docker Engine y el plugin `docker compose`.

### Archivo `.dockerignore`

Si no aparece en la carpeta, créalo con este contenido. Evita copiar a la imagen archivos innecesarios:

```
.venv
__pycache__
imagenes
resultados
resultados_bench
salida
```

### Ejecutar las pruebas

Abre una terminal dentro de `Prueba_rendimiento_modelo`, con Docker Desktop abierto:

```powershell
# 1. Construir la imagen (solo la primera vez o si cambias algún archivo)
docker compose build

# 2. Prueba de latencia: cuánto tarda una foto
docker compose run --rm bench python benchmark.py latencia --imgsz 1280 --presupuesto-s 60

# 3. Prueba de estrés: 1, 2 y 4 fotos analizadas al mismo tiempo
docker compose run --rm bench python benchmark.py estres --pasos 1,2,4 --duracion 30 --presupuesto-s 60

# 4. Prueba sostenida: 10 minutos analizando fotos sin parar
docker compose run --rm bench python benchmark.py sostenido --duracion 600 --presupuesto-s 60
```

Recomendaciones para obtener resultados confiables:

- Conecta el portátil a la corriente y cierra los programas pesados.
- Mientras corre una prueba, puedes ver el consumo en otra terminal con `docker stats`.

### Resultados

Cada prueba crea una carpeta en `resultados_bench/bench_<prueba>_<fecha>/` con estos archivos:

| Archivo | Contenido |
|---|---|
| `resumen.json` | Resumen de la prueba y datos del sistema |
| `latencias.csv` | Tiempo de cada foto por etapa (lectura, inferencia, filtros) |
| `muestras_recursos.csv` | Consumo de CPU y memoria cada 0.5 s |
| `grafico.png` | Gráfica de CPU y memoria en el tiempo |

La terminal muestra además una tabla con:

- tiempo típico (p50) y peor caso habitual (p95);
- uso de CPU (400 % = 4 núcleos);
- memoria máxima;
- una estimación del tiempo en la Raspberry Pi 5.

---

## Resultados obtenidos en mi equipo

| Prueba | Portátil (Intel i5-10300H, 4 núcleos) | Raspberry Pi 5 (estimado ×3) |
|---|---|---|
| Latencia (p95) | 1.52 s por foto | ≈ 4.6 s |
| Estrés, 1 foto a la vez | 1.15 s | ≈ 3.5 s |
| Estrés, 4 fotos a la vez | 5.90 s | ≈ 17.7 s |
| Sostenida, 10 min (649 fotos, 0 errores) | 1.10 s | ≈ 3.3 s |
| Memoria máxima (1 análisis) | 911 MB | ≈ 0.9 GB de 8 GB |

**Conclusión:** el modelo es viable para la Raspberry Pi 5. La configuración recomendada es:

- `imgsz` = 1280;
- el modelo cargado de forma permanente;
- las fotos procesadas en cola, una a la vez.


### Referencia

Modelo tomato_3classes_2000images_21_4_2023.pt tomado del siguiente repositorio: https://github.com/shubha-pandey/Tomato-Plant-Growth-Monitoring
