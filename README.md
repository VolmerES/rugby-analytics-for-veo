# Rugby Vision Analyst — MVP

Este primer MVP lee un MP4 local, detecta **personas** con YOLO11n preentrenado y les asigna IDs temporales con ByteTrack. Produce un vídeo con cajas e IDs y un JSONL por observación. Todavía no distingue jugadores, árbitros, equipos ni identidades reales. Un `TRACK_ID` puede cambiar tras una oclusión o un cambio de plano.

## Instalación en Windows

Requiere Python 3.10–3.13. En este equipo se ha usado Python 3.12. Se recomienda `uv`:

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe -e ".[dev]"
```

Para usar solo CPU, se puede instalar PyTorch siguiendo [su selector oficial](https://pytorch.org/get-started/locally/) y ejecutar con `--device cpu`. El código selecciona CUDA automáticamente cuando `torch.cuda.is_available()` es verdadero. La GPU física por sí sola no garantiza que PyTorch pueda usarla.

PyAV incluye las bibliotecas de FFmpeg necesarias para leer metadatos y decodificar; el ejecutable `ffmpeg` externo es opcional para este MVP. OpenCV escribe el vídeo de salida como MP4 `mp4v`. El primer procesamiento descarga automáticamente `yolo11n.pt` si no está presente; requiere Internet. Véase el [modelo YOLO11](https://docs.ultralytics.com/models/yolo11) y la [documentación de tracking](https://docs.ultralytics.com/modes/track).

## Primer análisis

Desde la raíz del proyecto:

```powershell
.venv\Scripts\python.exe -m rugby_vision.cli system-info
.venv\Scripts\python.exe -m rugby_vision.cli inspect-video "C:\ruta\partido.mp4"
.venv\Scripts\python.exe -m rugby_vision.cli process "C:\ruta\partido.mp4"
```

Si el entorno está activado, también puedes usar `rugby process "C:\ruta\partido.mp4"`. Para una prueba rápida antes de procesar el partido entero:

```powershell
.venv\Scripts\python.exe -m rugby_vision.cli process "C:\ruta\partido.mp4" --max-frames 300
```

`detect` exporta cajas sin IDs. `track`, `process` y `visualize` hacen tracking y exportan los mismos artefactos del MVP. `--device auto|cpu|cuda`, `--config` y `--output-dir` permiten cambiar la ejecución.

## Archivos de salida

| Archivo | Contenido |
| --- | --- |
| `data/processed/video_metadata.json` | FPS, resolución, codec, duración y otros datos del archivo fuente. |
| `data/processed/tracking_overlay.mp4` | Vídeo anotado con `TRACK_ID`, caja y confianza. No conserva el audio. |
| `data/processed/tracks.jsonl` | Una observación por persona y frame, con caja, tiempo, confianza e ID cuando ByteTrack lo asigna. |
| `data/processed/analysis_run.json` | Modelo, versiones, GPU, parámetros y rendimiento de la ejecución. |

Cada ejecución sobrescribe los archivos de su carpeta de salida. Usa `--output-dir` para conservar varias ejecuciones. El vídeo se procesa frame a frame, sin cargar el partido entero en memoria. Un frame sin personas no produce líneas JSONL. El tiempo de cada registro viene del PTS del frame cuando existe; si falta, se estima con el FPS. El overlay usa FPS constante, así que un vídeo de entrada de FPS variable puede mostrar pequeñas diferencias de tiempo.

## Configuración y estructura

`config/default.yaml` define modelo, umbral de confianza, tamaño de inferencia, tracker, directorio y codec de salida. El código está en `src/rugby_vision/`: `video/` lee metadatos y frames; `pipeline.py` hace inferencia y escribe resultados; `visualization/` dibuja las cajas. Los vídeos originales pueden guardarse en `data/raw/`, aunque se puede pasar cualquier ruta local.

## Interpretación y problemas conocidos

El detector usa la clase `person` de COCO; también puede detectar árbitros, suplentes o espectadores visibles. El número mostrado es un ID de trayectoria, **no un dorsal**. No hay reidentificación entre cortes ni recuperación garantizada tras oclusiones. Jugadores pequeños o amontonados pueden perderse. La confianza es la del detector para una caja, no una probabilidad de identidad. El modelo puede variar en calidad según zoom, iluminación y resolución del vídeo Veo. `tracking_overlay.mp4` no incluye audio; revisa el original para contrastar.

Si el MP4 no se abre, revisa `inspect-video` y el codec. Si el vídeo de salida está vacío, prueba otro codec compatible en `config/default.yaml` o abre el archivo con VLC. Para inspeccionar un tramo corto usa `--max-frames 300`.

## Pruebas y siguiente fase

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Tras probar un fragmento real, revisaremos falsos positivos, jugadores perdidos, cambios de ID y velocidad de procesamiento. Solo con esa evidencia decidiremos si ajustar el detector/tracker o introducir clasificación por equipo y reidentificación.

## Licencia

© 2026 Juan Bautista Delorme Pinedo. Bajo licencia [GNU AGPL v3](LICENSE), la misma de [Ultralytics](https://github.com/ultralytics/ultralytics) (YOLO11), de la que depende.
