# Corridas archivadas — superadas, no citar

Movidas aquí el 2026-09-09 al ordenar el espacio de trabajo. Nada se borró; esto es solo para que
no se confundan con las corridas vigentes al listar la carpeta.

**Corrección el mismo día:** `A_25.7k_seed10_limpio_hist20260805` y
`A_25.7k_seed10_limpio_PRE_modelos_backup` se movieron aquí y **se devolvieron** a
`resultados_barrido/` al descubrir que `fig14_r2_arquitecturas.py` (`PAR_MISMA_SEMILLA`) las usa
activamente como el único par de corridas con semilla idéntica y configuración constante, para
medir el ruido de inicialización puro. No están archivadas — siguen en su lugar habitual.

| Carpeta | Por qué está aquí |
|---|---|
| `B_709.7k_seed10_limpio_hist20260805` | Corrida anterior al arreglo de determinismo del 2026-08-17 (`tf.keras.utils.set_random_seed`). Hasta esa fecha `np.random.seed` + `tf.random.set_seed` no fijaba la inicialización de pesos en Keras 3 — la semilla declarada no controlaba nada. Ver `CLAUDE.md`, sección «Reproducibilidad». Verificado que ningún script la referencia por nombre. |
| `C_2.97M_seed10_limpio_hist20260805` | Ídem, arquitectura C. |
| `D_12.21M_seed10_limpio_hist20260805` | Ídem, arquitectura D. |
| `A_25.7k_seed10_limpio_CONMODELOS_SEGURO` | Copia de seguridad de una corrida de A, con modelos. Redundante con `A_25.7k_seed10_limpio`. Verificado que ningún script la referencia por nombre. |
| `D_12.21M_seed20_limpio_DIAG_1pliegue_backup` | Diagnóstico de un solo pliegue para decidir si hacía falta identificar la semilla/versión de Keras exacta del modelo publicado. Cerrado — ver `CODE/CONTEXTO.md` §6, «Diagnóstico de semilla — cerrado, no perseguir más». Verificado que ningún script la referencia por nombre. |

Antes de archivar cualquier otra carpeta de aquí, correr
`grep -rn "<nombre_de_carpeta>" CODE/*.py CODE/**/*.py` — así se detectó el error de este mismo día.

Las corridas vigentes (las que sí se citan) son las carpetas `<arch>_<params>_seed<N>_limpio[_Rz]`
sin sufijo adicional, más las dos de arriba que quedaron fuera de este archivo, en el nivel de
arriba (`resultados_barrido/`).
