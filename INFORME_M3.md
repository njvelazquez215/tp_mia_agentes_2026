# Informe — Milestone 3: Evaluación sobre la sala de escape

**Materia:** MIA303 — Agentes Autónomos y Sistemas de Decisión
**Entrega:** Milestone 3

---

## Resumen

Ejecuté el agente del M1+M2 sobre los ocho escenarios de `mia_world` en seis
configuraciones distintas, con tres repeticiones cada una: **144 corridas**,
12,0 M tokens de entrada, 51 minutos y **US$ 0,77** de Bedrock con
`amazon.nova-lite-v1:0`.

En la configuración de referencia el agente resuelve **el 21 % de las corridas**
(5 de 24). El modo de fallo dominante no es equivocarse sino repetir: el 66 % de
las corridas fallidas queda atrapado llamando a la misma herramienta con los
mismos argumentos, y el 72 % de todas las corridas termina por agotar el
presupuesto de pasos, no porque el agente decida que terminó.

El hallazgo principal es que el system prompt corto supera al largo: la versión
mínima resuelve el 62 % contra el 21 % del prompt con procedimiento detallado.
La sección 4 documenta el mecanismo.

---

## 1. Aproximación

### 1.1 Alcance de los cambios al framework

`MyAgent` quedó exactamente como se entregó en el M2. No agregué un planner, no
modifiqué el bucle ni incorporé memoria de mundo o lógica específica de salas de
escape. Los tests de conformidad de M1 y M2 siguen pasando sin cambios.

El criterio fue que especializar el agente para este problema haría que los
experimentos midieran la especialización en lugar del framework. Lo único que
varía entre configuraciones es el diccionario que recibe `build_agent`:

```python
config = {
    "system_prompt": ...,          # experimento de prompting
    "max_iterations": ...,         # experimento de presupuesto de pasos
    "max_history_messages": ...,   # experimento de memoria
}
```

Las tres claves ya estaban previstas en `build_agent` desde el M2. La única
pieza que no pasa por configuración es la ablación de herramienta, que
reemplaza el callable de `look` antes de registrarlo.

### 1.2 Estructura de la infraestructura

```
eval/
  configs.py    escenarios, tarifas, prompts y las 6 configuraciones
  harness.py    una corrida -> RunRecord, con la traza completa
  metrics.py    taxonomía de fallos, pass@k y agregación
  judge.py      rúbrica evaluada por un LLM juez
  report.py     genera results/report.md
  run.py        punto de entrada
  results/      runs.jsonl (crudo), judgments.jsonl, report.md
tests/
  test_eval_harness.py   23 tests con MockLLMClient, sin API
```

Una corrida es la terna (escenario, configuración, repetición). El harness monta
un mundo nuevo, arma el agente, le registra las herramientas de
`make_world_tools(world)`, le pasa el `user_message` del escenario y registra
todo lo observable: cada llamada con sus argumentos y su salida, la respuesta
final, los tokens, el tiempo y el estado final del mundo.

### 1.3 Criterio de éxito

El veredicto lo da `check_goal(world, scenario.goal)` sobre el estado final del
mundo, nunca el texto de la respuesta. No es una formalidad: en 3 de las 106
corridas fallidas el agente cerró afirmando que había abierto la puerta cuando
el mundo indicaba lo contrario. Midiendo sobre su respuesta, esas tres contarían
como éxitos.

### 1.4 Aislamiento entre corridas

`Scenario.initial_world` es un objeto mutable y las herramientas lo modifican en
sitio. La primera versión del harness reutilizaba el escenario entre
repeticiones, de modo que la repetición 2 arrancaba con la puerta que había
abierto la repetición 1. Con ese defecto, toda corrida posterior a la primera
victoria habría contado como ganada.

Se resuelve con un `deepcopy` por corrida, cubierto por
`test_corridas_sucesivas_no_comparten_mundo`. Es un error que no se detecta
revisando los resultados, porque su único efecto visible es que mejoran.

---

## 2. Métricas

### 2.1 Métrica cuantitativa: goal rate

La métrica principal es la fracción de corridas donde `check_goal` devuelve
`True`. Elegí esto y no una medida de similitud de texto porque el problema
tiene una condición de victoria objetiva y verificable sobre el estado del
mundo. No hay que decidir si la respuesta está bien: la puerta está abierta o no
lo está.

Reporto además **pass@k**, la fracción de escenarios resueltos en al menos una de
las tres repeticiones. La brecha entre ambas métricas mide varianza: si pass@3
es mucho mayor que el goal rate, el agente puede resolver el escenario pero no
de manera fiable, que es un diagnóstico distinto de no poder resolverlo.

### 2.2 Eficiencia

```
eficiencia = llamadas óptimas / llamadas usadas
```

El numerador sale de la tabla del enunciado y coincide con las secuencias de
`test_m3_world.py`, que el test verifica que efectivamente ganan.

La calculo únicamente sobre las corridas ganadas. Gastar tres llamadas y
fracasar no es ser eficiente, y si esas corridas entraran al promedio, degradar
el agente mejoraría la métrica.

### 2.3 Métrica cualitativa: rúbrica con LLM juez

El goal rate indica si el agente ganó, no cómo jugó. Dos corridas con el mismo
veredicto binario pueden ser muy distintas: una que dedujo dónde estaba la llave
y fue directo, y otra que abrió los veinte cajones hasta encontrarla por
descarte. También ocurre lo inverso: una corrida perdida puede haber razonado
bien y quedarse a un paso.

Un juez (`us.amazon.nova-pro-v1:0`) puntúa cuatro dimensiones de 1 a 5:
exploración sistemática, uso de memoria, fidelidad del reporte y recuperación de
errores. Tres decisiones de diseño:

1. **El juez es un modelo distinto del actor.** `nova-pro` evalúa a `nova-lite`.
   Un modelo que se evalúa a sí mismo tiende a preferir su propia salida;
   separarlos elimina ese sesgo y el sobrecosto es marginal.
2. **El juez no recibe el veredicto de la meta.** Solo la traza y la respuesta
   final. Si conociera el desenlace puntuaría el proceso hacia arriba por el
   solo hecho de que terminó bien.
3. **La salida se valida con `structured_call`** del M2 contra un modelo
   Pydantic, lo que garantiza enteros en rango y ejercita el framework propio
   contra un proveedor real.

### 2.4 Validación del juez

Como el juez no accede al desenlace, la correlación con el resultado sirve como
control de validez:

| | exploración | memoria | fidelidad | recuperación |
|---|---:|---:|---:|---:|
| corridas ganadas (n=38) | 4,55 | 4,39 | 4,82 | 4,50 |
| corridas perdidas (n=106) | 2,95 | 2,48 | 2,78 | 2,47 |

La separación es consistente en las cuatro dimensiones. Si la rúbrica fuera
ruido, ambas filas serían equivalentes.

---

## 3. Resultados

### 3.1 Configuración de referencia

| | valor |
|---|---:|
| corridas | 24 (8 escenarios x 3 repeticiones) |
| goal rate | **21 %** (5/24) |
| pass@3 | 38 % (3 de 8 escenarios) |
| eficiencia media (ganadas) | 0,55 |
| llamadas con error | 34 % |

### 3.2 Desglose por escenario

| escenario | dif. | óptimo | tope | éxitos | calls | efic. | err/call |
|---|---|---:|---:|---:|---:|---:|---:|
| `study-with-key` | easy | 3 | 7 | 3/3 | 5,3 | 0,57 | 6 % |
| `apartment-keys` | medium | 7 | 14 | 1/3 | 13,3 | 0,58 | 25 % |
| `color-locks` | medium | 11 | 22 | 0/3 | 22,0 | — | 12 % |
| `library-search` | hard | 7 | 17 | 0/3 | 17,0 | — | 45 % |
| `office-sequence` | hard | 13 | 26 | 1/3 | 26,7 | 0,46 | 12 % |
| `extreme-archive` | extreme | 4 | 27 | 0/3 | 45,7 | — | 74 % |
| `backtracking-vault` | extreme | 18 | 36 | 0/3 | 36,0 | — | 18 % |
| `vault-combination` | extreme | 21 | 42 | 0/3 | 42,7 | — | 30 % |

Las llamadas pueden superar el tope porque `max_iterations` acota las llamadas
al LLM, no a las herramientas: Nova a veces emite varios `tool_calls` en una
misma respuesta.

### 3.3 Desglose por dificultad

| dificultad | éxitos | goal rate | pass@3 |
|---|---:|---:|---:|
| easy | 3/3 | 100 % | 100 % |
| medium | 1/6 | 17 % | 50 % |
| hard | 1/6 | 17 % | 50 % |
| extreme | 0/9 | 0 % | 0 % |

El rendimiento no se degrada gradualmente: cae de golpe apenas el escenario deja
de ser trivial.

`office-sequence` es un caso a destacar. Es `hard`, tiene meta compuesta y
ordenada (hay que tener el documento antes de abrir la puerta) y aun así se
resolvió una vez de tres. `color-locks`, que es `medium`, no se resolvió nunca.
La dificultad declarada del escenario no predice bien el rendimiento del agente;
lo que sí lo predice es la cantidad de llamadas encadenadas que exige sin margen
de error.

### 3.4 Análisis de errores

Clasifico cada corrida en dos ejes independientes. Colapsarlos en uno pierde
información: una corrida puede agotar el presupuesto y además haber acumulado
errores de identificador, y cada cosa sugiere una corrección distinta.

**Terminación (las 144 corridas)**

| terminación | n | % |
|---|---:|---:|
| presupuesto agotado | 104 | 72 % |
| éxito | 38 | 26 % |
| rendición temprana | 2 | 1 % |
| crash | 0 | 0 % |

El agente casi nunca se detiene por decisión propia: sigue actuando hasta que se
lo corta. Ninguna excepción escapó del agente en 144 corridas, lo que corresponde
al manejo de errores implementado en el M2.

**Fricción dominante (las 106 corridas fallidas)**

| fricción | n | % |
|---|---:|---:|
| repetición | 70 | 66 % |
| id inexistente | 23 | 22 % |
| ninguna | 7 | 7 % |
| inventario vacío | 3 | 3 % |
| objeto no visible | 2 | 2 % |
| argumentos inválidos | 1 | 1 % |

Dos categorías explican el 88 % de los fallos, y ambas responden al mismo
problema de fondo: el agente no utiliza la información que el mundo ya le
entregó.

**Repetición.** Invoca tres o más veces la misma herramienta con los mismos
argumentos. En `color-locks` examinó `cofre_rojo` cinco veces seguidas; el mundo
respondía que estaba cerrado con llave y, en lugar de buscar la llave, volvía a
examinarlo.

**Identificadores inexistentes.** Inventa identificadores plausibles en lugar de
leerlos del mundo. El caso más claro es `color-locks`, donde detectó el patrón
`llave_X` → `cofre_X` y extrapoló nombres de contenedores inexistentes:

```
take(llave_plata)  use(llave_plata → puerta_principal)   ← llave correcta, blanco equivocado
examine(cofre_plateado)   ← no existe
examine(cofre_gris)       ← no existe
examine(cofre_azul)       ← no existe
examine(cofre_plata)      ← este sí
```

Seis llamadas dedicadas a adivinar, con `look` disponible y sin costo. En
`backtracking-vault` solicitó `take llave_de_la_reja` nueve veces (el
identificador real es `llave_boveda`), ignorando nueve mensajes de error
consecutivos que lo indicaban.

Esto explica el 74 % de llamadas con error de `extreme-archive`: veinte
expedientes con nombres largos son veinte oportunidades de inventar un id.

### 3.5 Discrepancia entre la respuesta y el estado del mundo

En 3 de las 106 corridas fallidas el agente cerró afirmando haber abierto la
puerta con la puerta cerrada. Es una proporción baja, pero no nula, y justifica
empíricamente la decisión de medir sobre el estado del mundo.

### 3.6 Coste

| | |
|---|---:|
| tokens | 12,0 M entrada / 198 K salida |
| coste del barrido | US$ 0,77 |
| tiempo | 51 min |

La entrada domina por dos órdenes de magnitud sobre la salida porque el bucle
reenvía el prompt completo en cada iteración, de modo que el consumo crece con
el cuadrado de los pasos. Un agente que entra en bucle no solo no avanza: además
incrementa el costo.

---

## 4. Experimentos

Cinco configuraciones contra la referencia. Cada una modifica una sola variable.

| configuración | goal rate | pass@3 | efic. | err/call | qué cambia |
|---|---:|---:|---:|---:|---|
| `baseline` | 21 % | 38 % | 0,55 | 34 % | — |
| `steps-justo` | **0 %** | 0 % | — | 16 % | tope = solución óptima |
| `steps-holgado` | 54 % | 88 % | 0,48 | 19 % | tope = 2x el generoso |
| `memoria-corta` | 21 % | 38 % | 0,53 | 22 % | ventana 50 → 12 mensajes |
| `prompt-minimo` | **62 %** | 88 % | 0,51 | 24 % | prompt sin procedimiento |
| `look-noop` | **0 %** | 0 % | — | 33 % | `look` devuelve texto vacío |

Rúbrica del juez sobre las mismas corridas:

| configuración | goal | exploración | memoria | fidelidad | recuperación |
|---|---:|---:|---:|---:|---:|
| `baseline` | 21 % | 3,42 | 2,71 | 2,88 | 2,21 |
| `steps-justo` | 0 % | **4,21** | 3,58 | 3,83 | **3,96** |
| `steps-holgado` | 54 % | 3,67 | 3,29 | 3,50 | 3,42 |
| `memoria-corta` | 21 % | 3,21 | 3,04 | 3,00 | 2,54 |
| `prompt-minimo` | 62 % | 4,08 | 3,67 | **4,33** | 3,79 |
| `look-noop` | 0 % | **1,67** | **1,62** | 2,38 | 2,12 |

### 4.1 Experimento 1 — Presupuesto de pasos

Tres puntos de la curva: tope igual al óptimo, generoso (referencia) y el doble
del generoso. El resultado es monótono y de gran magnitud: **0 % → 21 % → 54 %**.

**Corrección metodológica previa.** La primera definición del presupuesto lo
fijaba en 2x las llamadas óptimas. Un piloto de ocho corridas mostró que
`library-search` y `extreme-archive` fallaban con cero errores, explorando de
forma ordenada, cuando se les agotaban los pasos. El problema es que el óptimo
de `library-search` (7 llamadas) presupone saber cuál de los ocho libros esconde
la llave; un agente que no lo sabe necesita revisarlos. Con ese tope estaba
midiendo el presupuesto asignado y no el razonamiento del agente. El presupuesto
pasó a ser `max(2 x óptimo, exhaustivo + 4)`, tomando la columna de fuerza bruta
del enunciado.

**`steps-justo` obtiene 0 % con las mejores notas de proceso del barrido** (4,21
en exploración, 3,96 en recuperación de errores). Las tres corridas de
`study-with-key`, el escenario más simple, con tope 3:

```
look{} → examine{escritorio} → examine{alfombra}
look{} → examine{alfombra}   → use{llave_oro, puerta_principal}
look{} → examine{escritorio} → examine{alfombra}
```

Las tres comienzan con `look`, que no forma parte del camino óptimo. La segunda
es la más ilustrativa: examinó la alfombra y pasó directamente a usar la llave,
es decir que dedujo correctamente dónde estaba, y perdió igual porque le faltó
el `take`.

Este caso justifica la inclusión de una dimensión cualitativa. Con el goal rate
como única métrica, `steps-justo` es un fracaso total. Con la rúbrica es una
configuración que ejecutó el mejor proceso del barrido y perdió por presupuesto.
Son diagnósticos opuestos que conducen a decisiones opuestas.

**Conclusión.** El presupuesto de pasos es el factor de mayor peso. El agente es
capaz pero muy ineficiente: necesita aproximadamente el doble del presupuesto
calificado como generoso. Aun así, ampliar el presupuesto no es suficiente:
`steps-holgado` alcanza el 54 % global pero se queda en 22 % en los escenarios
`extreme`, de modo que también existe un techo de razonamiento.

### 4.2 Experimento 2 — Prompting

Comparé dos system prompts. El estratégico (referencia) describe las mecánicas
del mundo y aporta un procedimiento de cinco pasos. El mínimo se limita a
indicar qué son las herramientas.

**El prompt mínimo resuelve el 62 % y el estratégico el 21 %.** Es la única
diferencia entre ambas configuraciones, de modo que la comparación es limpia. El
juez, que no conoce ni la configuración ni el desenlace, otorga al mínimo la
mejor fidelidad del barrido (4,33).

Las trazas muestran el mecanismo. Mismo escenario, `color-locks`:

```
[estratégico]  take(llave_plata) use(llave_plata,puerta_principal)
               examine(cofre_plateado) examine(cofre_gris) examine(cofre_azul) ...
[mínimo]       look() examine(cofre_azul) examine(cofre_plata)
               use(llave_plata,cofre_plata) ...  → resuelto
```

El prompt estratégico indica textualmente *"1. Empezá con `look` para
orientarte"*, y el agente que lo recibe comienza con `take`, sin observar. El
prompt mínimo, que no lo menciona, sí comienza con `look()`.

El uso de herramientas confirma el patrón sobre las 24 corridas de cada
configuración:

| | `look` | `examine` | `use` | repetición |
|---|---:|---:|---:|---:|
| estratégico | 21 % | 38 % | 16 % | 47 % |
| mínimo | 20 % | 23 % | 24 % | 37 % |

El estratégico examina mucho y actúa poco; el mínimo actúa.

La interpretación es que describir las mecánicas del mundo en detalle le
proporcionó un modelo mental, y el agente razonó a partir de esa descripción en
lugar de observar. El prompt mínimo lo deja sin otra fuente de información que
las herramientas, y las utiliza. El efecto es consistente en las tres
repeticiones y en las cuatro dificultades: el prompt mínimo gana en medium, hard
y extreme.

### 4.3 Experimento 3 — Ventana de historial

Recorté la ventana del M2 de 50 a 12 mensajes. La predicción era que el
rendimiento caería en los escenarios multi-sala, donde hay que recordar el mapa.

**El resultado no varía: 21 % y pass@3 de 38 %, idénticos a la referencia.** Por
dificultad hay movimiento (medium sube de 17 % a 33 %, hard baja de 17 % a 0 %),
pero con tres repeticiones esa magnitud es ruido.

La conclusión es más informativa que la predicción: el agente no estaba
utilizando el historial largo. Su comportamiento es reactivo y decide con las
últimas observaciones. Esto es coherente con que su fallo dominante sea la
repetición: si consultara su historial, detectaría que ya realizó esa llamada.

Un detalle secundario: con ventana corta comenzó a invocar `calculator`,
`read_text_file` y `count_words`, las herramientas del M1, que están registradas
en todas las corridas porque `build_agent` las registra siempre. Son 41 llamadas
sobre 3522 en todo el barrido (1,2 %), y aparecen únicamente cuando pierde el
hilo.

### 4.4 Experimento 4 — Herramienta no-op

Reemplacé `look` por una función que devuelve `"No observás nada nuevo."`. El
esquema sigue anunciado al modelo: la herramienta existe pero no informa.

**El rendimiento cae del 21 % al 0 %**, incluido `study-with-key`, que se
resolvía 3 de 3. La rúbrica acompaña con las peores notas del barrido (1,67 en
exploración, 1,62 en memoria).

El dato relevante es el comportamiento ante la herramienta inutilizada: **el 50 %
de las llamadas siguen dirigidas a `look`**, con 68 % de repetición. El agente
recibe el mismo texto vacío una y otra vez y no infiere que debe cambiar de
estrategia.

`examine`, `take` y `use` continuaban operativas. Un agente que recordara la
salida del primer `look` habría podido resolver `study-with-key` sin volver a
observar. Este experimento y el anterior convergen en la misma conclusión desde
lados opuestos: la conducta del agente está anclada en la última observación y
no en la memoria de lo observado.

---

## 5. Limitaciones y trabajo futuro

### 5.1 Limitaciones del agente

**El bucle de repetición no tiene freno.** Concentra el 66 % de los fallos y el
framework no lo detecta. El agente puede invocar cincuenta veces la misma
herramienta con los mismos argumentos sin que el bucle de `run` intervenga.

**No aprovecha la memoria disponible.** La ventana del M2 funciona, pero el
experimento 3 muestra que reducirla a la cuarta parte no altera el resultado: el
agente decide con la última observación.

**No aprende de los mensajes de error.** Las herramientas de `mia_world`
devuelven errores accionables ("no existe ningún objeto con id X"), igual que las
implementadas en el M2. En `backtracking-vault` ignoró nueve consecutivos.

### 5.2 Limitaciones de la evaluación

**El juez no está instrumentado.** No registro sus tokens ni su costo, de modo
que los US$ 0,77 reportados corresponden únicamente al barrido del agente.

**Tres repeticiones son pocas.** Con n=3 por celda, una diferencia de un solo
éxito desplaza el goal rate 33 puntos. Las tendencias de gran magnitud (prompt,
`look-noop`, presupuesto) son sólidas por su tamaño y consistencia; las
variaciones menores, como el movimiento por dificultad de `memoria-corta`, no
las considero señal.

**El agente ve tres herramientas irrelevantes.** `build_agent` registra siempre
las del M1. No lo modifiqué porque rompería la conformidad del M1, pero
constituye un factor de confusión: son 41 llamadas desaprovechadas.

**Un solo modelo.** Todo el barrido usa `nova-lite`, por lo que no puedo separar
qué parte de lo medido corresponde al framework y qué parte al modelo. El
resultado del prompt mínimo, en particular, requiere verificación en otros
modelos antes de generalizarlo.

**La taxonomía de fricción se basa en patrones de texto.** Funciona porque
`mia_world` es fijo y sus mensajes de error son estables, pero no se traslada a
otro mundo sin reescribirla.

### 5.3 Trabajo futuro

1. **Detector de repetición en el bucle.** Si la misma llamada con los mismos
   argumentos aparece dos veces, inyectar un mensaje de sistema que lo señale
   antes de la tercera. Es de bajo costo y ataca el 66 % de los fallos.
2. **Registro estructurado del mundo.** Una estructura que el agente actualice
   con lo observado (mapa, inventario, contenedores ya abiertos) y que viaje en
   cada llamada fuera de la ventana de mensajes. Los experimentos 3 y 4 apuntan
   en esa dirección.
3. **Barrido sobre más modelos.** Repetir la evaluación con `nova-micro` y
   `nova-pro` para separar los efectos del framework de los del modelo. La
   infraestructura ya lo soporta: solo requiere cambiar `BEDROCK_MODEL_ID`.
4. **Aumentar las repeticiones a diez** en las celdas relevantes, dado que el
   barrido completo cuesta menos de un dólar.

---

## Apéndice — Cómo ejecutarlo

```bash
# (Windows PowerShell, desde la raíz del repo)

# Tests de la infraestructura de evaluación (deterministas, sin API)
.\.venv\Scripts\python.exe -m pytest tests/test_eval_harness.py -v

# Toda la suite: conformidad M1/M2/M3, escenarios propios y eval
.\.venv\Scripts\python.exe -m pytest -q

# Validar credenciales antes de gastar
.\.venv\Scripts\python.exe eval/run.py --preflight

# Coste estimado del barrido, sin llamar al modelo
.\.venv\Scripts\python.exe eval/run.py --estimar

# Barrido completo: 144 corridas + rúbrica + informe
.\.venv\Scripts\python.exe eval/run.py

# Variantes
.\.venv\Scripts\python.exe eval/run.py --solo-baseline --reps 1 --sin-juez
.\.venv\Scripts\python.exe eval/run.py --escenarios easy medium
.\.venv\Scripts\python.exe eval/run.py --solo-juez      # rúbrica sobre runs.jsonl existente
.\.venv\Scripts\python.exe eval/run.py --solo-informe   # regenera report.md sin gastar
```

El barrido escribe tres archivos en `eval/results/`: `runs.jsonl` con la traza
completa de cada corrida, `judgments.jsonl` con los veredictos del juez y
`report.md` con las tablas. Todos los números de este informe provienen de esos
archivos y se regeneran con `--solo-informe`.

Requiere un `.env` con `BEDROCK_MODEL_ID`, `AWS_REGION` y credenciales de
Bedrock. Usé `amazon.nova-lite-v1:0` en `us-east-2` para el agente y
`us.amazon.nova-pro-v1:0` para el juez; este último necesita el prefijo `us.`
porque Nova Pro no admite invocación on-demand con el identificador simple.
