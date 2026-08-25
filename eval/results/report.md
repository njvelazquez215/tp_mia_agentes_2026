# Resultados de la evaluación — Milestone 3

Modelo del agente: `amazon.nova-lite-v1:0`. 144 corridas = 8 escenarios x 6 configuraciones x repeticiones. Coste total estimado: **$0.7683**. Tiempo de cómputo: 51.5 min.


## 1. Resultado principal (baseline)

El agente resuelve **21%** de las corridas (pass@3 = 38% de los escenarios). Cuando gana alcanza 0.55 de eficiencia respecto de la solución óptima, y **34%** de sus llamadas a herramienta devuelven error.


|  | n | goal rate | pass@k | eficiencia | calls | err/call | coste |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 24 | 21% ± 41% | 38% | 0.55 | 26.1 | 34% | $0.1310 |


## 2. Desglose por escenario (baseline)

`óptimo` es la longitud de la solución perfecta; `calls`, lo que gastó el agente. La eficiencia solo se define en corridas ganadas.


| escenario | dif. | óptimo | tope | éxitos | calls | efic. | err/call |
| --- | --- | --- | --- | --- | --- | --- | --- |
| study-with-key | easy | 3 | 7 | 3/3 | 5.3 | 0.57 | 6% |
| apartment-keys | medium | 7 | 14 | 1/3 | 13.3 | 0.58 | 25% |
| color-locks | medium | 11 | 22 | 0/3 | 22.0 | — | 12% |
| library-search | hard | 7 | 17 | 0/3 | 17.0 | — | 45% |
| office-sequence | hard | 13 | 26 | 1/3 | 26.7 | 0.46 | 12% |
| extreme-archive | extreme | 4 | 27 | 0/3 | 45.7 | — | 74% |
| backtracking-vault | extreme | 18 | 36 | 0/3 | 36.0 | — | 18% |
| vault-combination | extreme | 21 | 42 | 0/3 | 42.7 | — | 30% |


## 3. Desglose por dificultad (baseline)

|  | n | goal rate | pass@k | eficiencia | calls | err/call | coste |
| --- | --- | --- | --- | --- | --- | --- | --- |
| easy | 3 | 100% ± 0% | 100% | 0.57 | 5.3 | 6% | $0.0026 |
| medium | 6 | 17% ± 37% | 50% | 0.58 | 17.7 | 17% | $0.0197 |
| hard | 6 | 17% ± 37% | 50% | 0.46 | 21.8 | 25% | $0.0268 |
| extreme | 9 | 0% ± 0% | 0% | — | 41.4 | 43% | $0.0820 |


## 4. Análisis de errores

Dos ejes independientes. **Terminación** es por qué se detuvo el agente; **fricción**, qué tipo de error predominó en su traza. Una corrida puede agotar el presupuesto y además haber estado peleando con ids inexistentes: son dos hechos distintos y cada uno sugiere un arreglo distinto.


### 4.1 Cómo terminaron las corridas (todas las configuraciones)


| terminación | n | % | qué significa |
| --- | --- | --- | --- |
| presupuesto_agotado | 104 | 72% | Se acabaron las iteraciones sin cumplir la meta: el agente seguía actuando cuando lo cortamos. |
| exito | 38 | 26% | El mundo cumple la meta al terminar la corrida. |
| rendicion_temprana | 2 | 1% | Devolvió una respuesta final sin cumplir la meta, teniendo presupuesto de sobra. |


### 4.2 Fricción dominante en las corridas fallidas


| fricción | n | % | qué significa |
| --- | --- | --- | --- |
| repeticion | 70 | 66% | Repitió la misma llamada con los mismos argumentos tres veces o más. |
| id_inexistente | 23 | 22% | Ids que no existen: usó el nombre en prosa ('llave dorada') en vez del id ('llave_oro'), o inventó un objeto. |
| ninguna | 7 | 7% | La traza no acumuló errores de herramienta relevantes. |
| inventario_vacio | 3 | 3% | Intentó `use` sin haber hecho `take` antes. |
| objeto_no_visible | 2 | 2% | Actuó sobre objetos que existen pero no son visibles desde donde está: no examinó el contenedor que los oculta, o está en otra sala. |
| argumentos_invalidos | 1 | 1% | El framework rechazó la llamada antes de llegar al mundo: herramienta inexistente, JSON malformado o firma incompatible. |


**Éxito alucinado:** en 3 de 106 corridas fallidas el agente afirmó haber abierto la puerta mientras el mundo decía lo contrario. Es la razón por la que la métrica se calcula con `check_goal` sobre el estado del mundo y no sobre el texto del agente.


## 5. Dimensión cualitativa: rúbrica del juez LLM

Juez: `us.amazon.nova-pro-v1:0`, distinto del actor. Recibe la traza y la respuesta final, pero no si la corrida cumplió la meta. Escala 1 (muy malo) a 5 (muy bueno).


| configuración | n juzgadas | exploracion sistematica | uso de memoria | fidelidad del reporte | recuperacion de errores |
| --- | --- | --- | --- | --- | --- |
| baseline | 24 | 3.42 | 2.71 | 2.88 | 2.21 |
| steps-justo | 24 | 4.21 | 3.58 | 3.83 | 3.96 |
| steps-holgado | 24 | 3.67 | 3.29 | 3.50 | 3.42 |
| memoria-corta | 24 | 3.21 | 3.04 | 3.00 | 2.54 |
| prompt-minimo | 24 | 4.08 | 3.67 | 4.33 | 3.79 |
| look-noop | 24 | 1.67 | 1.62 | 2.38 | 2.12 |


## 6. Experimentos

Cada fila cambia una sola cosa respecto del baseline. `MyAgent` es el mismo en todas: lo que cambia es la configuración que recibe de `build_agent`.


|  | n | goal rate | pass@k | eficiencia | calls | err/call | coste |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 24 | 21% ± 41% | 38% | 0.55 | 26.1 | 34% | $0.1310 |
| steps-justo | 24 | 0% ± 0% | 0% | — | 10.9 | 16% | $0.0441 |
| steps-holgado | 24 | 54% ± 50% | 88% | 0.48 | 38.2 | 19% | $0.2698 |
| memoria-corta | 24 | 21% ± 41% | 38% | 0.53 | 24.2 | 22% | $0.0944 |
| prompt-minimo | 24 | 62% ± 48% | 88% | 0.51 | 23.4 | 24% | $0.1130 |
| look-noop | 24 | 0% ± 0% | 0% | — | 23.9 | 33% | $0.1160 |


### Qué cambia cada configuración


- **`baseline`** — Referencia: prompt con estrategia explícita, presupuesto generoso, ventana de historial de 50 mensajes y herramientas intactas.

- **`steps-justo`** — Presupuesto igual a la solución óptima: cero margen de error y, en los escenarios de búsqueda, ni siquiera lo justo para buscar.

- **`steps-holgado`** — El doble del presupuesto generoso. Si el goal rate no sube respecto del baseline, el cuello de botella no son los pasos.

- **`memoria-corta`** — Ventana de historial recortada a 12 mensajes. Mide cuánto aporta la estrategia de memoria del M2.

- **`prompt-minimo`** — System prompt sin procedimiento: qué herramientas hay, pero no cómo encarar el problema.

- **`look-noop`** — look devuelve texto vacío. Mide la dependencia de reobservar el mundo frente a recordarlo.


### Efecto por dificultad


| configuración | easy | medium | hard | extreme |
| --- | --- | --- | --- | --- |
| baseline | 100% | 17% | 17% | 0% |
| steps-justo | 0% | 0% | 0% | 0% |
| steps-holgado | 100% | 67% | 67% | 22% |
| memoria-corta | 100% | 33% | 0% | 0% |
| prompt-minimo | 100% | 83% | 33% | 56% |
| look-noop | 0% | 0% | 0% | 0% |
