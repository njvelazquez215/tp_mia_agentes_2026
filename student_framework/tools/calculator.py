"""
Calculadora Simple.

Sigue el patrón del M1: callable tipado con `Annotated` + `Field` y
`ToolSchema.from_callable(...)` para derivar el JSON Schema. Sin `eval`
ni expresiones arbitrarias: solo la operación binaria indicada.

Nota sobre operadores: el ENUNCIADO_M1.md pide `+ - * %` (módulo) y la
consigna del campus pide `+ - * /` (división). Para cubrir ambas variantes
del enunciado se soportan los cinco operadores (`+ - * / %`).

M2: ningún error de uso lanza excepción. Todos devuelven un mensaje
accionable (qué parámetro falló, qué llegó y qué se esperaba) para que el
LLM pueda corregir los argumentos y reintentar.
"""

from __future__ import annotations
from typing import Annotated, Any
from pydantic import Field
from mia_agents.types import ToolSchema

_OPERADORES = ("+", "-", "*", "/", "%")


class _OperandoInvalido(ValueError):
    """Un operando no pudo interpretarse como número."""


def _format_number(value: float) -> str:
    """Muestra enteros sin `.0` y decimales tal cual (resultado siempre str)."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _as_number(value: Any, parametro: str) -> float:
    """Coacciona a float lo que haya mandado el LLM, o explica por qué no.

    Los modelos suelen mandar el número como string ("17", "17.5") o,
    cuando alucinan, en palabras ("diecisiete"). Lo primero se acepta;
    lo segundo se rechaza indicando el parámetro exacto.
    """
    if isinstance(value, bool) or value is None:
        raise _OperandoInvalido(
            f"el parámetro '{parametro}' recibió {value!r} "
            f"({type(value).__name__}) y no es un número"
        )
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        limpio = value.strip().replace(",", ".")
        try:
            return float(limpio)
        except ValueError:
            raise _OperandoInvalido(
                f"el parámetro '{parametro}' recibió el texto {value!r}, "
                "que no representa un número"
            ) from None
    raise _OperandoInvalido(
        f"el parámetro '{parametro}' recibió un valor de tipo "
        f"{type(value).__name__}, y se esperaba un número"
    )


def calculator(
    left_operand: Annotated[
        float, Field(description="Primer operando numérico (lado izquierdo).")
    ],
    right_operand: Annotated[
        float, Field(description="Segundo operando numérico (lado derecho).")
    ],
    operator: Annotated[
        str,
        Field(
            description=(
                "Operador binario a aplicar. Uno de: '+' (suma), '-' (resta), "
                "'*' (producto), '/' (división), '%' (módulo)."
            )
        ),
    ],
) -> str:
    """Realiza una única operación aritmética binaria entre dos números.

    Recibe dos operandos numéricos y un operador (uno de +, -, *, /, %) y
    devuelve el resultado como cadena. No interpreta expresiones: aplica
    exactamente la operación indicada. Ante argumentos inválidos devuelve
    un mensaje que empieza con "Error:" y explica cómo corregirlos.
    """
    try:
        left = _as_number(left_operand, "left_operand")
        right = _as_number(right_operand, "right_operand")
    except _OperandoInvalido as exc:
        return (
            f"Error: {exc}. Pasá los operandos como números JSON, "
            'por ejemplo {"left_operand": 17, "right_operand": 23, "operator": "*"}.'
        )

    op = operator.strip() if isinstance(operator, str) else operator
    permitidos = ", ".join(_OPERADORES)

    if op == "+":
        result = left + right
    elif op == "-":
        result = left - right
    elif op == "*":
        result = left * right
    elif op == "/":
        if right == 0:
            return (
                "Error: división por cero. El parámetro 'right_operand' no puede "
                "valer 0 cuando el operador es '/'. Cambiá el divisor o usá otra "
                "operación."
            )
        result = left / right
    elif op == "%":
        if right == 0:
            return (
                "Error: módulo por cero. El resto de una división por 0 no está "
                "definido: 'right_operand' debe ser distinto de 0 con el operador "
                "'%'."
            )
        result = left % right
    else:
        return (
            f"Error: operador no soportado {operator!r}. "
            f"Usá exactamente uno de estos: {permitidos}."
        )

    return _format_number(result)


calculator_schema = ToolSchema.from_callable(calculator)
