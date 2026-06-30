"""
Calculadora Simple.

Sigue el patrón del M1: callable tipado con `Annotated` + `Field` y
`ToolSchema.from_callable(...)` para derivar el JSON Schema. Sin `eval`
ni expresiones arbitrarias: solo la operación binaria indicada.

Nota sobre operadores: el ENUNCIADO_M1.md pide `+ - * %` (módulo) y la 
consigna del campus pide `+ - * /` (división). Para cubrir ambas variantes 
del enunciado se soportan los cinco operadores (`+ - * / %`).
"""

from __future__ import annotations
from typing import Annotated
from pydantic import Field
from mia_agents.types import ToolSchema


def _format_number(value: float) -> str:
    """Muestra enteros sin `.0` y decimales tal cual (resultado siempre str)."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


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
    exactamente la operación indicada.
    """
    op = operator.strip()

    if op == "+":
        result = left_operand + right_operand
    elif op == "-":
        result = left_operand - right_operand
    elif op == "*":
        result = left_operand * right_operand
    elif op == "/":
        if right_operand == 0:
            return "Error: división por cero."
        result = left_operand / right_operand
    elif op == "%":
        if right_operand == 0:
            return "Error: módulo por cero."
        result = left_operand % right_operand
    else:
        return (
            f"Error: operador no soportado '{operator}'. "
            "Usá uno de: +, -, *, /, %."
        )

    return _format_number(result)


calculator_schema = ToolSchema.from_callable(calculator)
