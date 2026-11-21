"""Task id -> class. Resolving an id imports only that module."""

from importlib import import_module
from inspect import signature
from typing import Any

from wsc_bench.tasks.base import Target

TARGET_PATHS: dict[str, str] = {
    "add-10digit": "wsc_bench.tasks.add_digits:AddDigits10",
    "sort-numbers": "wsc_bench.tasks.sort_numbers:SortNumbers",
    "gcd-10bit": "wsc_bench.tasks.gcd:Gcd10",
    "subleq-step": "wsc_bench.tasks.subleq:SubleqStep",
    "bowling": "wsc_bench.tasks.bowling:Bowling",
    "multiply-32": "wsc_bench.tasks.multiply:Multiply32",
    "arith-expr": "wsc_bench.tasks.arith_expr:ArithExpr",
    "chess-mate-in-one": "wsc_bench.tasks.chess_mate:ChessMateInOne",
    "regex-match": "wsc_bench.tasks.regex_match:RegexMatch",
    "decimal-period": "wsc_bench.tasks.decimal_period:DecimalPeriod",
    "sudoku-any": "wsc_bench.tasks.sudoku_any:SudokuAny",
    "inverse-bwt": "wsc_bench.tasks.inverse_bwt:InverseBwt",
    "empty-circumcircle": "wsc_bench.tasks.empty_circumcircle:EmptyCircumcircle",
    "reed-solomon-decode": "wsc_bench.tasks.reed_solomon:ReedSolomonDecode",
}

TARGETS: dict[str, type[Target]] = {}


def register_target(cls: type[Target]) -> type[Target]:
    """Class decorator kept from the original code; modules register on import."""
    TARGETS[cls.id] = cls
    return cls


def resolve_target_class(target_id: str) -> type[Target]:
    module_name, attribute = TARGET_PATHS[target_id].split(":")
    return getattr(import_module(module_name), attribute)


def get_target(*, target_id: str, params: dict[str, Any]) -> Target:
    """`max_nonzero` caps nonzero weights (wsc_bench.tasks.nonzero_cap) unless the class takes the cap itself."""
    target_class = resolve_target_class(target_id)
    if "max_nonzero" not in params or "max_nonzero" in signature(target_class.__init__).parameters:
        return target_class(**params)
    from wsc_bench.tasks.nonzero_cap import nonzero_capped_class

    return nonzero_capped_class(target_class)(**params)
