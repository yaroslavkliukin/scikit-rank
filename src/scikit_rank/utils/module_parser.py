"""Spec-string parser for encoder / loss registries.

Grammar
-------
::

    spec    := name (":" pairs)?
    pairs   := pair (";" pair)*
    pair    := key "=" value
    value   := python-literal
             | "true" | "false" | "none" | "null"     -- case-insensitive
             | bare-string                            -- e.g. ``silu``, ``all_pairs``

``name`` and ``key`` are case-folded to lowercase. ``;`` (not ``,``) separates
pairs so list/tuple/dict literals like ``hidden_dims=[32,64,128]`` need no
quoting.

A value that *looks* like a literal (starts with a digit, sign, quote or
bracket) must parse cleanly: e.g. ``rate=0..5`` raises instead of silently
becoming the string ``"0..5"``. Anything else is kept as a bare string.
"""

from __future__ import annotations
import ast
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

_LITERAL_PREFIXES = "0123456789-+.\"'[({"


def _parse_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    looks_literal = bool(value) and value[0] in _LITERAL_PREFIXES
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError) as exc:
        if looks_literal:
            raise ValueError(
                f"invalid literal in spec value {value!r}",
            ) from exc
        return value


class ModuleParserSpec:
    """Parsed ``name[:key=value;key=value;...]`` spec.

    Parameters
    ----------
    spec :
        The spec string.
    allowed :
        Optional iterable of accepted module names; if given, the parsed
        ``name`` is validated against it and a :class:`ValueError` is raised on
        mismatch.

    """

    def __init__(
        self,
        spec: str,
        *,
        allowed: Iterable[str] | None = None,
    ) -> None:
        name, sep, raw_kwargs = spec.partition(":")
        name = name.strip().lower()
        if not name:
            raise ValueError(f"empty module name in spec {spec!r}")
        if allowed is not None:
            allowed_set = set(allowed)
            if name not in allowed_set:
                raise ValueError(
                    f"unknown module {name!r}; expected one of {sorted(allowed_set)}",
                )

        self._module_name = name
        self._module_kwargs: dict[str, Any] = {}
        if sep and raw_kwargs.strip():
            for item in raw_kwargs.split(";"):
                item = item.strip()
                if not item:
                    continue
                key, eq, value = item.partition("=")
                if not eq:
                    raise ValueError(
                        f"invalid spec option {item!r}; expected key=value",
                    )
                self._module_kwargs[key.strip().lower()] = _parse_value(
                    value.strip(),
                )

    def module_name(self) -> str:
        return self._module_name

    def kwargs(self) -> Mapping[str, Any]:
        """Return the parsed kwargs as an immutable view."""
        return MappingProxyType(self._module_kwargs)

    def get(self, name: str, default: Any = None) -> Any:
        return self._module_kwargs.get(name, default)

    def get_first(self, *names: str, default: Any = None) -> Any:
        """Return the value of the first ``name`` present in the parsed kwargs.

        Useful for accepting aliases (e.g. ``"dim"`` / ``"embedding_dim"``).
        """
        for n in names:
            if n in self._module_kwargs:
                return self._module_kwargs[n]
        return default

    def to_spec(self) -> str:
        if not self._module_kwargs:
            return self._module_name
        body = ";".join(f"{k}={v!r}" for k, v in self._module_kwargs.items())
        return f"{self._module_name}:{body}"

    def __repr__(self) -> str:
        return f"ModuleParserSpec({self.to_spec()!r})"
