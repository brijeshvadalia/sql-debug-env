"""
tests/conftest.py — pytest configuration for SQL Debug Environment v3.0

Provides:
  - Pydantic stub for environments without pydantic installed
  - sys.path setup for project root imports
  - Shared fixtures
"""
import sys
import os

# Project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Pydantic stub — handles Field defaults properly so tests work without
# installing pydantic. Real pydantic is used on the deployed HF Space.
# ---------------------------------------------------------------------------

try:
    import pydantic
    # Real pydantic available — use it
    HAS_PYDANTIC = True
except ImportError:
    import types

    class _Field:
        """Minimal Field stub that returns the default value."""
        def __init__(self, default=None, default_factory=None, *args, **kwargs):
            self._default = default
            self._factory = default_factory

        def __repr__(self):
            return f"Field(default={self._default})"

    class _ModelMeta(type):
        """Metaclass that processes Field descriptors on class creation."""
        def __new__(mcs, name, bases, namespace):
            fields = {}
            annotations = namespace.get("__annotations__", {})

            # Collect Field defaults from all bases
            for base in bases:
                if hasattr(base, "_field_defaults"):
                    fields.update(base._field_defaults)

            # Process class-level attributes that are Field instances
            for attr, val in list(namespace.items()):
                if isinstance(val, _Field):
                    if val._factory is not None:
                        fields[attr] = ("factory", val._factory)
                    else:
                        fields[attr] = ("value", val._default)
                    namespace[attr] = val._default  # set class default

            namespace["_field_defaults"] = fields
            cls = super().__new__(mcs, name, bases, namespace)
            return cls

    class _BaseModel(metaclass=_ModelMeta):
        def __init__(self, **kwargs):
            # Apply field defaults first
            for attr, spec in getattr(self.__class__, "_field_defaults", {}).items():
                kind, val = spec
                if attr not in kwargs:
                    setattr(self, attr, val() if kind == "factory" else val)

            # Apply provided kwargs
            for k, v in kwargs.items():
                setattr(self, k, v)

        def model_dump(self):
            return {k: v for k, v in self.__dict__.items()
                    if not k.startswith("_")}

        def model_dump_json(self, **kw):
            import json
            return json.dumps(self.model_dump(), indent=kw.get("indent"))

    pydantic_mod = types.ModuleType("pydantic")
    pydantic_mod.BaseModel = _BaseModel
    pydantic_mod.Field = _Field
    sys.modules["pydantic"] = pydantic_mod
    HAS_PYDANTIC = False
