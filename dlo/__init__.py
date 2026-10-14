"""Five-method OpenFWI demonstration package (GPU operator imported lazily)."""
from .solver import METHODS, invert

__all__ = ["METHODS", "invert"]
