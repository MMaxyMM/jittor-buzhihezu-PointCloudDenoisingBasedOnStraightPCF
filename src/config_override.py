"""Small helpers for applying command-line overrides to config dictionaries."""

from copy import deepcopy
from typing import Dict, Iterable

from omegaconf import OmegaConf


def apply_overrides(config: Dict, overrides: Iterable[str]) -> Dict:
    """Return a copy of *config* with OmegaConf dot-list overrides applied."""
    values = list(overrides or [])
    if not values:
        return deepcopy(config)

    base = OmegaConf.create(deepcopy(config))
    updated = OmegaConf.merge(base, OmegaConf.from_dotlist(values))
    result = OmegaConf.to_container(updated, resolve=True)
    if not isinstance(result, dict):
        raise ValueError("overridden config must remain a mapping")
    return result
