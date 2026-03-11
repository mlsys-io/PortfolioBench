"""Root conftest — ensure the editable-installed freqtrade package is used
instead of the git-submodule directory being picked up as a namespace package."""

import importlib
import sys


def _fix_freqtrade_import() -> None:
    """If 'freqtrade' resolved to a namespace package (the submodule root),
    remove it so the editable-install finder can provide the real package."""
    mod = sys.modules.get("freqtrade")
    if mod is not None and getattr(mod, "__file__", None) is None:
        # Namespace package — drop it so a proper import can succeed
        del sys.modules["freqtrade"]

    # Also remove any cached sub-modules that might have been loaded
    stale = [k for k in sys.modules if k.startswith("freqtrade.")]
    for k in stale:
        if getattr(sys.modules[k], "__file__", None) is None:
            del sys.modules[k]

    # Re-import to let the editable finder (or sys.path) pick up the real package
    importlib.invalidate_caches()

    # Ensure the submodule's inner directory is on sys.path as a fallback
    import os
    ft_inner = os.path.join(os.path.dirname(__file__), "freqtrade")
    if os.path.isfile(os.path.join(ft_inner, "freqtrade", "__init__.py")):
        if ft_inner not in sys.path:
            sys.path.insert(0, ft_inner)


_fix_freqtrade_import()
