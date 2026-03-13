"""PortfolioBench CLI entry-point wrapper.

Delegates to ``freqtrade.main:main``.
"""

from __future__ import annotations


def main() -> None:
    from freqtrade.main import main as _ft_main
    _ft_main()


if __name__ == "__main__":
    main()
