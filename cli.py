"""PortfolioBench CLI entry-point wrapper.

Patches a known argparse conflict in the freqtrade submodule before
delegating to ``freqtrade.main:main``.

The ``portfolio`` subcommand's ``ARGS_PORTFOLIO`` list includes
``datadir_portfolio`` whose ``--data-dir`` CLI flag duplicates the
identical flag already inherited from ``ARGS_COMMON`` via the shared
``_common_parser`` parent.  Removing the duplicate entry prevents the
``ArgumentError`` that otherwise breaks **every** portbench command.
"""

from __future__ import annotations


def main() -> None:
    from freqtrade.commands import arguments as _args

    try:
        _args.ARGS_PORTFOLIO.remove("datadir_portfolio")
    except ValueError:
        pass  # already removed

    from freqtrade.main import main as _ft_main
    _ft_main()


if __name__ == "__main__":
    main()
