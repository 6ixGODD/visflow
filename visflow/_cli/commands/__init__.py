from __future__ import annotations

import argparse

from visflow import __version__


def parse_args() -> argparse.Namespace:
    from visflow._cli.commands import train, gradcam

    parser = argparse.ArgumentParser(
        description='VisFlow',
        prog="python -m visflow",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        '--version', '-v',
        action='version',
        version=f'%(prog)s {__version__}',
        help='Show the version of ModX',
    )
    subparser = parser.add_subparsers(
        title='subcommands',
        description='Available subcommands',
        dest='command',
    )
    train.register(subparser)
    gradcam.register(subparser)

    def _print_help(args_: argparse.Namespace) -> None:
        parser.print_help()
        if args_.command is None:
            print("\nPlease specify a subcommand. Use -h for help.")

    parser.set_defaults(func=_print_help)
    args = parser.parse_args()
    return args
