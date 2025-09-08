from __future__ import annotations

import argparse

from vis_tool import __version__


def parse_args() -> argparse.Namespace:
    import graphragx.cli.cmd.chat as chat
    import graphragx.cli.cmd.migrate as migrate
    import graphragx.cli.cmd.query as query

    parser = argparse.ArgumentParser(
        description="GraphRAG X",
        prog='python -m graphragx',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--version', '-v',
        action='version',
        version=f'%(prog)s {__version__}',
        help='Show the version of GraphRAG X',
    )
    parser.add_argument(
        '--log-level', '-l',
        type=str,
        default='DEBUG',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set the logging level (default: %(default)s)',
    )
    parser.add_argument(
        '--logfile', '-f',
        type=str,
        default=None,
        help='Path to the log file (default: %(default)s)',
    )
    parser.add_argument(
        '--exclude-namespace', '-x',
        type=str,
        nargs='*',
        default=('neo4j.notifications',
                 'neo4j.pool',
                 'neo4j.io',
                 'neo4j',
                 'httpcore.http11',
                 'httpcore.connection',
                 'openai._base_client'),
    )
    parser.add_argument(
        '--log-format', '-lf',
        type=str,
        default='[%(asctime)s] [%(name)s] [%(levelname)s] => %(message)s',
        help='Set the logging format (default: %(default)s)',
    )
    parser.add_argument(
        '--log-datefmt', '-ld',
        type=str,
        default='%Y-%m-%d %H:%M:%S',
        help='Set the date format for logging (default: %(default)s)',
    )
    subparser = parser.add_subparsers(
        title='subcommands',
        description='Available subcommands',
        dest='command',
    )

    query.register(subparser)
    migrate.register(subparser)
    chat.register(subparser)

    def _print_help(args_: argparse.Namespace) -> None:
        parser.print_help()
        if args_.command is None:
            print("\nPlease specify a subcommand. Use -h for help.")

    parser.set_defaults(func=_print_help)
    args = parser.parse_args()

    u.setup_logging(
        level=args.log_level,
        exclude_namespace=args.exclude_namespace,
        format=args.log_format,
        datefmt=args.log_datefmt,
        logfile=args.logfile,
    )

    return args
