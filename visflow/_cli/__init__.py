from __future__ import annotations

import visflow._cli.commands as cmd
import visflow._cli.exceptions as exc


def main() -> int:
    try:
        _main()
    except KeyboardInterrupt:
        return 130
    except exc.CLIException as e:
        return e.exit_code
    return 0


def _main() -> None:
    args = cmd.parse_args()
    args.func(args)
