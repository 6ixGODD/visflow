from __future__ import annotations

import asyncio as aio
import pathlib as p

from visflow.helpers.downloader import DownloadTask, HTTPDownloader
from visflow.pipelines import BasePipeline
from visflow.utils import spinner


class InitPipeline(BasePipeline):
    def __init__(self, proxy: str | None = None):
        self._completed = False
        self.downloader = HTTPDownloader(
            base_url="https://raw.githubusercontent.com/6ixGODD/visflow" "/master/",
            proxy=proxy,
        )

    def __call__(self) -> None:
        spinner.start("Initializing project...")
        aio.run(
            self.downloader.downloads(
                DownloadTask(
                    url=".config.example.yml",
                    save_to=(p.Path(""),),
                    method="GET",
                    save_url=".config.yml",
                ),
                DownloadTask(
                    url="data/train/README.md",
                    save_to=(p.Path(""),),
                    method="GET",
                    save_url="data/train/README.md",
                ),
                DownloadTask(
                    url="data/test/README.md",
                    save_to=(p.Path(""),),
                    method="GET",
                    save_url="data/test/README.md",
                ),
                DownloadTask(
                    url="data/val/README.md",
                    save_to=(p.Path(""),),
                    method="GET",
                    save_url="data/val/README.md",
                ),
            )
        )
        spinner.succeed(
            "Project initialized successfully. Please put your data in the "
            '"data" folder, then edit .config.yml to configure your project, '
            'then run "visflow train" to start training. Happy exploring!'
        )
        self._completed = True
