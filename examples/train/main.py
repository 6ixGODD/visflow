from __future__ import annotations

from visflow.resources.configs import TrainConfig
from visflow.pipelines.train import TrainPipeline


pipeline = TrainPipeline(TrainConfig.from_yaml('.train.yml'))

if __name__ == '__main__':
    pipeline()
