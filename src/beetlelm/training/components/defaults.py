from .base import DataPipeline
import torch.nn.functional as F
from .base import TrainingRegime


class CausalLMRegime(TrainingRegime):

    def compute_loss(self, model, batch, runtime_state):

        input_ids = batch["input_ids"]
        input_ids = input_ids[:, :-1]
        labels = batch["input_ids"][:, 1:]

        logits, _ = model(input_ids)
        logits = logits.transpose(1, 2)

        return F.cross_entropy(logits, labels)


class PicoDataPipeline(DataPipeline):

    def __init__(self, dataloader, pipeline_state=None):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        self.pipeline_state = pipeline_state

    def next_batch(self, runtime_state):
        batch = next(self.iterator)

        if self.pipeline_state is not None:
            runtime_state.pipeline_state = self.pipeline_state

        return batch
