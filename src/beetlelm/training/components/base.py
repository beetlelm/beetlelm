class DataPipeline:
    def next_batch(self, runtime_state):
        raise NotImplementedError

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass


class TrainingRegime:
    def compute_loss(self, model, batch, runtime_state):
        raise NotImplementedError

    def optimizer_step(self, optimizer, runtime_state):
        optimizer.step()
        optimizer.zero_grad()


class CheckpointPolicy:
    def should_checkpoint(self, runtime_state):
        return False


class Evaluator:
    def evaluate(self, model, runtime_state):
        return {}
