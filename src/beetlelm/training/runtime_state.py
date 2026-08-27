class RuntimeState:
    """
    Unified deterministic training ledger.

    Stores:
        - training step
        - token counter
        - pipeline_state (Pico streaming state)
        - curriculum state
    """

    def __init__(self):
        self.step = 0
        self.token_counter = 0
        self.pipeline_state = None
        self.curriculum_state = None

    # -------------------------

    def state_dict(self):
        return {
            "step": self.step,
            "token_counter": self.token_counter,
            "pipeline_state": (
                self.pipeline_state.state_dict()
                if self.pipeline_state is not None
                else None
            ),
            "curriculum_state": self.curriculum_state,
        }

    # -------------------------

    def load_state_dict(self, state):

        self.step = state["step"]
        self.token_counter = state["token_counter"]

        if (
            self.pipeline_state is not None
            and state["pipeline_state"] is not None
        ):
            self.pipeline_state.load_state_dict(
                state["pipeline_state"]
            )

        self.curriculum_state = state["curriculum_state"]
