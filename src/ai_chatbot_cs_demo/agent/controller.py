"""Controller layer: mediates between view and model."""

from .model import AgentModel


class AgentController:
    def __init__(self) -> None:
        self._model = AgentModel()

    def handle(self, user_input: str) -> str:
        user_input = user_input.strip()
        if not user_input:
            return ""
        return self._model.answer(user_input)
