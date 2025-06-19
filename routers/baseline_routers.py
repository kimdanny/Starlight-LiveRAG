from routers.RouterClass import Router
import random

random.seed(42)


class RandomRouter(Router):
    def __init__(self, M: int = 7) -> None:
        super().__init__(M)

    def batch_route(self, questions: list[str]) -> list[int]:
        action_set = [i for i in range(self.M)]
        return random.choices(action_set, k=len(questions))
