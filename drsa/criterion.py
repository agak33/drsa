class BaseCriterion:
    def __init__(self, name: str) -> None:
        """Base class for criteria."""
        self.name = name

    def __repr__(self) -> str:
        return self.name


class Criterion(BaseCriterion):
    def __init__(self, name: str, is_cost: bool = False) -> None:
        """Class for criteria."""
        self.name = name
        self.is_cost = is_cost

    def __repr__(self) -> str:
        return f"{self.name}; " f"{'cost' if self.is_cost else 'gain'} attr"


class DecisionCriterion(BaseCriterion):
    def __init__(self, name: str) -> None:
        """Class for decision criteria."""
        super().__init__(name)

    def __repr__(self) -> str:
        return f"{self.name}; decision attr"
