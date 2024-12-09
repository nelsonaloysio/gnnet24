import logging as log
import operator as op
from typing import Callable, Optional, Union

OP = {"==": "eq", "!=": "ne", ">":  "gt", "<":  "lt", ">=": "ge", "<=": "le"}
f = lambda x: f"{x:.3f}" if type(x) == float else x


class EarlyStop():
    """
    Class to implement early stopping based on an evaluation metric.

    :param patience: Number of epochs to wait for improvement.
    :param max_epoch: Maximum number of epochs to wait before early stop.
    :param min_epoch: Minimum number of epochs to wait before early stop.
    :param metric: Key to use if passing a dictionary of evaluation metrics.
    :param operator: Operator to use for comparison. Default: 'gt'.
        Choices: 'eq' (equal), 'ne' (not equal), 'gt' (greater than),
        'lt' (less than), 'ge' (greater or equal), 'le' (less or equal).
        Also accepts a string representing the operator, e.g., '>' for 'gt'.
    :param verbose: Whether to print information about the early stopping.
    :param format_func: Function to format the evaluation metric(s).
    """
    def __init__(
        self,
        patience: Optional[int] = None,
        max_epoch: Optional[int] = None,
        min_epoch: Optional[int] = None,
        metric: Optional[str] = None,
        operator: str = ">",
        verbose: bool = True,
        format_func: Optional[Callable] = f,
    ) -> None:
        assert operator is None or operator in list(OP) + list(OP.values()),\
            f"Invalid operator '{operator}'."

        self.patience = patience or 0
        self.max_epoch = max_epoch or 0
        self.min_epoch = min_epoch or 0
        self.metric = metric or "value"
        self.op = getattr(op, OP.get(operator, operator))
        self.verbose = verbose
        self.f = format_func or (lambda x: x)

    def __call__(self, eva: Union[float, dict], epoch: Optional[int] = None, **kwargs) -> bool:
        """
        Return True if the early stopping condition is reached.

        :param eva: Evaluation metric(s).
        :param epoch: Set current epoch. Optional.
        """
        self.epoch = self.epoch + 1 if epoch is None else epoch

        if not self.epoch or self.__is_better(eva, self.best):
            self.best = eva

        if self.verbose:
            log.info(
                "epoch: %s, %s%s%s",
                self.epoch,
                f"{', '.join([f'{k}: {self.f(v)}' for k, v in eva.items()])}"
                if type(eva) == dict else f"{self.metric}: {self.f(eva)}",
                f"{', ' + ', '.join([f'{k}: {self.f(v)}' for k, v in kwargs.items()])}"
                if kwargs else "",
                " (*)" if self.epoch == self.best_epoch else ""
            )

        # Return True to signal condition reached.
        if self.early_stop():
            log.info(
                "Early stop on epoch %s (best %s).",
                self.epoch,
                f"{', '.join([f'{k}: {self.f(v)}' for k, v in self.best.items()])}"
            )
            return True

        return False

    def __repr__(self) -> str:
        """
        Return a string representation of the early stopper.
        """
        best = self.best
        best = {', '.join([f'{k}: {self.f(v)}' for k, v in (best or {}).items()])}
        return f"epoch: {self.epoch}, best: {best}"

    def early_stop(self) -> bool:
        """ Return True if the early stopping condition is reached. """
        return (
            self.max_epoch and self.epoch >= self.max_epoch
            or (self.patience and self.epoch - self.best_epoch >= self.patience  and self.epoch >= self.min_epoch)
        )

    def reset(self) -> None:
        """ Reset the early stopper. """
        self.__dict__.pop("_best", None)
        self.__dict__.pop("_epoch", None)

    def __is_better(
        self,
        eva1: Union[float, dict],
        eva2: Optional[Union[float, dict]] = None
    ) -> bool:
        """
        Return True if `eva1` is better than `eva2`.

        :param eva1: Current evaluation.
        :param eva2: Previous evaluation.
        """
        if not eva2:
            return True

        if type(eva1) == dict:
            eva1 = eva1[self.metric]

        if type(eva2) == dict:
            eva2 = eva2[self.metric]

        return self.op(eva1, eva2)

    @property
    def best(self) -> Union[float, dict]:
        """ Return the best eval. """
        return self.__dict__.get("_best", None)

    @best.setter
    def best(self, value) -> None:
        """ Set the best eval. """
        self._best = {
            "epoch": self.epoch,
            **(value if type(value) == dict else {self.metric: value})}

    @property
    def best_epoch(self) -> Union[float, dict]:
        """ Return the best epoch. """
        best = self.best
        return best["epoch"] if best else None

    @property
    def epoch(self) -> int:
        """ Return the epoch. """
        return self.__dict__.get("_epoch", -1)

    @epoch.setter
    def epoch(self, value) -> None:
        """ Set the epoch. """
        self._epoch = value
