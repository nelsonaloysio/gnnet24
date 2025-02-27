import logging as log
import operator as op
from typing import Callable, Optional, Union

OPERATORS = {"==": "eq", "!=": "ne",
             ">":  "gt", "<":  "lt",
             ">=": "ge", "<=": "le"}

format_func = lambda x: f"{x:.4f}" if type(x) == float else x


class EarlyStop():
    """
    Class to implement early stopping based on an evaluation metric.

    :param patience: Number of epochs to wait before stopping the training.
        Set as None to disable early stopping and use `max_epoch` instead.
    :param max_epoch: Maximum number of epochs to train.
    :param min_epoch: Minimum number of epochs before stopping the training.
    :param monitor: Metric (key) to use if `eva` is a dictionary.
        If None, the evaluation metrics will be compared directly.
    :param operator: Operator to use for comparison. Default: 'gt'.
        Choices: 'eq' (equal), 'ne' (not equal), 'gt' (greater than),
        'lt' (less than), 'ge' (greater or equal), 'le' (less or equal).
        Also accepts a string representing the operator, e.g., '>' for 'gt'.
    """
    def __init__(
        self,
        patience: Optional[int] = None,
        max_epoch: Optional[int] = None,
        min_epoch: Optional[int] = None,
        monitor: Optional[str] = None,
        operator: Optional[str] = None,
        history: Optional[bool] = None,
        f: Optional[Callable] = None,
    ) -> None:
        assert operator is None or operator in list(OPERATORS.keys()) + list(OPERATORS.values()),\
               f"Invalid operator '{operator}'."

        self.patience = patience or 0
        self.max_epoch = max_epoch
        self.min_epoch = min_epoch or 0
        self.monitor = monitor or "acc"
        self.operator = getattr(op, OPERATORS.get(operator or ">", operator))
        self.history = {} if history else None
        self.f = f or format_func

    def __call__(
        self,
        eva: Union[float, dict],
        epoch: Optional[int] = None,
        quiet: bool = False,
        **kwargs
    ) -> bool:
        """
        Return True if the early stopping condition is reached.

        :param eva: Evaluation metric(s) to compare.
        :param epoch: Current epoch (optional).
        :param quiet: Whether to supress logging metrics.
        :param kwargs: Extra evaluation metrics to log.
        """
        self.epoch = self.epoch + 1 if epoch is None else epoch

        if not quiet:
            log.info(self._info(self.epoch, eva, self.last, self.best, kwargs))

        if not self.epoch or self.__is_better(eva, self.best):
            # Store best epoch and evaluation metric(s).
            self.best = eva
            self.counter = 0

        else:
            # Increment the stop counter.
            self.counter = self.counter + 1

        # Store last evaluation metric(s).
        self.last = eva

        # Store evaluation metric(s).
        if type(self.history) == dict:
            self.history[self.epoch] = self.last

        # Return True to signal condition reached.
        if self.epoch == self.max_epoch\
        or (self.patience and self.counter >= self.patience and self.epoch >= self.min_epoch):
            log.info(f"Early stopping condition reached ({self._info(self.best_epoch, eva)}).")
            return True

        return False

    def __repr__(self) -> str:
        """
        Return a string representation of the early stopping condition.
        """
        best = self._info(self.best_epoch, self.best) if self.best is not None else "epoch: None"
        return f"epoch: {self.epoch}, best {best}"

    def reset(self) -> None:
        """ Reset the early stopping condition. """
        self.best = None
        self.last = None
        self.counter = -1
        self.epoch = -1

    def _info(
        self,
        epoch: int,
        eva: Union[float, dict],
        last: Optional[Union[float, dict]] = None,
        best: Optional[Union[float, dict]] = None,
        extra: Optional[dict] = None,
    ) -> str:
        """
        Return info from evaluaton metrics.

        :param epoch: Current epoch.
        :param eva: Current evaluation metrics.
        :param last: Last evaluation metrics.
        :param best: Best evaluation metrics.
        :param extra: Extra evaluation metrics.
        """
        c = None

        if last is not None:
            c = "^" if self.__is_better(eva, last) else\
                "v" if self.__is_better(last, eva) else "="

        if best is not None:
            c = "*" if self.__is_better(eva, best) else c

        eva = f"{', '.join(f'{k}: {self.f(v)}' for k, v in eva.items())}"\
              if type(eva) == dict else\
              f"{self.monitor or 'eva'}: {eva:.4f}"\

        return f"epoch: {epoch}, {eva}"\
               f"{', ' + ', '.join(f'{k}: {self.f(v)}' for k, v in extra.items()) if extra else ''}"\
               f"{f' ({c})' if c else ''}"

    def __is_better(
        self,
        eva1: Union[float, dict],
        eva2: Optional[Union[float, dict]] = None
    ) -> bool:
        """
        Return True if `eva1` is better than `eva2`.

        :param eva1: Evaluation metric to compare.
        :param eva2: Evaluation metric to compare.
        """
        if not eva2:
            return True

        if type(eva1) == dict:
            eva1 = eva1[self.monitor]

        if type(eva2) == dict:
            eva2 = eva2[self.monitor]

        return self.operator(eva1, eva2)

    @property
    def best(self) -> Union[float, dict]:
        """ Return the best eval. """
        return self.__dict__.get("_best", None)

    @best.setter
    def best(self, value) -> None:
        """ Set the best eval. """
        self._best = {
            "epoch": self.epoch,
            **(value if type(value) == dict else {self.monitor or "value": value})}

    @property
    def best_epoch(self) -> Union[float, dict]:
        """ Return the best epoch. """
        best = self.best
        return best["epoch"] if best else None

    @property
    def best_value(self) -> Union[float, dict]:
        """ Return the best epoch. """
        best = self.best
        return best["value"] if best else None

    @property
    def counter(self) -> int:
        """ Return the counter. """
        return self.__dict__.get("_counter", -1)

    @counter.setter
    def counter(self, value) -> None:
        """ Set the counter. """
        self._counter = value

    @property
    def epoch(self) -> int:
        """ Return the epoch. """
        return self.__dict__.get("_epoch", -1)

    @epoch.setter
    def epoch(self, value) -> None:
        """ Set the epoch. """
        self._epoch = value

    @property
    def last(self) -> Union[float, dict]:
        """ Return the last eval. """
        return self.__dict__.get("_last", None)

    @last.setter
    def last(self, value) -> None:
        """ Set the last eval. """
        self._last = {
            "epoch": self.epoch,
            **(value if type(value) == dict else {self.monitor or "value": value})}
