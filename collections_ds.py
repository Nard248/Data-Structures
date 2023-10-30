from abc import ABC, abstractmethod


class Collection(ABC):

    @abstractmethod
    def empty(self) -> None:
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        pass

    @abstractmethod
    def size(self) -> int:
        pass

    def print(self) -> None:
        print(self)


class List(Collection):

    @abstractmethod
    def add_first(self, e) -> None:
        pass

    @abstractmethod
    def add_last(self, e) -> None:
        pass

    @abstractmethod
    def remove_first(self) -> bool:
        pass

    @abstractmethod
    def remove_last(self) -> bool:
        pass

    @abstractmethod
    def first(self) -> object:
        pass

    @abstractmethod
    def last(self) -> object:
        pass

    @abstractmethod
    def remove_min(self) -> object:
        pass


class Stack(Collection):

    @abstractmethod
    def push(self, e) -> None:
        pass

    @abstractmethod
    def pop(self) -> object:
        pass

    @abstractmethod
    def top(self) -> object:
        pass


class ArrayList(List):

    def __init__(self, capacity=10):
        self._array = [None] * capacity
        self._size = 0
        self._initial_capacity = capacity

    def get_at(self, index) -> object:
        if index < 0 or index >= self._size:
            raise IndexError("Array list index out of range!")
        return self._array[index]

    """ Task 1: Implement a method which removes the minimum element from the ArrayList. 
    In case if the list is empty, None should be returned."""

    def remove_min(self) -> object:
        if self.is_empty():
            return None
        else:
            if self._array[0] < self._array[1]:
                min = self._array[0]
            elif self._array[0] > self._array[1]:
                min = self._array[1]
            for i in range(1, self._size):
                x = self.get_at(i)
                if x < min:
                    self._array[i] = None
                    for i in range(i, self._size):
                        self._array[i] = self._array[i + 1]
                    self._size -= 1

    """ Task 3: Part 1: Implement a StepIterator iterator which receives an integer step and 
    performs forward iteration over every other <step> position element in the list.
    E.g. if the step is 2 and we have the following list [1, 2, 3, 4, 6, 7, 9] 
    the iterator will iterate over 1, 3, 6, 9 elements."""

    class StepIterator:

        def __init__(self, array, step):
            self._array = array
            self._step = step

        def __next__(self):
            if self._step <= 0:
                raise ValueError
            else:
                index = 0
                result_list = []
                while self._step < len(self._array):
                    result_list.append(self._array[index])
                    index += self._step
                return result_list

    def step_iterator(self, step: int) -> object:
        return ArrayList.StepIterator(self, step)

    """Task 3: Part 2: Make the ArrayList class iterable and use the StepIterator 
    to implement its default forward iteration. """

    def add_first(self, e) -> None:
        pass

    def add_last(self, e) -> None:
        pass

    def remove_first(self) -> bool:
        pass

    def remove_last(self) -> bool:
        pass

    def iter(self):
        return self

    def next(self):
        ArrayList.StepIterator(self, 1)

    def size(self) -> int:
        return self._size

    def is_empty(self) -> bool:
        return self._size == 0

    def empty(self) -> None:
        self._array = [None] * self._initial_capacity
        self._size = 0

    def first(self) -> object:
        if not self.is_empty():
            return self._array[0]
        return None

    def last(self) -> object:
        if not self.is_empty():
            return self._array[self._size - 1]
        return None
