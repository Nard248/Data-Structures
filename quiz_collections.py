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

    def __str__(self):
        return str(self._array)

    def resize(self) -> None:
        if self._size < len(self._array):
            return
        new_array = [None] * (len(self._array) * 2)

        for i in range(self._size):
            new_array[i] = self._array[i]

        self._array = new_array

        return

    def add_first(self, el: object) -> bool:
        self.resize()
        for i in range(self._size, 0, -1):
            self._array[i] = self._array[i - 1]
        self._array[0] = el
        self._size += 1
        return True

    def add_last(self, el: object) -> bool:
        self.resize()
        self._array[self._size] = el
        self._size += 1
        return True

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

    def remove_first(self) -> bool:
        if self.is_empty():
            return False
        for i in range(self._size - 1):
            self._array[i] = self._array[i + 1]
        self._array[self._size - 1] = None
        self._size -= 1
        return True

    def remove_last(self) -> bool:
        if self.is_empty():
            return False
        self._array[self._size - 1] = None
        self._size -= 1
        return True

    def remove_min(self):
        if self.is_empty():
            return

        min_index = 0
        for i in range(1, self._size):
            if self._array[i] < self._array[min_index]:
                min_index = i

        for i in range(min_index, self._size - 1):
            self._array[i] = self._array[i + 1]

        self._array[self._size - 1] = None
        self._size -= 1
        return

    def get_at(self, index) -> object:
        if index < 0 or index >= self._size:
            raise IndexError("Array list index out of range!")
        return self._array[index]

    class StepIterator:
        def __init__(self, array, step, size):
            self._array = array
            self._size = size
            self._step = step
            self._current = 0

        def __next__(self):
            if self._current >= self._size:
                raise StopIteration
            else:
                result = self._array[self._current]
                self._current += self._step
                return result

    def __iter__(self):
        return self.StepIterator(self._array, 1, self._size)


class SingleLinkedList(Collection):
    """List implementation using singly linked data."""

    def __init__(self):
        self._first: SingleLinkedList._Node = None
        self._last: SingleLinkedList._Node = None
        self._size: int = 0

    class _Node:
        def __init__(self, d, n=None):
            self._data: object = d
            self._next: SingleLinkedList._Node = n

    """ Task 2: Implement a method which removes the minimum element from the LinkedList. 
    In case if the list is empty, None should be returned."""

    def remove_min(self) -> object:
        pass

    """ Task 5: Implement a recursive method for which adds the element at given position. 
    The method should throw a ValueError exception in case if the index is out of boundaries.
    """

    def add_at(self, el: object, index: int) -> None:
        """A static private recursive helper method should be implemented for this task.
        The helper method should receive extra parameter of _Node type!"""
        pass

    def add_first(self, e) -> None:
        pass

    def add_last(self, e) -> None:
        if self.is_empty():
            self._last = self._first = SingleLinkedList._Node(e)
        else:
            self._last._next = SingleLinkedList._Node(e)
            self._last = self._last._next
        self._size += 1

    def remove_first(self) -> bool:
        pass

    def remove_last(self) -> bool:
        pass

    def first(self) -> object:
        if self._first:
            return self._first._data
        return None

    def last(self) -> object:
        if self._last:
            return self._last._data
        return None

    def empty(self) -> None:
        self._first = self._last = None
        self._size = 0

    def is_empty(self) -> bool:
        return self._size == 0

    def size(self) -> int:
        return self._size

    def __str__(self):
        current = self._first
        s = ""
        while current:
            s += str(current._data)
            if current._next:
                s += " -> "
            current = current._next
        return s


class DoubleLinkedList(Collection):
    """List implementation using doubly linked data."""

    def __init__(self):
        self._first: DoubleLinkedList.Node = None
        self._last: DoubleLinkedList.Node = None
        self._size: int = 0

    class _Node:
        def __init__(self, d, n=None, p=None):
            self._data: object = d
            self._next: DoubleLinkedList.Node = n
            self._prev: DoubleLinkedList.Node = p

    def add_first(self, e) -> None:
        pass

    def add_last(self, e) -> None:
        if self.is_empty():
            self._last = self._first = DoubleLinkedList._Node(e)
        else:
            self._last._next = DoubleLinkedList._Node(e, None, self._last)
            self._last = self._last._next
        self._size += 1

    def remove_first(self) -> bool:
        pass

    def remove_last(self) -> bool:
        pass

    def first(self) -> object:
        if self._first:
            return self._first._data
        return None

    def last(self) -> object:
        if self._last:
            return self._last._data
        return None

    def empty(self) -> None:
        self._first = self._last = None
        self._size = 0

    def is_empty(self) -> bool:
        return self._size == 0

    def size(self) -> int:
        return self._size

    def __str__(self):
        current = self._first
        s = ""
        while current:
            s += str(current._data)
            if current._next:
                s += " <-> "
            current = current._next
        return s


class LinkedStack(Stack):
    class _Node:
        def __init__(self, d=None, n=None):
            self._next = n
            self._data = d

    def __init__(self):
        self._first = None
        self._size = 0

    def push(self, e):
        self._first = LinkedStack._Node(e, self._first)
        self._size += 1

    def pop(self):
        if self._size == 0:
            return None
        t = self._first._data
        self._first = self._first._next
        self._size -= 1
        return t

    def top(self):
        if self._first:
            return self._first._data
        return None

    def empty(self) -> None:
        self._size = 0
        self._first = None

    def is_empty(self) -> bool:
        return self._size == 0

    def size(self) -> int:
        return self._size

    def __str__(self):
        current = self._first
        s = ""
        while current:
            s += str(current._data)
            if current._next:
                s += " <- "
            current = current._next
        return s


def get_max_element(stack: Stack) -> object:
    if stack.is_empty():
        return None

    top_elemet = stack.pop()
    print("Top Element", top_elemet)
    max_of_other = get_max_element(stack)
    print("Max of Other", max_of_other)
    print(stack)
    stack.push(top_elemet)

    if max_of_other is None:
        return top_elemet
    else:
        return max(top_elemet, max_of_other)


if __name__ == '__main__':
    stack = LinkedStack()
    stack.push(3)
    stack.push(1)
    stack.push(4)
    stack.push(1)
    stack.push(5)
    stack.push(9)
    print("Stack:", stack)
    print("Max Element:", get_max_element(stack))

