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
    class Node:
        def __init__(self, d, n=None):
            self._data = d
            self._next = n

        def set_next(self, node):
            self._next = node

    def __init__(self):
        self._first = None
        self._last = None
        self._size = 0

    def add_first(self, e) -> None:
        if self._size == 0:
            self._first = self.Node(e)
            self._last = self._first
            self._size += 1
            return
        new_node = self.Node(e)
        new_node.set_next(self._first)
        self._first = new_node
        self._size += 1
        return

    def add_last(self, e) -> None:
        if self._size == 0:
            self._first = self.Node(e)
            self._last = self._first
            self._size += 1
            return
        new_node = self.Node(e)
        self._last.set_next(new_node)
        self._last = new_node
        self._size += 1
        return

    def first(self) -> object:
        if self._first:
            return self._first._data
        return None

    def last(self) -> object:
        if self._last:
            return self._last._data
        return None

    def remove_first(self) -> bool:
        if not self._first:
            return False
        elif self._size == 1:
            self._first = None
            self._last = None
            self._size -= 1
            return True
        self._first = self._first._next
        self._size -= 1
        return True

    def remove_last(self) -> bool:
        if not self._first:
            return False
        elif self._size == 1:
            self._first = None
            self._last = None
            self._size -= 1
            return True
        temp = self._first
        while temp._next != self._last:
            temp = temp._next
        self._last = temp
        temp._next = None
        self._size -= 1
        return True

    def empty(self) -> None:
        self._first = self._last = None
        self._size = 0

    def is_empty(self) -> bool:
        return self._size == 0

    def size(self) -> int:
        return self._size

    def remove_element(self, e):
        if self.is_empty():
            return

        if self._first._data == e:
            self.remove_first()

        current = self._first
        while current._next:
            if current._next._data == e:
                current._next = current._next._next
                if not current._next:
                    self._last = current
                self._size -= 1
                return True
            current = current._next

    def remove_min(self) -> object:
        if self.is_empty():
            return "No element to remove"

        t = self._first
        min_value = self._first._data

        if self._size == 1:
            self.remove_first()

        while t._next is not None:
            if min_value < t._next._data:
                t = t._next
            else:
                min_value = t._next._data
                t = t._next

        p = self._first
        k = self._first._next
        prev = self._first
        if self._first._data == min_value:
            self.remove_first()
            return p
        else:
            for i in range(self._size):
                if k._data == min_value:
                    prev._next = k._next
                    if k == self._last:
                        self._last = prev
                    self._size -= 1
                else:
                    prev = prev._next
                    k = k._next

        return min_value

    def add_at(self, el, index):
        if index < 0 or index > self._size:
            raise ValueError("Index out of boundaries")

        if index == self._size:
            new_node = self.Node(el)
            if self._last is None:
                self._first = new_node
                self._last = new_node
            else:
                self._last._next = new_node
                self._last = new_node
            self._size += 1
            return

        self._first = self.add_at_recursively(self._first, el, index, 0)
        self._size += 1

    @staticmethod
    def add_at_recursively(current, el, index, current_index):
        print(f"Called add_at_recusively with values {current._data}, {el}, {index}, {current_index}")
        if current_index == index:
            new_node = SingleLinkedList.Node(el)
            new_node._next = current
            return new_node

        current._next = SingleLinkedList.add_at_recursively(current._next, el, index, current_index + 1)
        return current

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


def binomial_coefficient_recursive(n, k):
    if k == 0 or k == n:
        return 1
    return binomial_coefficient_recursive(n - 1, k - 1) + binomial_coefficient_recursive(n - 1, k)


def binomial_coefficient_memoized(n, k, memo={}):
    if k == 0 or k == n:
        return 1
    if (n, k) in memo:
        return memo[(n, k)]

    memo[(n, k)] = binomial_coefficient_memoized(n - 1, k - 1, memo) + binomial_coefficient_memoized(n - 1, k, memo)
    return memo[(n, k)]


def binomial_coefficient_bottom_up(n, k):
    dp = [[0 for _ in range(k + 1)] for _ in range(n + 1)]

    for i in range(n + 1):
        for j in range(min(i, k) + 1):
            if j == 0 or j == i:
                dp[i][j] = 1
            else:
                dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j]

    return dp[n][k]


def knapsack_recursive(weights, values, W, n):
    if n == 0 or W == 0:
        return 0

    if weights[n - 1] > W:
        return knapsack_recursive(weights, values, W, n - 1)
    else:
        return max(
            values[n - 1] + knapsack_recursive(weights, values, W - weights[n - 1], n - 1),
            knapsack_recursive(weights, values, W, n - 1)
        )


def knapsack_memoized(weights, values, W, n, memo={}):
    if n == 0 or W == 0:
        return 0

    if (n, W) in memo:
        return memo[(n, W)]

    if weights[n - 1] > W:
        memo[(n, W)] = knapsack_memoized(weights, values, W, n - 1, memo)
    else:
        memo[(n, W)] = max(
            values[n - 1] + knapsack_memoized(weights, values, W - weights[n - 1], n - 1, memo),
            knapsack_memoized(weights, values, W, n - 1, memo)
        )

    return memo[(n, W)]


def knapsack_tabulation(weights, values, W):
    n = len(weights)
    dp = [[0 for w in range(W + 1)] for i in range(n + 1)]

    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                dp[i][w] = 0
            elif weights[i - 1] <= w:
                dp[i][w] = max(values[i - 1] + dp[i - 1][w - weights[i - 1]], dp[i - 1][w])
            else:
                dp[i][w] = dp[i - 1][w]

    return dp[n][W]


def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left_half = arr[:mid]
    right_half = arr[mid:]

    left_half = merge_sort(left_half)
    right_half = merge_sort(right_half)

    merged = merge(left_half, right_half)

    return merged


def merge(left, right):
    result = []
    left_idx, right_idx = 0, 0

    while left_idx < len(left) and right_idx < len(right):
        if left[left_idx] < right[right_idx]:
            result.append(left[left_idx])
            left_idx += 1
        else:
            result.append(right[right_idx])
            right_idx += 1

    while left_idx < len(left):
        result.append(left[left_idx])
        left_idx += 1

    while right_idx < len(right):
        result.append(right[right_idx])
        right_idx += 1

    return result


# Testing the function
arr = [38, 27, 43, 3, 9, 82, 10]
sorted_arr = merge_sort(arr)
print(sorted_arr)

if __name__ == '__main__':
    weights = [10, 20, 30]
    values = [60, 100, 120]
    W = 50
    tab = knapsack_tabulation(weights, values, W)
    print(tab)
    weights = [10, 20, 30]
    values = [60, 100, 120]
    W = 50
    n = len(weights)

    max_val_recursive = knapsack_recursive(weights, values, W, n)
    print(f"Recursive: Maximum value = {max_val_recursive}")

    max_val_memoized = knapsack_memoized(weights, values, W, n)
    print(f"Memoized: Maximum value = {max_val_memoized}")