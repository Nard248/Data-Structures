from abc import ABC, abstractmethod
from my_online_shop import ProductType, AudioBook, PaperBook, Computer, Airpods
import collections


class CollectionADT(ABC):
    @abstractmethod
    def get_size(self):
        pass

    @abstractmethod
    def is_empty(self):
        pass

    @abstractmethod
    def empty(self):
        pass

    @abstractmethod
    def add(self, el):
        pass

    @abstractmethod
    def remove(self, el):
        pass

    @abstractmethod
    def printt(self):
        pass


class ListADT(CollectionADT):
    @abstractmethod
    def add_first(self, data):
        pass

    @abstractmethod
    def add_last(self, data):
        pass

    @abstractmethod
    def remove_first(self) -> bool:
        pass

    @abstractmethod
    def remove_last(self) -> bool:
        pass

    @abstractmethod
    def first(self):
        pass

    @abstractmethod
    def last(self):
        pass

    @abstractmethod
    def replace(self, el, new_el) -> bool:
        pass

    @abstractmethod
    def add_at(self, el, index):
        pass

    @abstractmethod
    def get_at(self, index):
        pass


class StackADT(CollectionADT):
    @abstractmethod
    def push(self, el):
        pass

    @abstractmethod
    def pop(self):
        pass

    @abstractmethod
    def top(self):
        pass


class QueueADT(CollectionADT):
    @abstractmethod
    def enqueue(self, el):
        pass

    @abstractmethod
    def dequeue(self):
        pass

    @abstractmethod
    def front(self):
        pass

    @abstractmethod
    def back(self):
        pass

    @abstractmethod
    def swap(self, el1, el2) -> bool:
        pass


class Deque(QueueADT):
    @abstractmethod
    def left_enqueue(self, el):
        pass

    @abstractmethod
    def right_dequeue(self):
        pass


class SetADT(CollectionADT):
    @abstractmethod
    def contains(self, el):
        pass

    @abstractmethod
    def keep_all(self, set):
        pass

    @abstractmethod
    def equals(self, set):
        pass

    @abstractmethod
    def get_element_at(self, index):
        pass


class MapADT(CollectionADT):
    @abstractmethod
    def put(self, key, value):
        pass

    @abstractmethod
    def get(self, key):
        pass

    @abstractmethod
    def key_set(self):
        pass


class SortedSet(CollectionADT):
    @abstractmethod
    def contains(self, value):
        pass

    @abstractmethod
    def equals(self, set):
        pass

    @abstractmethod
    def get_element_at(self, index):
        pass

    @abstractmethod
    def get_smallest_element(self):
        pass

    @abstractmethod
    def get_largest_element(self):
        pass


class SortedMap(CollectionADT):
    @abstractmethod
    def put(self, key, value):
        pass

    @abstractmethod
    def get(self, key):
        pass

    @abstractmethod
    def key_set(self):
        pass

    @abstractmethod
    def sub_map(self, key1, key2):
        pass


class WeightedGraphADT(CollectionADT):
    @abstractmethod
    def add_vertex(self, v):
        pass

    @abstractmethod
    def add_edge(self, v1, v2, w):
        pass

    @abstractmethod
    def remove_vertex(self, v):
        pass

    @abstractmethod
    def remove_edge(self, v1, v2):
        pass


class ArrayList(ListADT):
    def __init__(self):
        self._arr = [None]*10
        self._size = 0

    def resize(self, ):
        if self._size < len(self._arr):
            return
        new_array = [None] * (2 * len(self._arr))
        for i in range(len(self._arr)):
            new_array[i] = self._arr[i]
        self._arr = new_array

    def is_empty(self):
        return self._size == 0

    def empty(self):
        self._arr = [None] * len(self._arr)
        self._size = 0

    def first(self):
        return self._arr[0]

    def last(self):
        return self._arr[self._size-1]

    def add_first(self, data):
        self.resize()
        for i in range(self._size - 1, 0, -1):
            self._arr[i+1] = self._arr[i]
        self._arr[0] = data
        self._size += 1

    def get_size(self):
        return self._size

    def add_last(self, data):
        if len(self._arr) > self._size:
            self._arr[self._size] = data
            self._size += 1
        else:
            self.resize()
            self._arr[self._size] = data
            self._size += 1

    def add(self, data):
        ArrayList.add_last(self, data)

    def remove(self, el):
        pass

    def printt(self):
        for i in range(self._size):
            print(self._arr[i])

    def add_at(self, el, index):
        if index > self._size:
            return False
        ArrayList.resize(self)
        for i in range(self._size, index, -1):
            self._arr[i] = self._arr[i-1]
        self._arr[index] = el
        self._size += 1

    def get_at(self, index):
        if index >= len(self._arr):
            return False
        return self._arr[index]

    def remove_last(self):
        if self._size == 0:
            return False
        self._arr[self._size - 1] = None
        self._size -= 1
        return True

    def remove_first(self):
        if self.is_empty():
            return False
        for i in range(0, self._size-1):
            self._arr[i] = self._arr[i+1]
        self._size -= 1
        return True

    def replace(self, el, new_el):
        for i in range(self._size):
            if self._arr[i] == el:
                self._arr[i] = new_el
                return True
        return False

    class ArrayListIterator:
        def __init__(self, arr, s):
            self._arr = arr
            self._size = s
            self._counter = 0

        def __next__(self):
            self._counter += 1
            if self._counter == self._size + 1:
                raise StopIteration
            return self._arr[self._counter-1]

    class ArrayListEvenIterator:
        def __init__(self, arr, s):
            self._arr = arr
            self._size = s
            self._counter = 0

        def __next__(self):
            self._counter += 2
            if self._counter == self._size + 1:
                raise StopIteration
            return self._arr[self._counter-1]

    def __iter__(self):
        return ArrayList.ArrayListIterator(self._arr, self._size)


class LinkedList(ListADT):
    class Node:
        def __init__(self, data, next=None):
            self._data = data
            self._next = next

        def set_next(self, node):
            self._next = node

    def __init__(self):
        self._first = None
        self._last = None
        self._size = 0

    def get_size(self):
        return self._size

    def empty(self):
        while self._size != 0:
            self._first = self._first._next
            self._size -= 1
        return

    def is_empty(self):
        return self._size == 0

    def first(self):
        return self._first._data

    def last(self):
        return self._last._data

    def add_first(self, data):
        self._first = LinkedList.Node(data, self._first)
        if self.is_empty():
            self._last = self._first
        self._size += 1

    def add_last(self, data):
        new_node = LinkedList.Node(data, None)
        if self.is_empty():
            self._first = self._last = new_node
            self._size += 1
            return
        self._last._next = new_node
        self._last = new_node
        self._size += 1

    def add(self, data):
        LinkedList.add_last(self, data)

    def remove(self, el):
        pass

    def replace(self, el, new_el):
        curr = self._first
        while curr._next and curr._data != el:
            curr = curr._next
        if curr == self._last and self._last._data != el:
            return False
        curr._data = new_el
        return True

    def printt(self, linked_list):
        for el in linked_list:
            print(el)

    def remove_first(self):
        if self.is_empty():
            return False
        if self._size == 1:
            self._first = self._last = None
            self._size -= 1
            return True
        self._first = self._first._next
        self._size -= 1
        return True

    def remove_last(self):
        if self.is_empty():
            return False
        if self._size == 1:
            self._first = self._last = None
            self._size -= 1
            return True
        node = self._first
        prev = None
        while node._next:
            prev = node
            node = node._next
        self._last = prev
        self._last._next = None
        self._size -= 1
        return True

    def add_at(self, el, index):
        if index > self._size:
            return False
        elif index == 0:
            LinkedList.add_first(self, el)
        elif index == self._size:
            LinkedList.add_last(self, el)
        else:
            counter = 0
            curr = self._first
            while counter != index - 1:
                counter += 1
                curr = curr._next
            new_node = LinkedList.Node(el, curr._next)
            curr._next = new_node
            self._size += 1

    def get_at(self, index):
        if index >= self._size:
            return False
        if index == self._size - 1:
            return self._last
        counter = 0
        curr = self._first
        while counter != index:
            counter += 1
            curr = curr._next
        return curr

    def reverse_helper(self, node, prev_node):
        if node._next is None:
            node._next = prev_node
            return node
        next_node = node._next
        node._next = prev_node
        return self.reverse_helper(next_node, node)

    def reverse(self):
        if self.is_empty():
            return
        new_last = self._first
        new_last._next = None
        self._first = self.reverse_helper(self._first, None)
        self._last = new_last

    class LinkedListIterator:
        def __init__(self, first):
            self._current = first

        def __next__(self):
            if not self._current:
                raise StopIteration
            current = self._current._data
            self._current = self._current._next
            return current

    class LinkedListOddIterator:
        def __init__(self, first):
            self._current = first

        def __next__(self):
            if not self._current:
                raise StopIteration
            self._current = self._current._next
            if not self._current:
                raise StopIteration
            current = self._current._data
            self._current = self._current._next
            return current

    def __iter__(self):
        return LinkedList.LinkedListIterator(self._first)


class DoubleLinkedList(ListADT):
    class Node:
        def __init__(self, data, next=None, prev=None):
            self._data = data
            self._next = next
            self._prev = prev

        def set_next(self, node):
            self._next = node

    def __init__(self):
        self._first = None
        self._last = None
        self._size = 0

    def get_size(self):
        return self._size

    def empty(self):
        while self._size != 0:
            self._first = self._first._next
            self._size -= 1
        return

    def is_empty(self):
        return self._size == 0

    def first(self):
        return self._first._data

    def last(self):
        return self._last._data

    def remove(self, el):
        pass

    def add_first(self, data):
        new_node = DoubleLinkedList.Node(data=data, next=self._first)
        if self.is_empty():
            self._first = self._last = new_node
            self._size += 1
        else:
            self._first._prev = new_node
            self._first = new_node
            self._size += 1

    def add_last(self, data):
        new_node = DoubleLinkedList.Node(data, None, self._last)
        if self.is_empty():
            self._first = self._last = new_node
        else:
            self._last._next = new_node
            self._last = new_node
        self._size += 1

    def add(self, data):
        DoubleLinkedList.add_last(self, data)

    def replace(self, el, new_el):
        curr = self._first
        while curr._next and curr._data != el:
            curr = curr._next
        if curr == self._last and self._last._data != el:
            return False
        curr._data = new_el
        return True

    def printt(self):
        curr = self._first
        while curr:
            print(curr._data)
            curr = curr._next

    def remove_first(self):
        if self.is_empty():
            return False
        if self._size == 1:
            self._first = self._last = None
            self._size -= 1
            return True
        self._first = self._first._next
        self._first._prev = None
        self._size -= 1
        return True

    def remove_last(self):
        if self.is_empty():
            return False
        if self._size == 1:
            self._first = self._last = None
            self._size -= 1
            return True
        new_last = self._last._prev
        new_last._next = None
        self._last = new_last
        self._size -= 1
        return True

    def add_at(self, el, index):
        if index > self._size:
            return False
        elif index == 0:
            DoubleLinkedList.add_first(self, el)
        elif index == self._size:
            DoubleLinkedList.add_last(self, el)
        else:
            counter = 0
            curr = self._first
            while counter != index - 1:
                counter += 1
                curr = curr._next
            new_node = DoubleLinkedList.Node(el, curr._next, curr)
            curr._next = new_node
            new_node._next._prev = new_node

    def get_at(self, index):
        if index >= self._size:
            return False
        if index == self._size - 1:
            return self._last
        counter = 0
        curr = self._fisrt
        while counter != index:
            counter += 1
            curr = curr._next
        return curr

    def swap(self, el1, el2):
        if self.is_empty() or self._size == 1:
            return False
        curr = self._first
        n = 0
        while curr and n != 2:
            if curr._data == el1 or curr._data == el2:
                n += 1
            curr = curr._next
        if n != 2:
            return False
        n = 0
        curr = self._first
        while curr:
            if n == 0:
                if curr._data == el1:
                    already_swapped = el1
                    to_be_swapped = el2
                    curr._data = el2
                    n += 1
                elif curr._data == el2:
                    already_swapped = el2
                    to_be_swapped = el1
                    curr._data = el1
                    n += 1
            elif n == 1:
                if curr._data == to_be_swapped:
                    curr._data = already_swapped
                    break
            curr = curr._next
        return True


class LinkedListStack(StackADT):
    class Node:
        def __init__(self, data, next=None):
            self._data = data
            self._next = next

    def __init__(self):
        self._first = None
        self._size = 0

    def push(self, el):
        self._first = LinkedListStack.Node(el, self._first)
        self._size += 1

    def pop(self):
        if not self.is_empty():
            deleted_node = self._first
            self._first = self._first._next
            self._size -= 1
            return deleted_node
        return None

    def top(self):
        if not self.is_empty():
            return self._first._data

    def printt(self):
        curr = self._first
        while curr:
            print(curr._data)
            curr = curr._next

    def get_size(self):
        return self._size

    def is_empty(self):
        return self._size == 0

    def empty(self):
        while self._size != 0:
            self._first = self._first._next
            self._size -= 1
        return

    def add(self, el):
        LinkedListStack.push(self, el)

    def remove(self, el):
        pass

    class StackIterator:
        def __init__(self, first):
            self._current = first

        def __next__(self):
            if not self._current:
                raise StopIteration
            current = self._current._data
            self._current = self._current._next
            return current

    def __iter__(self):
        return LinkedListStack.StackIterator(self._first)


class ArrayQueue(QueueADT):
    def __init__(self, cap=10):
        self._arr = [None] * cap
        self._size = 0
        self._first_index = 0

    def is_empty(self):
        return self._size == 0

    def get_size(self):
        return self._size

    def remove(self, el):
        pass

    def add(self, el):
        self.enqueue(el)

    def back(self):
        pass

    def empty(self):
        pass

    def front(self):
        pass

    def printt(self):
        pass

    def swap(self):
        pass

    def enqueue(self, el):
        if self._size == len(self._arr):
            print('Queue capacity is full. Try later.')
            return
        self._arr[(self._first_index + self._size) % len(self._arr)] = el
        self._size += 1

    def dequeue(self):
        if self._size == 0:
            return None
        del_node = self._arr[self._first_index % len(self._arr)]
        self._arr[self._first_index % len(self._arr)] = None
        self._first_index = (self._first_index + 1) % len(self._arr)
        self._size -= 1
        return del_node

    def enqueue_at(self, el, index):
        if self._size == len(self._arr):
            print('Queue capacity is full. Try later.')
        for i in range(self._size, index, -1):
            self._arr[(self._first_index + i) % len(self._arr)] =\
                self._arr[(self._first_index + i - 1) % len(self._arr)]
        self._arr[(self._first_index + index) % len(self._arr)] = el
        self._size += 1

    class ArrayQueueIterator:
        def __init__(self, arr, s, first):
            self._arr = arr
            self._size = s
            self._counter = 0
            self._first = first

        def __next__(self):
            self._counter += 1
            if self._counter == self._size + 1:
                raise StopIteration
            return self._arr[(self._first + self._counter-1) % len(self._arr)]

    def __iter__(self):
        return ArrayQueue.ArrayQueueIterator(self._arr, self._size, self._first)


class DoubleLinkedListDeque(Deque):
    def __init__(self):
        self._queue = DoubleLinkedList()
        self._step = 3

    def get_size(self):
        return self._queue.get_size()

    def empty(self):
        self._queue.empty()

    def is_empty(self):
        return self._queue.is_empty()

    def enqueue(self, el):
        self._queue.add_last(el)
        self._queue._size += 1

    def remove(self, el):
        pass

    def dequeue(self):
        del_node = self._queue.first()
        self._queue.remove_first()
        self._queue._size -= 1
        return del_node

    def front(self):
        return self._queue.first()

    def back(self):
        return self._queue.last()

    def swap(self, el1, el2) -> bool:
        return self._queue.swap(el1, el2)

    def left_enqueue(self, el):
        self._queue.add_first(el)
        self._queue._size += 1

    def right_dequeue(self):
        del_node = self._queue.last()
        self._queue.remove_last()
        self._queue._size -= 1
        return del_node

    def add(self, el):
        DoubleLinkedListDeque.enqueue(self, el)

    def printt(self):
        self._queue.printt()

    def __call__(self, step):
        self._step = step
        return self

    class DequePositionIterator:
        def __init__(self, first, step):
            self._current = first
            self._step = step
            self._counter = 0

        def __next__(self):
            if not self._current:
                raise StopIteration
            el = self._current
            curr = self._current
            while curr._next and self._counter < self._step - 1:
                curr = curr._next
                self._counter += 1
            self._current = curr._next
            self._counter = 0
            return el._data

    def __iter__(self):
        return DoubleLinkedListDeque.DequePositionIterator(self._queue._first, self._step)


class ArrayDeque(QueueADT):
    def __init__(self, cap=10):
        self._arr = [None] * cap
        self._size = 0
        self._first_index = 0
        self._step = 3

    def is_empty(self):
        return self._size == 0

    def get_size(self):
        return self._size

    def empty(self):
        self._arr = [None] * len(self._arr)

    def enqueue(self, el):
        if self._size == len(self._arr):
            return False
        self._arr[(self._first_index + self._size) % len(self._arr)] = el
        self._size += 1

    def dequeue(self):
        if self._size == 0:
            return None
        del_node = self._arr[self._first_index % len(self._arr)]
        self._arr[self._first_index % len(self._arr)] = None
        self._first_index = (self._first_index + 1) % len(self._arr)
        self._size -= 1
        return del_node

    def front(self):
        return self._arr[self._first_index]

    def back(self):
        return self._arr[(self._first_index + self._size) % len(self._arr)]

    def swap(self, el1, el2) -> bool:
        if el1 not in self._arr or el2 not in self._arr:
            return False
        n = 0
        for i in range(0, self._size):
            if n == 2:
                break
            elif n == 0:
                if self._arr[(self._first_index + i) % len(self._arr)] == el1:
                    self._arr[(self._first_index + i) % len(self._arr)] = el2
                    already_swapped = el1
                    to_be_swapped = el2
                    n += 1
                elif self._arr[(self._first_index + i) % len(self._arr)] == el2:
                    self._arr[(self._first_index + i) % len(self._arr)] = el1
                    already_swapped = el2
                    to_be_swapped = el1
                    n += 1
            elif n == 1:
                if self._arr[(self._first_index + i) % len(self._arr)] == to_be_swapped:
                    self._arr[(self._first_index + i) %
                              len(self._arr)] = already_swapped
                    break

    def left_enqueue(self, el):
        if self._size == len(self._arr):
            return False
        self._arr[(self._first_index - 1) % len(self._arr)] = el
        self._first_index = (self._first_index - 1) % len(self._arr)
        self._size += 1

    def right_dequeue(self):
        if self._size == 0:
            return None
        del_node = self._arr[(self._first_index + self._size) % len(self._arr)]
        self._arr[(self._first_index + self._size) % len(self._arr)] = None
        self._size -= 1
        return del_node

    def add(self, el):
        return ArrayDeque.enqueue(self, el)

    def printt(self):
        for i in range(0, self._size):
            print(self._arr[(self._first_index + i) % len(self._arr)])

    def remove(self, el):
        if not el in self._arr:
            return False
        for i in range(0, self._size):
            if self._arr[(self._first_index + i) % len(self._arr)] == el:
                del_ind = i
                break
        for i in range(del_ind, self._size):
            self._arr[(self._first_index + i) % len(self._arr)
                      ] = self._arr[(self._first_index + i + 1) % len(self._arr)]

    def __call__(self, step):
        self._step = step
        return self

    class ArrayDequePositionIterator:
        def __init__(self, arr, size, first, step):
            self._arr = arr
            self._size = size
            self._step = step
            self._first_index = first
            self._counter = 0

        def __next__(self):
            if self._counter >= self._size:
                raise StopIteration
            el = self._arr[(self._first_index + self._counter) %
                           len(self._arr)]
            self._counter += self._step
            if el is None:
                raise StopIteration
            return el

    def __iter__(self):
        return ArrayDeque.ArrayDequePositionIterator(self._arr, self._size, self._first_index, self._step)


class HashSet(SetADT):
    class Node:
        def __init__(self, data=None, next=None):
            self._data = data
            self._next = next

    def __init__(self):
        self._hash_table = [None] * 26
        self._size = 0

    def _hash(el: str):
        return hash(el) % 26
        # return (ord(el[0].lower()) - 97) % 26

    def get_size(self):
        return self._size

    def is_empty(self):
        return self._size == 0

    def empty(self):
        self._size = 0
        for i in range(len(self._hash_table)):
            self._hash_table[i] = None

    def add(self, el):
        hash_index = HashSet._hash(el)
        if self._hash_table[hash_index] is None:
            self._hash_table[hash_index] = HashSet.Node(el)
        else:
            curr = self._hash_table[hash_index]
            prev = None
            while curr:
                if curr._data == el:
                    return False
                prev = curr
                curr = curr._next
            prev._next = HashSet.Node(el)
        self._size += 1
        return True

    def remove(self, el):
        hash_index = HashSet._hash(el)
        if self._hash_table[hash_index] is None:
            return False
        curr = self._hash_table[hash_index]
        prev = None
        while curr:
            if curr._data == el:
                if prev:
                    prev._next = curr._next
                else:
                    self._hash_table[hash_index] = self._hash_table[hash_index]._next
                self._size -= 1
                return True
            prev = curr
            curr = curr._next
        return False

    def remove_after(self, el):
        for i in range(0, len(self._hash_table)):
            next = None
            curr = self._hash_table[i]
            while curr:
                if curr._data == el and curr._next:
                    next = curr._next
                    curr._next = next._next
                    self._size -= 1
                    return True
                if curr._data == el:
                    for j in range(i + 1, len(self._hash_table)):
                        if self._hash_table[j]:
                            self._hash_table[j] = self._hash_table[j]._next
                            self._size -= 1
                            return True
                curr = curr._next
        return False

    def add_after(self, el, data):
        if HashSet._hash(el._data) != HashSet._hash(data):
            return False
        hash_index = HashSet._hash(el)
        curr = self._hash_table[hash_index]
        while curr and curr._data != el:
            curr = curr._next
        if not curr:
            return False
        new_node = HashSet.Node(data)
        new_node._next = curr._next
        curr._next = new_node
        self._size += 1
        return True

    def contains(self, el):
        hash_index = HashSet._hash(el)
        if self._hash_table[hash_index] is None:
            return False
        curr = self._hash_table[hash_index]
        while curr:
            if curr._data == el:
                return True
            curr = curr._next
        return False

    def keep_all(self, s):
        if self.is_empty():
            return
        if s.is_empty():
            return self.empty()
        for i in range(len(self._hash_table)):
            curr = self._hash_table[i]
            while curr:
                if not s.contains(curr._data):
                    self.remove(curr._data)
                curr = curr._next
        return

    def printt(self):
        output = ''
        for i in range(len(self._hash_table)):
            curr = self._hash_table[i]
            while curr:
                output += ', ' + str(curr._data)
                curr = curr._next
        print('(' + output[2:] + ')')

    def equals(self, set):
        if self._size != len(set):
            return False
        for i in range(len(self._hash_table)):
            curr = self._hash_table[i]
            while curr:
                if curr._data not in set:
                    return False
                curr = curr._next
        return True

    def get_element_at(self, index):
        if self.is_empty() or index < 0 or index >= self._size:
            return False
        counter = 0
        for i in range(len(self._hash_table)):
            curr = self._hash_table[i]
            while counter != index and curr:
                curr = curr._next
                counter += 1
            if counter == index and curr:
                return curr._data

    def remove_element_at(self, index):
        if self.is_empty() or index < 0 or index >= self._size:
            return False
        counter = 0
        for i in range(len(self._hash_table)):
            curr = self._hash_table[i]
            prev = None
            while counter != index and curr:
                prev = curr
                curr = curr._next
                counter += 1
            if counter == index and curr:
                prev._next = curr._next
                self._size -= 1
                return curr._data

    class ForwardIterator:
        def __init__(self, hash_table):
            self._hash_table = hash_table
            self.curr = None
            for i in range(len(self._hash_table)):
                if self._hash_table[i]:
                    self._curr = self._hash_table[i]
                    break

        def __next__(self):
            if not self._curr:
                raise StopIteration
            el = self._curr._data
            self._curr = self._curr._next
            if self._curr == None:
                for i in range(HashSet._hash(el) + 1, len(self._hash_table)):
                    if self._hash_table[i]:
                        self._curr = self._hash_table[i]
                        break
            return el

    def __iter__(self):
        return HashSet.ForwardIterator(self._hash_table)

    def product_type_iterator(self, type: ProductType):
        return ProductTypeIterator(self, type)


class ProductTypeIterator:
    def __init__(self, hash_set, type):
        self._hash_table = hash_set._hash_table
        if type == ProductType.AudioBook:
            self._type = AudioBook
        elif type == ProductType.PaperBook:
            self._type = PaperBook
        elif type == ProductType.Computer:
            self._type = Computer
        else:
            self._type = Airpods
        self._curr = None
        for i in range(len(self._hash_table)):
            self._curr = self._hash_table[i]
            while self._curr and not isinstance(self._curr._data, self._type):
                self._curr = self._curr._next
            if self._curr and isinstance(self._curr._data, self._type):
                break

    def __next__(self):
        if not self._curr:
            raise StopIteration
        el = self._curr._data
        self._curr = self._curr._next
        while self._curr and not isinstance(self._curr._data, self._type):
            self._curr = self._curr._next
        if self._curr is None:
            for i in range(HashSet._hash(el) + 1, len(self._hash_table)):
                self._curr = self._hash_table[i]
                while self._curr and not isinstance(self._curr._data, self._type):
                    self._curr = self._curr._next
                if self._curr and isinstance(self._curr._data, self._type):
                    break
        return el


class HashMap(MapADT):
    class Entry:
        def __init__(self, key, value, next=None):
            self._key = key
            self._value = value
            self._next = next

    def __init__(self):
        self._hash_table = [None] * 26
        self._size = 0

    def _hash(el: str):
        return (ord(el[0].lower()) - 97) % 26

    def get_size(self):
        return self._size

    def is_empty(self):
        return self._size == 0

    def empty(self):
        self._size = 0
        for i in range(len(self._hash_table)):
            self._hash_table[i] = None

    def add(self, key):
        self.put(key, None)

    def remove(self, el):
        hash_index = HashMap._hash(el)
        if self._hash_table[hash_index] is None:
            return False
        curr = self._hash_table[hash_index]
        prev = None
        while curr:
            if curr._key == el:
                val = curr._value
                if prev:
                    prev._next = curr._next
                else:
                    self._hash_table[hash_index] = self._hash_table[hash_index]._next
                self._size -= 1
                return val
            prev = curr
            curr = curr._next
        return False

    def printt(self):
        for entry in self._hash_table:
            print(entry._key, entry._value)

    def put(self, key, value):
        hash_index = HashMap._hash(key)
        if not self._hash_table[hash_index]:
            self._hash_table[hash_index] = HashMap.Entry(key, value)
            self._size += 1
            return value
        curr = self._hash_table[hash_index]
        while curr:
            if curr._key == key:
                old_val = curr._value
                curr._value = value
                return old_val
            curr = curr._next
        self._hash_table[hash_index] = HashMap.Entry(
            key, value, self._hash_table[hash_index])
        self._size += 1
        return value

    def get(self, key):
        hash_index = HashMap._hash(key)
        curr = self._hash_table[hash_index]
        while curr:
            if curr._key == key:
                return curr._value
            curr = curr._next
        return None

    def key_set(self):
        key_set = HashSet()
        col_iter = self.key_iter()
        while True:
            try:
                el = next(col_iter)
                key_set.add(el)
            except StopIteration:
                break
        return key_set

    class EntryForwardIterator:
        def __init__(self, hash_table):
            self._hash_table = hash_table
            self.curr = None
            for i in range(len(self._hash_table)):
                if self._hash_table[i]:
                    self._curr = self._hash_table[i]
                    break

        def __next__(self):
            if not self._curr:
                raise StopIteration
            el = self._curr
            self._curr = self._curr._next
            if self._curr == None:
                for i in range(HashMap._hash(el._key) + 1, len(self._hash_table)):
                    if self._hash_table[i]:
                        self._curr = self._hash_table[i]
                        break
            return el

    def __iter__(self):
        return HashMap.EntryForwardIterator(self._hash_table)

    class KeyForwardIterator:
        def __init__(self, hash_table):
            self._hash_table = hash_table
            self.curr = None
            for i in range(len(self._hash_table)):
                if self._hash_table[i]:
                    self._curr = self._hash_table[i]
                    break

        def __next__(self):
            if not self._curr:
                raise StopIteration
            el = self._curr._key
            self._curr = self._curr._next
            if self._curr == None:
                for i in range(HashMap._hash(el) + 1, len(self._hash_table)):
                    if self._hash_table[i]:
                        self._curr = self._hash_table[i]
                        break
            return el

    def key_iter(self):
        return HashMap.KeyForwardIterator(self._hash_table)

    class ColumnForwardIterator:
        def __init__(self, hash_table):
            self._hash_table = hash_table
            self._queue = ArrayQueue()
            for i in range(len(self._hash_table)):
                if self._hash_table[i]:
                    self._queue.enqueue(self._hash_table[i])

        def __next__(self):
            if self._queue.is_empty():
                raise StopIteration
            el = self._queue.dequeue()
            if el._next:
                self._queue.enqueue(el._next)
            return el._key

    def map_key_col_iter(self):
        return HashMap.ColumnForwardIterator(self._hash_table)

    class KeyOddPositionIterator:
        def __init__(self, hash_table):
            self._hash_table = hash_table
            self.curr = None
            for i in range(len(self._hash_table)):
                if self._hash_table[i]:
                    self._curr = self._hash_table[i]
                    break
            self._curr = self._curr._next
            if self._curr == None:
                for i in range(HashMap._hash(self._current._key) + 1, len(self._hash_table)):
                    if self._hash_table[i]:
                        self._curr = self._hash_table[i]
                        break

        def __next__(self):
            if not self._curr:
                raise StopIteration
            el, el2 = self._curr._key, self._curr
            self._curr = self._curr._next
            if self._curr == None:
                for i in range(HashMap._hash(el) + 1, len(self._hash_table)):
                    if self._hash_table[i]:
                        self._curr = self._hash_table[i]
                        break
            if self._curr:
                el2 = self._curr._key
            if self._curr:
                self._curr = self._curr._next
            if self._curr == None:
                for i in range(HashMap._hash(el2) + 1, len(self._hash_table)):
                    if self._hash_table[i]:
                        self._curr = self._hash_table[i]
                        break
            return el

    def odd_key_iter(self):
        return HashMap.KeyOddPositionIterator(self._hash_table)


class TreeSet(SortedSet):
    class Node:
        def __init__(self, data, left=None, right=None, parent=None):
            self._data = data
            self._left = left
            self._right = right
            self._parent = parent

    def __init__(self):
        self._root = None
        self._size = 0

    def depth(self, node):
        counter = 0
        curr = node
        while curr and curr._parent:
            counter += 1
            curr = curr._parent
        return counter

    def get_all_nodes(self):
        nodes = []
        self.get_all_nodes_rec_preorder(self._root, nodes)
        return nodes

    def get_all_nodes_rec_preorder(self, node, nodes_list):
        if not node:
            return
        nodes_list.append(node)
        self.get_all_nodes_rec_preorder(node._left, nodes_list)
        self.get_all_nodes_rec_preorder(node._right, nodes_list)

    def height(self):
        nodes = self.get_all_nodes()
        max_depth = 0
        for n in nodes:
            if max_depth < self.depth(n):
                max_depth = self.depth(n)
        return max_depth

    def node_height(self, node):
        if not node:
            return -1
        return 1 + max(self.node_height(node._left), self.node_height(node._right))

    def get_size(self):
        return self._size

    def is_empty(self):
        return self._root == None

    def empty(self):
        self._root = None
        self._size = 0

    def add(self, value):
        if not self._root:
            self._root = TreeSet.Node(value)
            self._size += 1
            return True
        curr = self._root
        while curr:
            if curr._data == value:
                return False
            if curr._data < value:
                if curr._right:
                    curr = curr._right
                else:
                    curr._right = TreeSet.Node(data=value, parent=curr)
                    self._size += 1
                    self.rebalance(curr)
                    return True
            else:
                if curr._left:
                    curr = curr._left
                else:
                    curr._left = TreeSet.Node(data=value, parent=curr)
                    self._size += 1
                    self.rebalance(curr)
                    return True
        return False

    def remove(self, value):
        if not self._root:
            return False
        curr = self._root
        while curr:
            if curr._data == value:
                if curr._right == None and curr._left == None:
                    parent = curr._parent
                    if not parent:
                        self._root = None
                    else:
                        if curr._data < parent._data:
                            parent._left = None
                        else:
                            parent._right = None
                        self.rebalance(parent)
                elif curr._left and curr._right:
                    largest = self.get_largest_element_node(curr._left)
                    val = largest._data
                    self.remove(val)
                    curr._data = val
                else:
                    if curr._right:
                        child = curr._right
                    else:
                        child = curr._left
                    parent = curr._parent
                    if not parent:
                        self._root = child
                    elif curr._data < parent._data:
                        parent._left = child
                    else:
                        parent._right = child
                    child._parent = parent
                    self.rebalance(parent)
                self._size -= 1
                return True
            elif curr._data < value:
                curr = curr._right
            else:
                curr = curr._left

    def print_inorder(self):
        self.print_inorder_rec(self._root)

    def print_inorder_rec(self, n):
        if not n:
            return
        self.print_inorder_rec(n._left)
        print(n._data)
        self.print_inorder_rec(n._right)

    def print_level_order(self):
        q = collections.deque()
        q.append(self._root)
        while len(q) != 0:
            next_level = collections.deque()
            for el in q:
                if el._parent:
                    print(el._data, f'({el._parent._data})', end=' ')
                else:
                    print(el._data, end=' ')
                if el._left:
                    next_level.append(el._left)
                if el._right:
                    next_level.append(el._right)
            q = next_level
            print('\n')

    def printt(self):
        self.print_level_order()

    def rebalance(self, node):
        if not node:
            return
        lh = self.node_height(node._left)
        rh = self.node_height(node._right)
        if abs(lh - rh) > 1:
            if lh > rh:
                y = node._left
                if self.node_height(y._right) > self.node_height(y._left):
                    self.rotate_left(y)
                self.rotate_right(node)
            else:
                y = node._right
                if self.node_height(y._left) > self.node_height(y._right):
                    self.rotate_right(y)
                self.rotate_left(node)
        self.rebalance(node._parent)

    def rotate_right(self, node):
        if self._root == node:
            self._root = node._left
        y_left_n = node._left
        T_y_right = y_left_n._right
        parent_n = node._parent
        y_left_n._right = node
        node._parent = y_left_n
        node._left = T_y_right
        if T_y_right:
            T_y_right._parent = node
        y_left_n._parent = parent_n
        if parent_n and node == parent_n._right:
            parent_n._right = y_left_n
        elif parent_n and node == parent_n._left:
            parent_n._left = y_left_n

    def rotate_left(self, node):
        if self._root == node:
            self._root = node._right
        y_right_n = node._right
        T_y_left = y_right_n._left
        parent_n = node._parent
        y_right_n._left = node
        node._parent = y_right_n
        node._right = T_y_left
        if T_y_left:
            T_y_left._parent = node
        y_right_n._parent = parent_n
        if parent_n and node == parent_n._left:
            parent_n._left = y_right_n
        elif parent_n and node == parent_n._right:
            parent_n._right = y_right_n

    def contains(self, el):
        for node in self:
            if node == el:
                return True
        return False

    def equals(self, set):
        if self._size != len(set) - 1:
            return False
        for node in self:
            if node not in set:
                return False
        return True

    def get_element_at(self, index):
        if index < 0 or index >= self._size:
            return None
        el = None
        counter = -1
        for node in self:
            if counter != index:
                el = node
                counter += 1
            else:
                break
        return el

    def get_largest_element_node(self, node):
        max = node
        while node and node._right:
            max = node._right
            node = node._right
        return max

    def get_smallest_element_node(self, node):
        min = node
        while node and node._left:
            min = node._left
            node = node._left
        return min

    def get_largest_element(self):
        node = self._root
        max = node
        while node and node._right:
            max = node._right
            node = node._right
        return max

    def get_smallest_element(self):
        node = self._root
        min = node
        while node and node._left:
            min = node._left
            node = node._left
        return min

    class InorderIterator:
        def __init__(self, root):
            self._current = TreeSet.get_smallest_element_node(self, root)

        def __next__(self):
            if not self._current:
                raise StopIteration
            curr = self._current
            if self._current._right:
                self._current = TreeSet.get_smallest_element_node(
                    self, self._current._right)
            else:
                parent = self._current._parent
                while parent and parent._right == self._current:
                    self._current = parent
                    parent = parent._parent
                self._current = parent
            return curr._data

    def __iter__(self):
        return TreeSet.InorderIterator(self._root)


class TreeMap(SortedMap):
    class Entry:
        def __init__(self, key, value, left=None, right=None, parent=None, color='R'):
            self._key = key
            self._value = value
            self._left = left
            self._right = right
            self._parent = parent
            self._color = color

    def __init__(self):
        self._root = None
        self._size = 0

    def depth(self, node):
        counter = 0
        curr = node
        while curr and curr._parent:
            counter += 1
            curr = curr._parent
        return counter

    def height(self):
        return self.node_height(self._root)

    def node_height(self, node):
        if not node:
            return -1
        return 1 + max(self.node_height(node._left), self.node_height(node._right))

    def get_size(self):
        return self._size

    def is_empty(self):
        return self._root == None

    def empty(self):
        self._root = None
        self._size = 0

    def print_level_order(self):
        q = collections.deque()
        q.append(self._root)
        while len(q) != 0:
            next_level = collections.deque()
            for el in q:
                s = ',' + el._color
                if el._parent:
                    s += '(' + str(el._parent._key) + ')'
                print(el._key, s, end=' ')
                if el._left:
                    next_level.append(el._left)
                if el._right:
                    next_level.append(el._right)
            q = next_level
            print('\n')

    def printt(self):
        self.print_level_order()

    def print_for_graphs(self):
        text = '{'
        for key, value in self:
            text += str(key) + ': '
            if isinstance(value, TreeMap):
                text += '{'
                for key1, value1 in value:
                    text += str(key1) + ': ' + str(value1) + ', '
                text = text[:-2]
                text += '}, '
            else:
                text += str(value) + ', '
        text = text[:-2]
        text += '}'
        print(text)

    def add(self, el):
        self.put(el, None)

    def remove(self, key):
        curr = self._root
        while curr:
            if curr._key == key:
                if not curr._left and not curr._right:
                    if curr._color == 'B':
                        curr._color = 'D'
                        self.resolve_deficit(curr)
                    val = curr._value
                    parent = curr._parent
                    if not parent:
                        self._root = None
                    else:
                        if curr._key < parent._key:
                            parent._left = None
                        else:
                            parent._right = None
                    self._size -= 1
                    return val
                else:
                    if curr._left:
                        rem = self.get_largest_element_node(curr._left)
                    else:
                        rem = self.get_smallest_element_node(curr._right)
                    key_delete = rem._key
                    val_delete = rem._value
                    self.remove(key_delete)
                    curr._key = key_delete
                    curr._value = val_delete
            elif curr._key < key:
                curr = curr._right
            else:
                curr = curr._left
        return None

    def put(self, key, value):
        if self._root == None:
            self._root = TreeMap.Entry(key=key, value=value, color='B')
            self._size += 1
            return value
        curr = self._root
        while curr:
            if curr._key == key:
                old_val = curr._value
                curr._value = value
                return old_val
            if curr._key < key:
                if curr._right:
                    curr = curr._right
                else:
                    curr._right = TreeMap.Entry(
                        key=key, value=value, parent=curr)
                    self._size += 1
                    self.rebalance(curr._right)
                    return value
            else:
                if curr._left:
                    curr = curr._left
                else:
                    curr._left = TreeMap.Entry(
                        key=key, value=value, parent=curr)
                    self._size += 1
                    self.rebalance(curr._left)
                    return value
        return None

    def has_black_children(self, node):
        if node._left and node._left._color == 'R':
            return False
        if node._right and node._right._color == 'R':
            return False
        return True

    def resolve_deficit(self, node):
        parent = node._parent
        if not parent:
            node._color = 'B'
            return
        if node == parent._right:
            sibling = parent._left
        else:
            sibling = parent._right
        if sibling._color == 'R':
            if node == parent._left:
                sibling._color, parent._color = parent._color, sibling._color
                self.rotate_left(parent)
                sibling = parent._right
            else:
                sibling._color, parent._color = parent._color, sibling._color
                self.rotate_right(parent)
                sibling = parent._left
        if self.has_black_children(sibling):
            sibling._color = 'R'
            node._color = 'B'
            if parent._color == 'R':
                parent._color = 'B'
            else:
                parent._color = 'D'
                self.resolve_deficit(parent)
        else:
            if sibling == parent._left:
                if not sibling._left or sibling._left._color == 'B':
                    sibling._color, sibling._right._color = sibling._right._color, sibling._color
                    self.rotate_left(sibling)
                    sibling = sibling._right
                sibling._color, parent._color = parent._color, sibling._color
                self.rotate_right(parent)
                sibling._left._color = 'B'
            else:
                if not sibling._right or sibling._right._color == 'B':
                    sibling._color, sibling._left._color = sibling._left._color, sibling._color
                    self.rotate_right(sibling)
                    sibling = sibling._left
                sibling._color, parent._color = parent._color, sibling._color
                self.rotate_left(parent)
                sibling._right._color = 'B'

    def rotate_right(self, node):
        if self._root == node:
            self._root = node._left
        y_left_n = node._left
        T_y_right = y_left_n._right
        parent_n = node._parent
        y_left_n._right = node
        node._parent = y_left_n
        node._left = T_y_right
        if T_y_right:
            T_y_right._parent = node
        y_left_n._parent = parent_n
        if parent_n and node == parent_n._right:
            parent_n._right = y_left_n
        elif parent_n and node == parent_n._left:
            parent_n._left = y_left_n

    def rotate_left(self, node):
        if self._root == node:
            self._root = node._right
        y_right_n = node._right
        T_y_left = y_right_n._left
        parent_n = node._parent
        y_right_n._left = node
        node._parent = y_right_n
        node._right = T_y_left
        if T_y_left:
            T_y_left._parent = node
        y_right_n._parent = parent_n
        if parent_n and node == parent_n._left:
            parent_n._left = y_right_n
        elif parent_n and node == parent_n._right:
            parent_n._right = y_right_n

    def rebalance(self, node):
        if not node:
            return
        if node._color == 'B':
            return
        parent = node._parent
        if not parent:
            node._color = 'B'
            return
        if parent._color == 'B':
            return
        g_parent = parent._parent
        if parent == g_parent._left:
            uncle = g_parent._right
        else:
            uncle = g_parent._left
        if uncle and uncle._color == 'R':
            uncle._color = 'B'
            parent._color = 'B'
            g_parent._color = 'R'
            self.rebalance(g_parent)
            return
        if parent == g_parent._right:
            if node == parent._left:
                self.rotate_right(parent)
            g_parent._right._color, g_parent._color = g_parent._color, g_parent._right._color
            self.rotate_left(g_parent)
        else:
            if parent == g_parent._left:
                if node == parent._right:
                    self.rotate_left(parent)
                g_parent._left._color, g_parent._color = g_parent._color, g_parent._left._color
                self.rotate_right(g_parent)

    def get(self, key):
        curr = self._root
        while curr:
            if curr._key < key:
                curr = curr._right
            elif curr._key > key:
                curr = curr._left
            else:
                return curr._value
        return None

    def key_set(self):
        key_set = TreeSet()
        for key, value in self:
            key_set.add(key)
        return key_set

    def sub_map(self, key1, key2):
        if key1 > key2:
            return False
        new_map = TreeMap()
        for key, value in self:
            if key >= key1 and key <= key2:
                new_map.put(key, value)
        return new_map

    def get_largest_element_node(self, node):
        max = node
        while node and node._right:
            max = node._right
            node = node._right
        return max

    def get_smallest_element_node(self, node):
        min = node
        while node and node._left:
            min = node._left
            node = node._left
        return min

    def is_tree_complete(self):
        if not self._root:
            return True
        smallest = self.get_smallest_element_node(self._root)
        leftest_nodes_depth = self.depth(smallest)
        min_depth = leftest_nodes_depth

        inorder_entry = self.inorder_iter()
        while True:
            try:
                entry = next(inorder_entry)
                if entry._right and not entry._left:
                    return False
                if not entry._right and entry._left:
                    if self.depth(entry._left) != leftest_nodes_depth:
                        return False
                if not entry._left and not entry._right:
                    if self.depth(entry) - min_depth == -1:
                        min_depth -= 1
                    elif self.depth(entry) != min_depth:
                        return False
            except StopIteration:
                break
        if leftest_nodes_depth - min_depth > 1:
            return False
        return True

    def is_tree_left_skewed(self):
        if not self._root:
            return True
        curr = self._root
        while curr:
            if curr._right:
                return False
            curr = curr._left
        return True

    def get_set_of_unique_values(self):
        val_set = TreeSet()
        for key, value in self:
            val_set.add(value)
        return val_set

    def higher_entry(self, key):
        curr = self.get_smallest_element_node(self._root)
        while curr and curr._key <= key:
            if curr._right:
                curr = TreeMap.get_smallest_element_node(self, curr._right)
            else:
                parent = curr._parent
                while parent and parent._right == curr:
                    curr = parent
                    parent = parent._parent
                curr = parent
        return curr

    def contains(self, k):
        for key, value in self:
            if key == k:
                return True
        return False

    class InorderEntryIterator:
        def __init__(self, node):
            self._current = node

        def get_next_element(self, node):
            if not node:
                return None
            if node._right:
                return TreeMap.get_smallest_element_node(self, node._right)
            else:
                parent = node._parent
                while parent and parent._right == node:
                    node = parent
                    parent = parent._parent
                return parent

        def __next__(self):
            if not self._current:
                raise StopIteration
            el = self._current
            self._current = self.get_next_element(self._current)
            return el

    def inorder_entry_iter(self):
        return TreeMap.InorderEntryIterator(self.get_smallest_element_node(self._root))

    class InorderIterator:
        def __init__(self, node):
            self._current = node

        def get_next_element(self, node):
            if not node:
                return None
            if node._right:
                return TreeMap.get_smallest_element_node(self, node._right)
            else:
                parent = node._parent
                while parent and parent._right == node:
                    node = parent
                    parent = parent._parent
                return parent

        def __next__(self):
            if not self._current:
                raise StopIteration
            el = self._current
            self._current = self.get_next_element(self._current)
            return el._key, el._value

    def __iter__(self):
        return TreeMap.InorderIterator(self.get_smallest_element_node(self._root))

    def inorder_iter(self):
        return TreeMap.InorderIterator(self.get_smallest_element_node(self._root))

    class InorderReverseIterator:
        def __init__(self, node):
            self._current = node

        def get_next_element(self, node):
            if not node:
                return None
            if node._left:
                return TreeMap.get_largest_element_node(self, node._left)
            else:
                parent = node._parent
                while parent and parent._left == node:
                    node = parent
                    parent = parent._parent
                return parent

        def __next__(self):
            if not self._current:
                raise StopIteration
            el = self._current
            self._current = self.get_next_element(self._current)
            return el._key, el._value

    def inorder_reverse_iter(self):
        return TreeMap.InorderReverseIterator(self.get_largest_element_node(self._root))

    class PreorderIterator:
        def __init__(self, root):
            self._current = root

        def get_next_element(self, node):
            if node == None:
                return None
            if node._left:
                return node._left
            elif node._right:
                return node._right
            else:
                parent = node._parent
                while parent and parent._right and parent._right == node:
                    node = parent
                    parent = parent._parent
                if parent and parent._right:
                    return parent._right
                elif parent:
                    while parent and not parent._right:
                        parent = parent._parent
                    if parent:
                        return parent._right
                    else:
                        return None
                else:
                    return None

        def __next__(self):
            if not self._current:
                raise StopIteration
            el = self._current
            self._current = self.get_next_element(self._current)
            return el._key, el._value

    def preorder_iter(self):
        return TreeMap.PreorderIterator(self._root)

    class PreorderReverseIterator:
        def __init__(self, root):
            self._current = root
            while self._current and (self._current._left or self._current._right):
                if self._current._right:
                    self._current = self._current._right
                else:
                    self._current = self._current._left

        def get_next_element(self, node):
            if not node or not node._parent:
                return None
            elif node._parent._left and node != node._parent._left:
                node = node._parent._left
                while node and (node._right or node._left):
                    if node._right:
                        node = node._right
                    else:
                        node = node._left
                return node
            elif node == node._parent._left or (node == node._parent._right and not node._parent._left):
                return node._parent

        def __next__(self):
            if not self._current:
                raise StopIteration
            el = self._current
            self._current = self.get_next_element(self._current)
            return el._key, el._value

    def preorder_reverse_iter(self):
        return TreeMap.PreorderReverseIterator(self._root)

    class PreorderOddPositionIterator:
        def __init__(self, root):
            self._current = self.get_next_element(root)

        def get_next_element(self, node):
            if node == None:
                return None
            if node._left:
                return node._left
            elif node._right:
                return node._right
            else:
                parent = node._parent
                while parent and parent._right and parent._right == node:
                    node = parent
                    parent = parent._parent
                if parent and parent._right:
                    return parent._right
                elif parent:
                    while parent and not parent._right:
                        parent = parent._parent
                    if parent:
                        return parent._right
                    else:
                        return None
                else:
                    return None

        def __next__(self):
            if not self._current:
                raise StopIteration
            el = self._current
            self._current = self.get_next_element(self._current)
            self._current = self.get_next_element(self._current)
            return el._key, el._value

    def preorder_odd_iter(self):
        return TreeMap.PreorderOddPositionIterator(self._root)

    class PostorderIterator:
        def __init__(self, root):
            self._current = root
            while self._current and (self._current._left or self._current._right):
                if self._current._left:
                    self._current = self._current._left
                else:
                    self._current = self._current._right

        def get_next_element(self, node):
            if not node or not node._parent:
                return None
            elif node._parent._right and node != node._parent._right:
                node = node._parent._right
                while node and (node._left or node._right):
                    if node._left:
                        node = node._left
                    else:
                        node = node._right
                return node
            elif node == node._parent._right or (node == node._parent._left and not node._parent._right):
                return node._parent

        def __next__(self):
            if not self._current:
                raise StopIteration
            el = self._current
            self._current = self.get_next_element(self._current)
            return el._key, el._value

    def postorder_iter(self):
        return TreeMap.PostorderIterator(self._root)

    class PostorderReverseIterator:
        def __init__(self, root):
            self._current = root

        def get_next_element(self, node):
            if node == None:
                return None
            if node._right:
                return node._right
            elif node._left:
                return node._left
            else:
                parent = node._parent
                while parent and parent._left and parent._left == node:
                    node = parent
                    parent = parent._parent
                if parent and parent._left:
                    return parent._left
                elif parent:
                    while parent and not parent._left:
                        parent = parent._parent
                    if parent:
                        return parent._left
                    else:
                        return None
                else:
                    return None

        def __next__(self):
            if not self._current:
                raise StopIteration
            el = self._current
            self._current = self.get_next_element(self._current)
            return el._key, el._value

    def postorder_reverse_iter(self):
        return TreeMap.PostorderReverseIterator(self._root)

    class InorderIndexIterator:
        def __init__(self, node, index):
            self._current = node
            counter = 0
            while counter != index and self._current:
                self._current = self.higher_entry(self._current._key)
                counter += 1

        def higher_entry(self, k):
            curr = TreeMap.get_smallest_element_node(self, self._current)
            while curr and curr._key <= k:
                if curr._right:
                    curr = TreeMap.get_smallest_element_node(self, curr._right)
                else:
                    parent = curr._parent
                    while parent and parent._right == curr:
                        curr = parent
                        parent = parent._parent
                    curr = parent
            return curr

        def __next__(self):
            if not self._current:
                raise StopIteration
            el = self._current
            self._current = self.higher_entry(self._current._key)
            return el._key, el._value

    def inorder_index_entry_iter(self, index):
        return TreeMap.InorderIndexIterator(self.get_smallest_element_node(self._root), index)


class WeightedGraphMap(WeightedGraphADT):
    def __init__(self):
        self._graph = {}

    def get_size(self):
        return len(self._graph)

    def is_empty(self):
        return len(self._graph) == 0

    def empty(self):
        self._graph = {}

    def add(self, v):
        self.add_vertex(v)

    def remove(self, v):
        self.remove_vertex(v)

    def printt(self):
        print(self._graph)

    def add_vertex(self, v):
        if v in self._graph:
            return False
        self._graph[v] = {}

    def add_edge(self, v1, v2, w):
        if v1 in self._graph and v2 in self._graph and v1 in self._graph[v2] and v2 in self._graph[v1]:
            return False
        self.add_vertex(v1)
        self.add_vertex(v2)
        self._graph[v1][v2] = w
        self._graph[v2][v1] = w

    def remove_vertex(self, v):
        if v not in self._graph:
            return False
        self._graph.pop(v)
        for connections in self._graph:
            connections.pop(v)
        return True

    def remove_edge(self, v1, v2):
        if v1 not in self._graph or v2 not in self._graph:
            return False
        self._graph[v1].pop(v2)
        self._graph[v2].pop(v1)

    def dijkstra(self, start):
        costs = {}
        visited = []
        for node in self._graph:
            costs[node] = float('inf')
        costs[start] = 0
        node = self.get_cheapest_node(costs, visited)
        while node is not None:
            neighbours = self._graph[node]
            for n in neighbours:
                new_cost = costs[node] + neighbours[n]
                if new_cost < costs[n]:
                    costs[n] = new_cost
            visited.append(node)
            node = self.get_cheapest_node(costs, visited)
        return costs

    def get_cheapest_node(self, costs, visited):
        cheapest_value = float('inf')
        cheapest_node = None
        for node in costs:
            if costs[node] < cheapest_value and node not in visited:
                cheapest_value = costs[node]
                cheapest_node = node
        return cheapest_node

    def kruskal(self):
        min_span_tree = WeightedGraphMap()
        edges = []
        for node in self._graph:
            for neighbour in self._graph[node]:
                if (self._graph[node][neighbour], neighbour, node) not in edges:
                    edges.append((self._graph[node][neighbour], node, neighbour))
        edges.sort(key = lambda x: x[0])
        for weight, start, end in edges:
            min_span_tree.add_edge(start, end, weight)
            if self.check_is_there_cycle(min_span_tree._graph):
                min_span_tree.remove_edge(start, end)
        return min_span_tree

    def check_is_there_cycle(self, graph):
        visited = {node: False for node in graph}
        for node in graph:
            if not visited[node]:
                if self.check_is_there_cycle_helper(graph, node, visited, None):
                    return True
        return False

    def check_is_there_cycle_helper(self, graph, node, visited, parent):
         visited[node] = True
         for n in graph[node]:
             if not visited[n]:
                 if self.check_is_there_cycle_helper(graph, n, visited, node):
                     return True
             elif parent != n:
                return True
         return False

    def prim(self, start):
        min_span_tree = WeightedGraphMap()
        priority_queue = {start: 0}
        parents = {node: None for node in self._graph}
        mst = []
        
        while priority_queue:
            node = min(priority_queue, key = priority_queue.get)
            priority_queue.pop(node)
            if node not in mst:
                parent = parents[node]
                if parent is not None:
                    min_span_tree.add_edge(parent, node, self._graph[parent][node])
                mst.append(node)
                
                for neighbour in self._graph[node]:
                    new_cost = self._graph[node][neighbour]
                    if neighbour not in mst:
                        if neighbour not in priority_queue or priority_queue[neighbour] > new_cost:
                            priority_queue[neighbour] = new_cost
                            parents[neighbour] = node
        return min_span_tree
        
    def floyd_warshall(self):
        solution_matrix = [[float('inf') for _ in range(len(self._graph))] for _ in range(len(self._graph))]
        for start in self._graph:
            solution_matrix[start][start] = 0
            for end in self._graph[start]:
                solution_matrix[start][end] = self._graph[start][end]
        
        for k in range(len(self._graph)):
            for i in range(len(self._graph)):
                for j in range(len(self._graph)):
                    solution_matrix[i][j] = min(solution_matrix[i][j], 
                                                solution_matrix[i][k] + solution_matrix[k][j])
        return solution_matrix
    

class WeightedDirectedGraphAdjMap(WeightedGraphADT):
    def __init__(self):
        self._graph = TreeMap()

    def get_size(self):
        return self._graph.get_size()

    def is_empty(self):
        return self._graph.is_empty()

    def empty(self):
        self._graph.empty()

    def add(self, v):
        self.add_vertex(v)

    def remove(self, v):
        self.remove_vertex(v)

    def printt(self):
        self._graph.print_for_graphs()

    def add_vertex(self, v):
        if self._graph.contains(v):
            return False
        self._graph.put(v, TreeMap())

    def add_edge(self, v1, v2, w):
        self.add_vertex(v1)
        self.add_vertex(v2)
        self._graph.get(v1).put(v2, w)

    def remove_vertex(self, v):
        if not self._graph.contains(v):
            return False
        self._graph.remove(v)
        for vertices, connections in self._graph:
            connections.remove(v)
        return True

    def remove_edge(self, v1, v2):
        if not self._graph.contains(v1) or not self._graph.contains(v2):
            return False
        self._graph.get(v1).remove(v2)

    def shortest_path(self, start):
        costs = TreeMap()
        visited = []
        for node, connections in self._graph:
            costs.put(node, float('inf'))
        costs.put(start, 0)
        node = self.get_cheapest_node(costs, visited)
        while node is not None:
            neighbours = self._graph.get(node)
            for n, weight in neighbours:
                new_cost = costs.get(node) + weight
                if new_cost < costs.get(n):
                    costs.put(n, new_cost)
            visited.append(node)
            node = self.get_cheapest_node(costs, visited)
        return costs

    def get_cheapest_node(self, costs, visited):
        cheapest_value = float('inf')
        cheapest_node = None
        for node, node_costs in costs:
            if node_costs < cheapest_value and node not in visited:
                cheapest_value = node_costs
                cheapest_node = node
        return cheapest_node

    def bellman_ford(self, start):
        costs = {node: float('inf') for node, val in self._graph}
        costs[start] = 0 
        edges = []
        for node, neighbours in self._graph:
            for neighbour, weight in neighbours: 
                edges.append((node, neighbour, weight))
                

        for _ in range(self._graph.get_size() - 1):
            for start, end, weight in edges:
                if costs[start] != float('inf') and costs[start] + weight < costs[end]:
                    costs[end] = costs[start] + weight
        
        for start, end, weight in edges:
            if costs[start] != float('inf') and costs[start] + weight < costs[end]:
                raise ValueError
        return costs


class MinPriorityQueue(QueueADT):
    def __init__(self):
        self._heap = [None] * 20
        self._size = 0
    
    def get_size(self):
        return self._size

    def is_empty(self):
        return self._size == 0

    def empty(self):
        self._heap = []
        self._size = 0

    def add(self, el):
        self.enqueue(el)

    def remove(self, el):
        pass

    def printt(self):
        for i in range(self._size):
            print(self._heap[i])
    
    def resize(self):
        if len(self._heap) == self._size + 1:
            self._heap = self._heap + [None] * len(self._heap)
        
    def enqueue(self, el):
        if el in self._heap:
            return False
        self.resize()
        ind = self._size
        self._heap[self._size] = el
        while self._heap[(ind - 1) // 2] and self._heap[ind] < self._heap[(ind - 1) // 2]:
            self.swap(ind, (ind - 1) // 2)
            ind = (ind - 1) // 2
        self._size += 1

    def dequeue(self):
        deq = self._heap[0]
        self._heap[0], self._heap[self._size - 1] = self._heap[self._size - 1], None
        ind = 0
        while (self._heap[2 * ind + 1] and self._heap[ind] > self._heap[2 * ind + 1]) \
            or (self._heap[2 * ind + 2] and self._heap[ind] > self._heap[2 * ind + 2]):
            if self._heap[2 * ind + 2] and self._heap[2 * ind + 1] > self._heap[2 * ind + 2]:
                self.swap(ind, 2 * ind + 2)
                ind = 2 * ind + 2
            else:
                self.swap(ind, 2 * ind + 1)
                ind = 2 * ind + 1 
        self._size -= 1
        return deq
    
    def front(self):
        return self._heap[0]

    def back(self):
        pass

    def swap(self, ind1, ind2):
        self._heap[ind1], self._heap[ind2] = self._heap[ind2], self._heap[ind1] 

    def get_left_child(self, ind):
        if self._heap[2 * ind + 1]:
            return self._heap[2 * ind + 1]
        else:
            return None

    def get_right_child(self, ind):
        if self._heap[2 * ind + 2]:
            return self._heap[2 * ind + 2]
        else:
            return None
        
    def get_parent(self, ind):
        if self._heap[(ind - 1) // 2]:
            return self._heap[(ind - 1) // 2]
        else:
            return None
        
    def get_leftest_childs_ind(self):
        ind = 0
        while self.get_left_child(ind) is not None:
            ind = 2 * ind + 1
        return ind
            
    def get_number_of_products_rating_higher(self, rating):
        num = 0
        for i in self:
            if i.get_rating() > rating:
                num += 1
        return num
    
    class InorderIterator:
        def __init__(self, heap, leftest_ind):
            self.heap = heap
            self.current_ind = leftest_ind
        
        def get_next_index(self, ind):
            if not self.heap[ind]:
                return None
            if self.heap[ind * 2 + 2]:
                temp_ind = ind * 2 + 2
                while self.heap[temp_ind * 2 + 1]:
                    temp_ind = temp_ind * 2 + 1
                return temp_ind
            else:
                parent_ind = (ind - 1) // 2
                while self.heap[parent_ind] and self.heap[parent_ind * 2 + 2] == self.heap[ind]:
                    ind = parent_ind
                    parent_ind = (parent_ind - 1) // 2
                return parent_ind

        def __next__(self):
            if not self.heap[self.current_ind]:
                raise StopIteration
            el = self.heap[self.current_ind]
            self.current_ind = self.get_next_index(self.current_ind)
            return el
        
    def __iter__(self):
        return MinPriorityQueue.InorderIterator(self._heap, self.get_leftest_childs_ind())