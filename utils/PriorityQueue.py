import heapq


class PriorityQueue():
    """Implementation of a priority queue."""

    def __init__(self):
        self.queue = []
        self.current = 0

    def next(self):

        if self.current >= len(self.queue):
            self.current
            raise StopIteration

        out = self.queue[self.current]
        self.current += 1
        return out

    def pop(self):
        try:
            return heapq.heappop(self.queue)
        except:
            raise

    def remove(self, nodeId):
        raise NotImplementedError

    def __iter__(self):
        return self

    def __str__(self):
        return 'PQ:[%s]' % (', '.join([str(i) for i in self.queue]))

    def append(self, node):
        heapq.heappush(self.queue, node)

    def __contains__(self, key):
        self.current = 0
        for x in self.queue:
            if x[-1] == key:
                return True
        return False
        #  return key in [n for v,n in self.queue]

    def __eq__(self, other):
        return self == other

    def size(self):
        return len(self.queue)

    def clear(self):
        self.queue = []

    def top(self):
        return self.queue[0]

    __next__ = next
