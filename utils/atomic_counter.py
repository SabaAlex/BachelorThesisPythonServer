import threading

class FastReadCounter(object):
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()
        
    def increment(self):
        with self._lock:
            self._value += 1

    def decrement(self):
        with self._lock:
            self._value -= 1

    def get_count(self):
        return self._value