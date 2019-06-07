from basics.base import Base


class SlidingWindow(Base):

    def __init__(self, length, init_window_values, **kwargs):
        super().__init__(**kwargs)

        if length % 2 != 1:
            length += 1
            print("Warning: given window length is even, increasing length to %d" % length)

        self.length = length
        self.center_index = self.length // 2

        if hasattr(init_window_values, "__getitem__"):
            self._log.debug("Using given initial values to fill the window as much as possible.")

            l = len(init_window_values)
            l = l if l < self.length else self.length

            self.window = init_window_values[-l:][:]
        else:
            self.window = list()

    def get_window(self):
        return self.window

    def flush(self):
        self.window.clear()

    def center(self):
        if not self.is_filled():
            print("Warning: window not completely filled yet, unable to get center")
            return None

        return self.window[self.center_index]

    def last(self):
        return self.window[-1]

    def first(self):
        return self.window[0]

    def is_filled(self):
        return len(self.window) >= self.length

    def is_empty(self):
        return len(self.window) == 0

    def slide(self, value):
        if self.is_filled():
            self.window = self.window[1:]

        self.window.append(value)
