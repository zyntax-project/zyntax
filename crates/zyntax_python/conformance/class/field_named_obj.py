class Wrapper:
    def __init__(self, obj, left):
        self.obj = obj
        self.left = left

    def take(self):
        self.left -= 1
        return self.obj


w = Wrapper("z", 3)
print(w.take(), w.take(), w.left, Wrapper(9, 1).take(), Wrapper([1], 0).obj)
