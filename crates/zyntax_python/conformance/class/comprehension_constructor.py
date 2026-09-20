# constructor arguments typed inside comprehensions
class Sq(object):
    def __init__(self, board, pos):
        self.board = board
        self.pos = pos

    def right(self):
        return self.board.squares[(self.pos + 1) % 4]


class Board(object):
    def __init__(self):
        self.squares = [Sq(self, pos) for pos in range(4)]
        self.pairs = {pos: Sq(self, pos * 10) for pos in range(2)}
        self.gen = sum(Sq(self, p).pos for p in range(3))


b = Board()
print(b.squares[2].pos, b.squares[3].right().pos, b.pairs[1].pos, b.gen)
