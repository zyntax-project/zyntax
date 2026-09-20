import random
import go

random.seed(1)
board = go.Board()
pos = go.computer_move(board)
print(pos, go.MOVES, go.TIMESTAMP)
random.seed(7)
board = go.Board()
for i in range(20):
    p = board.random_move()
    if p != go.PASS:
        board.move(p)
print(board.score(go.BLACK), board.score(go.WHITE), len(board.history))
print(board)
