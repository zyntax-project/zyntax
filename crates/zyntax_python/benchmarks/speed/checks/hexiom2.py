import hexiom2
import io
import sys
board = hexiom2.Hex(3)
print(board.size, board.count, len(board.nodes_by_id))
hexiom2.run_level36()
captured = io.StringIO()
original = sys.stdout
sys.stdout = captured
print("captured line", 1)
print("second")
sys.stdout = original
print(repr(captured.getvalue()))
print("done")
