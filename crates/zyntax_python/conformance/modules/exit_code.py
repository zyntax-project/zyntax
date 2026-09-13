# sys.exit with a status
import sys
print("before")
if len(sys.argv) > 5:
    sys.exit(1)
sys.exit(3)
