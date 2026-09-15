import richards
r = richards.Richards()
print(r.run(2))
print(richards.taskWorkArea.holdCount, richards.taskWorkArea.qpktCount)
print(r.run(iterations=1))
