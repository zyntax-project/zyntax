-- warn writes to stderr only once turned on; its arguments must be
-- strings or numbers.
warn("not shown")
warn("@on")
warn("shown ", "in ", "pieces ", 1, 2.5)
warn("@unknown control")
warn("@off")
warn("hidden again")
print(pcall(warn))
print(pcall(warn, "a", {}))
print(pcall(warn, "a", nil))
print(pcall(warn, 1))
print(select("#", warn("@on")))
print("ok")
