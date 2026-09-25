-- What a step reports in each mode: generational steps never end a
-- cycle; an incremental step over a small heap ends one.
collectgarbage()
print(collectgarbage("step"), collectgarbage("step", 0))
print(collectgarbage("incremental"))
collectgarbage()
print(collectgarbage("step"))
print(collectgarbage("step", 100))
print(collectgarbage("generational"))
print(collectgarbage("step"))
