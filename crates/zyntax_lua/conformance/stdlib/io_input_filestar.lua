-- io.input and io.output take a name or a file; anything else is
-- checked as a file, and a closed file is refused before it becomes
-- the default.
local XX = setmetatable({}, {__name = "My Type"})
print(pcall(function () io.input(XX) end))
print(pcall(function () io.input({}) end))
print(pcall(function () io.input(true) end))
print(pcall(function () io.input(print) end))
print(pcall(function () io.output(XX) end))
print(pcall(function () io.output({}) end))
print(pcall(function () io.output(true) end))
print(pcall(function () io.output(print) end))
print(io.input() == io.stdin, io.output() == io.stdout)
print(pcall(function () io.input("/nonexistent/zylua/file") end))
local closed = io.tmpfile()
closed:close()
print(pcall(function () io.input(closed) end))
print(pcall(function () io.output(closed) end))
print(io.type(io.input()), io.type(io.output()))
