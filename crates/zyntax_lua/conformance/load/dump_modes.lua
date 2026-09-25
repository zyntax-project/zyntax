-- load's mode, string.dump of a library function, and binary files.

local d = string.dump(function () return 1 end)
print(load(d, nil, "t"))
print(load(d, "=x", "t"))
print(load("return 1", nil, "b"))
print(load("return 1", "=x", "b"))
print(load(d, nil, "b")(), load(d, nil, "bt")(), load(d)())
print(load(string.dump(function () return 1 end), nil, "b", {})())

print(pcall(string.dump, print))
print(pcall(string.dump, string.dump))
print(select(2, pcall(string.dump, {})):match("%(.*%)"))
print(select(2, pcall(string.dump)):match("%(.*%)"))

local file = os.tmpname()
local function write(s)
  local h = assert(io.open(file, "wb"))
  h:write(s)
  h:close()
end

write(string.dump(function () return 10, "\0alo\255", "hi" end))
local a, b, c = assert(loadfile(file))()
print(a, #b, c)

-- no upvalues, an empty environment
write(string.dump(function () return 1 end))
local f = assert(loadfile(file, "b", {}))
print(type(f), f())
print(loadfile(file, "t"))

-- a '#' line before a binary chunk
write("#this is a comment for a binary file\0\n" ..
      string.dump(function () return 20, "\0\0\0" end))
a, b, c = assert(loadfile(file))()
print(a, #b, c)
print(dofile(file))

-- a '#' line before text keeps the line numbers
write("# a comment\nreturn require'debug'.getinfo(1).currentline")
print(assert(loadfile(file))())

os.remove(file)
