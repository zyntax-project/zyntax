-- io.lines(name) closes its file however the loop is left.
local file = os.tmpname()
local f = assert(io.open(file, "w"))
f:write("one\ntwo\nthree\n")
f:close()

-- The closing value is the loop's fourth hidden local.
local function gettoclose (lv)
  lv = lv + 1
  local stvar = 0
  for i = 1, 1000 do
    local n, v = debug.getlocal(lv, i)
    if n == "(for state)" then
      stvar = stvar + 1
      if stvar == 4 then return v end
    end
  end
end

local h
for l in io.lines(file) do
  h = gettoclose(1)
  print(l, io.type(h))
  break
end
print("after break", io.type(h))

local function foo (name)
  for l in io.lines(name) do
    h = gettoclose(1)
    print(l, io.type(h))
    error(h)
  end
end
local st, msg = pcall(foo, file)
print("after error", st, io.type(msg))

local function bar (name)
  for l in io.lines(name) do
    h = gettoclose(1)
    if l == "two" then return l end
  end
end
print("returned", bar(file), io.type(h))

local n = 0
for l in io.lines(file) do n = n + 1; h = gettoclose(1) end
print("to the end", n, io.type(h))
-- Only io.lines with a name gives a closing value.
local g = assert(io.open(file))
print("counts", select("#", io.lines(file)), select("#", g:lines()))
for l in g:lines() do end
print("f:lines leaves it open", io.type(g))
assert(g:close())
assert(os.remove(file))
