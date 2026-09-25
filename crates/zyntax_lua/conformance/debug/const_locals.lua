-- A <const> local whose value is known where it is declared has no
-- slot: it is neither a local nor an upvalue to the debug library.
local K <const> = 10
local S <const> = "s"
local N <const> = -2 * 3 + 1
local function getK() return K, S, N end
print(debug.getupvalue(getK, 1))
print(debug.setupvalue(getK, 1, 5), getK())

-- One computed at run time is an upvalue like any other.
local R <const> = tostring(4)
local function getR() return R end
print(debug.getupvalue(getR, 1))
print(debug.setupvalue(getR, 1, "5"), getR())

local function locals()
  local a <const> = 1
  local b = 2
  local c <const> = {}
  local d <const> = a
  local i = 1
  while true do
    local name, value = debug.getlocal(1, i)
    if not name then break end
    print(i, name, type(value))
    i = i + 1
  end
end
locals()

-- Several names: only the last can be folded.
local function multi()
  local p <const>, q <const> = 1, 2
  local up = function() return p + q end
  print(debug.getupvalue(up, 1))
  print(debug.getupvalue(up, 2))
  print(debug.getlocal(1, 1))
  local name, value = debug.getlocal(1, 2)
  print(name, type(value))
end
multi()
