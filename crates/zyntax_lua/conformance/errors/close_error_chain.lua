-- Errors in closing methods: every pending one still runs, each seeing
-- the error in flight.
local function func2close (f) return setmetatable({}, {__close = f}) end

-- The original error is in a closing method.
local function foo ()
  local x <close> = func2close(function (_, msg)
    print("x sees", msg); error("@x", 0)
  end)
  local x1 <close> = func2close(function (_, msg) print("x1 sees", msg) end)
  local y <close> = func2close(function (_, msg)
    print("y sees", msg); error("@y", 0)
  end)
  local z <close> = func2close(function (_, msg)
    print("z sees", msg); error("@z", 0)
  end)
  return 200
end
print(pcall(foo))

-- The original error is not.
local function bar ()
  local x <close> = func2close(function (_, msg) print("x sees", msg) end)
  local x1 <close> = func2close(function (_, msg)
    print("x1 sees", msg); error("@x1", 0)
  end)
  local y <close> = func2close(function (_, msg)
    print("y sees", msg); error("@y", 0)
  end)
  local first = true
  local z <close> = func2close(function (_, msg)
    print("z sees", msg, first); first = false; error("@z", 0)
  end)
  error(4)
end
print(pcall(bar))

-- An invalid metamethod is an error too.
local function baz ()
  local a1 <close> = func2close(function (_, msg)
    print("a1 sees", msg); error(12)
  end)
  local a2 <close> = setmetatable({}, {__close = print})
  local a3 <close> = func2close(function (_, msg)
    print("a3 sees", msg); error(123)
  end)
  getmetatable(a2).__close = 4
end
print(pcall(baz))

-- Closing methods with variables of their own.
local track = {}
local function qux ()
  local x0 <close> = func2close(function (_, msg)
    track[#track + 1] = "x0:" .. tostring(msg)
  end)
  local x <close> = func2close(function ()
    local xx <close> = func2close(function (_, msg)
      track[#track + 1] = "xx:" .. tostring(msg)
      error(202, 0)
    end)
    track[#track + 1] = "x"
    error(101, 0)
  end)
  track[#track + 1] = "qux"
  return 20, 30, 40
end
print(pcall(qux))
print(table.concat(track, " "))
