-- debug.traceback: the stack from the caller down, as lua5.4 writes it.
print(debug.traceback("at the top"))

local function leaf(msg)
  print(debug.traceback(msg, 1))
end
leaf("from a local")

local t = {}
function t.field() leaf("through a field") end
t.field()

function global_fn() t.field() end
global_fn()

local obj = {}
function obj:method() print(debug.traceback("in a method")) end
obj:method()

local function up() leaf("through an upvalue") end
local function outer() up() end
outer()

print(pcall(function() print(debug.traceback("under pcall")) end))

local function tail() print(debug.traceback("tail called")) end
local function caller() return tail() end
caller()

-- The message: a number is text, any other non-string comes back.
print(debug.traceback(42))
print(type(debug.traceback({})))
print(debug.traceback(nil))
print(debug.traceback())

-- The level: 2 starts at the caller of the function asking.
local function two() print(debug.traceback("level 2", 2)) end
local function one() two() end
one()
print(debug.traceback("past the stack", 50))
