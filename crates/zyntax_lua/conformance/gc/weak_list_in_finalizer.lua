-- Finalizers append values that die and values that live to a
-- weak-value list; a traversal afterwards sees only live values, even
-- when a collection clears one after the traversal has found it.
local W = setmetatable({}, {__mode = "v"})
local hold = {}
local runs = 0
local function mk()
  setmetatable({}, {__gc = function()
    runs = runs + 1
    for i = 1, 3000 do
      local o = {i}
      W[#W + 1] = o
      if i % 100 == 0 then hold[#hold + 1] = o end
    end
  end})
end
for r = 1, 3 do mk() collectgarbage() end
local bad = 0
for k, v in pairs(W) do
  if type(k) ~= 'number' or type(v) ~= 'table' or type(v[1]) ~= 'number' then bad = bad + 1 end
end
for i = 1, #W do
  local v = W[i]
  if v ~= nil and (type(v) ~= 'table' or type(v[1]) ~= 'number') then bad = bad + 1 end
end
print('runs>=2', runs >= 2, 'bad', bad)
