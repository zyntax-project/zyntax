-- Upvalues are numbered in the order the reference's parser meets
-- them: an assignment's targets before its values.
local a, b = 20, 30
local function k () a = b end
print(debug.getupvalue(k, 1))
print(debug.getupvalue(k, 2))

local t, x, y = {}, 1, 2
local function g () t[x] = y end
for i = 1, 3 do print((debug.getupvalue(g, i))) end

local p, q, r = 1, 2, 3
local function h () p, q = r, p end
for i = 1, 3 do print(debug.getupvalue(h, i)) end

local u, w = {}, 5
local function m () u.f.g = w end
for i = 1, 2 do print((debug.getupvalue(m, i))) end
