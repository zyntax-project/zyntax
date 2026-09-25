-- require finds a module written as a binary chunk, with or without a
-- '#' first line.

package.path = "./?.lua;" .. package.path
local function write(name, s)
  local h = assert(io.open(name .. ".lua", "wb"))
  h:write(s)
  h:close()
end

local body = string.dump(load("local name, path = ...; return {name = name, path = path, n = 42}"))
write("zl_binary_mod", body)
write("zl_binary_hash", "#!/usr/bin/env lua\n" .. body)

local ok, m = pcall(require, "zl_binary_mod")
print(ok, m.name, m.path, m.n)
local h = require("zl_binary_hash")
print(h.name, h.n)

write("zl_binary_bad", body:sub(1, 20))
print(select(2, pcall(require, "zl_binary_bad")):match("truncated") ~= nil)

os.remove("zl_binary_mod.lua")
os.remove("zl_binary_hash.lua")
os.remove("zl_binary_bad.lua")
