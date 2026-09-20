-- An error raised with a fresh message and caught every step.
local function fail(i)
  error("step " .. i)
end

local function main(n)
  local total = 0
  for i = 1, n do
    local ok, msg = pcall(fail, i)
    if not ok then total = total + #msg end
  end
  return total
end

print(main(tonumber(arg[1])))
