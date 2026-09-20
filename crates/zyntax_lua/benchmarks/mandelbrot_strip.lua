-- The Zyntax mandelbrot recurrence over a five-row strip, 875 wide,
-- up to 1000 iterations a point, summed through a palette.
-- Returns what the ZynML kernel returns.

local function palette_sum(iteration, max_iter)
  local fraction = iteration / max_iter
  local r = math.floor(fraction * 255.0)
  local g = math.floor((1.0 - fraction) * 255.0)
  local b = math.floor((0.5 - math.abs(fraction - 0.5)) * 2.0 * 255.0)
  return r + g + b
end

local function main()
  local size = 25
  local max_iter = 1000
  local max_rad = 65536.0
  local width = 35 * size
  local height = 96
  local scale = 0.004
  local checksum = 0
  for y = 92, height - 1 do
    for x = 0, width - 1 do
      local cx = x * scale - 2.5
      local cy = y * scale - 1.0
      local zx, zy, zx2, zy2 = 0.0, 0.0, 0.0, 0.0
      local iter = 0
      while iter < max_iter do
        if zx2 + zy2 >= max_rad then break end
        local new_zx = zx2 - zy2 + cx
        local new_zy = 2.0 * zx * zy + cy
        zx = new_zx
        zy = new_zy
        zx2 = zx * zx
        zy2 = zy * zy
        iter = iter + 1
      end
      checksum = checksum + palette_sum(iter, max_iter)
    end
  end
  return checksum
end

local start = os.clock()
local result = main()
local elapsed = os.clock() - start
print(string.format("result: %d", result))
print(string.format("elapsed: %s", elapsed))
