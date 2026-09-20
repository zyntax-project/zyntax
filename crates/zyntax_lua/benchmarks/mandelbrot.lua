local RGB = {}
RGB.__index = RGB

function RGB.new(r, g, b)
  return setmetatable({ r = r, g = g, b = b }, RGB)
end

local Complex = {}
Complex.__index = Complex

function Complex.new(i, j)
  return setmetatable({ i = i, j = j }, Complex)
end

local function trunc(x)
  if x >= 0 then return math.floor(x) end
  return math.ceil(x)
end

local function palette(fraction)
  local r = trunc(fraction * 255)
  local g = trunc((1 - fraction) * 255)
  local b = trunc((0.5 - math.abs(fraction - 0.5)) * 2 * 255)
  return RGB.new(r, g, b)
end

local function run()
  local size = 25
  local max_iterations = 200
  local max_rad = 65536
  local width = 350
  local height = 200
  local pal = {}
  for i = 0, max_iterations do
    pal[i] = palette(i / max_iterations)
  end
  local scale = 0.25 / size
  local checksum = 0
  for y = 0, height - 1 do
    for x = 0, width - 1 do
      local iteration = 0
      local offset = Complex.new(x * scale - 2.5, y * scale - 1)
      local val = Complex.new(0.0, 0.0)
      while val.i * val.i + val.j * val.j < max_rad and iteration < max_iterations do
        val = Complex.new(val.i * val.i - val.j * val.j + offset.i, 2.0 * val.i * val.j + offset.j)
        iteration = iteration + 1
      end
      local color = pal[iteration]
      checksum = checksum + color.r + color.g + color.b
    end
  end
  return checksum
end

local start = os.clock()
local checksum = run()
local elapsed = os.clock() - start
print(string.format("checksum: %d", checksum))
print(string.format("elapsed: %s", elapsed))
