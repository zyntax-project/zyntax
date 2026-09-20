local PI = 3.141592653589793
local SOLAR_MASS = 4 * PI * PI
local DAYS_PER_YEAR = 365.24

local Body = {}
Body.__index = Body

function Body.new(x, y, z, vx, vy, vz, mass)
  return setmetatable({ x = x, y = y, z = z, vx = vx, vy = vy, vz = vz, mass = mass }, Body)
end

function Body:offset_momentum(px, py, pz)
  self.vx = -px / SOLAR_MASS
  self.vy = -py / SOLAR_MASS
  self.vz = -pz / SOLAR_MASS
end

local function jupiter()
  return Body.new(4.84143144246472090e+00, -1.16032004402742839e+00, -1.03622044471123109e-01,
    1.66007664274403694e-03 * DAYS_PER_YEAR, 7.69901118419740425e-03 * DAYS_PER_YEAR,
    -6.90460016972063023e-05 * DAYS_PER_YEAR, 9.54791938424326609e-04 * SOLAR_MASS)
end

local function saturn()
  return Body.new(8.34336671824457987e+00, 4.12479856412430479e+00, -4.03523417114321381e-01,
    -2.76742510726862411e-03 * DAYS_PER_YEAR, 4.99852801234917238e-03 * DAYS_PER_YEAR,
    2.30417297573763929e-05 * DAYS_PER_YEAR, 2.85885980666130812e-04 * SOLAR_MASS)
end

local function uranus()
  return Body.new(1.28943695621391310e+01, -1.51111514016986312e+01, -2.23307578892655734e-01,
    2.96460137564761618e-03 * DAYS_PER_YEAR, 2.37847173959480950e-03 * DAYS_PER_YEAR,
    -2.96589568540237556e-05 * DAYS_PER_YEAR, 4.36624404335156298e-05 * SOLAR_MASS)
end

local function neptune()
  return Body.new(1.53796971148509165e+01, -2.59193146099879641e+01, 1.79258772950371181e-01,
    2.68067772490389322e-03 * DAYS_PER_YEAR, 1.62824170038242295e-03 * DAYS_PER_YEAR,
    -9.51592254519715870e-05 * DAYS_PER_YEAR, 5.15138902046611451e-05 * SOLAR_MASS)
end

local function sun()
  return Body.new(0, 0, 0, 0, 0, 0, SOLAR_MASS)
end

local NBody = {}
NBody.__index = NBody

function NBody.new()
  local self = setmetatable({}, NBody)
  self.bodies = { sun(), jupiter(), saturn(), uranus(), neptune() }
  local px, py, pz = 0.0, 0.0, 0.0
  for _, b in ipairs(self.bodies) do
    px = px + b.vx * b.mass
    py = py + b.vy * b.mass
    pz = pz + b.vz * b.mass
  end
  self.bodies[1]:offset_momentum(px, py, pz)
  return self
end

function NBody:advance(dt)
  local bodies = self.bodies
  local size = #bodies
  for i = 1, size do
    local a = bodies[i]
    for j = i + 1, size do
      local b = bodies[j]
      local dx = a.x - b.x
      local dy = a.y - b.y
      local dz = a.z - b.z
      local distance = math.sqrt(dx * dx + dy * dy + dz * dz)
      local mag = dt / (distance * distance * distance)
      a.vx = a.vx - dx * b.mass * mag
      a.vy = a.vy - dy * b.mass * mag
      a.vz = a.vz - dz * b.mass * mag
      b.vx = b.vx + dx * a.mass * mag
      b.vy = b.vy + dy * a.mass * mag
      b.vz = b.vz + dz * a.mass * mag
    end
  end
  for _, body in ipairs(bodies) do
    body.x = body.x + dt * body.vx
    body.y = body.y + dt * body.vy
    body.z = body.z + dt * body.vz
  end
end

function NBody:energy()
  local e = 0.0
  local bodies = self.bodies
  local n = #bodies
  for i = 1, n do
    local a = bodies[i]
    e = e + 0.5 * a.mass * (a.vx * a.vx + a.vy * a.vy + a.vz * a.vz)
    for j = i + 1, n do
      local b = bodies[j]
      local dx = b.x - a.x
      local dy = b.y - a.y
      local dz = b.z - a.z
      e = e - (a.mass * b.mass) / math.sqrt(dx * dx + dy * dy + dz * dz)
    end
  end
  return e
end

local function trunc(x)
  if x >= 0 then return math.floor(x) end
  return math.ceil(x)
end

local start = os.clock()
local sim = NBody.new()
for _ = 1, 500000 do
  sim:advance(0.01)
end
local energy = trunc(sim:energy() * 1000000)
local elapsed = os.clock() - start
print(string.format("energy: %d", energy))
print(string.format("elapsed: %s", elapsed))
