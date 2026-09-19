print("before")
local function fail() error("uncaught here") end
fail()
print("never")
