local M = {}
function M.attach(t)
  setmetatable(t, {__index = function(_, k) return "from the module: " .. k end,
                   __len = function() return 99 end})
  return t
end
return M
