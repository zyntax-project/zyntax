local helper = require "lib.mod"
return { twice = function(x) return helper.add(x, x) end, tag = "util.deep" }
