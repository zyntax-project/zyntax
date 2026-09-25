/* A versioned file name: the open function is named for the part of
   the module's name before the hyphen. */
#include "lua.h"
#include "lauxlib.h"

LUAMOD_API int luaopen_cload(lua_State *L) {
  lua_pushfstring(L, "v2 as %s from %s", lua_tostring(L, 1),
                  lua_tostring(L, 2));
  return 1;
}
