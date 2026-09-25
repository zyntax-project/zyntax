/* Open functions for the searchers: the module's own, one for a
   submodule the all-in-one searcher finds, and one that raises. */
#include "lua.h"
#include "lauxlib.h"

static int twice(lua_State *L) {
  lua_pushinteger(L, 2 * luaL_checkinteger(L, 1));
  return 1;
}

/* What the loader was called with. */
static int opened(lua_State *L, const char *which) {
  lua_createtable(L, 0, 3);
  lua_pushstring(L, which);
  lua_setfield(L, -2, "which");
  lua_pushvalue(L, 1);
  lua_setfield(L, -2, "name");
  lua_pushvalue(L, 2);
  lua_setfield(L, -2, "file");
  return 1;
}

LUAMOD_API int cload_twice(lua_State *L) { return twice(L); }

LUAMOD_API int luaopen_cload(lua_State *L) { return opened(L, "cload"); }

LUAMOD_API int luaopen_cload_sub(lua_State *L) {
  return opened(L, "cload.sub");
}

LUAMOD_API int luaopen_cload_bad(lua_State *L) {
  return luaL_error(L, "cannot open %s", lua_tostring(L, 1));
}
