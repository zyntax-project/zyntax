/* A C function called often enough that its callers are compiled. */
#include "lua.h"
#include "lauxlib.h"

static int add(lua_State *L) {
  lua_pushinteger(L, luaL_checkinteger(L, 1) + luaL_checkinteger(L, 2));
  return 1;
}

static int pair(lua_State *L) {
  lua_pushvalue(L, 2);
  lua_pushvalue(L, 1);
  return 2;
}

static const luaL_Reg funcs[] = {{"add", add}, {"pair", pair}, {NULL, NULL}};

LUAMOD_API int luaopen_ctiers(lua_State *L) {
  luaL_newlib(L, funcs);
  return 1;
}
