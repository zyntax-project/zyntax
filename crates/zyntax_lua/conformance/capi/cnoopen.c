/* A library with no open function for its name. */
#include "lua.h"

LUAMOD_API int cnoopen_other(lua_State *L) { return lua_gettop(L); }
