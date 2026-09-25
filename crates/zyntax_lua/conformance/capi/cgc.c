/* Values C code holds only on its stack, in upvalues or in the
   registry, while the program allocates. */
#include <string.h>

#include "lua.h"
#include "lauxlib.h"

/* Push many values, call the churn function, then read them back. */
static int hold(lua_State *L) {
  int n = (int)luaL_checkinteger(L, 2), i;
  lua_Integer sum = 0;
  luaL_checkstack(L, n + 4, "no room");
  for (i = 0; i < n; i++) {
    lua_createtable(L, 0, 1);
    lua_pushfstring(L, "value %d", i);
    lua_setfield(L, -2, "s");
    lua_pushinteger(L, i);
    lua_setfield(L, -2, "i");
  }
  lua_pushvalue(L, 1);
  lua_call(L, 0, 0);
  for (i = 0; i < n; i++) {
    lua_getfield(L, 3 + i, "i");
    sum += lua_tointeger(L, -1);
    lua_pop(L, 1);
    lua_getfield(L, 3 + i, "s");
    if (lua_type(L, -1) != LUA_TSTRING) return luaL_error(L, "lost %d", i);
    lua_pop(L, 1);
  }
  lua_pushinteger(L, sum);
  return 1;
}

static int stash(lua_State *L) {
  lua_createtable(L, 0, 0);
  lua_pushstring(L, "kept in the registry");
  lua_setfield(L, -2, "text");
  lua_pushinteger(L, luaL_ref(L, LUA_REGISTRYINDEX));
  return 1;
}

static int unstash(lua_State *L) {
  lua_rawgeti(L, LUA_REGISTRYINDEX, luaL_checkinteger(L, 1));
  lua_getfield(L, -1, "text");
  return 1;
}

static int upvalue_read(lua_State *L) {
  lua_getfield(L, lua_upvalueindex(1), "text");
  return 1;
}

static int make_reader(lua_State *L) {
  lua_createtable(L, 0, 0);
  lua_pushstring(L, "kept in an upvalue");
  lua_setfield(L, -2, "text");
  lua_pushcclosure(L, upvalue_read, 1);
  return 1;
}

typedef struct {
  char text[32];
} Payload;

static int payload(lua_State *L) {
  Payload *p = (Payload *)lua_newuserdatauv(L, sizeof(Payload), 0);
  strcpy(p->text, "payload intact");
  lua_pushvalue(L, 1);
  lua_call(L, 0, 0);
  lua_pushstring(L, p->text);
  return 1;
}

static const luaL_Reg funcs[] = {{"hold", hold},       {"stash", stash},
                                 {"unstash", unstash}, {"reader", make_reader},
                                 {"payload", payload}, {NULL, NULL}};

LUAMOD_API int luaopen_cgc(lua_State *L) {
  luaL_newlib(L, funcs);
  return 1;
}
