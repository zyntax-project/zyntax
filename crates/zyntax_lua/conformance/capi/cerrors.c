/* Errors raised in C, and errors passing through C. */
#include "lua.h"
#include "lauxlib.h"

static int boom(lua_State *L) {
  return luaL_error(L, "boom %d", (int)luaL_optinteger(L, 1, 42));
}

static int throw_value(lua_State *L) {
  lua_settop(L, 1);
  return lua_error(L);
}

/* Call its first argument with the rest, unprotected. */
static int call(lua_State *L) {
  lua_call(L, lua_gettop(L) - 1, LUA_MULTRET);
  return lua_gettop(L);
}

/* Call its first argument protected: the status, then what it gave. */
static int pcall(lua_State *L) {
  int status = lua_pcall(L, lua_gettop(L) - 1, LUA_MULTRET, 0);
  lua_pushinteger(L, status);
  lua_insert(L, 1);
  return lua_gettop(L);
}

static int handler(lua_State *L) {
  lua_pushfstring(L, "handled: %s", lua_tostring(L, 1));
  return 1;
}

/* Call its argument protected under a message handler. */
static int xpcall(lua_State *L) {
  int status;
  lua_pushcfunction(L, handler);
  lua_insert(L, 1);
  status = lua_pcall(L, 0, 1, 1);
  lua_pushinteger(L, status);
  lua_insert(L, -2);
  return 2;
}

/* A C function whose error is caught by a C caller in the same call. */
static int inner(lua_State *L) {
  lua_pushstring(L, "inner");
  return lua_error(L);
}

static int outer(lua_State *L) {
  int status;
  lua_pushcfunction(L, inner);
  status = lua_pcall(L, 0, 0, 0);
  lua_pushinteger(L, status);
  lua_insert(L, -2);
  lua_pushstring(L, "after");
  return 3;
}

/* Many values left on the stack when an error leaves. */
static int crowded(lua_State *L) {
  int i;
  luaL_checkstack(L, 100, "no room");
  for (i = 0; i < 100; i++) lua_pushinteger(L, i);
  return luaL_error(L, "crowded");
}

static int checkint(lua_State *L) {
  lua_pushinteger(L, luaL_checkinteger(L, 1) * 2);
  return 1;
}

static const luaL_Reg funcs[] = {
    {"boom", boom},       {"throw", throw_value}, {"call", call},
    {"pcall", pcall},     {"xpcall", xpcall},     {"outer", outer},
    {"crowded", crowded}, {"checkint", checkint}, {NULL, NULL}};

LUAMOD_API int luaopen_cerrors(lua_State *L) {
  luaL_newlib(L, funcs);
  return 1;
}
