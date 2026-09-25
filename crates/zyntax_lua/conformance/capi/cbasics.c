/* The stack, value and table functions of the C API. */
#include <string.h>

#include "lua.h"
#include "lauxlib.h"

/* The arguments' types and the top, as one string. */
static int types(lua_State *L) {
  int n = lua_gettop(L), i;
  luaL_Buffer b;
  luaL_buffinit(L, &b);
  for (i = 1; i <= n; i++) {
    luaL_addstring(&b, luaL_typename(L, i));
    luaL_addchar(&b, i < n ? ',' : ';');
  }
  lua_pushfstring(L, "%d", n);
  luaL_addvalue(&b);
  luaL_pushresult(&b);
  return 1;
}

/* The stack after settop, rotate, insert, remove, replace, copy. */
static int shuffle(lua_State *L) {
  lua_settop(L, 0);
  lua_pushinteger(L, 1);
  lua_pushinteger(L, 2);
  lua_pushinteger(L, 3);
  lua_pushinteger(L, 4);
  lua_pushinteger(L, 5);
  lua_rotate(L, 2, 1);   /* 1 5 2 3 4 */
  lua_insert(L, 1);      /* 4 1 5 2 3 */
  lua_remove(L, 2);      /* 4 5 2 3 */
  lua_pushinteger(L, 9);
  lua_replace(L, 3);     /* 4 5 9 3 */
  lua_copy(L, 1, 4);     /* 4 5 9 4 */
  lua_settop(L, 6);      /* 4 5 9 4 nil nil */
  lua_pushinteger(L, lua_absindex(L, -1));
  return lua_gettop(L);
}

/* Formats of lua_pushfstring. */
static int fstrings(lua_State *L) {
  lua_pushfstring(L, "[%s|%d|%I|%f|%f|%c|%U|%%]", "str", -42,
                  (lua_Integer)1 << 40, 3.5, 2.0, 'x', (long)0x20AC);
  lua_pushfstring(L, "%s and %s", "one", "two");
  lua_pushfstring(L, "%f", 1e100);
  return 3;
}

/* Conversions of its argument. */
static int convert(lua_State *L) {
  int isnum = 0;
  lua_Integer i = lua_tointegerx(L, 1, &isnum);
  lua_Number n;
  size_t len = 0;
  const char *s;
  lua_pushboolean(L, isnum);
  lua_pushinteger(L, i);
  n = lua_tonumberx(L, 1, &isnum);
  lua_pushboolean(L, isnum);
  lua_pushnumber(L, n);
  lua_pushboolean(L, lua_isinteger(L, 1));
  lua_pushboolean(L, lua_isstring(L, 1));
  lua_pushvalue(L, 1);
  s = lua_tolstring(L, -1, &len);
  if (s != NULL) {
    lua_pushinteger(L, (lua_Integer)len);
    lua_pushinteger(L, (lua_Integer)strlen(s));
    lua_pushboolean(L, s[len] == '\0');
  } else {
    lua_pushnil(L);
    lua_pushnil(L);
    lua_pushnil(L);
  }
  lua_remove(L, -4);
  lua_pushboolean(L, lua_toboolean(L, 1));
  return 10;
}

/* Tables built, read and walked from C. */
static int tables(lua_State *L) {
  lua_Integer sum = 0;
  lua_createtable(L, 4, 2);
  lua_pushstring(L, "a");
  lua_setfield(L, -2, "x");
  lua_pushinteger(L, 10);
  lua_seti(L, -2, 1);
  lua_pushinteger(L, 20);
  lua_rawseti(L, -2, 2);
  lua_pushstring(L, "k");
  lua_pushinteger(L, 30);
  lua_settable(L, -3);
  lua_pushstring(L, "r");
  lua_pushinteger(L, 40);
  lua_rawset(L, -3);
  lua_getfield(L, -1, "x");
  lua_geti(L, -2, 2);
  lua_rawgeti(L, -3, 1);
  lua_pushstring(L, "k");
  lua_gettable(L, -5);
  lua_pushstring(L, "r");
  lua_rawget(L, -6);
  /* t "a" 20 10 30 40 */
  lua_pushnil(L);
  while (lua_next(L, 1) != 0) {
    if (lua_isinteger(L, -1)) sum += lua_tointeger(L, -1);
    lua_pop(L, 1);
  }
  lua_pushinteger(L, sum);
  lua_pushinteger(L, (lua_Integer)lua_rawlen(L, 1));
  lua_len(L, 1);
  lua_remove(L, 1);
  return 8;
}

/* Globals, set and read by name. */
static int globals(lua_State *L) {
  lua_pushinteger(L, 99);
  lua_setglobal(L, "from_c");
  lua_getglobal(L, "to_c");
  lua_getglobal(L, "from_c");
  return 2;
}

/* Arithmetic, comparison, concatenation and length. */
static int arith(lua_State *L) {
  lua_pushinteger(L, 7);
  lua_pushinteger(L, 2);
  lua_arith(L, LUA_OPIDIV);
  lua_pushnumber(L, 7);
  lua_pushinteger(L, 2);
  lua_arith(L, LUA_OPDIV);
  lua_pushinteger(L, 5);
  lua_arith(L, LUA_OPUNM);
  lua_pushinteger(L, 5);
  lua_pushinteger(L, 3);
  lua_arith(L, LUA_OPMOD);
  lua_pushinteger(L, 2);
  lua_pushinteger(L, 10);
  lua_arith(L, LUA_OPPOW);
  lua_pushinteger(L, lua_compare(L, 1, 2, LUA_OPLT));
  lua_pushinteger(L, lua_compare(L, 1, 1, LUA_OPEQ));
  lua_pushinteger(L, lua_compare(L, 2, 1, LUA_OPLE));
  lua_pushinteger(L, lua_rawequal(L, 1, 2));
  lua_pushstring(L, "ab");
  lua_pushinteger(L, 12);
  lua_pushnumber(L, 3.5);
  lua_concat(L, 3);
  lua_pushinteger(L, (lua_Integer)lua_rawlen(L, -1));
  return lua_gettop(L);
}

/* A counter kept in an upvalue. */
static int counter(lua_State *L) {
  lua_Integer n = lua_tointeger(L, lua_upvalueindex(1)) + 1;
  lua_pushinteger(L, n);
  lua_copy(L, -1, lua_upvalueindex(1));
  lua_pushvalue(L, lua_upvalueindex(2));
  return 2;
}

static int newcounter(lua_State *L) {
  lua_pushinteger(L, 0);
  lua_pushvalue(L, 1);
  lua_pushcclosure(L, counter, 2);
  return 1;
}

/* The same C function without upvalues is the same value. */
static int same(lua_State *L) {
  lua_pushcfunction(L, types);
  lua_pushcfunction(L, types);
  lua_pushboolean(L, lua_rawequal(L, -1, -2));
  lua_pushboolean(L, lua_iscfunction(L, -2));
  lua_pushboolean(L, lua_tocfunction(L, -3) == types);
  return 3;
}

/* A value kept in the registry under a reference. */
static int keep(lua_State *L) {
  lua_settop(L, 1);
  lua_pushinteger(L, luaL_ref(L, LUA_REGISTRYINDEX));
  return 1;
}

static int fetch(lua_State *L) {
  lua_rawgeti(L, LUA_REGISTRYINDEX, luaL_checkinteger(L, 1));
  return 1;
}

static int drop(lua_State *L) {
  luaL_unref(L, LUA_REGISTRYINDEX, (int)luaL_checkinteger(L, 1));
  return 0;
}

/* The registry's reserved slots. */
static int registry(lua_State *L) {
  int globals, same_print, thread, main, same_string;
  lua_settop(L, 0);
  lua_rawgeti(L, LUA_REGISTRYINDEX, LUA_RIDX_GLOBALS);
  globals = lua_type(L, -1) == LUA_TTABLE;
  lua_getfield(L, -1, "print");
  lua_getglobal(L, "print");
  same_print = lua_rawequal(L, -1, -2);
  lua_settop(L, 0);
  lua_rawgeti(L, LUA_REGISTRYINDEX, LUA_RIDX_MAINTHREAD);
  thread = lua_type(L, -1) == LUA_TTHREAD;
  main = lua_pushthread(L);
  lua_settop(L, 0);
  lua_getfield(L, LUA_REGISTRYINDEX, LUA_LOADED_TABLE);
  lua_getfield(L, -1, "string");
  lua_getglobal(L, "string");
  same_string = lua_rawequal(L, -1, -2);
  lua_settop(L, 0);
  lua_pushboolean(L, globals);
  lua_pushboolean(L, same_print);
  lua_pushboolean(L, thread);
  lua_pushboolean(L, main);
  lua_pushboolean(L, same_string);
  return 5;
}

/* Metamethods of a userdata written in C. */
typedef struct {
  lua_Integer v;
} Box;

static int box_new(lua_State *L) {
  Box *b = (Box *)lua_newuserdatauv(L, sizeof(Box), 1);
  b->v = luaL_checkinteger(L, 1);
  lua_pushvalue(L, 2);
  lua_setiuservalue(L, -2, 1);
  luaL_setmetatable(L, "Box");
  return 1;
}

static int box_index(lua_State *L) {
  Box *b = (Box *)luaL_checkudata(L, 1, "Box");
  const char *k = luaL_checkstring(L, 2);
  if (strcmp(k, "v") == 0)
    lua_pushinteger(L, b->v);
  else if (strcmp(k, "tag") == 0)
    lua_getiuservalue(L, 1, 1);
  else
    lua_pushnil(L);
  return 1;
}

static int box_tostring(lua_State *L) {
  Box *b = (Box *)luaL_checkudata(L, 1, "Box");
  lua_pushfstring(L, "Box(%I)", b->v);
  return 1;
}

static int box_add(lua_State *L) {
  Box *a = (Box *)luaL_checkudata(L, 1, "Box");
  lua_Integer k = luaL_checkinteger(L, 2);
  lua_pushinteger(L, a->v + k);
  return 1;
}

static int box_len(lua_State *L) {
  Box *b = (Box *)luaL_checkudata(L, 1, "Box");
  lua_pushinteger(L, b->v * 2);
  return 1;
}

static int box_check(lua_State *L) {
  lua_pushboolean(L, luaL_testudata(L, 1, "Box") != NULL);
  lua_pushboolean(L, lua_isuserdata(L, 1));
  lua_pushinteger(L, (lua_Integer)lua_rawlen(L, 1));
  return 3;
}

static int light(lua_State *L) {
  static int anchor;
  lua_pushlightuserdata(L, &anchor);
  lua_pushlightuserdata(L, &anchor);
  lua_pushboolean(L, lua_rawequal(L, -1, -2));
  lua_pushboolean(L, lua_touserdata(L, -2) == &anchor);
  lua_pushstring(L, luaL_typename(L, -3));
  return 3;
}

static const luaL_Reg box_meta[] = {
    {"__index", box_index},
    {"__tostring", box_tostring},
    {"__add", box_add},
    {"__len", box_len},
    {NULL, NULL}};

static const luaL_Reg funcs[] = {
    {"types", types},       {"shuffle", shuffle},   {"fstrings", fstrings},
    {"convert", convert},   {"tables", tables},     {"globals", globals},
    {"arith", arith},       {"newcounter", newcounter}, {"same", same},
    {"keep", keep},         {"fetch", fetch},       {"drop", drop},
    {"registry", registry}, {"box", box_new},       {"check", box_check},
    {"light", light},       {NULL, NULL}};

LUAMOD_API int luaopen_cbasics(lua_State *L) {
  luaL_newmetatable(L, "Box");
  luaL_setfuncs(L, box_meta, 0);
  lua_pop(L, 1);
  luaL_newlib(L, funcs);
  return 1;
}
