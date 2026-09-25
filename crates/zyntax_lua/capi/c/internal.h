/*
 * The Rust cores behind the veneers of lapi.c.
 *
 * A core never raises. One that can fail returns ZLC_OK, or ZLC_RAISED
 * with the error pending, and its veneer then raises; its results go
 * through pointers. The others return their result.
 */
#ifndef ZLC_INTERNAL_H
#define ZLC_INTERNAL_H

#include <stddef.h>

#include "lua.h"
#include "zrtl_native.h"

#define ZLC_OK 0
#define ZLC_RAISED 1

/*
 * The State's chain of protect frames is the word the lua_State
 * pointer points at; the State's first word, LUA_EXTRASPACE bytes
 * before it, is the extra space. src/state.rs asserts the same layout.
 */
#define ZLC_CHAIN(L) ((zrtl_native_frame **)(void *)(L))
typedef char zlc_extraspace_is_one_word[
    LUA_EXTRASPACE == sizeof(void *) ? 1 : -1];

int zlc_invoke(lua_State *L, lua_CFunction f, int *n);

/* stack */
int zlc_absindex(lua_State *L, int idx);
int zlc_gettop(lua_State *L);
void zlc_settop(lua_State *L, int idx);
void zlc_pushvalue(lua_State *L, int idx);
void zlc_rotate(lua_State *L, int idx, int n);
void zlc_copy(lua_State *L, int fromidx, int toidx);
int zlc_checkstack(lua_State *L, int n);
void zlc_xmove(lua_State *from, lua_State *to, int n);

/* access */
int zlc_type(lua_State *L, int idx);
int zlc_isnumber(lua_State *L, int idx);
int zlc_isstring(lua_State *L, int idx);
int zlc_isinteger(lua_State *L, int idx);
int zlc_iscfunction(lua_State *L, int idx);
int zlc_isuserdata(lua_State *L, int idx);
lua_Number zlc_tonumberx(lua_State *L, int idx, int *isnum);
lua_Integer zlc_tointegerx(lua_State *L, int idx, int *isnum);
int zlc_toboolean(lua_State *L, int idx);
const char *zlc_tolstring(lua_State *L, int idx, size_t *len);
lua_Unsigned zlc_rawlen(lua_State *L, int idx);
lua_CFunction zlc_tocfunction(lua_State *L, int idx);
void *zlc_touserdata(lua_State *L, int idx);
lua_State *zlc_tothread(lua_State *L, int idx);
const void *zlc_topointer(lua_State *L, int idx);

/* pushes */
void zlc_pushnil(lua_State *L);
void zlc_pushnumber(lua_State *L, lua_Number n);
void zlc_pushinteger(lua_State *L, lua_Integer n);
const char *zlc_pushlstring(lua_State *L, const char *s, size_t len);
void zlc_pushboolean(lua_State *L, int b);
void zlc_pushlightuserdata(lua_State *L, void *p);
int zlc_pushthread(lua_State *L);
void zlc_pushcclosure(lua_State *L, lua_CFunction fn, int n);

/* tables: *type is the pushed value's type */
int zlc_getglobal(lua_State *L, const char *name, int *type);
int zlc_gettable(lua_State *L, int idx, int *type);
int zlc_getfield(lua_State *L, int idx, const char *k, int *type);
int zlc_geti(lua_State *L, int idx, lua_Integer n, int *type);
int zlc_rawget(lua_State *L, int idx, int *type);
int zlc_rawgeti(lua_State *L, int idx, lua_Integer n, int *type);
int zlc_rawgetp(lua_State *L, int idx, const void *p, int *type);
void zlc_createtable(lua_State *L, int narr, int nrec);
int zlc_setglobal(lua_State *L, const char *name);
int zlc_settable(lua_State *L, int idx);
int zlc_setfield(lua_State *L, int idx, const char *k);
int zlc_seti(lua_State *L, int idx, lua_Integer n);
int zlc_rawset(lua_State *L, int idx);
int zlc_rawseti(lua_State *L, int idx, lua_Integer n);
int zlc_rawsetp(lua_State *L, int idx, const void *p);
int zlc_next(lua_State *L, int idx, int *more);
int zlc_rawequal(lua_State *L, int idx1, int idx2);
int zlc_compare(lua_State *L, int idx1, int idx2, int op, int *result);
int zlc_arith(lua_State *L, int op);
int zlc_concat(lua_State *L, int n);
int zlc_len(lua_State *L, int idx);
size_t zlc_stringtonumber(lua_State *L, const char *s);
int zlc_getmetatable(lua_State *L, int objindex);
int zlc_setmetatable(lua_State *L, int objindex);

/* userdata */
void *zlc_newuserdatauv(lua_State *L, size_t sz, int nuvalue);
int zlc_getiuservalue(lua_State *L, int idx, int n);
int zlc_setiuservalue(lua_State *L, int idx, int n);
int zlc_getupvalue(lua_State *L, int funcindex, int n, int set);

/* calls and errors */
int zlc_call(lua_State *L, int nargs, int nresults);
int zlc_pcall(lua_State *L, int nargs, int nresults, int errfunc, int *status);
void zlc_error(lua_State *L);
void zlc_panic_text(lua_State *L, char *buf, size_t size);
lua_CFunction zlc_atpanic(lua_State *L, lua_CFunction panicf);
lua_CFunction zlc_panicf(lua_State *L);

/* the collector and the debug interface */
int zlc_gc(lua_State *L, int what, int arg);
int zlc_level(lua_State *L, int level);
int zlc_caller(lua_State *L, char *buf, size_t size);
void zlc_push_function(lua_State *L, int level);

#endif
