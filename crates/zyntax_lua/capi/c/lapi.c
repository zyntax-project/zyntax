/*
 * The functions of lua.h, as veneers over the Rust cores of
 * internal.h.
 *
 * Only C raises. A veneer whose core reports an error pending jumps to
 * the protect frame the running C function was called under, crossing
 * only this file's frames, lauxlib's and the module's own. What a
 * format string, a va_list or a continuation needs is C here too.
 */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"

#include "internal.h"
#include "zrtl_native.h"

const char lua_ident[] =
    "$LuaVersion: " LUA_COPYRIGHT " $"
    "$LuaAuthors: " LUA_AUTHORS " $";

/* ---- raising ------------------------------------------------------- */

/* With no protect frame: the panic function, the message, the end. */
static void zlc_panic(lua_State *L) {
  char msg[512];
  lua_CFunction panicf = zlc_panicf(L);
  zlc_panic_text(L, msg, sizeof msg);
  if (panicf != NULL) {
    zlc_pushlstring(L, msg, strlen(msg));
    panicf(L);
  }
  fprintf(stderr, "PANIC: unprotected error in call to Lua API (%s)\n", msg);
  fflush(stderr);
  abort();
}

/* The pending error, to the innermost protect frame of L. */
static void zlc_raise(lua_State *L) {
  zrtl_native_raise(ZLC_CHAIN(L), ZLC_RAISED, 0);
  zlc_panic(L);
}

#define CHECKED(call) \
  do { if ((call) != ZLC_OK) zlc_raise(L); } while (0)

/* Raise a message about a function this build does not provide. */
static int zlc_unprovided(lua_State *L, const char *name) {
  const char *s = " is not provided by zylua yet";
  char msg[128];
  size_t n = strlen(name);
  if (n > 64) n = 64;
  memcpy(msg, name, n);
  memcpy(msg + n, s, strlen(s) + 1);
  zlc_pushlstring(L, msg, strlen(msg));
  zlc_error(L);
  zlc_raise(L);
  return 0;
}

/* Run a C function under a protect frame on L's chain. */
struct zlc_call_ctx {
  lua_State *L;
  lua_CFunction f;
};

static int invoke_body(void *p) {
  struct zlc_call_ctx *c = (struct zlc_call_ctx *)p;
  return c->f(c->L);
}

int zlc_invoke(lua_State *L, lua_CFunction f, int *n) {
  struct zlc_call_ctx c;
  c.L = L;
  c.f = f;
  return zrtl_native_protect(ZLC_CHAIN(L), invoke_body, &c, n);
}

/* ---- state --------------------------------------------------------- */

LUA_API lua_State *lua_newstate(lua_Alloc f, void *ud) {
  (void)f;
  (void)ud;
  return NULL;
}

LUA_API void lua_close(lua_State *L) { (void)L; }

LUA_API lua_State *lua_newthread(lua_State *L) {
  zlc_unprovided(L, "lua_newthread");
  return NULL;
}

LUA_API int lua_closethread(lua_State *L, lua_State *from) {
  (void)from;
  return zlc_unprovided(L, "lua_closethread");
}

LUA_API int lua_resetthread(lua_State *L) {
  return zlc_unprovided(L, "lua_resetthread");
}

LUA_API lua_CFunction lua_atpanic(lua_State *L, lua_CFunction panicf) {
  return zlc_atpanic(L, panicf);
}

LUA_API lua_Number lua_version(lua_State *L) {
  (void)L;
  return LUA_VERSION_NUM;
}

/* ---- the stack ----------------------------------------------------- */

LUA_API int lua_absindex(lua_State *L, int idx) { return zlc_absindex(L, idx); }
LUA_API int lua_gettop(lua_State *L) { return zlc_gettop(L); }
LUA_API void lua_settop(lua_State *L, int idx) { zlc_settop(L, idx); }
LUA_API void lua_pushvalue(lua_State *L, int idx) { zlc_pushvalue(L, idx); }
LUA_API void lua_rotate(lua_State *L, int idx, int n) { zlc_rotate(L, idx, n); }
LUA_API void lua_copy(lua_State *L, int fromidx, int toidx) {
  zlc_copy(L, fromidx, toidx);
}
LUA_API int lua_checkstack(lua_State *L, int n) { return zlc_checkstack(L, n); }
LUA_API void lua_xmove(lua_State *from, lua_State *to, int n) {
  zlc_xmove(from, to, n);
}

/* ---- access -------------------------------------------------------- */

LUA_API int lua_isnumber(lua_State *L, int idx) { return zlc_isnumber(L, idx); }
LUA_API int lua_isstring(lua_State *L, int idx) { return zlc_isstring(L, idx); }
LUA_API int lua_iscfunction(lua_State *L, int idx) {
  return zlc_iscfunction(L, idx);
}
LUA_API int lua_isinteger(lua_State *L, int idx) { return zlc_isinteger(L, idx); }
LUA_API int lua_isuserdata(lua_State *L, int idx) {
  return zlc_isuserdata(L, idx);
}
LUA_API int lua_type(lua_State *L, int idx) { return zlc_type(L, idx); }

LUA_API const char *lua_typename(lua_State *L, int t) {
  static const char *const names[] = {
      "no value", "nil", "boolean", "userdata", "number",
      "string", "table", "function", "userdata", "thread"};
  (void)L;
  if (t < LUA_TNONE || t > LUA_TTHREAD) return "?";
  return names[t + 1];
}

LUA_API lua_Number lua_tonumberx(lua_State *L, int idx, int *isnum) {
  return zlc_tonumberx(L, idx, isnum);
}
LUA_API lua_Integer lua_tointegerx(lua_State *L, int idx, int *isnum) {
  return zlc_tointegerx(L, idx, isnum);
}
LUA_API int lua_toboolean(lua_State *L, int idx) { return zlc_toboolean(L, idx); }
LUA_API const char *lua_tolstring(lua_State *L, int idx, size_t *len) {
  return zlc_tolstring(L, idx, len);
}
LUA_API lua_Unsigned lua_rawlen(lua_State *L, int idx) { return zlc_rawlen(L, idx); }
LUA_API lua_CFunction lua_tocfunction(lua_State *L, int idx) {
  return zlc_tocfunction(L, idx);
}
LUA_API void *lua_touserdata(lua_State *L, int idx) {
  return zlc_touserdata(L, idx);
}
LUA_API lua_State *lua_tothread(lua_State *L, int idx) {
  return zlc_tothread(L, idx);
}
LUA_API const void *lua_topointer(lua_State *L, int idx) {
  return zlc_topointer(L, idx);
}

/* ---- comparison and arithmetic ------------------------------------- */

LUA_API void lua_arith(lua_State *L, int op) { CHECKED(zlc_arith(L, op)); }

LUA_API int lua_rawequal(lua_State *L, int idx1, int idx2) {
  return zlc_rawequal(L, idx1, idx2);
}

LUA_API int lua_compare(lua_State *L, int idx1, int idx2, int op) {
  int result = 0;
  CHECKED(zlc_compare(L, idx1, idx2, op, &result));
  return result;
}

/* ---- pushes -------------------------------------------------------- */

LUA_API void lua_pushnil(lua_State *L) { zlc_pushnil(L); }
LUA_API void lua_pushnumber(lua_State *L, lua_Number n) { zlc_pushnumber(L, n); }
LUA_API void lua_pushinteger(lua_State *L, lua_Integer n) {
  zlc_pushinteger(L, n);
}

LUA_API const char *lua_pushlstring(lua_State *L, const char *s, size_t len) {
  return zlc_pushlstring(L, s, len);
}

LUA_API const char *lua_pushstring(lua_State *L, const char *s) {
  if (s == NULL) {
    zlc_pushnil(L);
    return NULL;
  }
  return zlc_pushlstring(L, s, strlen(s));
}

/* A growing byte buffer on the C heap. */
typedef struct {
  char *p;
  size_t n;
  size_t cap;
  char fixed[256];
} zlc_buf;

static void buf_init(zlc_buf *b) {
  b->p = b->fixed;
  b->n = 0;
  b->cap = sizeof b->fixed;
}

static void buf_free(zlc_buf *b) {
  if (b->p != b->fixed) free(b->p);
}

static void buf_add(zlc_buf *b, const char *s, size_t len) {
  if (b->n + len > b->cap) {
    size_t cap = b->cap * 2;
    char *p;
    while (cap < b->n + len) cap *= 2;
    p = (char *)malloc(cap);
    if (p == NULL) abort();
    memcpy(p, b->p, b->n);
    buf_free(b);
    b->p = p;
    b->cap = cap;
  }
  memcpy(b->p + b->n, s, len);
  b->n += len;
}

/* A float as lua_Number prints: LUA_NUMBER_FMT, with ".0" when that reads as
   an integer. */
static int number_text(char *out, size_t size, lua_Number n) {
  int len = snprintf(out, size, LUA_NUMBER_FMT, (LUAI_UACNUMBER)n);
  if (len > 0 && (size_t)len + 2 < size &&
      out[strspn(out, "-0123456789")] == '\0') {
    out[len++] = '.';
    out[len++] = '0';
    out[len] = '\0';
  }
  return len;
}

/* A code point as UTF-8, up to 0x7FFFFFFF, written backwards from the
   end of an 8-byte buffer: the bytes written. */
static int utf8_escape(char *buff, unsigned long x) {
  int n = 1;
  if (x < 0x80)
    buff[7] = (char)x;
  else {
    unsigned int mfb = 0x3f;
    do {
      buff[8 - (n++)] = (char)(0x80 | (x & 0x3f));
      x >>= 6;
      mfb >>= 1;
    } while (x > mfb);
    buff[8 - n] = (char)((~mfb << 1) | x);
  }
  return n;
}

LUA_API const char *lua_pushvfstring(lua_State *L, const char *fmt,
                                     va_list argp) {
  zlc_buf b;
  const char *e;
  const char *result;
  char num[64];
  buf_init(&b);
  while ((e = strchr(fmt, '%')) != NULL) {
    int len;
    buf_add(&b, fmt, (size_t)(e - fmt));
    switch (*(e + 1)) {
      case 's': {
        const char *s = va_arg(argp, char *);
        if (s == NULL) s = "(null)";
        buf_add(&b, s, strlen(s));
        break;
      }
      case 'c': {
        char c = (char)(unsigned char)va_arg(argp, int);
        buf_add(&b, &c, 1);
        break;
      }
      case 'd': {
        len = snprintf(num, sizeof num, LUA_INTEGER_FMT,
                       (LUAI_UACINT)va_arg(argp, int));
        buf_add(&b, num, (size_t)len);
        break;
      }
      case 'I': {
        len = snprintf(num, sizeof num, LUA_INTEGER_FMT,
                       (LUAI_UACINT)(lua_Integer)va_arg(argp, LUAI_UACINT));
        buf_add(&b, num, (size_t)len);
        break;
      }
      case 'f': {
        len = number_text(num, sizeof num,
                          (lua_Number)va_arg(argp, LUAI_UACNUMBER));
        buf_add(&b, num, (size_t)len);
        break;
      }
      case 'p': {
        void *p = va_arg(argp, void *);
        len = snprintf(num, sizeof num, "%p", p);
        buf_add(&b, num, (size_t)len);
        break;
      }
      case 'U': {
        char bf[8];
        len = utf8_escape(bf, (unsigned long)va_arg(argp, long));
        buf_add(&b, bf + 8 - len, (size_t)len);
        break;
      }
      case '%': {
        buf_add(&b, "%", 1);
        break;
      }
      default: {
        char msg[64];
        buf_free(&b);
        snprintf(msg, sizeof msg, "invalid option '%%%c' to 'lua_pushfstring'",
                 *(e + 1));
        zlc_pushlstring(L, msg, strlen(msg));
        zlc_error(L);
        zlc_raise(L);
        return NULL;
      }
    }
    fmt = e + 2;
  }
  buf_add(&b, fmt, strlen(fmt));
  result = zlc_pushlstring(L, b.p, b.n);
  buf_free(&b);
  return result;
}

LUA_API const char *lua_pushfstring(lua_State *L, const char *fmt, ...) {
  const char *ret;
  va_list argp;
  va_start(argp, fmt);
  ret = lua_pushvfstring(L, fmt, argp);
  va_end(argp);
  return ret;
}

LUA_API void lua_pushcclosure(lua_State *L, lua_CFunction fn, int n) {
  zlc_pushcclosure(L, fn, n);
}

LUA_API void lua_pushboolean(lua_State *L, int b) { zlc_pushboolean(L, b); }
LUA_API void lua_pushlightuserdata(lua_State *L, void *p) {
  zlc_pushlightuserdata(L, p);
}
LUA_API int lua_pushthread(lua_State *L) { return zlc_pushthread(L); }

/* ---- get functions ------------------------------------------------- */

LUA_API int lua_getglobal(lua_State *L, const char *name) {
  int t = LUA_TNIL;
  CHECKED(zlc_getglobal(L, name, &t));
  return t;
}

LUA_API int lua_gettable(lua_State *L, int idx) {
  int t = LUA_TNIL;
  CHECKED(zlc_gettable(L, idx, &t));
  return t;
}

LUA_API int lua_getfield(lua_State *L, int idx, const char *k) {
  int t = LUA_TNIL;
  CHECKED(zlc_getfield(L, idx, k, &t));
  return t;
}

LUA_API int lua_geti(lua_State *L, int idx, lua_Integer n) {
  int t = LUA_TNIL;
  CHECKED(zlc_geti(L, idx, n, &t));
  return t;
}

LUA_API int lua_rawget(lua_State *L, int idx) {
  int t = LUA_TNIL;
  CHECKED(zlc_rawget(L, idx, &t));
  return t;
}

LUA_API int lua_rawgeti(lua_State *L, int idx, lua_Integer n) {
  int t = LUA_TNIL;
  CHECKED(zlc_rawgeti(L, idx, n, &t));
  return t;
}

LUA_API int lua_rawgetp(lua_State *L, int idx, const void *p) {
  int t = LUA_TNIL;
  CHECKED(zlc_rawgetp(L, idx, p, &t));
  return t;
}

LUA_API void lua_createtable(lua_State *L, int narr, int nrec) {
  zlc_createtable(L, narr, nrec);
}

LUA_API void *lua_newuserdatauv(lua_State *L, size_t sz, int nuvalue) {
  return zlc_newuserdatauv(L, sz, nuvalue);
}

LUA_API int lua_getmetatable(lua_State *L, int objindex) {
  return zlc_getmetatable(L, objindex);
}

LUA_API int lua_getiuservalue(lua_State *L, int idx, int n) {
  return zlc_getiuservalue(L, idx, n);
}

/* ---- set functions ------------------------------------------------- */

LUA_API void lua_setglobal(lua_State *L, const char *name) {
  CHECKED(zlc_setglobal(L, name));
}

LUA_API void lua_settable(lua_State *L, int idx) { CHECKED(zlc_settable(L, idx)); }

LUA_API void lua_setfield(lua_State *L, int idx, const char *k) {
  CHECKED(zlc_setfield(L, idx, k));
}

LUA_API void lua_seti(lua_State *L, int idx, lua_Integer n) {
  CHECKED(zlc_seti(L, idx, n));
}

LUA_API void lua_rawset(lua_State *L, int idx) { CHECKED(zlc_rawset(L, idx)); }

LUA_API void lua_rawseti(lua_State *L, int idx, lua_Integer n) {
  CHECKED(zlc_rawseti(L, idx, n));
}

LUA_API void lua_rawsetp(lua_State *L, int idx, const void *p) {
  CHECKED(zlc_rawsetp(L, idx, p));
}

LUA_API int lua_setmetatable(lua_State *L, int objindex) {
  CHECKED(zlc_setmetatable(L, objindex));
  return 1;
}

LUA_API int lua_setiuservalue(lua_State *L, int idx, int n) {
  return zlc_setiuservalue(L, idx, n);
}

/* ---- calls --------------------------------------------------------- */

LUA_API void lua_callk(lua_State *L, int nargs, int nresults,
                       lua_KContext ctx, lua_KFunction k) {
  (void)ctx;
  (void)k;
  CHECKED(zlc_call(L, nargs, nresults));
}

LUA_API int lua_pcallk(lua_State *L, int nargs, int nresults, int errfunc,
                       lua_KContext ctx, lua_KFunction k) {
  int status = LUA_OK;
  (void)ctx;
  (void)k;
  CHECKED(zlc_pcall(L, nargs, nresults, errfunc, &status));
  return status;
}

LUA_API int lua_load(lua_State *L, lua_Reader reader, void *dt,
                     const char *chunkname, const char *mode) {
  (void)reader;
  (void)dt;
  (void)chunkname;
  (void)mode;
  return zlc_unprovided(L, "lua_load");
}

LUA_API int lua_dump(lua_State *L, lua_Writer writer, void *data, int strip) {
  (void)writer;
  (void)data;
  (void)strip;
  return zlc_unprovided(L, "lua_dump");
}

/* ---- coroutines ---------------------------------------------------- */

LUA_API int lua_yieldk(lua_State *L, int nresults, lua_KContext ctx,
                       lua_KFunction k) {
  (void)nresults;
  (void)ctx;
  (void)k;
  return zlc_unprovided(L, "lua_yieldk");
}

LUA_API int lua_resume(lua_State *L, lua_State *from, int narg, int *nres) {
  (void)from;
  (void)narg;
  (void)nres;
  return zlc_unprovided(L, "lua_resume");
}

LUA_API int lua_status(lua_State *L) {
  (void)L;
  return LUA_OK;
}

LUA_API int lua_isyieldable(lua_State *L) {
  (void)L;
  return 0;
}

/* Warnings are off: nothing a C module warns about is shown. */
LUA_API void lua_setwarnf(lua_State *L, lua_WarnFunction f, void *ud) {
  (void)L;
  (void)f;
  (void)ud;
}

LUA_API void lua_warning(lua_State *L, const char *msg, int tocont) {
  (void)L;
  (void)msg;
  (void)tocont;
}

/* ---- the collector ------------------------------------------------- */

LUA_API int lua_gc(lua_State *L, int what, ...) {
  int arg = 0;
  va_list argp;
  va_start(argp, what);
  switch (what) {
    case LUA_GCSTEP:
    case LUA_GCSETPAUSE:
    case LUA_GCSETSTEPMUL:
      arg = va_arg(argp, int);
      break;
    default:
      break;
  }
  va_end(argp);
  return zlc_gc(L, what, arg);
}

/* ---- miscellaneous ------------------------------------------------- */

LUA_API int lua_error(lua_State *L) {
  zlc_error(L);
  zlc_raise(L);
  return 0;
}

LUA_API int lua_next(lua_State *L, int idx) {
  int more = 0;
  CHECKED(zlc_next(L, idx, &more));
  return more;
}

LUA_API void lua_concat(lua_State *L, int n) { CHECKED(zlc_concat(L, n)); }

LUA_API void lua_len(lua_State *L, int idx) { CHECKED(zlc_len(L, idx)); }

LUA_API size_t lua_stringtonumber(lua_State *L, const char *s) {
  return zlc_stringtonumber(L, s);
}

/* Memory C code asks the state for is the C heap's. */
static void *zlc_alloc(void *ud, void *ptr, size_t osize, size_t nsize) {
  (void)ud;
  (void)osize;
  if (nsize == 0) {
    free(ptr);
    return NULL;
  }
  return realloc(ptr, nsize);
}

LUA_API lua_Alloc lua_getallocf(lua_State *L, void **ud) {
  (void)L;
  if (ud) *ud = NULL;
  return zlc_alloc;
}

LUA_API void lua_setallocf(lua_State *L, lua_Alloc f, void *ud) {
  (void)L;
  (void)f;
  (void)ud;
}

/*
 * A slot marked to be closed is closed by lua_closeslot; one still
 * open when its frame raises is left to the collector.
 */
LUA_API void lua_toclose(lua_State *L, int idx) {
  (void)L;
  (void)idx;
}

LUA_API void lua_closeslot(lua_State *L, int idx) {
  idx = lua_absindex(L, idx);
  if (luaL_getmetafield(L, idx, "__close") != LUA_TNIL) {
    lua_pushvalue(L, idx);
    lua_pushnil(L);
    lua_call(L, 2, 0);
  }
  lua_pushnil(L);
  lua_replace(L, idx);
}

/* ---- the debug interface ------------------------------------------- */

/*
 * Level 0 is the running C function and level 1 the code that called
 * it; ar->i_ci carries the level plus one.
 */
LUA_API int lua_getstack(lua_State *L, int level, lua_Debug *ar) {
  if (level < 0 || !zlc_level(L, level)) return 0;
  ar->i_ci = (struct CallInfo *)(size_t)(level + 1);
  return 1;
}

static void set_c_source(lua_Debug *ar) {
  ar->source = "=[C]";
  ar->srclen = 4;
  ar->linedefined = -1;
  ar->lastlinedefined = -1;
  ar->what = "C";
  strcpy(ar->short_src, "[C]");
}

LUA_API int lua_getinfo(lua_State *L, const char *what, lua_Debug *ar) {
  int level = -1;
  int line = -1;
  char chunk[LUA_IDSIZE];
  int ok = 1;
  if (*what == '>') {
    zlc_settop(L, -2);
    what++;
  } else {
    level = (int)(size_t)ar->i_ci - 1;
  }
  chunk[0] = '\0';
  if (level == 1) {
    line = zlc_caller(L, chunk, sizeof chunk);
    if (line <= 0) line = -1;
  }
  for (; *what; what++) {
    switch (*what) {
      case 'S':
        if (level == 1 && line > 0) {
          ar->source = "=?";
          ar->srclen = 2;
          ar->linedefined = 0;
          ar->lastlinedefined = -1;
          ar->what = "Lua";
          strncpy(ar->short_src, chunk, LUA_IDSIZE - 1);
          ar->short_src[LUA_IDSIZE - 1] = '\0';
        } else {
          set_c_source(ar);
        }
        break;
      case 'l':
        ar->currentline = line;
        break;
      case 'u':
        ar->nups = 0;
        ar->nparams = 0;
        ar->isvararg = 1;
        break;
      case 'n':
        ar->name = NULL;
        ar->namewhat = "";
        break;
      case 't':
        ar->istailcall = 0;
        break;
      case 'r':
        ar->ftransfer = 0;
        ar->ntransfer = 0;
        break;
      case 'f':
        zlc_push_function(L, level);
        break;
      case 'L':
        zlc_pushnil(L);
        break;
      default:
        ok = 0;
        break;
    }
  }
  return ok;
}

LUA_API const char *lua_getlocal(lua_State *L, const lua_Debug *ar, int n) {
  (void)L;
  (void)ar;
  (void)n;
  return NULL;
}

LUA_API const char *lua_setlocal(lua_State *L, const lua_Debug *ar, int n) {
  (void)L;
  (void)ar;
  (void)n;
  return NULL;
}

LUA_API const char *lua_getupvalue(lua_State *L, int funcindex, int n) {
  return zlc_getupvalue(L, funcindex, n, 0) ? "" : NULL;
}

LUA_API const char *lua_setupvalue(lua_State *L, int funcindex, int n) {
  return zlc_getupvalue(L, funcindex, n, 1) ? "" : NULL;
}

LUA_API void *lua_upvalueid(lua_State *L, int fidx, int n) {
  (void)fidx;
  (void)n;
  zlc_unprovided(L, "lua_upvalueid");
  return NULL;
}

LUA_API void lua_upvaluejoin(lua_State *L, int fidx1, int n1, int fidx2,
                             int n2) {
  (void)fidx1;
  (void)n1;
  (void)fidx2;
  (void)n2;
  zlc_unprovided(L, "lua_upvaluejoin");
}

LUA_API void lua_sethook(lua_State *L, lua_Hook func, int mask, int count) {
  (void)mask;
  (void)count;
  if (func != NULL) zlc_unprovided(L, "lua_sethook");
}

LUA_API lua_Hook lua_gethook(lua_State *L) {
  (void)L;
  return NULL;
}

LUA_API int lua_gethookmask(lua_State *L) {
  (void)L;
  return 0;
}

LUA_API int lua_gethookcount(lua_State *L) {
  (void)L;
  return 0;
}

LUA_API int lua_setcstacklimit(lua_State *L, unsigned int limit) {
  (void)L;
  (void)limit;
  return 200;
}

/* ---- lualib.h ------------------------------------------------------ */

LUALIB_API void luaL_openlibs(lua_State *L) {
  zlc_unprovided(L, "luaL_openlibs");
}
