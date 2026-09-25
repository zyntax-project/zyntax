/*
 * The protect frame and the raise that jumps to it. See zrtl_native.h.
 */
#include <setjmp.h>

#include "zrtl_native.h"

#if defined(_MSC_VER) && defined(_M_X64)
/*
 * A zero Frame makes longjmp restore the registers without unwinding,
 * so it never walks frames that carry no unwind information or a stack
 * the thread's stack limits do not describe (a fiber's).
 */
typedef char zrtl_native_jb_holds_jump_buffer[
    sizeof(jmp_buf) >= sizeof(_JUMP_BUFFER) ? 1 : -1];
#define ZRTL_SETJMP(jb) setjmp(jb)
#define ZRTL_LONGJMP(jb) longjmp((jb), 1)
#define ZRTL_NO_UNWIND(jb) (((_JUMP_BUFFER *)(jb))->Frame = 0)
#elif defined(_WIN32)
#define ZRTL_SETJMP(jb) setjmp(jb)
#define ZRTL_LONGJMP(jb) longjmp((jb), 1)
#define ZRTL_NO_UNWIND(jb) ((void)0)
#else
/* The forms that leave the signal mask alone: no system call. */
#define ZRTL_SETJMP(jb) _setjmp(jb)
#define ZRTL_LONGJMP(jb) _longjmp((jb), 1)
#define ZRTL_NO_UNWIND(jb) ((void)0)
#endif

struct zrtl_native_frame {
  jmp_buf jb;
  zrtl_native_frame *prev;
  volatile int status;
  volatile int value;
};

int zrtl_native_protect(zrtl_native_frame **chain, int (*body)(void *),
                        void *ctx, int *out) {
  zrtl_native_frame frame;
  frame.prev = *chain;
  frame.status = 0;
  frame.value = 0;
  *chain = &frame;
  if (ZRTL_SETJMP(frame.jb) == 0) {
    int result;
    ZRTL_NO_UNWIND(frame.jb);
    result = body(ctx);
    *chain = frame.prev;
    *out = result;
    return 0;
  }
  *chain = frame.prev;
  *out = frame.value;
  return frame.status;
}

int zrtl_native_raise(zrtl_native_frame **chain, int status, int value) {
  zrtl_native_frame *frame = *chain;
  if (frame == 0) {
    return ZRTL_NATIVE_NO_FRAME;
  }
  frame->status = status;
  frame->value = value;
  ZRTL_LONGJMP(frame->jb);
  return ZRTL_NATIVE_NO_FRAME;
}
