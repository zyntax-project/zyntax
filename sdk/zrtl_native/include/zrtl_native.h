/*
 * Native interop for Zyntax runtimes: an error exit that crosses only
 * foreign frames.
 *
 * A protect frame is a jump target on the stack of the function that
 * made it. zrtl_native_protect pushes one onto a chain the caller keeps
 * (one word, initially NULL), runs a body under it and pops it again;
 * zrtl_native_raise from anywhere the body reaches jumps back to the
 * innermost frame on the chain. The jump restores registers and does
 * nothing else: no destructor, no unwind handler and no signal mask is
 * run or restored, so every frame it crosses must be one that can be
 * discarded (C code with nothing to release), and the target must be
 * older on the same stack.
 */
#ifndef ZRTL_NATIVE_H
#define ZRTL_NATIVE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct zrtl_native_frame zrtl_native_frame;

/* What zrtl_native_raise returns when the chain holds no frame. */
#define ZRTL_NATIVE_NO_FRAME (-1)

/*
 * Run body(ctx) under a new protect frame on *chain. Returns 0 with
 * body's result in *out when body returns, or the nonzero status a
 * raise carried with its value in *out. *chain is as it was on entry
 * either way.
 */
int zrtl_native_protect(zrtl_native_frame **chain, int (*body)(void *),
                        void *ctx, int *out);

/*
 * Jump to the innermost frame on *chain with a nonzero status and a
 * value. Never returns while the chain holds a frame; returns
 * ZRTL_NATIVE_NO_FRAME when it holds none.
 */
int zrtl_native_raise(zrtl_native_frame **chain, int status, int value);

#ifdef __cplusplus
}
#endif

#endif
