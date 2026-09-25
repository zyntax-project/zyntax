/*
 * C frames for the boundary's tests: bodies that raise from some depth
 * of plain C calls below the protect frame.
 */
#include "zrtl_native.h"

struct zn_test_ctx {
  zrtl_native_frame **chain;
  int depth;
  int value;
  int reached;
};

static int zn_test_descend(struct zn_test_ctx *ctx, int depth) {
  volatile int local = depth;
  if (depth == 0) {
    ctx->reached = 1;
    zrtl_native_raise(ctx->chain, 7, ctx->value);
    ctx->reached = 2;
    return -1;
  }
  return zn_test_descend(ctx, depth - 1) + local;
}

/* A body that calls `depth` frames deep and raises status 7 there. */
int zn_test_raise_at_depth(void *p) {
  struct zn_test_ctx *ctx = (struct zn_test_ctx *)p;
  return zn_test_descend(ctx, ctx->depth);
}

/* A body that returns its value without raising. */
int zn_test_return(void *p) {
  struct zn_test_ctx *ctx = (struct zn_test_ctx *)p;
  return ctx->value;
}

struct zn_test_nested {
  zrtl_native_frame **chain;
  int inner_status;
  int inner_value;
  int outer_value;
};

static int zn_test_inner(void *p) {
  struct zn_test_nested *n = (struct zn_test_nested *)p;
  zrtl_native_raise(n->chain, 3, 11);
  return 0;
}

/*
 * A body that protects an inner body which raises, records what the
 * inner frame caught, then raises to the outer frame.
 */
int zn_test_nested_body(void *p) {
  struct zn_test_nested *n = (struct zn_test_nested *)p;
  int out = 0;
  n->inner_status = zrtl_native_protect(n->chain, zn_test_inner, n, &out);
  n->inner_value = out;
  zrtl_native_raise(n->chain, 5, n->outer_value);
  return 0;
}

/* A raise with no frame on the chain, which returns. */
int zn_test_raise_unprotected(zrtl_native_frame **chain) {
  return zrtl_native_raise(chain, 9, 1);
}

struct zn_test_pause {
  zrtl_native_frame **chain;
  void (*pause)(void);
  int value;
};

/* A body that calls `pause`, which may switch stacks and back, then
   raises status 7 three C frames below its own. */
int zn_test_pause_then_raise(void *p) {
  struct zn_test_pause *ctx = (struct zn_test_pause *)p;
  struct zn_test_ctx inner;
  ctx->pause();
  inner.chain = ctx->chain;
  inner.depth = 3;
  inner.value = ctx->value;
  inner.reached = 0;
  zn_test_descend(&inner, inner.depth);
  return -1;
}
