/*
 * ZRTL (Zyntax Runtime Library) SDK
 *
 * This header provides the interface for creating ZRTL plugins
 * that can be loaded by the Zyntax compiler at runtime.
 *
 * Similar to HashLink's HDLL format, ZRTL files are native dynamic
 * libraries that export runtime symbols for the JIT/AOT compiler.
 *
 * Example usage:
 *
 *   #include "zrtl.h"
 *
 *   // Define your runtime functions
 *   int32_t my_add(int32_t a, int32_t b) {
 *       return a + b;
 *   }
 *
 *   void my_print(const char* msg) {
 *       printf("%s\n", msg);
 *   }
 *
 *   // Export symbol table (required)
 *   ZRTL_SYMBOLS_BEGIN
 *       ZRTL_SYMBOL("$MyRuntime$add", my_add),
 *       ZRTL_SYMBOL("$MyRuntime$print", my_print),
 *   ZRTL_SYMBOLS_END
 *
 *   // Export plugin info (required)
 *   ZRTL_PLUGIN_INFO("my_runtime")
 *
 * Build commands:
 *
 *   Linux:   gcc -shared -fPIC -o my_runtime.zrtl my_runtime.c
 *   macOS:   clang -shared -fPIC -o my_runtime.zrtl my_runtime.c
 *   Windows: cl /LD my_runtime.c /Fe:my_runtime.zrtl
 *
 * Load in zyntax:
 *
 *   zyntax compile --runtime my_runtime.zrtl --source input.hx
 */

#ifndef ZRTL_H
#define ZRTL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ZRTL format version - must match the compiler's expected version */
#define ZRTL_VERSION 1

/* Platform-specific export macro */
#ifdef _WIN32
    #define ZRTL_EXPORT __declspec(dllexport)
#else
    #define ZRTL_EXPORT __attribute__((visibility("default")))
#endif

/* Symbol entry structure */
typedef struct {
    const char* name;   /* Symbol name, e.g., "$Array$push" */
    void* ptr;          /* Function pointer */
} ZrtlSymbol;

/* Plugin info structure */
typedef struct {
    uint32_t version;   /* Must be ZRTL_VERSION */
    const char* name;   /* Plugin name */
} ZrtlInfo;

/*
 * Macros for defining the symbol table
 *
 * Usage:
 *   ZRTL_SYMBOLS_BEGIN
 *       ZRTL_SYMBOL("$Type$method", function_ptr),
 *       ZRTL_SYMBOL("$Type$other", other_ptr),
 *   ZRTL_SYMBOLS_END
 */
#define ZRTL_SYMBOLS_BEGIN \
    ZRTL_EXPORT ZrtlSymbol _zrtl_symbols[] = {

#define ZRTL_SYMBOL(name, func) \
    { name, (void*)(func) }

#define ZRTL_SYMBOLS_END \
        { NULL, NULL } \
    };

/*
 * Macro for defining plugin info
 *
 * Usage:
 *   ZRTL_PLUGIN_INFO("my_plugin_name")
 */
#define ZRTL_PLUGIN_INFO(plugin_name) \
    ZRTL_EXPORT ZrtlInfo _zrtl_info = { \
        .version = ZRTL_VERSION, \
        .name = plugin_name \
    };

/* ============================================================
 * Type Tags for Dynamic Type Identification
 * ============================================================
 *
 * Type tags identify the runtime type of opaque values passed between
 * Zyntax and native code. This enables type-safe interop with boxed values.
 *
 * Tag layout (32-bit):
 *   Bits 0-7:   Category (primitive, struct, array, function, etc.)
 *   Bits 8-23:  Type ID within category (registered at runtime)
 *   Bits 24-31: Flags (nullable, mutable, etc.)
 */

/* Type categories */
typedef enum {
    ZRTL_CAT_VOID       = 0x00,  /* void / unit type */
    ZRTL_CAT_BOOL       = 0x01,  /* boolean */
    ZRTL_CAT_INT        = 0x02,  /* signed integers (i8, i16, i32, i64) */
    ZRTL_CAT_UINT       = 0x03,  /* unsigned integers (u8, u16, u32, u64) */
    ZRTL_CAT_FLOAT      = 0x04,  /* floating point (f32, f64) */
    ZRTL_CAT_STRING     = 0x05,  /* string */
    ZRTL_CAT_ARRAY      = 0x06,  /* array/vector */
    ZRTL_CAT_MAP        = 0x07,  /* hashmap/dictionary */
    ZRTL_CAT_STRUCT     = 0x08,  /* struct/record */
    ZRTL_CAT_CLASS      = 0x09,  /* class instance */
    ZRTL_CAT_ENUM       = 0x0A,  /* enum variant */
    ZRTL_CAT_UNION      = 0x0B,  /* tagged union */
    ZRTL_CAT_FUNCTION   = 0x0C,  /* function pointer/closure */
    ZRTL_CAT_POINTER    = 0x0D,  /* raw pointer */
    ZRTL_CAT_OPTIONAL   = 0x0E,  /* optional/nullable */
    ZRTL_CAT_RESULT     = 0x0F,  /* result/error type */
    ZRTL_CAT_TUPLE      = 0x10,  /* tuple */
    ZRTL_CAT_TRAIT_OBJ  = 0x11,  /* trait object / dyn */
    ZRTL_CAT_OPAQUE     = 0x12,  /* opaque/foreign type */
    ZRTL_CAT_CUSTOM     = 0xFF,  /* custom user-defined category */
} ZrtlTypeCategory;

/* Type flags */
typedef enum {
    ZRTL_FLAG_NONE      = 0x00,
    ZRTL_FLAG_NULLABLE  = 0x01,  /* Type can be null */
    ZRTL_FLAG_MUTABLE   = 0x02,  /* Value is mutable */
    ZRTL_FLAG_BOXED     = 0x04,  /* Value is heap-allocated */
    ZRTL_FLAG_ARC       = 0x08,  /* Value uses ARC */
    ZRTL_FLAG_WEAK      = 0x10,  /* Weak reference */
    ZRTL_FLAG_PINNED    = 0x20,  /* Value cannot be moved */
} ZrtlTypeFlags;

/* Primitive type IDs (within INT/UINT/FLOAT categories) */
typedef enum {
    ZRTL_PRIM_8   = 0x01,  /* 8-bit (i8, u8) */
    ZRTL_PRIM_16  = 0x02,  /* 16-bit (i16, u16) */
    ZRTL_PRIM_32  = 0x03,  /* 32-bit (i32, u32, f32) */
    ZRTL_PRIM_64  = 0x04,  /* 64-bit (i64, u64, f64) */
    ZRTL_PRIM_PTR = 0x05,  /* pointer-sized (isize, usize) */
} ZrtlPrimitiveSize;

/* Type tag - 32-bit packed type identifier */
typedef uint32_t ZrtlTypeTag;

/* Construct a type tag */
#define ZRTL_MAKE_TAG(category, type_id, flags) \
    ((ZrtlTypeTag)(((flags) << 24) | ((type_id) << 8) | (category)))

/* Extract components from a type tag */
#define ZRTL_TAG_CATEGORY(tag) ((ZrtlTypeCategory)((tag) & 0xFF))
#define ZRTL_TAG_TYPE_ID(tag)  ((uint16_t)(((tag) >> 8) & 0xFFFF))
#define ZRTL_TAG_FLAGS(tag)    ((ZrtlTypeFlags)(((tag) >> 24) & 0xFF))

/* Pre-defined type tags for primitives */
#define ZRTL_TAG_VOID    ZRTL_MAKE_TAG(ZRTL_CAT_VOID, 0, 0)
#define ZRTL_TAG_BOOL    ZRTL_MAKE_TAG(ZRTL_CAT_BOOL, 0, 0)
#define ZRTL_TAG_I8      ZRTL_MAKE_TAG(ZRTL_CAT_INT, ZRTL_PRIM_8, 0)
#define ZRTL_TAG_I16     ZRTL_MAKE_TAG(ZRTL_CAT_INT, ZRTL_PRIM_16, 0)
#define ZRTL_TAG_I32     ZRTL_MAKE_TAG(ZRTL_CAT_INT, ZRTL_PRIM_32, 0)
#define ZRTL_TAG_I64     ZRTL_MAKE_TAG(ZRTL_CAT_INT, ZRTL_PRIM_64, 0)
#define ZRTL_TAG_ISIZE   ZRTL_MAKE_TAG(ZRTL_CAT_INT, ZRTL_PRIM_PTR, 0)
#define ZRTL_TAG_U8      ZRTL_MAKE_TAG(ZRTL_CAT_UINT, ZRTL_PRIM_8, 0)
#define ZRTL_TAG_U16     ZRTL_MAKE_TAG(ZRTL_CAT_UINT, ZRTL_PRIM_16, 0)
#define ZRTL_TAG_U32     ZRTL_MAKE_TAG(ZRTL_CAT_UINT, ZRTL_PRIM_32, 0)
#define ZRTL_TAG_U64     ZRTL_MAKE_TAG(ZRTL_CAT_UINT, ZRTL_PRIM_64, 0)
#define ZRTL_TAG_USIZE   ZRTL_MAKE_TAG(ZRTL_CAT_UINT, ZRTL_PRIM_PTR, 0)
#define ZRTL_TAG_F32     ZRTL_MAKE_TAG(ZRTL_CAT_FLOAT, ZRTL_PRIM_32, 0)
#define ZRTL_TAG_F64     ZRTL_MAKE_TAG(ZRTL_CAT_FLOAT, ZRTL_PRIM_64, 0)
#define ZRTL_TAG_STRING  ZRTL_MAKE_TAG(ZRTL_CAT_STRING, 0, 0)

/* ============================================================
 * DynamicBox - Runtime Boxed Value
 * ============================================================
 *
 * DynamicBox is the standard way to pass opaque/polymorphic values
 * between Zyntax code and native runtime functions.
 *
 * Layout (24 bytes on 64-bit):
 *   - tag:     Type tag identifying the contained type
 *   - size:    Size of the data in bytes
 *   - data:    Pointer to the actual data
 *   - dropper: Optional destructor function
 *
 * Usage in native code:
 *
 *   void process_value(ZrtlDynamicBox* box) {
 *       switch (ZRTL_TAG_CATEGORY(box->tag)) {
 *           case ZRTL_CAT_INT:
 *               // Handle integer
 *               int64_t value = zrtl_box_as_i64(box);
 *               break;
 *           case ZRTL_CAT_STRING:
 *               // Handle string
 *               ZrtlString* str = (ZrtlString*)box->data;
 *               break;
 *           case ZRTL_CAT_STRUCT:
 *               // Handle struct by type ID
 *               if (ZRTL_TAG_TYPE_ID(box->tag) == MY_POINT_TYPE_ID) {
 *                   MyPoint* pt = (MyPoint*)box->data;
 *               }
 *               break;
 *       }
 *   }
 */

/* Destructor function type for boxed values */
typedef void (*ZrtlDropFn)(void* data);

/* ============================================================
 * Memory management helpers (defined early - used by other helpers)
 * ============================================================ */

/* Allocate memory (use platform malloc) */
static inline void* zrtl_alloc(size_t size) {
    extern void* malloc(size_t);
    return malloc(size);
}

/* Free memory (use platform free) */
static inline void zrtl_free(void* ptr) {
    extern void free(void*);
    free(ptr);
}

/* Reallocate memory */
static inline void* zrtl_realloc(void* ptr, size_t size) {
    extern void* realloc(void*, size_t);
    return realloc(ptr, size);
}

/* Dynamic boxed value */
typedef struct {
    ZrtlTypeTag tag;       /* Type tag identifying the type */
    uint32_t size;         /* Size of data in bytes */
    void* data;            /* Pointer to the data */
    ZrtlDropFn dropper;    /* Optional destructor (NULL if no cleanup needed) */
} ZrtlDynamicBox;

/* Check if a box is null/empty */
#define ZRTL_BOX_IS_NULL(box) ((box)->data == NULL)

/* Check box type category */
#define ZRTL_BOX_IS_CATEGORY(box, cat) (ZRTL_TAG_CATEGORY((box)->tag) == (cat))

/* Type-safe accessors for primitive types */
static inline int8_t zrtl_box_as_i8(const ZrtlDynamicBox* box) {
    return box->data ? *(int8_t*)box->data : 0;
}

static inline int16_t zrtl_box_as_i16(const ZrtlDynamicBox* box) {
    return box->data ? *(int16_t*)box->data : 0;
}

static inline int32_t zrtl_box_as_i32(const ZrtlDynamicBox* box) {
    return box->data ? *(int32_t*)box->data : 0;
}

static inline int64_t zrtl_box_as_i64(const ZrtlDynamicBox* box) {
    return box->data ? *(int64_t*)box->data : 0;
}

static inline uint8_t zrtl_box_as_u8(const ZrtlDynamicBox* box) {
    return box->data ? *(uint8_t*)box->data : 0;
}

static inline uint16_t zrtl_box_as_u16(const ZrtlDynamicBox* box) {
    return box->data ? *(uint16_t*)box->data : 0;
}

static inline uint32_t zrtl_box_as_u32(const ZrtlDynamicBox* box) {
    return box->data ? *(uint32_t*)box->data : 0;
}

static inline uint64_t zrtl_box_as_u64(const ZrtlDynamicBox* box) {
    return box->data ? *(uint64_t*)box->data : 0;
}

static inline float zrtl_box_as_f32(const ZrtlDynamicBox* box) {
    return box->data ? *(float*)box->data : 0.0f;
}

static inline double zrtl_box_as_f64(const ZrtlDynamicBox* box) {
    return box->data ? *(double*)box->data : 0.0;
}

static inline uint8_t zrtl_box_as_bool(const ZrtlDynamicBox* box) {
    return box->data ? *(uint8_t*)box->data : 0;
}

/* Get data as typed pointer (caller must verify tag first) */
#define ZRTL_BOX_AS(box, type) ((type*)((box)->data))

/* Create a box for primitive values (stack-allocated data) */
static inline ZrtlDynamicBox zrtl_box_i32(int32_t* value) {
    ZrtlDynamicBox box = { ZRTL_TAG_I32, sizeof(int32_t), value, NULL };
    return box;
}

static inline ZrtlDynamicBox zrtl_box_i64(int64_t* value) {
    ZrtlDynamicBox box = { ZRTL_TAG_I64, sizeof(int64_t), value, NULL };
    return box;
}

static inline ZrtlDynamicBox zrtl_box_f32(float* value) {
    ZrtlDynamicBox box = { ZRTL_TAG_F32, sizeof(float), value, NULL };
    return box;
}

static inline ZrtlDynamicBox zrtl_box_f64(double* value) {
    ZrtlDynamicBox box = { ZRTL_TAG_F64, sizeof(double), value, NULL };
    return box;
}

static inline ZrtlDynamicBox zrtl_box_bool(uint8_t* value) {
    ZrtlDynamicBox box = { ZRTL_TAG_BOOL, sizeof(uint8_t), value, NULL };
    return box;
}

/* Create a heap-allocated box */
static inline ZrtlDynamicBox zrtl_box_alloc(ZrtlTypeTag tag, uint32_t size) {
    ZrtlDynamicBox box;
    box.tag = tag;
    box.size = size;
    box.data = zrtl_alloc(size);
    box.dropper = zrtl_free;
    return box;
}

/* Free a boxed value */
static inline void zrtl_box_free(ZrtlDynamicBox* box) {
    if (box->data && box->dropper) {
        box->dropper(box->data);
    }
    box->data = NULL;
    box->size = 0;
    box->tag = ZRTL_TAG_VOID;
    box->dropper = NULL;
}

/* Clone a box (deep copy) */
static inline ZrtlDynamicBox zrtl_box_clone(const ZrtlDynamicBox* box) {
    extern void* memcpy(void*, const void*, size_t);
    ZrtlDynamicBox clone;
    clone.tag = box->tag;
    clone.size = box->size;
    if (box->data && box->size > 0) {
        clone.data = zrtl_alloc(box->size);
        memcpy(clone.data, box->data, box->size);
        clone.dropper = zrtl_free;
    } else {
        clone.data = NULL;
        clone.dropper = NULL;
    }
    return clone;
}

/* ============================================================
 * Generic Type Support
 * ============================================================
 *
 * For generic types like Array<T>, Map<K,V>, Optional<T>, etc.,
 * we need to track the type arguments. This is done via a
 * GenericTypeArgs structure that accompanies the DynamicBox.
 *
 * Example: Array<Int32> would have:
 *   - tag: ZRTL_CAT_ARRAY
 *   - type_args: [ZRTL_TAG_I32]
 *
 * Example: Map<String, Array<Int32>> would have:
 *   - tag: ZRTL_CAT_MAP
 *   - type_args: [ZRTL_TAG_STRING, ZRTL_MAKE_TAG(ZRTL_CAT_ARRAY, ...)]
 *   - nested GenericTypeArgs for the Array<Int32>
 */

/* Maximum type arguments for a generic type */
#define ZRTL_MAX_TYPE_ARGS 8

/* Generic type arguments */
typedef struct ZrtlGenericTypeArgs {
    uint8_t count;                                /* Number of type arguments */
    ZrtlTypeTag args[ZRTL_MAX_TYPE_ARGS];         /* Type tags for each argument */
    struct ZrtlGenericTypeArgs* nested[ZRTL_MAX_TYPE_ARGS]; /* Nested generics (NULL if not generic) */
} ZrtlGenericTypeArgs;

/* Extended DynamicBox with generic type info */
typedef struct {
    ZrtlDynamicBox base;                          /* Base box (tag, size, data, dropper) */
    ZrtlGenericTypeArgs* type_args;               /* Generic type arguments (NULL if non-generic) */
} ZrtlGenericBox;

/* Check if a box is a generic type */
#define ZRTL_BOX_IS_GENERIC(gbox) ((gbox)->type_args != NULL && (gbox)->type_args->count > 0)

/* Get type argument at index */
static inline ZrtlTypeTag zrtl_gbox_type_arg(const ZrtlGenericBox* gbox, uint8_t index) {
    if (!gbox->type_args || index >= gbox->type_args->count) {
        return ZRTL_TAG_VOID;
    }
    return gbox->type_args->args[index];
}

/* Create generic type args (caller owns memory) */
static inline ZrtlGenericTypeArgs* zrtl_type_args_new(uint8_t count) {
    extern void* memset(void*, int, size_t);
    ZrtlGenericTypeArgs* args = (ZrtlGenericTypeArgs*)zrtl_alloc(sizeof(ZrtlGenericTypeArgs));
    memset(args, 0, sizeof(ZrtlGenericTypeArgs));
    args->count = count;
    return args;
}

/* Free generic type args (including nested) */
static inline void zrtl_type_args_free(ZrtlGenericTypeArgs* args) {
    if (!args) return;
    for (uint8_t i = 0; i < args->count; i++) {
        if (args->nested[i]) {
            zrtl_type_args_free(args->nested[i]);
        }
    }
    zrtl_free(args);
}

/* Create a generic box for Array<T> */
static inline ZrtlGenericBox zrtl_gbox_array(void* data, uint32_t size, ZrtlTypeTag element_type) {
    ZrtlGenericBox gbox;
    gbox.base.tag = ZRTL_MAKE_TAG(ZRTL_CAT_ARRAY, 0, 0);
    gbox.base.size = size;
    gbox.base.data = data;
    gbox.base.dropper = NULL;
    gbox.type_args = zrtl_type_args_new(1);
    gbox.type_args->args[0] = element_type;
    return gbox;
}

/* Create a generic box for Map<K, V> */
static inline ZrtlGenericBox zrtl_gbox_map(void* data, uint32_t size, ZrtlTypeTag key_type, ZrtlTypeTag value_type) {
    ZrtlGenericBox gbox;
    gbox.base.tag = ZRTL_MAKE_TAG(ZRTL_CAT_MAP, 0, 0);
    gbox.base.size = size;
    gbox.base.data = data;
    gbox.base.dropper = NULL;
    gbox.type_args = zrtl_type_args_new(2);
    gbox.type_args->args[0] = key_type;
    gbox.type_args->args[1] = value_type;
    return gbox;
}

/* Create a generic box for Optional<T> */
static inline ZrtlGenericBox zrtl_gbox_optional(void* data, uint32_t size, ZrtlTypeTag inner_type) {
    ZrtlGenericBox gbox;
    gbox.base.tag = ZRTL_MAKE_TAG(ZRTL_CAT_OPTIONAL, 0, 0);
    gbox.base.size = size;
    gbox.base.data = data;
    gbox.base.dropper = NULL;
    gbox.type_args = zrtl_type_args_new(1);
    gbox.type_args->args[0] = inner_type;
    return gbox;
}

/* Create a generic box for Result<T, E> */
static inline ZrtlGenericBox zrtl_gbox_result(void* data, uint32_t size, ZrtlTypeTag ok_type, ZrtlTypeTag err_type) {
    ZrtlGenericBox gbox;
    gbox.base.tag = ZRTL_MAKE_TAG(ZRTL_CAT_RESULT, 0, 0);
    gbox.base.size = size;
    gbox.base.data = data;
    gbox.base.dropper = NULL;
    gbox.type_args = zrtl_type_args_new(2);
    gbox.type_args->args[0] = ok_type;
    gbox.type_args->args[1] = err_type;
    return gbox;
}

/* Free a generic box */
static inline void zrtl_gbox_free(ZrtlGenericBox* gbox) {
    zrtl_box_free(&gbox->base);
    if (gbox->type_args) {
        zrtl_type_args_free(gbox->type_args);
        gbox->type_args = NULL;
    }
}

/* ============================================================
 * Type Descriptor - Full Type Information
 * ============================================================
 *
 * For complex types, we may need more than just a tag.
 * TypeDescriptor provides full structural type information.
 */

/* Forward declaration */
struct ZrtlTypeDescriptor;

/* Field descriptor for struct types */
typedef struct {
    const char* name;                       /* Field name */
    uint32_t offset;                        /* Byte offset within struct */
    struct ZrtlTypeDescriptor* type;        /* Field type descriptor */
} ZrtlFieldDescriptor;

/* Type descriptor - complete type information */
typedef struct ZrtlTypeDescriptor {
    ZrtlTypeTag tag;                        /* Base type tag */
    const char* name;                       /* Type name (e.g., "Array", "MyStruct") */
    uint32_t size;                          /* Size in bytes */
    uint32_t alignment;                     /* Alignment requirement */
    ZrtlDropFn dropper;                     /* Destructor */

    /* For generic types */
    uint8_t type_param_count;               /* Number of type parameters */
    struct ZrtlTypeDescriptor** type_params;/* Type parameter descriptors */

    /* For struct/class types */
    uint16_t field_count;                   /* Number of fields */
    ZrtlFieldDescriptor* fields;            /* Field descriptors */

    /* For enum types */
    uint16_t variant_count;                 /* Number of variants */
    const char** variant_names;             /* Variant names */

    /* For function types */
    struct ZrtlTypeDescriptor* return_type; /* Return type */
    uint8_t param_count;                    /* Number of parameters */
    struct ZrtlTypeDescriptor** param_types;/* Parameter types */
} ZrtlTypeDescriptor;

/* ============================================================
 * Type Registry - For Registering Custom Types
 * ============================================================
 *
 * Plugins can register custom types to get unique type IDs.
 * This allows type-safe handling of opaque struct types.
 */

/* Type info for registered custom types */
typedef struct {
    const char* name;          /* Type name (e.g., "MyPoint") */
    uint32_t size;             /* Size of the type in bytes */
    uint32_t alignment;        /* Alignment requirement */
    ZrtlDropFn dropper;        /* Optional destructor */
    ZrtlTypeCategory category; /* Type category */
} ZrtlTypeInfo;

/* Maximum number of custom types per plugin */
#define ZRTL_MAX_CUSTOM_TYPES 256

/* Register a custom type (returns type ID, or 0 on failure) */
/* Note: Implementation provided by the Zyntax runtime */
typedef uint16_t (*ZrtlRegisterTypeFn)(const ZrtlTypeInfo* info);

/* Create a tag for a registered custom type */
#define ZRTL_CUSTOM_TAG(type_id, flags) \
    ZRTL_MAKE_TAG(ZRTL_CAT_STRUCT, type_id, flags)

/* Macro to define a custom struct type */
#define ZRTL_DEFINE_TYPE(type_name, struct_type) \
    static ZrtlTypeInfo type_name##_type_info = { \
        .name = #type_name, \
        .size = sizeof(struct_type), \
        .alignment = _Alignof(struct_type), \
        .dropper = NULL, \
        .category = ZRTL_CAT_STRUCT \
    }

/* Macro to define a custom type with destructor */
#define ZRTL_DEFINE_TYPE_WITH_DROP(type_name, struct_type, drop_fn) \
    static ZrtlTypeInfo type_name##_type_info = { \
        .name = #type_name, \
        .size = sizeof(struct_type), \
        .alignment = _Alignof(struct_type), \
        .dropper = (ZrtlDropFn)(drop_fn), \
        .category = ZRTL_CAT_STRUCT \
    }

/* ============================================================
 * Common type definitions for runtime functions
 * ============================================================ */

/* ============================================================
 * ZrtlString - Canonical String Format
 * ============================================================
 *
 * IMPORTANT: Zyntax uses a length-prefixed inline string format.
 * This is NOT a struct with a pointer - it's a contiguous memory block:
 *
 *   Memory layout: [length: i32][utf8_bytes...]
 *   - First 4 bytes: length as little-endian i32 (byte count, NOT char count)
 *   - Remaining bytes: UTF-8 encoded string data
 *   - NO null terminator (use length to determine end)
 *
 * Example: "Hello" is stored as:
 *   [0x05, 0x00, 0x00, 0x00, 'H', 'e', 'l', 'l', 'o']
 *    ^--- length = 5 ---^    ^--- UTF-8 bytes ---^
 *
 * In C, strings are passed as `int32_t*` pointing to the length field.
 * Use the helper macros below to access string data.
 */

/* String pointer type - points to the length header */
typedef int32_t* ZrtlStringPtr;
typedef const int32_t* ZrtlStringConstPtr;

/* Get string length from a ZrtlStringPtr */
#define ZRTL_STRING_LENGTH(str_ptr) (*(str_ptr))

/* Get pointer to UTF-8 bytes from a ZrtlStringPtr */
#define ZRTL_STRING_DATA(str_ptr) ((const char*)((str_ptr) + 1))

/* Get mutable pointer to UTF-8 bytes */
#define ZRTL_STRING_DATA_MUT(str_ptr) ((char*)((str_ptr) + 1))

/* Calculate total allocation size for a string of given byte length */
#define ZRTL_STRING_ALLOC_SIZE(byte_len) (sizeof(int32_t) + (byte_len))

/* Create a new string from C string (null-terminated) */
static inline ZrtlStringPtr zrtl_string_new(const char* cstr) {
    extern size_t strlen(const char*);
    extern void* memcpy(void*, const void*, size_t);

    if (!cstr) return NULL;

    int32_t len = (int32_t)strlen(cstr);
    ZrtlStringPtr ptr = (ZrtlStringPtr)zrtl_alloc(ZRTL_STRING_ALLOC_SIZE(len));
    if (!ptr) return NULL;

    *ptr = len;
    memcpy(ZRTL_STRING_DATA_MUT(ptr), cstr, len);
    return ptr;
}

/* Create a new string from bytes with known length */
static inline ZrtlStringPtr zrtl_string_from_bytes(const char* bytes, int32_t len) {
    extern void* memcpy(void*, const void*, size_t);

    if (len < 0) return NULL;

    ZrtlStringPtr ptr = (ZrtlStringPtr)zrtl_alloc(ZRTL_STRING_ALLOC_SIZE(len));
    if (!ptr) return NULL;

    *ptr = len;
    if (bytes && len > 0) {
        memcpy(ZRTL_STRING_DATA_MUT(ptr), bytes, len);
    }
    return ptr;
}

/* Create an empty string */
static inline ZrtlStringPtr zrtl_string_empty(void) {
    ZrtlStringPtr ptr = (ZrtlStringPtr)zrtl_alloc(sizeof(int32_t));
    if (ptr) *ptr = 0;
    return ptr;
}

/* Free a ZrtlString */
static inline void zrtl_string_free(ZrtlStringPtr str) {
    if (str) zrtl_free(str);
}

/* Copy a ZrtlString */
static inline ZrtlStringPtr zrtl_string_copy(ZrtlStringConstPtr str) {
    extern void* memcpy(void*, const void*, size_t);

    if (!str) return zrtl_string_empty();

    int32_t len = ZRTL_STRING_LENGTH(str);
    size_t total_size = ZRTL_STRING_ALLOC_SIZE(len);
    ZrtlStringPtr copy = (ZrtlStringPtr)zrtl_alloc(total_size);
    if (!copy) return NULL;

    memcpy(copy, str, total_size);
    return copy;
}

/* Print a ZrtlString to stdout */
static inline void zrtl_string_print(ZrtlStringConstPtr str) {
    extern int putchar(int);

    if (!str) return;

    int32_t len = ZRTL_STRING_LENGTH(str);
    const char* data = ZRTL_STRING_DATA(str);

    for (int32_t i = 0; i < len; i++) {
        putchar((unsigned char)data[i]);
    }
}

/* Compare two ZrtlStrings for equality */
static inline int zrtl_string_equals(ZrtlStringConstPtr a, ZrtlStringConstPtr b) {
    extern int memcmp(const void*, const void*, size_t);

    if (a == b) return 1;
    if (!a || !b) return 0;

    int32_t len_a = ZRTL_STRING_LENGTH(a);
    int32_t len_b = ZRTL_STRING_LENGTH(b);

    if (len_a != len_b) return 0;

    return memcmp(ZRTL_STRING_DATA(a), ZRTL_STRING_DATA(b), len_a) == 0;
}

/* ============================================================
 * ZrtlStringView - Non-owning String Reference (for SDK use)
 * ============================================================
 *
 * For SDK code that needs to work with string data without
 * dealing with the inline format, use ZrtlStringView.
 * This is NOT the format used by the compiler - only for SDK convenience.
 */
typedef struct {
    const char* data;   /* Pointer to UTF-8 bytes */
    int32_t length;     /* Length in bytes */
} ZrtlStringView;

/* Create a view from a ZrtlStringPtr */
static inline ZrtlStringView zrtl_string_view(ZrtlStringConstPtr str) {
    ZrtlStringView view = { NULL, 0 };
    if (str) {
        view.length = ZRTL_STRING_LENGTH(str);
        view.data = ZRTL_STRING_DATA(str);
    }
    return view;
}

/* ============================================================
 * ZrtlArray - Canonical Array Format
 * ============================================================
 *
 * Arrays use a similar inline format for the header:
 *
 *   Memory layout: [capacity: i32][length: i32][elem0, elem1, ...]
 *   - First 4 bytes: allocated capacity
 *   - Next 4 bytes: current length (number of elements)
 *   - Remaining: element data
 *
 * Arrays are passed as `int32_t*` pointing to the capacity field.
 */

/* Array pointer type - points to the capacity header */
typedef int32_t* ZrtlArrayPtr;
typedef const int32_t* ZrtlArrayConstPtr;

/* Get array capacity */
#define ZRTL_ARRAY_CAPACITY(arr_ptr) ((arr_ptr)[0])

/* Get array length */
#define ZRTL_ARRAY_LENGTH(arr_ptr) ((arr_ptr)[1])

/* Get pointer to array elements (typed) */
#define ZRTL_ARRAY_DATA(arr_ptr, elem_type) ((elem_type*)((arr_ptr) + 2))

/* Header size in i32 units */
#define ZRTL_ARRAY_HEADER_SIZE 2

/* Calculate allocation size for array */
#define ZRTL_ARRAY_ALLOC_SIZE(capacity, elem_size) \
    (sizeof(int32_t) * ZRTL_ARRAY_HEADER_SIZE + (capacity) * (elem_size))

/* Zyntax optional type */
typedef struct {
    uint8_t has_value;  /* 0 = none, 1 = some */
    void* value;        /* Pointer to value if has_value == 1 */
} ZrtlOptional;

/* ============================================================
 * Iterator Protocol
 * ============================================================
 *
 * The iterator protocol allows runtime functions to iterate over
 * collections in a type-safe manner, with hasNext() + next() methods.
 *
 * Two variants are provided:
 * 1. ZrtlIterator - Generic iterator using DynamicBox values
 * 2. ZrtlIteratorI32 - Specialized iterator for i32 values (common case)
 *
 * Usage example (C):
 *
 *   void print_all(ZrtlArrayConstPtr arr) {
 *       ZrtlArrayIterator it = zrtl_array_iterator(arr);
 *       while (zrtl_array_iterator_has_next(&it)) {
 *           int32_t value = zrtl_array_iterator_next_i32(&it);
 *           printf("%d\n", value);
 *       }
 *   }
 */

/* Generic iterator state */
typedef struct {
    const void* collection;     /* Pointer to the collection being iterated */
    int32_t index;              /* Current position */
    int32_t length;             /* Total number of elements */
    ZrtlTypeTag element_type;   /* Type of elements (for generic containers) */
} ZrtlIterator;

/* Check if iterator has more elements */
static inline int zrtl_iterator_has_next(const ZrtlIterator* it) {
    return it && it->index < it->length;
}

/* Get current index */
static inline int32_t zrtl_iterator_index(const ZrtlIterator* it) {
    return it ? it->index : -1;
}

/* Reset iterator to beginning */
static inline void zrtl_iterator_reset(ZrtlIterator* it) {
    if (it) it->index = 0;
}

/* ============================================================
 * Array Iterator (specialized for inline array format)
 * ============================================================ */

/* Array iterator state */
typedef struct {
    ZrtlArrayConstPtr array;    /* Pointer to the array */
    int32_t index;              /* Current position */
} ZrtlArrayIterator;

/* Create iterator for an array */
static inline ZrtlArrayIterator zrtl_array_iterator(ZrtlArrayConstPtr arr) {
    ZrtlArrayIterator it = { arr, 0 };
    return it;
}

/* Check if array iterator has more elements */
static inline int zrtl_array_iterator_has_next(const ZrtlArrayIterator* it) {
    if (!it || !it->array) return 0;
    return it->index < ZRTL_ARRAY_LENGTH(it->array);
}

/* Get next i32 element and advance iterator */
static inline int32_t zrtl_array_iterator_next_i32(ZrtlArrayIterator* it) {
    if (!zrtl_array_iterator_has_next(it)) return 0;
    int32_t value = ZRTL_ARRAY_DATA(it->array, int32_t)[it->index];
    it->index++;
    return value;
}

/* Get next element as pointer (for generic element types) */
static inline const void* zrtl_array_iterator_next_ptr(ZrtlArrayIterator* it, size_t elem_size) {
    if (!zrtl_array_iterator_has_next(it)) return NULL;
    const char* data = (const char*)(it->array + ZRTL_ARRAY_HEADER_SIZE);
    const void* elem = data + (it->index * elem_size);
    it->index++;
    return elem;
}

/* Reset array iterator */
static inline void zrtl_array_iterator_reset(ZrtlArrayIterator* it) {
    if (it) it->index = 0;
}

/* ============================================================
 * String Iterator (iterate over UTF-8 bytes or codepoints)
 * ============================================================ */

/* String iterator state (byte-level) */
typedef struct {
    ZrtlStringConstPtr string;  /* Pointer to the string */
    int32_t index;              /* Current byte position */
} ZrtlStringIterator;

/* Create iterator for a string */
static inline ZrtlStringIterator zrtl_string_iterator(ZrtlStringConstPtr str) {
    ZrtlStringIterator it = { str, 0 };
    return it;
}

/* Check if string iterator has more bytes */
static inline int zrtl_string_iterator_has_next(const ZrtlStringIterator* it) {
    if (!it || !it->string) return 0;
    return it->index < ZRTL_STRING_LENGTH(it->string);
}

/* Get next byte and advance iterator */
static inline uint8_t zrtl_string_iterator_next_byte(ZrtlStringIterator* it) {
    if (!zrtl_string_iterator_has_next(it)) return 0;
    uint8_t byte = (uint8_t)ZRTL_STRING_DATA(it->string)[it->index];
    it->index++;
    return byte;
}

/* Get next UTF-8 codepoint and advance iterator
 * Returns the codepoint value, or 0 on end/error
 * Handles 1-4 byte UTF-8 sequences
 */
static inline uint32_t zrtl_string_iterator_next_codepoint(ZrtlStringIterator* it) {
    if (!zrtl_string_iterator_has_next(it)) return 0;

    const uint8_t* data = (const uint8_t*)ZRTL_STRING_DATA(it->string);
    int32_t len = ZRTL_STRING_LENGTH(it->string);
    int32_t i = it->index;

    uint8_t b0 = data[i];
    uint32_t cp;

    if ((b0 & 0x80) == 0) {
        /* 1-byte ASCII */
        cp = b0;
        it->index = i + 1;
    } else if ((b0 & 0xE0) == 0xC0 && i + 1 < len) {
        /* 2-byte sequence */
        cp = ((b0 & 0x1F) << 6) | (data[i+1] & 0x3F);
        it->index = i + 2;
    } else if ((b0 & 0xF0) == 0xE0 && i + 2 < len) {
        /* 3-byte sequence */
        cp = ((b0 & 0x0F) << 12) | ((data[i+1] & 0x3F) << 6) | (data[i+2] & 0x3F);
        it->index = i + 3;
    } else if ((b0 & 0xF8) == 0xF0 && i + 3 < len) {
        /* 4-byte sequence */
        cp = ((b0 & 0x07) << 18) | ((data[i+1] & 0x3F) << 12) |
             ((data[i+2] & 0x3F) << 6) | (data[i+3] & 0x3F);
        it->index = i + 4;
    } else {
        /* Invalid UTF-8 or truncated, return replacement char */
        cp = 0xFFFD;
        it->index = i + 1;
    }

    return cp;
}

/* Reset string iterator */
static inline void zrtl_string_iterator_reset(ZrtlStringIterator* it) {
    if (it) it->index = 0;
}

/* ============================================================
 * Iterator Vtable (for custom iterable types)
 * ============================================================
 *
 * For custom types that want to be iterable, implement this vtable.
 * This allows runtime functions to iterate over user-defined collections.
 */

/* Forward declaration */
struct ZrtlIterableVtable;

/* Iterable object header (embed at start of custom iterable structs) */
typedef struct {
    const struct ZrtlIterableVtable* vtable;
} ZrtlIterableHeader;

/* Vtable for iterable types */
typedef struct ZrtlIterableVtable {
    /* Returns non-zero if there are more elements */
    int (*has_next)(void* iterator_state);

    /* Returns next element as DynamicBox, advances iterator */
    ZrtlDynamicBox (*next)(void* iterator_state);

    /* Creates iterator state for this collection (caller must free) */
    void* (*create_iterator)(const void* collection);

    /* Frees iterator state */
    void (*free_iterator)(void* iterator_state);
} ZrtlIterableVtable;

/* ============================================================
 * Helper macros for common patterns
 * ============================================================ */

/* Define a method with standard naming convention */
#define ZRTL_METHOD(type, name) \
    ZRTL_SYMBOL("$" #type "$" #name, type##_##name)

/* Define multiple methods for a type */
#define ZRTL_TYPE_BEGIN(type) /* Start type definition */
#define ZRTL_TYPE_END /* End type definition */

/* ============================================================
 * Array creation helpers
 * ============================================================ */

/* Create a new array with given capacity (for i32 elements) */
static inline ZrtlArrayPtr zrtl_array_new_i32(int32_t initial_capacity) {
    int32_t cap = initial_capacity > 0 ? initial_capacity : 8;
    size_t size = ZRTL_ARRAY_ALLOC_SIZE(cap, sizeof(int32_t));
    ZrtlArrayPtr ptr = (ZrtlArrayPtr)zrtl_alloc(size);
    if (!ptr) return NULL;

    ZRTL_ARRAY_CAPACITY(ptr) = cap;
    ZRTL_ARRAY_LENGTH(ptr) = 0;
    return ptr;
}

/* Free an array */
static inline void zrtl_array_free(ZrtlArrayPtr arr) {
    if (arr) zrtl_free(arr);
}

/* Push an i32 element to array (may reallocate, returns new ptr) */
static inline ZrtlArrayPtr zrtl_array_push_i32(ZrtlArrayPtr arr, int32_t value) {
    if (!arr) return NULL;

    int32_t cap = ZRTL_ARRAY_CAPACITY(arr);
    int32_t len = ZRTL_ARRAY_LENGTH(arr);

    /* Check if we need to grow */
    if (len + ZRTL_ARRAY_HEADER_SIZE >= cap) {
        int32_t new_cap = cap * 2;
        size_t new_size = ZRTL_ARRAY_ALLOC_SIZE(new_cap, sizeof(int32_t));
        ZrtlArrayPtr new_arr = (ZrtlArrayPtr)zrtl_realloc(arr, new_size);
        if (!new_arr) return NULL;

        arr = new_arr;
        ZRTL_ARRAY_CAPACITY(arr) = new_cap;
    }

    /* Add element */
    ZRTL_ARRAY_DATA(arr, int32_t)[len] = value;
    ZRTL_ARRAY_LENGTH(arr) = len + 1;

    return arr;
}

/* Get element from array */
static inline int32_t zrtl_array_get_i32(ZrtlArrayConstPtr arr, int32_t index) {
    if (!arr || index < 0 || index >= ZRTL_ARRAY_LENGTH(arr)) {
        return 0;
    }
    return ZRTL_ARRAY_DATA(arr, int32_t)[index];
}

/* ============================================================
 * Async/Await Support
 * ============================================================
 *
 * This section provides the C ABI for async functions that can be
 * called from Zyntax-based languages. Native async functions return
 * a ZrtlPromise which can be polled until completion.
 *
 * # Promise ABI Convention
 *
 * Async functions return a pointer to ZrtlPromise:
 *   async fn foo(a: i32) -> i64  =>  ZrtlPromise* foo(int32_t a)
 *
 * The poll function returns an i64:
 *   - 0 = Pending (operation not complete, poll again)
 *   - positive = Ready(value) - operation completed successfully
 *   - negative = Failed(error_code) - operation failed
 *
 * # Usage Example (C)
 *
 *   // Define async state machine
 *   typedef struct {
 *       ZrtlStateMachineHeader header;
 *       int32_t counter;
 *       int32_t target;
 *   } CounterState;
 *
 *   // Poll function
 *   int64_t counter_poll(void* state) {
 *       CounterState* s = (CounterState*)state;
 *       if (s->counter >= s->target) {
 *           s->header.state = ZRTL_ASYNC_COMPLETED;
 *           return s->counter;  // Ready(counter)
 *       }
 *       s->counter++;
 *       return 0;  // Pending
 *   }
 *
 *   // Create async function
 *   ZrtlPromise* async_count_to(int32_t target) {
 *       CounterState* state = malloc(sizeof(CounterState));
 *       state->header = ZRTL_STATE_MACHINE_INIT;
 *       state->counter = 0;
 *       state->target = target;
 *
 *       ZrtlPromise* promise = malloc(sizeof(ZrtlPromise));
 *       promise->state_machine = state;
 *       promise->poll_fn = counter_poll;
 *       return promise;
 *   }
 *
 *   // Await (blocking)
 *   int64_t result = zrtl_promise_block_on(async_count_to(100));
 */

/* Async state values */
typedef enum {
    ZRTL_ASYNC_INITIAL   = 0,          /* Initial state, not started */
    ZRTL_ASYNC_RESUME1   = 1,          /* Resumed after first await */
    ZRTL_ASYNC_RESUME2   = 2,          /* Resumed after second await */
    ZRTL_ASYNC_RESUME3   = 3,
    ZRTL_ASYNC_RESUME4   = 4,
    ZRTL_ASYNC_RESUME5   = 5,
    ZRTL_ASYNC_RESUME6   = 6,
    ZRTL_ASYNC_RESUME7   = 7,
    ZRTL_ASYNC_COMPLETED = 0xFFFFFFFE, /* Completed successfully */
    ZRTL_ASYNC_FAILED    = 0xFFFFFFFF, /* Failed with error */
} ZrtlAsyncState;

/* State machine header - must be first field in async state structs */
typedef struct {
    uint32_t state;      /* Current async state (ZrtlAsyncState) */
    uint32_t _reserved;  /* Reserved for alignment */
} ZrtlStateMachineHeader;

/* Initial value for state machine header */
#define ZRTL_STATE_MACHINE_INIT { ZRTL_ASYNC_INITIAL, 0 }

/* Poll function signature: takes state pointer, returns i64 ABI result */
typedef int64_t (*ZrtlPollFn)(void* state_machine);

/* Promise structure - returned by async functions */
typedef struct {
    void* state_machine;    /* Pointer to the state machine */
    ZrtlPollFn poll_fn;     /* Poll function */
} ZrtlPromise;

/* Poll result values */
#define ZRTL_POLL_PENDING 0

/* Check if poll result is pending */
#define ZRTL_POLL_IS_PENDING(result) ((result) == 0)

/* Check if poll result is ready (positive = success) */
#define ZRTL_POLL_IS_READY(result) ((result) > 0)

/* Check if poll result is failed (negative = error) */
#define ZRTL_POLL_IS_FAILED(result) ((result) < 0)

/* Extract error code from failed result */
#define ZRTL_POLL_ERROR_CODE(result) ((int32_t)(result))

/* Poll a promise once */
static inline int64_t zrtl_promise_poll(ZrtlPromise* promise) {
    if (!promise || !promise->poll_fn) return -1;
    return promise->poll_fn(promise->state_machine);
}

/* Block until promise completes (busy-wait) */
static inline int64_t zrtl_promise_block_on(ZrtlPromise* promise) {
    if (!promise) return -1;
    int64_t result;
    while ((result = zrtl_promise_poll(promise)) == ZRTL_POLL_PENDING) {
        /* Spin - could add yield here for better CPU usage */
    }
    return result;
}

/* Free a promise and its state machine */
static inline void zrtl_promise_free(ZrtlPromise* promise) {
    if (promise) {
        if (promise->state_machine) {
            zrtl_free(promise->state_machine);
        }
        zrtl_free(promise);
    }
}

/* Check if state machine is finished */
static inline int zrtl_state_is_finished(const ZrtlStateMachineHeader* header) {
    return header->state >= ZRTL_ASYNC_COMPLETED;
}

/* Advance state machine to next resume point */
static inline void zrtl_state_advance(ZrtlStateMachineHeader* header) {
    if (header->state < ZRTL_ASYNC_COMPLETED) {
        header->state++;
    }
}

/* Mark state machine as completed */
static inline void zrtl_state_complete(ZrtlStateMachineHeader* header) {
    header->state = ZRTL_ASYNC_COMPLETED;
}

/* Mark state machine as failed */
static inline void zrtl_state_fail(ZrtlStateMachineHeader* header) {
    header->state = ZRTL_ASYNC_FAILED;
}

/* ============================================================
 * Promise Combinators
 * ============================================================
 *
 * These helpers allow combining multiple promises.
 *
 * - zrtl_promise_all: Wait for all promises to complete
 * - zrtl_promise_race: Wait for first promise to complete
 */

/* Promise array for combinators */
typedef struct {
    ZrtlPromise** promises;  /* Array of promise pointers */
    uint32_t count;          /* Number of promises */
    int64_t* results;        /* Results array (for all) */
    uint32_t completed;      /* Number completed (for all) */
    int32_t winner;          /* Index of first completed (for race), -1 if none */
} ZrtlPromiseGroup;

/* Create a promise group for Promise.all */
static inline ZrtlPromiseGroup* zrtl_promise_group_new(uint32_t count) {
    ZrtlPromiseGroup* group = (ZrtlPromiseGroup*)zrtl_alloc(sizeof(ZrtlPromiseGroup));
    if (!group) return NULL;

    group->promises = (ZrtlPromise**)zrtl_alloc(sizeof(ZrtlPromise*) * count);
    group->results = (int64_t*)zrtl_alloc(sizeof(int64_t) * count);
    group->count = count;
    group->completed = 0;
    group->winner = -1;

    if (!group->promises || !group->results) {
        if (group->promises) zrtl_free(group->promises);
        if (group->results) zrtl_free(group->results);
        zrtl_free(group);
        return NULL;
    }

    for (uint32_t i = 0; i < count; i++) {
        group->promises[i] = NULL;
        group->results[i] = 0;
    }

    return group;
}

/* Add a promise to the group */
static inline void zrtl_promise_group_add(ZrtlPromiseGroup* group, uint32_t index, ZrtlPromise* promise) {
    if (group && index < group->count) {
        group->promises[index] = promise;
    }
}

/* Poll all promises (Promise.all pattern) - returns 1 when all complete */
static inline int zrtl_promise_all_poll(ZrtlPromiseGroup* group) {
    if (!group) return 1;

    for (uint32_t i = 0; i < group->count; i++) {
        if (group->results[i] == ZRTL_POLL_PENDING && group->promises[i]) {
            int64_t result = zrtl_promise_poll(group->promises[i]);
            if (result != ZRTL_POLL_PENDING) {
                group->results[i] = result;
                group->completed++;
            }
        }
    }

    return group->completed >= group->count;
}

/* Poll for first completion (Promise.race pattern) - returns winner index or -1 */
static inline int32_t zrtl_promise_race_poll(ZrtlPromiseGroup* group) {
    if (!group || group->winner >= 0) return group ? group->winner : -1;

    for (uint32_t i = 0; i < group->count; i++) {
        if (group->promises[i]) {
            int64_t result = zrtl_promise_poll(group->promises[i]);
            if (result != ZRTL_POLL_PENDING) {
                group->results[i] = result;
                group->winner = (int32_t)i;
                return group->winner;
            }
        }
    }

    return -1;  /* None completed yet */
}

/* Free a promise group (does NOT free individual promises) */
static inline void zrtl_promise_group_free(ZrtlPromiseGroup* group) {
    if (group) {
        if (group->promises) zrtl_free(group->promises);
        if (group->results) zrtl_free(group->results);
        zrtl_free(group);
    }
}

/* ============================================================
 * Test Harness Macros
 * ============================================================
 *
 * These macros provide a minimal test framework for ZRTL plugins.
 * Tests can be run standalone or integrated with cargo test.
 *
 * Usage:
 *
 *   #include "zrtl.h"
 *
 *   // Initialize test state (must be at file scope, before tests)
 *   ZRTL_TEST_INIT();
 *
 *   ZRTL_TEST(test_array_push) {
 *       ZrtlArrayPtr arr = zrtl_array_new_i32(4);
 *       arr = zrtl_array_push_i32(arr, 42);
 *
 *       ZRTL_ASSERT(arr != NULL);
 *       ZRTL_ASSERT_EQ(ZRTL_ARRAY_LENGTH(arr), 1);
 *       ZRTL_ASSERT_EQ(zrtl_array_get_i32(arr, 0), 42);
 *
 *       zrtl_array_free(arr);
 *       ZRTL_PASS();
 *   }
 *
 *   ZRTL_TEST(test_string_create) {
 *       ZrtlStringPtr str = zrtl_string_new("Hello");
 *       ZRTL_ASSERT_NOT_NULL(str);
 *       ZRTL_ASSERT_EQ(ZRTL_STRING_LENGTH(str), 5);
 *
 *       zrtl_string_free(str);
 *       ZRTL_PASS();
 *   }
 *
 *   ZRTL_TEST_MAIN(
 *       ZRTL_RUN(test_array_push),
 *       ZRTL_RUN(test_string_create)
 *   )
 *
 * Build and run:
 *
 *   gcc -o test_runtime test_runtime.c && ./test_runtime
 *
 * Output:
 *
 *   [PASS] test_array_push
 *   [PASS] test_string_create
 *   ========================
 *   2 passed, 0 failed
 */

/* Test result codes */
#define ZRTL_TEST_PASS 1
#define ZRTL_TEST_FAIL 0

/* Test function signature */
typedef int (*ZrtlTestFn)(void);

/* Test entry for registration */
typedef struct {
    const char* name;
    ZrtlTestFn fn;
} ZrtlTestEntry;

/* Global test state (used by macros) */
typedef struct {
    int passed;
    int failed;
    const char* current_test;
    const char* failure_file;
    int failure_line;
    const char* failure_msg;
} ZrtlTestState;

/* Declare a test function
 * Note: ZRTL_TEST_INIT() must be used before any ZRTL_TEST declarations
 */
#define ZRTL_TEST(name) \
    static int name(void)

/* Pass the test */
#define ZRTL_PASS() return ZRTL_TEST_PASS

/* Fail the test with message */
#define ZRTL_FAIL(msg) do { \
    _zrtl_test_state.failure_file = __FILE__; \
    _zrtl_test_state.failure_line = __LINE__; \
    _zrtl_test_state.failure_msg = msg; \
    return ZRTL_TEST_FAIL; \
} while(0)

/* Basic assertion */
#define ZRTL_ASSERT(cond) do { \
    if (!(cond)) { \
        _zrtl_test_state.failure_file = __FILE__; \
        _zrtl_test_state.failure_line = __LINE__; \
        _zrtl_test_state.failure_msg = "Assertion failed: " #cond; \
        return ZRTL_TEST_FAIL; \
    } \
} while(0)

/* Assertion with custom message */
#define ZRTL_ASSERT_MSG(cond, msg) do { \
    if (!(cond)) { \
        _zrtl_test_state.failure_file = __FILE__; \
        _zrtl_test_state.failure_line = __LINE__; \
        _zrtl_test_state.failure_msg = msg; \
        return ZRTL_TEST_FAIL; \
    } \
} while(0)

/* Assert equality (integers) */
#define ZRTL_ASSERT_EQ(a, b) do { \
    if ((a) != (b)) { \
        _zrtl_test_state.failure_file = __FILE__; \
        _zrtl_test_state.failure_line = __LINE__; \
        _zrtl_test_state.failure_msg = "Assertion failed: " #a " == " #b; \
        return ZRTL_TEST_FAIL; \
    } \
} while(0)

/* Assert inequality */
#define ZRTL_ASSERT_NE(a, b) do { \
    if ((a) == (b)) { \
        _zrtl_test_state.failure_file = __FILE__; \
        _zrtl_test_state.failure_line = __LINE__; \
        _zrtl_test_state.failure_msg = "Assertion failed: " #a " != " #b; \
        return ZRTL_TEST_FAIL; \
    } \
} while(0)

/* Assert less than */
#define ZRTL_ASSERT_LT(a, b) do { \
    if (!((a) < (b))) { \
        _zrtl_test_state.failure_file = __FILE__; \
        _zrtl_test_state.failure_line = __LINE__; \
        _zrtl_test_state.failure_msg = "Assertion failed: " #a " < " #b; \
        return ZRTL_TEST_FAIL; \
    } \
} while(0)

/* Assert less than or equal */
#define ZRTL_ASSERT_LE(a, b) do { \
    if (!((a) <= (b))) { \
        _zrtl_test_state.failure_file = __FILE__; \
        _zrtl_test_state.failure_line = __LINE__; \
        _zrtl_test_state.failure_msg = "Assertion failed: " #a " <= " #b; \
        return ZRTL_TEST_FAIL; \
    } \
} while(0)

/* Assert greater than */
#define ZRTL_ASSERT_GT(a, b) do { \
    if (!((a) > (b))) { \
        _zrtl_test_state.failure_file = __FILE__; \
        _zrtl_test_state.failure_line = __LINE__; \
        _zrtl_test_state.failure_msg = "Assertion failed: " #a " > " #b; \
        return ZRTL_TEST_FAIL; \
    } \
} while(0)

/* Assert greater than or equal */
#define ZRTL_ASSERT_GE(a, b) do { \
    if (!((a) >= (b))) { \
        _zrtl_test_state.failure_file = __FILE__; \
        _zrtl_test_state.failure_line = __LINE__; \
        _zrtl_test_state.failure_msg = "Assertion failed: " #a " >= " #b; \
        return ZRTL_TEST_FAIL; \
    } \
} while(0)

/* Assert not null */
#define ZRTL_ASSERT_NOT_NULL(ptr) do { \
    if ((ptr) == NULL) { \
        _zrtl_test_state.failure_file = __FILE__; \
        _zrtl_test_state.failure_line = __LINE__; \
        _zrtl_test_state.failure_msg = "Assertion failed: " #ptr " != NULL"; \
        return ZRTL_TEST_FAIL; \
    } \
} while(0)

/* Assert null */
#define ZRTL_ASSERT_NULL(ptr) do { \
    if ((ptr) != NULL) { \
        _zrtl_test_state.failure_file = __FILE__; \
        _zrtl_test_state.failure_line = __LINE__; \
        _zrtl_test_state.failure_msg = "Assertion failed: " #ptr " == NULL"; \
        return ZRTL_TEST_FAIL; \
    } \
} while(0)

/* Assert string equality */
#define ZRTL_ASSERT_STR_EQ(a, b) do { \
    if (!zrtl_string_equals(a, b)) { \
        _zrtl_test_state.failure_file = __FILE__; \
        _zrtl_test_state.failure_line = __LINE__; \
        _zrtl_test_state.failure_msg = "Assertion failed: strings not equal"; \
        return ZRTL_TEST_FAIL; \
    } \
} while(0)

/* Assert float approximately equal (within epsilon) */
#define ZRTL_ASSERT_FLOAT_EQ(a, b, epsilon) do { \
    double _diff = (a) - (b); \
    if (_diff < 0) _diff = -_diff; \
    if (_diff > (epsilon)) { \
        _zrtl_test_state.failure_file = __FILE__; \
        _zrtl_test_state.failure_line = __LINE__; \
        _zrtl_test_state.failure_msg = "Assertion failed: " #a " ~= " #b; \
        return ZRTL_TEST_FAIL; \
    } \
} while(0)

/* Run a test and track results */
#define ZRTL_RUN(test_fn) { #test_fn, test_fn }

/* Main test runner macro
 * Note: Use ZRTL_TEST_INIT() at file scope before defining tests,
 * then ZRTL_TEST_MAIN() at the end to generate main().
 */
#define ZRTL_TEST_MAIN(...) \
    int main(void) { \
        extern int printf(const char*, ...); \
        ZrtlTestEntry _tests[] = { __VA_ARGS__ }; \
        int _num_tests = sizeof(_tests) / sizeof(_tests[0]); \
        \
        for (int _i = 0; _i < _num_tests; _i++) { \
            _zrtl_test_state.current_test = _tests[_i].name; \
            _zrtl_test_state.failure_file = NULL; \
            _zrtl_test_state.failure_line = 0; \
            _zrtl_test_state.failure_msg = NULL; \
            \
            int _result = _tests[_i].fn(); \
            \
            if (_result == ZRTL_TEST_PASS) { \
                printf("[PASS] %s\n", _tests[_i].name); \
                _zrtl_test_state.passed++; \
            } else { \
                printf("[FAIL] %s\n", _tests[_i].name); \
                if (_zrtl_test_state.failure_file) { \
                    printf("       at %s:%d\n", \
                           _zrtl_test_state.failure_file, \
                           _zrtl_test_state.failure_line); \
                } \
                if (_zrtl_test_state.failure_msg) { \
                    printf("       %s\n", _zrtl_test_state.failure_msg); \
                } \
                _zrtl_test_state.failed++; \
            } \
        } \
        \
        printf("========================\n"); \
        printf("%d passed, %d failed\n", \
               _zrtl_test_state.passed, \
               _zrtl_test_state.failed); \
        \
        return _zrtl_test_state.failed > 0 ? 1 : 0; \
    }

/* Alternative: Test runner without main() for embedding in other test frameworks */
#define ZRTL_RUN_TESTS(...) do { \
    extern int printf(const char*, ...); \
    ZrtlTestEntry _tests[] = { __VA_ARGS__ }; \
    int _num_tests = sizeof(_tests) / sizeof(_tests[0]); \
    \
    for (int _i = 0; _i < _num_tests; _i++) { \
        _zrtl_test_state.current_test = _tests[_i].name; \
        _zrtl_test_state.failure_file = NULL; \
        _zrtl_test_state.failure_line = 0; \
        _zrtl_test_state.failure_msg = NULL; \
        \
        int _result = _tests[_i].fn(); \
        \
        if (_result == ZRTL_TEST_PASS) { \
            printf("[PASS] %s\n", _tests[_i].name); \
            _zrtl_test_state.passed++; \
        } else { \
            printf("[FAIL] %s\n", _tests[_i].name); \
            if (_zrtl_test_state.failure_file) { \
                printf("       at %s:%d\n", \
                       _zrtl_test_state.failure_file, \
                       _zrtl_test_state.failure_line); \
            } \
            if (_zrtl_test_state.failure_msg) { \
                printf("       %s\n", _zrtl_test_state.failure_msg); \
            } \
            _zrtl_test_state.failed++; \
        } \
    } \
} while(0)

/* Initialize test state (for ZRTL_RUN_TESTS without main) */
#define ZRTL_TEST_INIT() \
    static ZrtlTestState _zrtl_test_state = { 0, 0, NULL, NULL, 0, NULL }

/* Get test summary after ZRTL_RUN_TESTS */
#define ZRTL_TEST_PASSED() (_zrtl_test_state.passed)
#define ZRTL_TEST_FAILED() (_zrtl_test_state.failed)

#ifdef __cplusplus
}
#endif

#endif /* ZRTL_H */
