//! `ZyntaxValue` re-export — the canonical type now lives in
//! [`zyntax_compiler::value`] so the BC interpreter can use it
//! directly. This module exists to keep `zyntax_embed::ZyntaxValue`
//! resolvable for existing consumers; new code should import from
//! [`zyntax_compiler::value`] directly.

pub use zyntax_compiler::value::{StructBuilder, ZyntaxValue};

#[cfg(test)]
mod tests {
    use super::ZyntaxValue;
    use zyntax_compiler::zrtl::TypeCategory;

    #[test]
    fn test_type_category() {
        assert_eq!(ZyntaxValue::Int(42).type_category(), TypeCategory::Int);
        assert_eq!(
            ZyntaxValue::String("hi".into()).type_category(),
            TypeCategory::String
        );
        assert_eq!(
            ZyntaxValue::Array(vec![]).type_category(),
            TypeCategory::Array
        );
    }

    #[test]
    fn test_accessors() {
        let int_val = ZyntaxValue::Int(42);
        assert_eq!(int_val.as_int(), Some(42));
        assert_eq!(int_val.as_float(), Some(42.0));
        assert_eq!(int_val.as_str(), None);

        let str_val = ZyntaxValue::String("hello".into());
        assert_eq!(str_val.as_str(), Some("hello"));
        assert_eq!(str_val.as_int(), None);
    }

    #[test]
    fn test_struct_builder() {
        let point = ZyntaxValue::new_struct("Point")
            .field("x", 10i32)
            .field("y", 20i32)
            .build();

        assert!(matches!(point, ZyntaxValue::Struct { .. }));
        assert_eq!(point.get_field("x"), Some(&ZyntaxValue::Int(10)));
        assert_eq!(point.get_field("y"), Some(&ZyntaxValue::Int(20)));
    }

    #[test]
    fn test_from_impls() {
        let v: ZyntaxValue = 42i32.into();
        assert!(matches!(v, ZyntaxValue::Int(42)));

        let v: ZyntaxValue = "hello".into();
        assert!(matches!(v, ZyntaxValue::String(s) if s == "hello"));

        let v: ZyntaxValue = vec![1i32, 2, 3].into();
        assert!(matches!(v, ZyntaxValue::Array(_)));
    }
}
