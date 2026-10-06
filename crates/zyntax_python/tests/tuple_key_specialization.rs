use zyntax_typed_ast::typed_ast::TypedDeclaration;
use zyntax_typed_ast::{PrimitiveType, Type};

#[test]
fn an_overridden_tuple_key_method_has_a_typed_dispatcher() {
    let source = include_str!("../benchmarks/speed/checks/scimark_sor.py");
    let modules = |name: &str| match name {
        "scimark" => Some(include_str!("../benchmarks/speed/kernels/scimark.py").to_string()),
        "util" => Some(include_str!("../benchmarks/speed/shims/util.py").to_string()),
        "optparse" => Some(include_str!("../benchmarks/speed/shims/optparse.py").to_string()),
        _ => None,
    };
    let program = zyntax_python::parse_program_with(source, "scimark_sor.py", &modules)
        .expect("tuple-key program lowers");
    let functions: Vec<_> = program
        .declarations
        .iter()
        .filter_map(|d| match &d.node {
            TypedDeclaration::Function(f) => Some(f),
            _ => None,
        })
        .collect();
    let float = Type::Primitive(PrimitiveType::F64);
    let int = Type::Primitive(PrimitiveType::I64);
    let get = functions
        .iter()
        .find(|f| {
            f.name.resolve_global().is_some_and(|n| {
                n.starts_with("scimark$Array2D$__getitem__$s") && !n.ends_with("$dispatch")
            }) && f.params.get(1).is_some_and(
                |p| matches!(&p.ty, Type::Tuple(ts) if ts == &[int.clone(), int.clone()]),
            )
        })
        .expect("Array2D has a (int, int) indexing instance");
    assert_eq!(get.return_type, float);
    let dispatch = format!("{}$dispatch", get.name.resolve_global().unwrap());
    let dispatcher = functions
        .iter()
        .find(|f| f.name.resolve_global().as_deref() == Some(&dispatch))
        .expect("typed indexing keeps override dispatch");
    assert_eq!(dispatcher.return_type, float);
    assert_eq!(dispatcher.params[1].ty, get.params[1].ty);
}

#[test]
fn tuple_coordinates_keep_the_index_calculation_integer_typed() {
    let program =
        zyntax_python::parse_program(include_str!("../conformance/class/tuple_key_overrides.py"))
            .expect("tuple-key program lowers");
    let int = Type::Primitive(PrimitiveType::I64);
    assert!(
        program.declarations.iter().any(|d| {
            let TypedDeclaration::Function(f) = &d.node else {
                return false;
            };
            f.name
                .resolve_global()
                .is_some_and(|n| n.starts_with("Grid$_idx$s"))
                && f.params.iter().skip(1).all(|p| p.ty == int)
                && f.return_type == int
        }),
        "index calculation stays integer-typed"
    );
}
