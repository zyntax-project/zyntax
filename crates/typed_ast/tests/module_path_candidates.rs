use std::path::PathBuf;
use zyntax_typed_ast::import_resolver::ModuleArchitecture;

#[test]
fn typescript_paths_distinguish_source_extensions_and_emitted_javascript() {
    let architecture = ModuleArchitecture::typescript();
    let base = PathBuf::from("project");
    let candidates = |path: &str| architecture.module_to_paths(&[path.to_owned()], &base);
    assert_eq!(candidates("./module.ts"), vec![base.join("./module.ts")]);
    assert_eq!(
        candidates("./module.js"),
        [
            "./module.ts",
            "./module.tsx",
            "./module.d.ts",
            "./module.js"
        ]
        .map(|p| base.join(p))
    );
    assert_eq!(
        candidates("./module"),
        [
            "./module.ts",
            "./module.tsx",
            "./module.d.ts",
            "./module.js",
            "./module/index.ts",
            "./module/index.tsx",
            "./module/index.d.ts",
            "./module/index.js",
        ]
        .map(|p| base.join(p))
    );
    assert_eq!(
        candidates("../module.d.ts"),
        vec![PathBuf::from("module.d.ts")]
    );
    assert_eq!(candidates("./absent/../module"), candidates("./module"));
    assert_eq!(
        candidates("./module/"),
        [
            "module/index.ts",
            "module/index.tsx",
            "module/index.d.ts",
            "module/index.js"
        ]
        .map(|p| base.join(p))
    );
    assert_eq!(
        candidates("."),
        ["index.ts", "index.tsx", "index.d.ts", "index.js"].map(|p| base.join(p))
    );
    let javascript = ModuleArchitecture::NodeStyle {
        extensions: vec![".json".into(), ".js".into()],
        index_name: "index".into(),
    };
    assert_eq!(
        javascript.module_to_paths(&["./module.js".into()], &base),
        vec![base.join("./module.js")]
    );
}
