//! What a module exports, by Lua's convention: the value its chunk
//! returns. Read from the types the compiler gives a chunk `load`
//! compiles, without compiling or running it, so a host can describe a
//! module before anything the module requires exists.
//!
//! A table's fields are its string keys, less the metatable's own
//! (`__index`, `__call`, ...), which are Lua's protocol rather than the
//! table's content. A function declared with `:` takes `self` first and
//! is a method; the tables `setmetatable` gives a table as their
//! metatable are its instances. The types a LuaLS annotation declares
//! come with what it annotates (see [`crate::annotation`]).

use std::collections::HashMap;

use crate::annotation::{LuaType, Returned};
use crate::scope::{FuncId, Holder, Scopes};
use crate::types::{Inferred, ShapeId, Ty};
use crate::{Error, Result};

/// A value a chunk returns, or a field of a table it returns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Exported {
    Function(ExportedFunction),
    /// A table, by its index in [`Exports::tables`]: tables refer to
    /// each other, and to themselves.
    Table(usize),
    /// Any other value, or one the types do not follow; with the type a
    /// `@type` declares for it.
    Value(Option<LuaType>),
}

/// A function: its parameters' names, and whether `...` follows them. A
/// method takes `self` first, as one declared with `:` does, and
/// `params` names it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExportedFunction {
    pub params: Vec<String>,
    pub variadic: bool,
    pub method: bool,
    /// What its `@param` and `@return` annotations declare, when it has
    /// any.
    pub signature: Option<Signature>,
}

/// A function's declared types: one per parameter in order (`any` where
/// none is declared), the type of `...`, and the results, each with the
/// name its `@return` gives it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Signature {
    pub params: Vec<LuaType>,
    pub variadic: Option<LuaType>,
    pub returns: Vec<Returned>,
}

/// A table a chunk returns, or reaches through the fields of one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExportedTable {
    /// Its fields under constant string keys, metafields left out: the
    /// constructor's, in sorted order, then those the stores add.
    pub fields: Vec<(String, Exported)>,
    /// Whether stores under keys not known at compile time may hold
    /// names the fields do not list.
    pub open: bool,
    /// The tables whose metatable it is, by index in
    /// [`Exports::tables`].
    pub instances: Vec<usize>,
    /// The class a `@class` makes it, when one does.
    pub class: Option<DeclaredClass>,
}

/// A `@class` and the `@field`s it declares for its instances.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeclaredClass {
    pub name: String,
    pub fields: Vec<(String, LuaType)>,
}

/// What a chunk exports: its first result, and the tables it reaches.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Exports {
    pub value: Exported,
    pub tables: Vec<ExportedTable>,
}

impl Exports {
    /// The table `value` is, when it is one.
    pub fn table(&self, value: &Exported) -> Option<&ExportedTable> {
        match value {
            Exported::Table(i) => self.tables.get(*i),
            _ => None,
        }
    }
}

/// Whether `key` names one of a metatable's own fields, which Lua
/// reserves: two underscores first.
pub fn is_metafield(key: &str) -> bool {
    key.starts_with("__")
}

pub(crate) fn of(scopes: &Scopes, inferred: &Inferred) -> Result<Exports> {
    let declared = &scopes.declared;
    if let Some((span, message)) = declared.problems.first() {
        return Err(Error::Annotation {
            message: message.clone(),
            span: *span,
        });
    }
    // The shape each annotated variable or global holds.
    let shape_of = |h: &Holder| match h {
        Holder::Var(v) => inferred.var(*v),
        Holder::Global(name) => inferred.global(name),
    };
    let mut classes = HashMap::new();
    for (holder, class) in &declared.classes {
        if let Ty::Shape(k) = shape_of(holder) {
            classes.insert(
                k,
                DeclaredClass {
                    name: class.name.clone(),
                    fields: class.fields.clone(),
                },
            );
        }
    }
    let mut field_types = HashMap::new();
    for ((holder, field), ty) in &declared.fields {
        if let Ty::Shape(k) = shape_of(holder) {
            field_types.insert((k, field.clone()), ty.clone());
        }
    }
    let first = inferred.chunk.as_ref().map_or(Ty::Nil, |r| r.first());
    let mut reader = Reader {
        scopes,
        inferred,
        classes,
        field_types,
        tables: Vec::new(),
        seen: HashMap::new(),
    };
    let value = reader.exported(first, None);
    Ok(Exports {
        value,
        tables: reader.tables,
    })
}

struct Reader<'a> {
    scopes: &'a Scopes,
    inferred: &'a Inferred,
    classes: HashMap<ShapeId, DeclaredClass>,
    field_types: HashMap<(ShapeId, String), LuaType>,
    tables: Vec<ExportedTable>,
    /// Each shape's table, once read.
    seen: HashMap<ShapeId, usize>,
}

impl Reader<'_> {
    /// The export a value of type `ty` is; `declared` is the `@type` of
    /// the field that holds it.
    fn exported(&mut self, ty: Ty, declared: Option<LuaType>) -> Exported {
        match ty {
            Ty::Func(f) => Exported::Function(self.function(f)),
            Ty::Shape(k) => Exported::Table(self.table(k)),
            _ => Exported::Value(declared),
        }
    }

    fn function(&self, f: FuncId) -> ExportedFunction {
        let info = self.scopes.func(f);
        let params: Vec<String> = info
            .params
            .iter()
            .map(|&v| self.scopes.var(v).name.clone())
            .collect();
        let signature = self.scopes.declared.funcs.get(&f).map(|sig| {
            let declared = |name: &str| {
                sig.params
                    .iter()
                    .find(|(n, _)| n == name)
                    .map(|(_, ty)| ty.clone())
            };
            Signature {
                params: params
                    .iter()
                    .map(|p| declared(p).unwrap_or(LuaType::Any))
                    .collect(),
                variadic: declared("..."),
                returns: sig.returns.clone(),
            }
        });
        ExportedFunction {
            method: params.first().is_some_and(|p| p == "self"),
            params,
            variadic: info.is_vararg,
            signature,
        }
    }

    fn table(&mut self, k: ShapeId) -> usize {
        if let Some(&i) = self.seen.get(&k) {
            return i;
        }
        let inferred = self.inferred;
        let shape = inferred.shape(k);
        let i = self.tables.len();
        self.seen.insert(k, i);
        self.tables.push(ExportedTable {
            fields: Vec::new(),
            open: shape.dynamic_keys || (shape.escapes && inferred.blind_dynamic_stores),
            instances: Vec::new(),
            class: self.classes.get(&k).cloned(),
        });
        let fields = shape
            .fields
            .iter()
            .filter(|(name, _)| !is_metafield(name))
            .map(|(name, &ty)| {
                let declared = self.field_types.get(&(k, name.clone())).cloned();
                (name.clone(), self.exported(ty, declared))
            })
            .collect();
        self.tables[i].fields = fields;
        let instances: Vec<usize> = inferred
            .shapes
            .iter()
            .enumerate()
            .filter(|(_, o)| o.classes.contains(&k))
            .map(|(o, _)| self.table(ShapeId(o as u32)))
            .collect();
        self.tables[i].instances = instances;
        i
    }
}
