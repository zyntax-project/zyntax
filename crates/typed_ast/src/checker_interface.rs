//! Body-free source typing facts carried by compiled imports and snapshots.

use crate::typed_ast::ParamOwnership;
use crate::{InternedString, Type};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CheckerInterface {
    pub module: InternedString,
    pub exports: FxHashMap<InternedString, CheckerExport>,
    pub types: FxHashMap<InternedString, CheckerType>,
    pub functions: FxHashMap<InternedString, Arc<CheckerFunction>>,
    pub views: FxHashMap<InternedString, Arc<CheckerView>>,
}

/// The original declaration behind an exported name, shared across forwarding paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckerExport {
    pub origin: InternedString,
    pub type_only: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckerType {
    pub ty: Type,
    pub view: Option<InternedString>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckerView {
    pub ty: Type,
    pub fields: Vec<CheckerField>,
}

impl CheckerView {
    pub fn field(&self, name: InternedString) -> Option<&CheckerField> {
        self.fields.iter().find(|field| field.name == name)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckerField {
    pub name: InternedString,
    pub ty: Type,
    pub readonly: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckerFunction {
    pub native: InternedString,
    pub type_params: Vec<(InternedString, Type)>,
    pub params: Vec<CheckerParameter>,
    /// `None` only while a frontend is resolving an inferred signature.
    pub ret: Option<Type>,
    pub ret_view: Option<InternedString>,
    pub array_result: Option<ArrayReturn>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckerParameter {
    pub name: InternedString,
    pub ty: Type,
    pub view: Option<InternedString>,
    pub ownership: ParamOwnership,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ArrayReturn {
    pub minimum: u64,
    pub parameter: Option<usize>,
}
