//! `Date.now()` and `performance.now()`, backed by native clocks.

use super::*;

impl<'ast, 'program> Lowerer<'ast, 'program> {
    /// True when `name` refers to the global object, not a local, import or namespace.
    fn is_global_object(&self, object: &js::Expression<'ast>, name: &str) -> bool {
        let js::Expression::Identifier(id) = object else {
            return false;
        };
        id.name == name
            && !self.namespace_references.contains_key(&id.node_id.get())
            && !self
                .imported_exports
                .contains_key(&self.name(id.name.as_str()))
            && !self
                .scopes
                .iter()
                .any(|s| s.contains_key(&self.name(id.name.as_str())))
    }

    pub(super) fn timing_call(
        &mut self,
        c: &js::CallExpression<'ast>,
        m: &js::StaticMemberExpression<'ast>,
    ) -> Result<Option<Expr>> {
        let symbol = if m.property.name == "now" && self.is_global_object(&m.object, "Date") {
            "$TS$date_now"
        } else if m.property.name == "now" && self.is_global_object(&m.object, "performance") {
            "$TS$performance_now"
        } else {
            return Ok(None);
        };
        if !c.arguments.is_empty() || c.type_arguments.is_some() {
            return self.error(ErrorKind::Type, "now takes no arguments", c.span);
        }
        Ok(Some(self.call(symbol, vec![], number(), c.span)))
    }
}
