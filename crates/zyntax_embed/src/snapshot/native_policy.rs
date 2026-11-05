/// Runtime-only settings leave the snapshot's existing cleanup valid.
pub(super) fn allows_key(key: &str) -> bool {
    !key.starts_with("ZYNTAX_")
        || key.starts_with("ZYNTAX_TRACE_")
        || key.starts_with("ZYNTAX_DUMP_")
        || matches!(
            key,
            "ZYNTAX_COMPILE_WORKERS"
                | "ZYNTAX_DISABLE_WARM_UP"
                | "ZYNTAX_OPTIMIZE_QUIET_MS"
                | "ZYNTAX_PORTABLE_COMPILE_AT_LOAD"
                // Only newly prepared runtime bodies defer releases.
                | "ZYNTAX_GC_MANAGED_DROPS"
        )
}

#[cfg(test)]
mod tests {
    use super::allows_key;

    #[test]
    fn runtime_drop_policy_preserves_precompiled_cleanup() {
        assert!(allows_key("ZYNTAX_GC_MANAGED_DROPS"));
        assert!(allows_key("ZYNTAX_TRACE_PORTABLE"));
        assert!(allows_key("ZYNTAX_COMPILE_WORKERS"));
    }

    #[test]
    fn codegen_overrides_still_decline_precompiled_code() {
        assert!(!allows_key("ZYNTAX_DISABLE_PORTABLE_CODE"));
        assert!(!allows_key("ZYNTAX_DISABLE_AUTOMATIC_RELEASE"));
        assert!(!allows_key("ZYNTAX_DROP_GLUE"));
        assert!(!allows_key("ZYNTAX_UNKNOWN_OPTIMIZATION"));
    }
}
