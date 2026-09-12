//! What every Python program starts with, written in Python: the
//! exception hierarchy. It is parsed like the program and its classes
//! come first, so `raise ValueError("x")` names an ordinary class and a
//! library error becomes an instance of one.

/// The classes the library's error kinds map onto, by name.
pub(crate) const EXCEPTION_KINDS: &[&str] = &[
    "ValueError",
    "TypeError",
    "KeyError",
    "IndexError",
    "ZeroDivisionError",
    "AttributeError",
    "RuntimeError",
    "NotImplementedError",
    "StopIteration",
    "AssertionError",
    "NameError",
    "OverflowError",
    "Exception",
];

pub(crate) const SOURCE: &str = r#"
class BaseException:
    def __init__(self, message: str = ""):
        self.message = message
    def __str__(self) -> str:
        return self.message
class Exception(BaseException):
    pass
class ArithmeticError(Exception):
    pass
class ZeroDivisionError(ArithmeticError):
    pass
class OverflowError(ArithmeticError):
    pass
class LookupError(Exception):
    pass
class IndexError(LookupError):
    pass
class KeyError(LookupError):
    def __str__(self) -> str:
        return repr(self.message)
class ValueError(Exception):
    pass
class TypeError(Exception):
    pass
class AttributeError(Exception):
    pass
class NameError(Exception):
    pass
class RuntimeError(Exception):
    pass
class NotImplementedError(RuntimeError):
    pass
class StopIteration(Exception):
    pass
class AssertionError(Exception):
    pass
"#;
