//! Source code management and location tracking

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents a source file with its content and metadata
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceFile {
    pub name: String,
    pub content: String,
    pub line_starts: Vec<usize>, // Byte offsets of line starts
}

impl SourceFile {
    pub fn new(name: String, content: String) -> Self {
        let mut line_starts = vec![0];

        for (i, &byte) in content.as_bytes().iter().enumerate() {
            if byte == b'\n' {
                line_starts.push(i + 1);
            }
        }

        Self {
            name,
            content,
            line_starts,
        }
    }

    pub fn line_count(&self) -> usize {
        self.line_starts.len()
    }

    pub fn get_location(&self, offset: usize) -> Location {
        let line = self
            .line_starts
            .binary_search(&offset)
            .unwrap_or_else(|i| i.saturating_sub(1));

        let line_start = self.line_starts[line];
        let column = offset - line_start;

        Location {
            line: line + 1,     // 1-based
            column: column + 1, // 1-based
            offset,
        }
    }

    pub fn get_line(&self, line_number: usize) -> Option<&str> {
        if line_number == 0 || line_number > self.line_count() {
            return None;
        }

        let start = self.line_starts[line_number - 1];
        let end = if line_number < self.line_count() {
            self.line_starts[line_number] - 1
        } else {
            self.content.len()
        };

        Some(&self.content[start..end])
    }
}

/// A span of source code from start to end position
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    pub fn empty(pos: usize) -> Self {
        Self {
            start: pos,
            end: pos,
        }
    }

    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    pub fn contains(&self, pos: usize) -> bool {
        self.start <= pos && pos < self.end
    }

    pub fn merge(&self, other: Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }

    pub fn slice<'a>(&self, source: &'a str) -> &'a str {
        &source[self.start..self.end]
    }
}

impl Default for Span {
    fn default() -> Self {
        Self { start: 0, end: 0 }
    }
}

/// A specific location in source code
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Location {
    pub line: usize,   // 1-based
    pub column: usize, // 1-based
    pub offset: usize, // 0-based byte offset
}

impl Location {
    pub fn new(line: usize, column: usize, offset: usize) -> Self {
        Self {
            line,
            column,
            offset,
        }
    }
}

/// Manages multiple source files for a compilation session
#[derive(Debug, Default)]
pub struct SourceMap {
    files: HashMap<String, SourceFile>,
    file_ids: HashMap<String, usize>,
    next_file_id: usize,
}

impl SourceMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_file(&mut self, name: String, content: String) -> usize {
        let file = SourceFile::new(name.clone(), content);
        let file_id = self.next_file_id;

        self.files.insert(name.clone(), file);
        self.file_ids.insert(name, file_id);
        self.next_file_id += 1;

        file_id
    }

    pub fn get_file(&self, name: &str) -> Option<&SourceFile> {
        self.files.get(name)
    }

    pub fn get_file_by_id(&self, id: usize) -> Option<&SourceFile> {
        self.file_ids
            .iter()
            .find(|(_, &file_id)| file_id == id)
            .and_then(|(name, _)| self.files.get(name))
    }

    pub fn get_location(&self, file_name: &str, offset: usize) -> Option<Location> {
        self.get_file(file_name)
            .map(|file| file.get_location(offset))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_file_location() {
        let content = "line 1\nline 2\nline 3";
        let file = SourceFile::new("test.txt".to_string(), content.to_string());

        assert_eq!(file.line_count(), 3);
        assert_eq!(file.get_location(0), Location::new(1, 1, 0));
        assert_eq!(file.get_location(7), Location::new(2, 1, 7));
        assert_eq!(file.get_location(14), Location::new(3, 1, 14));
    }

    #[test]
    fn test_span_operations() {
        let span1 = Span::new(5, 10);
        let span2 = Span::new(8, 15);

        assert_eq!(span1.len(), 5);
        assert!(span1.contains(7));
        assert!(!span1.contains(12));

        let merged = span1.merge(span2);
        assert_eq!(merged, Span::new(5, 15));
    }

    #[test]
    fn test_source_map() {
        let mut source_map = SourceMap::new();

        let file1_id = source_map.add_file("file1.txt".to_string(), "content1".to_string());
        let file2_id = source_map.add_file("file2.txt".to_string(), "content2".to_string());

        assert_eq!(file1_id, 0);
        assert_eq!(file2_id, 1);

        assert!(source_map.get_file("file1.txt").is_some());
        assert!(source_map.get_file_by_id(0).is_some());
    }
}
