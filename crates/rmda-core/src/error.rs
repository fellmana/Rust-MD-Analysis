//! Structured error types with context-rich diagnostics.
//!
//! The parser and analyses must never panic on malformed input; they return
//! [`Error`] with file/byte offsets so the CLI can report precisely.

use std::fmt;

/// Crate-wide result alias.
pub type Result<T> = std::result::Result<T, Error>;

/// Parse / analysis failure with optional source location context.
#[derive(Debug, Clone)]
pub enum Error {
    /// Malformed input (file, offset, line, message).
    Parse {
        source: Option<String>,
        offset: Option<u64>,
        line: Option<u64>,
        msg: String,
    },
    /// Unsupported or unknown format / feature.
    Unsupported(String),
    /// Inconsistent cell / data (e.g. non-finite coordinates).
    InvalidData(String),
    /// Explicitly refused operation (e.g. unimplemented convention).
    NotImplemented(String),
    /// I/O failure (wraps std::io::Error text; we keep the string so Error stays Send+Sync cheap).
    Io(String),
    /// Analysis-level error.
    Analysis(String),
    /// The input cannot be used: a file that cannot be opened or recognised,
    /// or a frame that fails [`crate::data::Frame::check_input`]. The CLI
    /// exits 3 for this and for [`Error::Parse`] (design §21), so a batch
    /// script can tell a bad frame from a failed analysis.
    Input(String),
    /// The command line or a run config asks for something invalid; the CLI
    /// exits 2, as for its own argument errors.
    Usage(String),
}

impl Error {
    pub fn parse(msg: impl Into<String>) -> Self {
        Error::Parse {
            source: None,
            offset: None,
            line: None,
            msg: msg.into(),
        }
    }

    pub fn at(source: impl Into<String>, offset: u64, line: u64, msg: impl Into<String>) -> Self {
        Error::Parse {
            source: Some(source.into()),
            offset: Some(offset),
            line: Some(line),
            msg: msg.into(),
        }
    }

    /// Parse error located at a line number (no byte offset).
    pub fn at_line(source: impl Into<String>, line: usize, msg: impl Into<String>) -> Self {
        Error::Parse {
            source: Some(source.into()),
            offset: None,
            line: Some(line as u64),
            msg: msg.into(),
        }
    }

    pub fn unsupported(msg: impl Into<String>) -> Self {
        Error::Unsupported(msg.into())
    }

    pub fn invalid(msg: impl Into<String>) -> Self {
        Error::InvalidData(msg.into())
    }

    pub fn not_implemented(msg: impl Into<String>) -> Self {
        Error::NotImplemented(msg.into())
    }

    pub fn input(msg: impl Into<String>) -> Self {
        Error::Input(msg.into())
    }
    pub fn usage(msg: impl Into<String>) -> Self {
        Error::Usage(msg.into())
    }
    /// This error as an input error: what opening or reading the input
    /// failed on (an unreadable file, an unrecognised format, a frame the
    /// reader refuses) is the input's fault. Parse errors keep their location.
    pub fn into_input(self) -> Self {
        match self {
            Error::Io(m) | Error::Unsupported(m) | Error::InvalidData(m) => Error::Input(m),
            e => e,
        }
    }
    /// The CLI's exit code for this error (design §21): 3 when the input
    /// is unusable, 2 for an invalid request, 1 otherwise.
    pub fn exit_code(&self) -> i32 {
        match self {
            Error::Parse { .. } | Error::Input(_) => 3,
            Error::Usage(_) => 2,
            _ => 1,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Parse {
                source,
                offset,
                line,
                msg,
            } => {
                write!(f, "parse error")?;
                if let Some(s) = source {
                    write!(f, " in {s}")?;
                }
                // Print whichever location fields are present independently:
                // `at_line` sets only `line`, so requiring both would hide the
                // line number of every line-located error.
                match (offset, line) {
                    (Some(o), Some(l)) => write!(f, " (byte {o}, line {l})")?,
                    (Some(o), None) => write!(f, " (byte {o})")?,
                    (None, Some(l)) => write!(f, " (line {l})")?,
                    (None, None) => {}
                }
                write!(f, ": {msg}")
            }
            Error::Unsupported(m) => write!(f, "unsupported: {m}"),
            Error::InvalidData(m) => write!(f, "invalid data: {m}"),
            Error::NotImplemented(m) => write!(f, "not implemented: {m}"),
            Error::Io(m) => write!(f, "io error: {m}"),
            Error::Analysis(m) => write!(f, "analysis error: {m}"),
            Error::Input(m) => write!(f, "invalid input: {m}"),
            Error::Usage(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A line-located parse error (the common `at_line` path) must show its
    /// line number even though it has no byte offset.
    #[test]
    fn line_only_parse_error_shows_line_number() {
        let e = Error::at_line("traj.dump", 42, "bad atom id");
        let s = e.to_string();
        assert!(s.contains("line 42"), "missing line number: {s}");
        assert!(s.contains("traj.dump"), "missing source: {s}");
        assert!(s.contains("bad atom id"), "missing message: {s}");
        assert!(!s.contains("byte"), "unexpected byte offset: {s}");
    }

    #[test]
    fn offset_only_parse_error_shows_byte_offset() {
        // Offset with no line: the Display must render the byte position
        // without requiring a line number (the inverse of the `at_line` case).
        let e = Error::Parse {
            source: Some("traj.dump".into()),
            offset: Some(1024),
            line: None,
            msg: "truncated".into(),
        };
        let s = e.to_string();
        assert!(s.contains("byte 1024"), "missing byte offset: {s}");
        assert!(!s.contains("line"), "unexpected line number: {s}");
    }

    #[test]
    fn both_location_fields_shown_together() {
        let e = Error::at("traj.dump", 1024, 7, "truncated");
        let s = e.to_string();
        assert!(s.contains("byte 1024") && s.contains("line 7"), "got: {s}");
    }

    #[test]
    fn bare_parse_error_has_no_location() {
        let s = Error::parse("no atoms").to_string();
        assert!(!s.contains("line") && !s.contains("byte"), "got: {s}");
        assert!(s.contains("no atoms"));
    }
}
