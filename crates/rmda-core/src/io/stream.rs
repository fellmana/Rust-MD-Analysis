//! Streaming gzip input (design §8.1 `ByteSource::Gzip`).
//!
//! A `MultiGzDecoder`'s byte stream (so concatenated members, as LAMMPS
//! `dump *.gz` and parallel writers produce, read as one stream; inflated in
//! parallel by `io::gzpar` when it can) feeds a buffer from which one
//! frame's bytes at a time are cut, using only the frame header to find where
//! the frame ends. Peak memory is one decompressed frame plus the read-ahead,
//! not the decompressed file.

use crate::error::{Error, Result};
use crate::io::engine::{count_newlines, fast_i64};
use crate::io::format::Format;
use std::io::Read;
use std::path::Path;

/// Bytes pulled from the decoder per refill.
const READ_AHEAD: usize = 4 << 20;

pub struct GzFrames {
    dec: Box<dyn Read + Send>,
    /// Decompressed bytes not yet handed out; `buf[..start]` is consumed.
    buf: Vec<u8>,
    start: usize,
    eof: bool,
    /// Lines handed out so far (the next frame's first line is `line + 1`).
    line: usize,
    pub path: String,
}

impl GzFrames {
    pub fn open(path: &Path) -> Result<Self> {
        Ok(Self::from_reader(
            crate::io::gzpar::open(path)?,
            &path.display().to_string(),
        ))
    }

    pub fn from_reader(dec: Box<dyn Read + Send>, name: &str) -> Self {
        GzFrames {
            dec,
            buf: Vec::new(),
            start: 0,
            eof: false,
            line: 0,
            path: name.to_string(),
        }
    }

    /// Pull more decompressed bytes; `false` at end of stream.
    fn fill(&mut self) -> Result<bool> {
        if self.eof {
            return Ok(false);
        }
        let old = self.buf.len();
        self.buf.resize(old + READ_AHEAD, 0);
        let mut got = 0;
        while got < READ_AHEAD {
            match self.dec.read(&mut self.buf[old + got..]) {
                Ok(0) => {
                    self.eof = true;
                    break;
                }
                Ok(k) => got += k,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => {
                    self.buf.truncate(old);
                    return Err(Error::Io(format!("cannot decompress {}: {e}", self.path)));
                }
            }
        }
        self.buf.truncate(old + got);
        Ok(got > 0)
    }

    /// End offset (exclusive, through its newline) of the line starting at
    /// `at`, refilling as needed; `None` when the stream ends before `at`.
    fn line_end(&mut self, at: usize) -> Result<Option<usize>> {
        let mut from = at;
        loop {
            if let Some(p) = self.buf[from..].iter().position(|&b| b == b'\n') {
                return Ok(Some(from + p + 1));
            }
            from = self.buf.len();
            if !self.fill()? {
                return Ok(if at < self.buf.len() {
                    Some(self.buf.len())
                } else {
                    None
                });
            }
        }
    }

    /// End offset after `n` more lines from `at`.
    fn lines_end(&mut self, at: usize, n: usize) -> Result<Option<usize>> {
        let mut pos = at;
        let mut left = n;
        while left > 0 {
            let avail = &self.buf[pos..];
            let nl = count_newlines(avail);
            if nl >= left {
                // The `left`-th newline in `avail`.
                let mut seen = 0;
                for (i, &b) in avail.iter().enumerate() {
                    if b == b'\n' {
                        seen += 1;
                        if seen == left {
                            return Ok(Some(pos + i + 1));
                        }
                    }
                }
            }
            left -= nl;
            pos = self.buf.len();
            if !self.fill()? {
                // A last line without a newline still counts.
                return Ok(if left == 1 && pos > at && self.buf[pos - 1] != b'\n' {
                    Some(pos)
                } else {
                    None
                });
            }
        }
        Ok(Some(pos))
    }

    /// Cut `buf[start..end]` out as a frame.
    fn take(&mut self, end: usize) -> (Vec<u8>, usize) {
        let rest = self.buf.split_off(end);
        let frame = std::mem::replace(&mut self.buf, rest);
        let first = self.line;
        self.line +=
            count_newlines(&frame) + usize::from(frame.last().map_or(false, |&b| b != b'\n'));
        (frame, first)
    }

    fn truncated(&self, what: &str) -> Error {
        Error::at_line(
            &self.path,
            self.line,
            format!("stream ends inside a frame ({what})"),
        )
    }

    /// The next frame's bytes and the file line number before its first line,
    /// or `None` at the end of the stream.
    pub fn next_frame(&mut self, fmt: Format) -> Result<Option<(Vec<u8>, usize)>> {
        // Blank lines between frames belong to no frame.
        loop {
            let Some(end) = self.line_end(self.start)? else {
                return Ok(None);
            };
            let line = &self.buf[self.start..end];
            if line.iter().all(|b| b.is_ascii_whitespace()) {
                self.start = end;
                self.line += 1;
                continue;
            }
            break;
        }
        // Frames are cut from the front of the buffer, so offsets below stay
        // valid while it grows.
        if self.start > 0 {
            self.buf.drain(..self.start);
            self.start = 0;
        }
        match fmt {
            Format::LammpsData => {
                while self.fill()? {}
                let end = self.buf.len();
                Ok(Some(self.take(end)))
            }
            Format::Xyz => {
                let l1 = self
                    .line_end(self.start)?
                    .ok_or_else(|| self.truncated("count"))?;
                let n = fast_i64(&self.buf[self.start..l1])
                    .filter(|&n| n >= 0)
                    .ok_or_else(|| Error::at_line(&self.path, self.line + 1, "bad atom count"))?;
                let end = self
                    .lines_end(l1, n as usize + 1)?
                    .ok_or_else(|| self.truncated("atom block"))?;
                Ok(Some(self.take(end)))
            }
            Format::LammpsDump => {
                let mut at = self.start;
                let mut natoms: Option<usize> = None;
                loop {
                    let e = self.line_end(at)?.ok_or_else(|| self.truncated("header"))?;
                    let line = &self.buf[at..e];
                    let item = String::from_utf8_lossy(line);
                    let item = item.trim();
                    at = e;
                    if item == "ITEM: NUMBER OF ATOMS" {
                        let v = self.line_end(at)?.ok_or_else(|| self.truncated("header"))?;
                        natoms = fast_i64(&self.buf[at..v]).map(|n| n.max(0) as usize);
                        at = v;
                    } else if item.starts_with("ITEM: ATOMS") {
                        break;
                    }
                }
                // Without a count the parser reports the header error itself.
                let end = match natoms {
                    Some(n) => self
                        .lines_end(at, n)?
                        .ok_or_else(|| self.truncated("atom block"))?,
                    None => at,
                };
                Ok(Some(self.take(end)))
            }
            Format::LammpsDumpBinary => Err(Error::Unsupported(format!(
                "{}: gzipped binary dumps are not supported",
                self.path
            ))),
        }
    }
}
