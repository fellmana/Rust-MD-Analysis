//! Atom selection expressions (design §7.4, first subset).
//!
//! Grammar: clauses joined by `&&`, each `FIELD OP NUMBER` with `FIELD` one of
//! `x y z` (Cartesian, Å), `sx sy sz` (fractional in the cell), `type`, `id`,
//! and `OP` one of `< <= > >= == !=`. Example: `z > 500 && type != 3`.

use crate::cell::{Coord3, UnitCell};
use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq)]
enum Field {
    X(usize),
    Frac(usize),
    Type,
    Id,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Op {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

#[derive(Debug, Clone, PartialEq)]
struct Clause {
    field: Field,
    op: Op,
    value: f64,
}

/// A parsed selection.
#[derive(Debug, Clone, PartialEq)]
pub struct Selection {
    text: String,
    clauses: Vec<Clause>,
}

impl Selection {
    pub fn parse(text: &str) -> Result<Self> {
        let mut clauses = Vec::new();
        for part in text.split("&&") {
            let part = part.trim();
            if part.is_empty() {
                return Err(Error::invalid(format!(
                    "empty clause in selection '{text}'"
                )));
            }
            // Longest operators first, so `<=` is not read as `<`.
            let ops = [
                ("<=", Op::Le),
                (">=", Op::Ge),
                ("==", Op::Eq),
                ("!=", Op::Ne),
                ("<", Op::Lt),
                (">", Op::Gt),
            ];
            let Some((at, sym, op)) = ops
                .iter()
                .filter_map(|(s, o)| part.find(s).map(|i| (i, *s, *o)))
                .min_by_key(|(i, s, _)| (*i, std::cmp::Reverse(s.len())))
            else {
                return Err(Error::invalid(format!("no comparison in clause '{part}'")));
            };
            let name = part[..at].trim();
            let value = part[at + sym.len()..].trim();
            let field = match name {
                "x" => Field::X(0),
                "y" => Field::X(1),
                "z" => Field::X(2),
                "sx" => Field::Frac(0),
                "sy" => Field::Frac(1),
                "sz" => Field::Frac(2),
                "type" => Field::Type,
                "id" => Field::Id,
                other => {
                    return Err(Error::invalid(format!(
                        "unknown field '{other}' (x y z sx sy sz type id)"
                    )))
                }
            };
            let value: f64 = value
                .parse()
                .map_err(|_| Error::invalid(format!("bad number '{value}' in '{part}'")))?;
            clauses.push(Clause { field, op, value });
        }
        Ok(Self {
            text: text.to_owned(),
            clauses,
        })
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    /// Whether an atom matches. `frac` is its fractional position.
    pub fn matches(&self, pos: Coord3, frac: Coord3, ty: u32, id: i64) -> bool {
        self.clauses.iter().all(|c| {
            let v = match c.field {
                Field::X(d) => pos[d],
                Field::Frac(d) => frac[d],
                Field::Type => ty as f64,
                Field::Id => id as f64,
            };
            match c.op {
                Op::Lt => v < c.value,
                Op::Le => v <= c.value,
                Op::Gt => v > c.value,
                Op::Ge => v >= c.value,
                Op::Eq => v == c.value,
                Op::Ne => v != c.value,
            }
        })
    }
}

/// Label atoms by the first of `regions` they match, `-1` for none.
pub fn label_regions(
    cell: &UnitCell,
    positions: &[Coord3],
    types: &[u32],
    ids: &[i64],
    regions: &[Selection],
) -> Result<Vec<i32>> {
    let inv_t = cell.inverse_transposed()?;
    Ok(positions
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            let f = cell.x2lamda_pre(p, &inv_t);
            let ty = types.get(i).copied().unwrap_or(0);
            let id = ids.get(i).copied().unwrap_or(i as i64 + 1);
            regions
                .iter()
                .position(|r| r.matches(p, f, ty, id))
                .map_or(-1, |k| k as i32)
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_matches() {
        let s = Selection::parse("z <= 500 && type != 3").unwrap();
        assert!(s.matches([0.0, 0.0, 500.0], [0.0; 3], 1, 1));
        assert!(!s.matches([0.0, 0.0, 500.1], [0.0; 3], 1, 1));
        assert!(!s.matches([0.0, 0.0, 10.0], [0.0; 3], 3, 1));
        let f = Selection::parse("sz>0.5").unwrap();
        assert!(f.matches([0.0; 3], [0.0, 0.0, 0.7], 1, 1));
    }

    #[test]
    fn rejects_bad_input() {
        for bad in ["", "z", "q < 1", "z < abc", "z < 1 &&"] {
            assert!(Selection::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn labels_by_first_matching_region() {
        let cell = UnitCell::orthorhombic(10.0, 10.0, 10.0);
        let pos = [[1.0, 1.0, 1.0], [1.0, 1.0, 6.0], [1.0, 1.0, 9.5]];
        let r = [
            Selection::parse("z < 5").unwrap(),
            Selection::parse("z < 9").unwrap(),
        ];
        assert_eq!(
            label_regions(&cell, &pos, &[1; 3], &[1, 2, 3], &r).unwrap(),
            vec![0, 1, -1]
        );
    }
}
