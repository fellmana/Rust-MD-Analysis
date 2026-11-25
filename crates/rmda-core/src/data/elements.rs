//! Element symbols for atom types: given names, the file's own labels, or a
//! type's mass looked up in a table of standard atomic weights.

/// Standard atomic weights (u), H to U.
const WEIGHTS: [(&str, f64); 92] = [
    ("H", 1.008),
    ("He", 4.0026),
    ("Li", 6.94),
    ("Be", 9.0122),
    ("B", 10.81),
    ("C", 12.011),
    ("N", 14.007),
    ("O", 15.999),
    ("F", 18.998),
    ("Ne", 20.180),
    ("Na", 22.990),
    ("Mg", 24.305),
    ("Al", 26.982),
    ("Si", 28.085),
    ("P", 30.974),
    ("S", 32.06),
    ("Cl", 35.45),
    ("Ar", 39.948),
    ("K", 39.098),
    ("Ca", 40.078),
    ("Sc", 44.956),
    ("Ti", 47.867),
    ("V", 50.942),
    ("Cr", 51.996),
    ("Mn", 54.938),
    ("Fe", 55.845),
    ("Co", 58.933),
    ("Ni", 58.693),
    ("Cu", 63.546),
    ("Zn", 65.38),
    ("Ga", 69.723),
    ("Ge", 72.630),
    ("As", 74.922),
    ("Se", 78.971),
    ("Br", 79.904),
    ("Kr", 83.798),
    ("Rb", 85.468),
    ("Sr", 87.62),
    ("Y", 88.906),
    ("Zr", 91.224),
    ("Nb", 92.906),
    ("Mo", 95.95),
    ("Tc", 97.907),
    ("Ru", 101.07),
    ("Rh", 102.91),
    ("Pd", 106.42),
    ("Ag", 107.87),
    ("Cd", 112.41),
    ("In", 114.82),
    ("Sn", 118.71),
    ("Sb", 121.76),
    ("Te", 127.60),
    ("I", 126.90),
    ("Xe", 131.29),
    ("Cs", 132.91),
    ("Ba", 137.33),
    ("La", 138.91),
    ("Ce", 140.12),
    ("Pr", 140.91),
    ("Nd", 144.24),
    ("Pm", 144.91),
    ("Sm", 150.36),
    ("Eu", 151.96),
    ("Gd", 157.25),
    ("Tb", 158.93),
    ("Dy", 162.50),
    ("Ho", 164.93),
    ("Er", 167.26),
    ("Tm", 168.93),
    ("Yb", 173.05),
    ("Lu", 174.97),
    ("Hf", 178.49),
    ("Ta", 180.95),
    ("W", 183.84),
    ("Re", 186.21),
    ("Os", 190.23),
    ("Ir", 192.22),
    ("Pt", 195.08),
    ("Au", 196.97),
    ("Hg", 200.59),
    ("Tl", 204.38),
    ("Pb", 207.2),
    ("Bi", 208.98),
    ("Po", 208.98),
    ("At", 209.99),
    ("Rn", 222.02),
    ("Fr", 223.02),
    ("Ra", 226.03),
    ("Ac", 227.03),
    ("Th", 232.04),
    ("Pa", 231.04),
    ("U", 238.03),
];

/// A mass matches an element within this, u: LAMMPS inputs give weights to
/// two decimals or more, and no two elements here are this close but Po/Bi.
const MASS_TOLERANCE: f64 = 0.1;

/// The element whose standard atomic weight is nearest `mass`, when it is
/// within [`MASS_TOLERANCE`] and no other element is.
pub fn element_for_mass(mass: f64) -> Option<&'static str> {
    if !mass.is_finite() {
        return None;
    }
    let mut hits = WEIGHTS
        .iter()
        .filter(|(_, w)| (w - mass).abs() < MASS_TOLERANCE);
    let first = hits.next()?;
    hits.next().is_none().then_some(first.0)
}

/// The name of each type `1..=ntypes` (index `t - 1`): `given` first, then
/// the file's labels, then the element of the type's mass, else `X`.
pub fn type_names(
    given: &[String],
    labels: &[String],
    masses: &[f64],
    ntypes: usize,
) -> Vec<String> {
    (0..ntypes)
        .map(|k| {
            if let Some(g) = given.get(k) {
                return g.clone();
            }
            if let Some(l) = labels.get(k).filter(|l| !l.is_empty()) {
                return l.clone();
            }
            masses
                .get(k)
                .and_then(|&m| element_for_mass(m))
                .unwrap_or("X")
                .to_owned()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masses_name_their_elements() {
        assert_eq!(element_for_mass(183.84), Some("W"));
        assert_eq!(element_for_mass(55.845), Some("Fe"));
        assert_eq!(element_for_mass(50.9415), Some("V"));
        assert_eq!(element_for_mass(1.0e3), None);
        assert_eq!(element_for_mass(f64::NAN), None);
    }

    #[test]
    fn given_names_beat_labels_and_labels_beat_masses() {
        let names = type_names(
            &["V".to_owned()],
            &[String::new(), "Mo".to_owned()],
            &[183.84, 95.95, 55.845, 999.0],
            4,
        );
        assert_eq!(names, ["V", "Mo", "Fe", "X"]);
    }
}
