//! Minimal 3-vector and column-major 4×4 matrix helpers.
//!
//! Deliberately dependency-free: this crate targets wasm32 for the ADR-324
//! demo surface, and a `nalgebra` pull would dominate the module size for
//! what is a handful of fixed-size operations. Matrices use the OpenGL /
//! three.js `Matrix4.elements` layout: column-major, `m[col * 4 + row]`.

/// A 3-component `f64` vector.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Vec3 {
    /// X component.
    pub x: f64,
    /// Y component.
    pub y: f64,
    /// Z component.
    pub z: f64,
}

impl Vec3 {
    /// Construct from components.
    #[inline]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Scale by `s`.
    #[inline]
    pub fn scale(self, s: f64) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }

    /// Dot product.
    #[inline]
    pub fn dot(self, rhs: Vec3) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// Cross product (right-handed).
    #[inline]
    pub fn cross(self, rhs: Vec3) -> Vec3 {
        Vec3::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    /// Euclidean length.
    #[inline]
    pub fn length(self) -> f64 {
        self.dot(self).sqrt()
    }

    /// Unit vector, or `None` when the length is (near) zero.
    #[inline]
    pub fn normalize(self) -> Option<Vec3> {
        let len = self.length();
        if len <= f64::EPSILON {
            None
        } else {
            Some(self.scale(1.0 / len))
        }
    }
}

impl core::ops::Sub for Vec3 {
    type Output = Vec3;
    #[inline]
    fn sub(self, rhs: Vec3) -> Vec3 {
        Vec3::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl core::ops::Add for Vec3 {
    type Output = Vec3;
    #[inline]
    fn add(self, rhs: Vec3) -> Vec3 {
        Vec3::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

/// Column-major 4×4 matrix, `m[col * 4 + row]` — the exact layout of
/// three.js `Matrix4.elements` / OpenGL, so the array can be passed to
/// `Matrix4.fromArray` untouched.
pub type Mat4 = [f64; 16];

/// The identity matrix.
pub const IDENTITY: Mat4 = [
    1.0, 0.0, 0.0, 0.0, //
    0.0, 1.0, 0.0, 0.0, //
    0.0, 0.0, 1.0, 0.0, //
    0.0, 0.0, 0.0, 1.0,
];

/// `a * b` (column-major).
#[inline]
pub fn mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = [0.0; 16];
    for col in 0..4 {
        for row in 0..4 {
            let mut acc = 0.0;
            for k in 0..4 {
                acc += a[k * 4 + row] * b[col * 4 + k];
            }
            out[col * 4 + row] = acc;
        }
    }
    out
}

/// Transform a point (`w = 1`); returns the transformed `(x, y, z)` and `w`
/// *before* the perspective divide, so callers can check clip-space signs.
#[inline]
pub fn transform_point(m: &Mat4, p: Vec3) -> (Vec3, f64) {
    let x = m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12];
    let y = m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13];
    let z = m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14];
    let w = m[3] * p.x + m[7] * p.y + m[11] * p.z + m[15];
    (Vec3::new(x, y, z), w)
}

/// Transform a point and apply the perspective divide, yielding normalized
/// device coordinates. Returns `None` when `w` is (near) zero.
#[inline]
pub fn project_point(m: &Mat4, p: Vec3) -> Option<Vec3> {
    let (v, w) = transform_point(m, p);
    if w.abs() <= f64::EPSILON {
        None
    } else {
        Some(v.scale(1.0 / w))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_transforms_are_noops() {
        let p = Vec3::new(1.5, -2.0, 3.25);
        let (q, w) = transform_point(&IDENTITY, p);
        assert_eq!(q, p);
        assert_eq!(w, 1.0);
        assert_eq!(mul(&IDENTITY, &IDENTITY), IDENTITY);
    }

    #[test]
    fn cross_is_right_handed() {
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        assert_eq!(x.cross(y), Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn normalize_rejects_zero() {
        assert!(Vec3::new(0.0, 0.0, 0.0).normalize().is_none());
        let n = Vec3::new(0.0, 3.0, 4.0).normalize().unwrap();
        assert!((n.length() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn mul_matches_manual_translation_composition() {
        // T(a) * T(b) == T(a + b) for translations.
        let mut ta = IDENTITY;
        ta[12] = 1.0;
        ta[13] = 2.0;
        ta[14] = 3.0;
        let mut tb = IDENTITY;
        tb[12] = -4.0;
        tb[13] = 0.5;
        tb[14] = 7.0;
        let tab = mul(&ta, &tb);
        assert_eq!(tab[12], -3.0);
        assert_eq!(tab[13], 2.5);
        assert_eq!(tab[14], 10.0);
    }
}
