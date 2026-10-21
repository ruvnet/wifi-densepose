//! Scalar elements.

use crate::TchError;

/// A single scalar value.
pub struct Scalar {
    pub(super) c_scalar: *mut torch_sys::C_scalar,
}

impl Scalar {
    /// Creates an integer scalar.
    pub fn int(v: i64) -> Scalar {
        let c_scalar = unsafe_torch!(torch_sys::ats_int(v));
        Scalar { c_scalar }
    }

    /// Creates a float scalar scalar.
    pub fn float(v: f64) -> Scalar {
        let c_scalar = unsafe_torch!(torch_sys::ats_float(v));
        Scalar { c_scalar }
    }

    /// Returns an integer value.
    pub fn to_int(&self) -> Result<i64, TchError> {
        let i = unsafe_torch_err!(torch_sys::ats_to_int(self.c_scalar));
        Ok(i)
    }

    /// Returns a float value.
    pub fn to_float(&self) -> Result<f64, TchError> {
        let f = unsafe_torch_err!(torch_sys::ats_to_float(self.c_scalar));
        Ok(f)
    }

    /// Returns a string representation of the scalar.
    pub fn to_string(&self) -> Result<String, TchError> {
        let s = unsafe_torch_err!({
            super::utils::ptr_to_string(torch_sys::ats_to_string(self.c_scalar))
        });
        match s {
            None => Err(TchError::Kind("nullptr representation".to_string())),
            Some(s) => Ok(s),
        }
    }
}

impl std::fmt::Debug for Scalar {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.to_string() {
            Err(_) => write!(f, "err"),
            Ok(s) => write!(f, "scalar<{s}>"),
        }
    }
}

impl Drop for Scalar {
    fn drop(&mut self) {
        unsafe_torch!(torch_sys::ats_free(self.c_scalar))
    }
}

impl From<i64> for Scalar {
    fn from(v: i64) -> Scalar {
        Scalar::int(v)
    }
}

impl From<f64> for Scalar {
    fn from(v: f64) -> Scalar {
        Scalar::float(v)
    }
}

impl From<Scalar> for i64 {
    fn from(s: Scalar) -> i64 {
        Self::from(&s)
    }
}

impl From<Scalar> for f64 {
    fn from(s: Scalar) -> f64 {
        Self::from(&s)
    }
}

impl From<&Scalar> for i64 {
    fn from(s: &Scalar) -> i64 {
        s.to_int().unwrap()
    }
}

impl From<&Scalar> for f64 {
    fn from(s: &Scalar) -> f64 {
        s.to_float().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::Scalar;
    #[test]
    fn scalar() {
        let pi = Scalar::float(std::f64::consts::PI);
        assert_eq!(i64::from(&pi), 3);
        assert_eq!(f64::from(&pi), std::f64::consts::PI);
        let leet = Scalar::int(1337);
        assert_eq!(i64::from(&leet), 1337);
        assert_eq!(f64::from(&leet), 1337.);
        assert_eq!(&format!("{pi:?}"), "scalar<3.14159>");
    }
}
