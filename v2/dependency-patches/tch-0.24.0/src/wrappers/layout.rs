/// A tensor layout.

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Layout {
    Strided,
    Sparse,
    SparseCsr,
    Mkldnn,
    SparseCsc,
    SparseBsr,
    SparseBsc,
    NumOptions,
}

impl Layout {
    // This should be kept in sync with include/c10/core/Layout.h
    pub fn to_i8(&self) -> i8 {
        match self {
            Self::Strided => 0,
            Self::Sparse => 1,
            Self::SparseCsr => 2,
            Self::Mkldnn => 3,
            Self::SparseCsc => 4,
            Self::SparseBsr => 5,
            Self::SparseBsc => 6,
            Self::NumOptions => 7,
        }
    }
}
