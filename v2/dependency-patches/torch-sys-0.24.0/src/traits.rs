macro_rules! list_trait {
    ($trait: ident, $ty: ident) => {
        pub trait $trait {
            fn as_ptr(&self) -> *const $ty;
            fn len_i32(&self) -> i32;
        }

        impl $trait for $ty {
            fn as_ptr(&self) -> *const $ty {
                self
            }
            fn len_i32(&self) -> i32 {
                1
            }
        }

        impl $trait for &[$ty] {
            fn as_ptr(&self) -> *const $ty {
                (*self).as_ptr()
            }
            fn len_i32(&self) -> i32 {
                self.len() as i32
            }
        }

        impl $trait for Vec<$ty> {
            fn as_ptr(&self) -> *const $ty {
                self.as_slice().as_ptr()
            }
            fn len_i32(&self) -> i32 {
                self.len() as i32
            }
        }

        impl $trait for &Vec<$ty> {
            fn as_ptr(&self) -> *const $ty {
                self.as_slice().as_ptr()
            }
            fn len_i32(&self) -> i32 {
                self.len() as i32
            }
        }

        impl $trait for &Box<[$ty]> {
            fn as_ptr(&self) -> *const $ty {
                (**self).as_ptr()
            }
            fn len_i32(&self) -> i32 {
                self.len() as i32
            }
        }

        impl $trait for [$ty] {
            fn as_ptr(&self) -> *const $ty {
                self.as_ptr()
            }
            fn len_i32(&self) -> i32 {
                self.len() as i32
            }
        }

        impl<const N: usize> $trait for &[$ty; N] {
            fn as_ptr(&self) -> *const $ty {
                self.as_slice().as_ptr()
            }
            fn len_i32(&self) -> i32 {
                self.len() as i32
            }
        }

        impl<const N: usize> $trait for [$ty; N] {
            fn as_ptr(&self) -> *const $ty {
                self.as_slice().as_ptr()
            }
            fn len_i32(&self) -> i32 {
                self.len() as i32
            }
        }
    };
}

list_trait!(DoubleList, f64);
list_trait!(IntList, i64);

pub trait IntListOption {
    fn as_ptr(&self) -> *const i64;
    fn len_i32(&self) -> i32;
}

trait IntListOption_: IntList {}

impl<T: IntListOption_> IntListOption for T {
    fn as_ptr(&self) -> *const i64 {
        self.as_ptr()
    }
    fn len_i32(&self) -> i32 {
        self.len_i32()
    }
}

impl<T: IntListOption_> IntListOption for Option<T> {
    fn as_ptr(&self) -> *const i64 {
        match self {
            Some(v) => v.as_ptr(),
            None => std::ptr::null(),
        }
    }

    fn len_i32(&self) -> i32 {
        self.as_ref().map_or(-1, |v| v.len_i32())
    }
}

impl IntListOption_ for i64 {}
impl IntListOption_ for [i64] {}
impl IntListOption_ for &[i64] {}
impl IntListOption_ for Vec<i64> {}
impl IntListOption_ for &Vec<i64> {}
impl IntListOption_ for &Box<[i64]> {}
