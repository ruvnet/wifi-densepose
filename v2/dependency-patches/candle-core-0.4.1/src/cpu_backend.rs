use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{DType, Error, IntDType, Layout, Result, Shape, WithDType};
use half::{bf16, f16};
use rayon::prelude::*;

const USE_IM2COL_CONV1D: bool = true;
const USE_IM2COL_CONV2D: bool = true;

// TODO: Maybe we should not implement [Clone] here and instead have an explicit allocator +
// intercept the oom errors to avoid panicking and provide a proper error.
#[derive(Debug, Clone)]
pub enum CpuStorage {
    U8(Vec<u8>),
    U32(Vec<u32>),
    I64(Vec<i64>),
    BF16(Vec<bf16>),
    F16(Vec<f16>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

#[derive(Debug, Clone)]
pub struct CpuDevice;

pub trait Map1 {
    fn f<T: WithDType>(&self, vs: &[T], layout: &Layout) -> Result<Vec<T>>;

    fn map(&self, vs: &CpuStorage, layout: &Layout) -> Result<CpuStorage> {
        match vs {
            CpuStorage::U8(vs) => Ok(CpuStorage::U8(self.f(vs, layout)?)),
            CpuStorage::U32(vs) => Ok(CpuStorage::U32(self.f(vs, layout)?)),
            CpuStorage::I64(vs) => Ok(CpuStorage::I64(self.f(vs, layout)?)),
            CpuStorage::BF16(vs) => Ok(CpuStorage::BF16(self.f(vs, layout)?)),
            CpuStorage::F16(vs) => Ok(CpuStorage::F16(self.f(vs, layout)?)),
            CpuStorage::F32(vs) => Ok(CpuStorage::F32(self.f(vs, layout)?)),
            CpuStorage::F64(vs) => Ok(CpuStorage::F64(self.f(vs, layout)?)),
        }
    }
}

pub trait Map1Any {
    fn f<T: WithDType, W: Fn(Vec<T>) -> CpuStorage>(
        &self,
        vs: &[T],
        layout: &Layout,
        wrap: W,
    ) -> Result<CpuStorage>;

    fn map(&self, vs: &CpuStorage, layout: &Layout) -> Result<CpuStorage> {
        match vs {
            CpuStorage::U8(vs) => Ok(self.f(vs, layout, CpuStorage::U8)?),
            CpuStorage::U32(vs) => Ok(self.f(vs, layout, CpuStorage::U32)?),
            CpuStorage::I64(vs) => Ok(self.f(vs, layout, CpuStorage::I64)?),
            CpuStorage::BF16(vs) => Ok(self.f(vs, layout, CpuStorage::BF16)?),
            CpuStorage::F16(vs) => Ok(self.f(vs, layout, CpuStorage::F16)?),
            CpuStorage::F32(vs) => Ok(self.f(vs, layout, CpuStorage::F32)?),
            CpuStorage::F64(vs) => Ok(self.f(vs, layout, CpuStorage::F64)?),
        }
    }
}

type C = CpuStorage;
pub trait Map2 {
    const OP: &'static str;
    fn f<T: WithDType>(&self, v1: &[T], l1: &Layout, v2: &[T], l2: &Layout) -> Result<Vec<T>>;

    fn map(
        &self,
        v1: &CpuStorage,
        l1: &Layout,
        v2: &CpuStorage,
        l2: &Layout,
    ) -> Result<CpuStorage> {
        match (v1, v2) {
            (C::U8(v1), C::U8(v2)) => Ok(C::U8(self.f(v1, l1, v2, l2)?)),
            (C::U32(v1), C::U32(v2)) => Ok(C::U32(self.f(v1, l1, v2, l2)?)),
            (C::I64(v1), C::I64(v2)) => Ok(C::I64(self.f(v1, l1, v2, l2)?)),
            (C::BF16(v1), C::BF16(v2)) => Ok(C::BF16(self.f(v1, l1, v2, l2)?)),
            (C::F16(v1), C::F16(v2)) => Ok(C::F16(self.f(v1, l1, v2, l2)?)),
            (C::F32(v1), C::F32(v2)) => Ok(C::F32(self.f(v1, l1, v2, l2)?)),
            (C::F64(v1), C::F64(v2)) => Ok(C::F64(self.f(v1, l1, v2, l2)?)),
            _ => Err(Error::DTypeMismatchBinaryOp {
                lhs: v1.dtype(),
                rhs: v2.dtype(),
                op: Self::OP,
            }
            .bt()),
        }
    }
}

pub trait Map2U8 {
    const OP: &'static str;
    fn f<T: WithDType>(&self, v1: &[T], l1: &Layout, v2: &[T], l2: &Layout) -> Result<Vec<u8>>;

    fn map(
        &self,
        v1: &CpuStorage,
        l1: &Layout,
        v2: &CpuStorage,
        l2: &Layout,
    ) -> Result<CpuStorage> {
        match (v1, v2) {
            (C::U8(v1), C::U8(v2)) => Ok(C::U8(self.f(v1, l1, v2, l2)?)),
            (C::U32(v1), C::U32(v2)) => Ok(C::U8(self.f(v1, l1, v2, l2)?)),
            (C::I64(v1), C::I64(v2)) => Ok(C::U8(self.f(v1, l1, v2, l2)?)),
            (C::BF16(v1), C::BF16(v2)) => Ok(C::U8(self.f(v1, l1, v2, l2)?)),
            (C::F16(v1), C::F16(v2)) => Ok(C::U8(self.f(v1, l1, v2, l2)?)),
            (C::F32(v1), C::F32(v2)) => Ok(C::U8(self.f(v1, l1, v2, l2)?)),
            (C::F64(v1), C::F64(v2)) => Ok(C::U8(self.f(v1, l1, v2, l2)?)),
            _ => Err(Error::DTypeMismatchBinaryOp {
                lhs: v1.dtype(),
                rhs: v2.dtype(),
                op: Self::OP,
            }
            .bt()),
        }
    }
}

struct Cmp(CmpOp);
impl Map2U8 for Cmp {
    const OP: &'static str = "cmp";
    #[inline(always)]
    fn f<T: WithDType>(
        &self,
        lhs: &[T],
        lhs_l: &Layout,
        rhs: &[T],
        rhs_l: &Layout,
    ) -> Result<Vec<u8>> {
        let dst = match self.0 {
            CmpOp::Eq => binary_map(lhs_l, rhs_l, lhs, rhs, |x, y| u8::from(x == y)),
            CmpOp::Ne => binary_map(lhs_l, rhs_l, lhs, rhs, |x, y| u8::from(x != y)),
            CmpOp::Lt => binary_map(lhs_l, rhs_l, lhs, rhs, |x, y| u8::from(x < y)),
            CmpOp::Le => binary_map(lhs_l, rhs_l, lhs, rhs, |x, y| u8::from(x <= y)),
            CmpOp::Gt => binary_map(lhs_l, rhs_l, lhs, rhs, |x, y| u8::from(x > y)),
            CmpOp::Ge => binary_map(lhs_l, rhs_l, lhs, rhs, |x, y| u8::from(x >= y)),
        };
        Ok(dst)
    }
}

struct WCond<'a, T: IntDType>(&'a [T], &'a Layout);

impl<'a, I: IntDType> Map2 for WCond<'a, I> {
    const OP: &'static str = "where";
    #[inline(always)]
    fn f<T: WithDType>(&self, t: &[T], t_l: &Layout, f: &[T], f_l: &Layout) -> Result<Vec<T>> {
        let vs = match (
            self.1.contiguous_offsets(),
            t_l.contiguous_offsets(),
            f_l.contiguous_offsets(),
        ) {
            (Some((o1, o2)), Some((o_t1, o_t2)), Some((o_f1, o_f2))) => {
                let pred = &self.0[o1..o2];
                let t = &t[o_t1..o_t2];
                let f = &f[o_f1..o_f2];
                pred.iter()
                    .zip(t.iter().zip(f.iter()))
                    .map(|(p, (&t, &f))| if p.is_true() { t } else { f })
                    .collect::<Vec<_>>()
            }
            _ => self
                .1
                .strided_index()
                .zip(t_l.strided_index().zip(f_l.strided_index()))
                .map(|(i_p, (i_t, i_f))| {
                    if self.0[i_p].is_true() {
                        t[i_t]
                    } else {
                        f[i_f]
                    }
                })
                .collect::<Vec<_>>(),
        };
        Ok(vs)
    }
}

struct ReduceIndex {
    reduce_dim_index: usize,
    use_min: bool,
    return_index: bool,
}

impl ReduceIndex {
    // The value gets replaced if f(s[current_acc], s[i]) returns true.
    #[inline(always)]
    fn fold_impl<T, U, F, G>(&self, src: &[T], src_l: &Layout, f: F, g: G) -> Result<Vec<U>>
    where
        T: Clone + Copy,
        U: Clone + Copy,
        F: Fn(T, T) -> bool,
        G: Fn(T, usize) -> U,
    {
        let reduce_dim_size = src_l.dims()[self.reduce_dim_index];
        let reduce_dim_stride = src_l.stride()[self.reduce_dim_index];
        let dst_len = src_l.shape().elem_count() / reduce_dim_size;
        let mut dst: Vec<U> = Vec::with_capacity(dst_len);
        let dst_to_set = dst.spare_capacity_mut();
        let dst_to_set = unsafe { std::mem::transmute::<_, &mut [U]>(dst_to_set) };
        match src_l.contiguous_offsets() {
            Some((o1, o2)) => {
                let src = &src[o1..o2];
                if reduce_dim_stride == 1 {
                    for (start_src_i, dst_v) in dst_to_set.iter_mut().enumerate() {
                        let start_src_i = start_src_i * reduce_dim_size;
                        let src = &src[start_src_i..start_src_i + reduce_dim_size];
                        let mut acc = 0;
                        let mut val = src[0];
                        for (src_i, &s) in src.iter().enumerate() {
                            if f(val, s) {
                                acc = src_i;
                                val = s
                            }
                        }
                        *dst_v = g(val, acc)
                    }
                } else {
                    for (start_src_i, dst_v) in dst_to_set.iter_mut().enumerate() {
                        let (p, q) = (
                            start_src_i / reduce_dim_stride,
                            start_src_i % reduce_dim_stride,
                        );
                        // start_src_i = p * reduce_dim_stride + q
                        let start_src_i = p * reduce_dim_stride * reduce_dim_size + q;
                        let src = &src[start_src_i..];
                        let mut acc = 0;
                        let mut val = src[0];
                        for src_i in 0..reduce_dim_size {
                            let s = src[src_i * reduce_dim_stride];
                            if f(val, s) {
                                acc = src_i;
                                val = s
                            }
                        }
                        *dst_v = g(val, acc)
                    }
                }
            }
            None => {
                let l = src_l.narrow(self.reduce_dim_index, 0, 1)?;
                for (unstr_index, src_index) in l.strided_index().enumerate() {
                    let src = &src[src_index..];
                    let mut acc = 0;
                    let mut val = src[0];
                    for src_i in 0..reduce_dim_size {
                        let s = src[src_i * reduce_dim_stride];
                        if f(val, s) {
                            acc = src_i;
                            val = s
                        }
                    }
                    dst_to_set[unstr_index] = g(val, acc)
                }
            }
        }
        unsafe { dst.set_len(dst_len) };
        Ok(dst)
    }
}

impl Map1Any for ReduceIndex {
    #[inline(always)]
    fn f<T: WithDType, W: Fn(Vec<T>) -> CpuStorage>(
        &self,
        src: &[T],
        src_l: &Layout,
        wrap: W,
    ) -> Result<CpuStorage> {
        if src_l.shape().elem_count() == 0 {
            Err(Error::EmptyTensor { op: "reduce" }.bt())?
        }
        let dst = match (self.return_index, self.use_min) {
            (false, true) => wrap(self.fold_impl(src, src_l, |x, y| x > y, |v, _i| v)?),
            (false, false) => wrap(self.fold_impl(src, src_l, |x, y| x < y, |v, _i| v)?),
            (true, true) => {
                CpuStorage::U32(self.fold_impl(src, src_l, |x, y| x > y, |_v, i| i as u32)?)
            }
            (true, false) => {
                CpuStorage::U32(self.fold_impl(src, src_l, |x, y| x < y, |_v, i| i as u32)?)
            }
        };
        Ok(dst)
    }
}

struct ReduceSum<'a> {
    dst_shape: &'a Shape,
    reduce_dims: &'a [usize],
    reduce_dims_and_stride: Vec<(usize, usize)>,
}

impl<'a> ReduceSum<'a> {
    #[inline(always)]
    fn fold_impl<T>(&self, src: &[T], src_l: &Layout, start_elt: T) -> Result<Vec<T>>
    where
        T: WithDType,
    {
        let mut dst = vec![start_elt; self.dst_shape.elem_count()];
        match src_l.contiguous_offsets() {
            Some((o1, o2)) => {
                let src = &src[o1..o2];
                // Handle the case where we reduce over the last dimensions separately as it is
                // fairly common and easy to optimize. This rely on the layout being contiguous!
                // reduce_dims is sorted, check if it is ranging from a to n-1.
                let reduce_over_last_dims = self
                    .reduce_dims
                    .iter()
                    .rev()
                    .enumerate()
                    .all(|(i, &v)| v == src_l.shape().rank() - 1 - i);
                if reduce_over_last_dims {
                    let reduce_sz = self
                        .reduce_dims_and_stride
                        .iter()
                        .map(|(u, _)| u)
                        .product::<usize>();
                    for (dst_i, dst_v) in dst.iter_mut().enumerate() {
                        let src_i = dst_i * reduce_sz;
                        unsafe {
                            T::vec_reduce_sum(
                                src[src_i..src_i + reduce_sz].as_ptr(),
                                dst_v,
                                reduce_sz,
                            )
                        };
                    }
                    return Ok(dst);
                };
                for (unstr_index, &src) in src.iter().enumerate() {
                    let mut dst_index = unstr_index;
                    // Set the reduce_dims indexes to 0.
                    for &(dim, stride) in self.reduce_dims_and_stride.iter() {
                        // The compiler is able to optimize the following in a single divmod op.
                        let (pre, post) = (dst_index / stride, dst_index % stride);
                        dst_index = (pre / dim) * stride + post;
                    }
                    dst[dst_index] += src;
                }
            }
            None => {
                for (unstr_index, src_index) in src_l.strided_index().enumerate() {
                    let mut dst_index = unstr_index;
                    // Set the reduce_dims indexes to 0.
                    for &(dim, stride) in self.reduce_dims_and_stride.iter() {
                        // The compiler is able to optimize the following in a single divmod op.
                        let (pre, post) = (dst_index / stride, dst_index % stride);
                        dst_index = (pre / dim) * stride + post;
                    }
                    dst[dst_index] += src[src_index];
                }
            }
        }
        Ok(dst)
    }
}

impl<'a> Map1 for ReduceSum<'a> {
    #[inline(always)]
    fn f<T: WithDType>(&self, src: &[T], src_l: &Layout) -> Result<Vec<T>> {
        self.fold_impl(src, src_l, T::zero())
    }
}

pub fn unary_map<T: Copy, U: Copy, F: FnMut(T) -> U>(
    vs: &[T],
    layout: &Layout,
    mut f: F,
) -> Vec<U> {
    match layout.strided_blocks() {
        crate::StridedBlocks::SingleBlock { start_offset, len } => vs
            [start_offset..start_offset + len]
            .iter()
            .map(|&v| f(v))
            .collect(),
        crate::StridedBlocks::MultipleBlocks {
            block_start_index,
            block_len,
        } => {
            let mut result = Vec::with_capacity(layout.shape().elem_count());
            // Specialize the case where block_len is one to avoid the second loop.
            if block_len == 1 {
                for index in block_start_index {
                    let v = unsafe { vs.get_unchecked(index) };
                    result.push(f(*v))
                }
            } else {
                for index in block_start_index {
                    for offset in 0..block_len {
                        let v = unsafe { vs.get_unchecked(index + offset) };
                        result.push(f(*v))
                    }
                }
            }
            result
        }
    }
}

pub fn unary_map_vec<T: Copy, U: Copy, F: FnMut(T) -> U, FV: FnMut(&[T], &mut [U])>(
    vs: &[T],
    layout: &Layout,
    mut f: F,
    mut f_vec: FV,
) -> Vec<U> {
    match layout.strided_blocks() {
        crate::StridedBlocks::SingleBlock { start_offset, len } => {
            let mut ys: Vec<U> = Vec::with_capacity(len);
            let ys_to_set = ys.spare_capacity_mut();
            let ys_to_set = unsafe { std::mem::transmute::<_, &mut [U]>(ys_to_set) };
            f_vec(&vs[start_offset..start_offset + len], ys_to_set);
            // SAFETY: values are all set by f_vec.
            unsafe { ys.set_len(len) };
            ys
        }
        crate::StridedBlocks::MultipleBlocks {
            block_start_index,
            block_len,
        } => {
            let el_count = layout.shape().elem_count();
            // Specialize the case where block_len is one to avoid the second loop.
            if block_len == 1 {
                let mut result = Vec::with_capacity(el_count);
                for index in block_start_index {
                    let v = unsafe { vs.get_unchecked(index) };
                    result.push(f(*v))
                }
                result
            } else {
                let mut ys: Vec<U> = Vec::with_capacity(el_count);
                let ys_to_set = ys.spare_capacity_mut();
                let ys_to_set = unsafe { std::mem::transmute::<_, &mut [U]>(ys_to_set) };
                let mut dst_index = 0;
                for src_index in block_start_index {
                    let vs = &vs[src_index..src_index + block_len];
                    let ys = &mut ys_to_set[dst_index..dst_index + block_len];
                    f_vec(vs, ys);
                    dst_index += block_len;
                }
                // SAFETY: values are all set by f_vec.
                unsafe { ys.set_len(el_count) };
                ys
            }
        }
    }
}

// This function maps over two strided index sequences.
pub fn binary_map<T: Copy, U: Copy, F: FnMut(T, T) -> U>(
    lhs_l: &Layout,
    rhs_l: &Layout,
    lhs: &[T],
    rhs: &[T],
    mut f: F,
) -> Vec<U> {
    match (lhs_l.contiguous_offsets(), rhs_l.contiguous_offsets()) {
        (Some((o_l1, o_l2)), Some((o_r1, o_r2))) => lhs[o_l1..o_l2]
            .iter()
            .zip(rhs[o_r1..o_r2].iter())
            .map(|(&l, &r)| f(l, r))
            .collect(),
        (Some((o_l1, o_l2)), None) => {
            // TODO: Maybe we want to avoid going through the layout twice.
            match rhs_l.offsets_b() {
                Some(ob) => {
                    let mut i_in_block = 0;
                    let mut i_right_broadcast = 0;
                    lhs[o_l1..o_l2]
                        .iter()
                        .map(|&l| {
                            let r = unsafe { rhs.get_unchecked(i_in_block + ob.start) };
                            i_right_broadcast += 1;
                            if i_right_broadcast >= ob.right_broadcast {
                                i_in_block += 1;
                                i_right_broadcast = 0;
                            }
                            if i_in_block >= ob.len {
                                i_in_block = 0
                            }
                            f(l, *r)
                        })
                        .collect()
                }
                None => lhs_l
                    .strided_index()
                    .zip(rhs_l.strided_index())
                    .map(|(lhs_i, rhs_i)| f(lhs[lhs_i], rhs[rhs_i]))
                    .collect(),
            }
        }
        (None, Some((o_r1, o_r2))) => {
            // TODO: Maybe we want to avoid going through the layout twice.
            match lhs_l.offsets_b() {
                Some(ob) => {
                    let mut i_in_block = 0;
                    let mut i_right_broadcast = 0;
                    rhs[o_r1..o_r2]
                        .iter()
                        .map(|&r| {
                            let l = unsafe { lhs.get_unchecked(i_in_block + ob.start) };
                            i_right_broadcast += 1;
                            if i_right_broadcast >= ob.right_broadcast {
                                i_in_block += 1;
                                i_right_broadcast = 0;
                            }
                            if i_in_block >= ob.len {
                                i_in_block = 0
                            }
                            f(*l, r)
                        })
                        .collect()
                }
                None => lhs_l
                    .strided_index()
                    .zip(rhs_l.strided_index())
                    .map(|(lhs_i, rhs_i)| f(lhs[lhs_i], rhs[rhs_i]))
                    .collect(),
            }
        }
        _ => lhs_l
            .strided_index()
            .zip(rhs_l.strided_index())
            .map(|(lhs_i, rhs_i)| f(lhs[lhs_i], rhs[rhs_i]))
            .collect(),
    }
}

// Similar to binary_map but with vectorized variants.
pub fn binary_map_vec<T: Copy, F: FnMut(T, T) -> T, FV: FnMut(&[T], &[T], &mut [T])>(
    lhs_l: &Layout,
    rhs_l: &Layout,
    lhs: &[T],
    rhs: &[T],
    mut f: F,
    mut f_vec: FV,
) -> Vec<T> {
    let el_count = lhs_l.shape().elem_count();
    match (lhs_l.contiguous_offsets(), rhs_l.contiguous_offsets()) {
        (Some((o_l1, o_l2)), Some((o_r1, o_r2))) => {
            let mut ys: Vec<T> = Vec::with_capacity(el_count);
            let ys_to_set = ys.spare_capacity_mut();
            let ys_to_set = unsafe { std::mem::transmute::<_, &mut [T]>(ys_to_set) };
            f_vec(&lhs[o_l1..o_l2], &rhs[o_r1..o_r2], ys_to_set);
            // SAFETY: values are all set by f_vec.
            unsafe { ys.set_len(el_count) };
            ys
        }
        (Some((o_l1, o_l2)), None) => match rhs_l.offsets_b() {
            Some(ob) if ob.right_broadcast == 1 => {
                let rhs = &rhs[ob.start..ob.start + ob.len];
                let mut ys: Vec<T> = Vec::with_capacity(el_count);
                let ys_to_set = ys.spare_capacity_mut();
                let ys_to_set = unsafe { std::mem::transmute::<_, &mut [T]>(ys_to_set) };
                let mut dst_i = 0;
                for src_i in (o_l1..o_l2).step_by(ob.len) {
                    f_vec(
                        &lhs[src_i..src_i + ob.len],
                        rhs,
                        &mut ys_to_set[dst_i..dst_i + ob.len],
                    );
                    dst_i += ob.len;
                }
                // SAFETY: values are all set by f_vec.
                unsafe { ys.set_len(el_count) };
                ys
            }
            Some(ob) => {
                let rhs = &rhs[ob.start..ob.start + ob.len];
                let mut ys = lhs[o_l1..o_l2].to_vec();
                for idx_l in 0..ob.left_broadcast {
                    let start = idx_l * ob.len * ob.right_broadcast;
                    for (i, &r) in rhs.iter().enumerate() {
                        let start = start + i * ob.right_broadcast;
                        for v in ys[start..start + ob.right_broadcast].iter_mut() {
                            *v = f(*v, r)
                        }
                    }
                }
                ys
            }
            None => lhs_l
                .strided_index()
                .zip(rhs_l.strided_index())
                .map(|(lhs_i, rhs_i)| f(lhs[lhs_i], rhs[rhs_i]))
                .collect(),
        },
        (None, Some((o_r1, o_r2))) => match lhs_l.offsets_b() {
            Some(ob) if ob.right_broadcast == 1 => {
                let lhs = &lhs[ob.start..ob.start + ob.len];
                let mut ys: Vec<T> = Vec::with_capacity(el_count);
                let ys_to_set = ys.spare_capacity_mut();
                let ys_to_set = unsafe { std::mem::transmute::<_, &mut [T]>(ys_to_set) };
                let mut dst_i = 0;
                for src_i in (o_r1..o_r2).step_by(ob.len) {
                    f_vec(
                        lhs,
                        &rhs[src_i..src_i + ob.len],
                        &mut ys_to_set[dst_i..dst_i + ob.len],
                    );
                    dst_i += ob.len;
                }
                // SAFETY: values are all set by f_vec.
                unsafe { ys.set_len(el_count) };
                ys
            }
            Some(ob) => {
                let lhs = &lhs[ob.start..ob.start + ob.len];
                let mut ys = rhs[o_r1..o_r2].to_vec();
                for idx_l in 0..ob.left_broadcast {
                    let start = idx_l * ob.len * ob.right_broadcast;
                    for (i, &l) in lhs.iter().enumerate() {
                        let start = start + i * ob.right_broadcast;
                        for v in ys[start..start + ob.right_broadcast].iter_mut() {
                            *v = f(l, *v)
                        }
                    }
                }
                ys
            }
            None => lhs_l
                .strided_index()
                .zip(rhs_l.strided_index())
                .map(|(lhs_i, rhs_i)| f(lhs[lhs_i], rhs[rhs_i]))
                .collect(),
        },
        _ => lhs_l
            .strided_index()
            .zip(rhs_l.strided_index())
            .map(|(lhs_i, rhs_i)| f(lhs[lhs_i], rhs[rhs_i]))
            .collect(),
    }
}

struct Affine(f64, f64);

impl Map1 for Affine {
    fn f<T: WithDType>(&self, vs: &[T], layout: &Layout) -> Result<Vec<T>> {
        let mul = T::from_f64(self.0);
        let add = T::from_f64(self.1);
        Ok(unary_map(vs, layout, |v| v * mul + add))
    }
}

struct AvgPool2D((usize, usize), (usize, usize));

impl Map1 for AvgPool2D {
    fn f<T: WithDType>(&self, src: &[T], layout: &Layout) -> Result<Vec<T>> {
        // https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html
        let (k_h, k_w) = self.0;
        let (s_h, s_w) = self.1;
        let (b_sz, c, h, w) = layout.shape().dims4()?;
        let stride = layout.stride();
        let (stride_h, stride_w) = (stride[2], stride[3]);
        let h_out = (h - k_h) / s_h + 1;
        let w_out = (w - k_w) / s_w + 1;
        let src_index = layout.start_offset();
        let mut dst = vec![T::zero(); b_sz * c * h_out * w_out];
        let scale = 1f64 / (k_h * k_w) as f64;
        let scale = T::from_f64(scale);
        for b_idx in 0..b_sz {
            let dst = &mut dst[b_idx * c * h_out * w_out..];
            let src_index = src_index + b_idx * stride[0];
            for c_idx in 0..c {
                let dst = &mut dst[c_idx * h_out * w_out..];
                let src_index = src_index + c_idx * stride[1];
                for h_idx in 0..h_out {
                    for w_idx in 0..w_out {
                        let mut sum = T::zero();
                        for m in 0..k_h {
                            for n in 0..k_w {
                                let m = s_h * h_idx + m;
                                let n = s_w * w_idx + n;
                                sum += src[src_index + m * stride_h + n * stride_w]
                            }
                        }
                        dst[h_idx * w_out + w_idx] = sum * scale;
                    }
                }
            }
        }
        Ok(dst)
    }
}

struct MaxPool2D((usize, usize), (usize, usize));

impl Map1 for MaxPool2D {
    fn f<T: WithDType>(&self, src: &[T], layout: &Layout) -> Result<Vec<T>> {
        // https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
        let (k_h, k_w) = self.0;
        let (s_h, s_w) = self.1;
        let (b_sz, c, h, w) = layout.shape().dims4()?;
        let stride = layout.stride();
        let (stride_h, stride_w) = (stride[2], stride[3]);
        let h_out = (h - k_h) / s_h + 1;
        let w_out = (w - k_w) / s_w + 1;
        let src_index = layout.start_offset();
        let mut dst = vec![T::zero(); b_sz * c * h_out * w_out];
        for b_idx in 0..b_sz {
            let dst = &mut dst[b_idx * c * h_out * w_out..];
            let src_index = src_index + b_idx * stride[0];
            for c_idx in 0..c {
                let dst = &mut dst[c_idx * h_out * w_out..];
                let src_index = src_index + c_idx * stride[1];
                for h_idx in 0..h_out {
                    for w_idx in 0..w_out {
                        let mut largest =
                            src[src_index + s_h * h_idx * stride_h + s_w * w_idx * stride_w];
                        for m in 0..k_h {
                            for n in 0..k_w {
                                let m = s_h * h_idx + m;
                                let n = s_w * w_idx + n;
                                if largest < src[src_index + m * stride_h + n * stride_w] {
                                    largest = src[src_index + m * stride_h + n * stride_w]
                                }
                            }
                        }
                        dst[h_idx * w_out + w_idx] = largest;
                    }
                }
            }
        }
        Ok(dst)
    }
}

struct UpsampleNearest1D(usize);

impl Map1 for UpsampleNearest1D {
    fn f<T: WithDType>(&self, src: &[T], layout: &Layout) -> Result<Vec<T>> {
        // TODO: Specialized implementation for the case 2*sz?
        let dst_sz = self.0;
        let (b_sz, c, src_sz) = layout.shape().dims3()?;
        let stride = layout.stride();
        let stride_sz = stride[2];
        let src_index = layout.start_offset();
        let scale_sz = src_sz as f64 / dst_sz as f64;
        let mut dst = vec![T::zero(); b_sz * c * dst_sz];
        let src_idxs = (0..dst_sz)
            .map(|idx| usize::min(src_sz - 1, (idx as f64 * scale_sz) as usize))
            .collect::<Vec<_>>();
        for b_idx in 0..b_sz {
            let dst = &mut dst[b_idx * c * dst_sz..];
            let src_index = src_index + b_idx * stride[0];
            for c_idx in 0..c {
                let dst = &mut dst[c_idx * dst_sz..];
                let src_index = src_index + c_idx * stride[1];
                for (idx, src_idx) in src_idxs.iter().enumerate() {
                    dst[idx] = src[src_index + src_idx * stride_sz]
                }
            }
        }
        Ok(dst)
    }
}

struct UpsampleNearest2D(usize, usize);

impl Map1 for UpsampleNearest2D {
    fn f<T: WithDType>(&self, src: &[T], layout: &Layout) -> Result<Vec<T>> {
        // TODO: Specialized implementation for the case 2*h, 2*w?
        let (dst_h, dst_w) = (self.0, self.1);
        let (b_sz, c, src_h, src_w) = layout.shape().dims4()?;
        let stride = layout.stride();
        let (stride_h, stride_w) = (stride[2], stride[3]);
        let src_index = layout.start_offset();
        let scale_h = src_h as f64 / dst_h as f64;
        let scale_w = src_w as f64 / dst_w as f64;
        let mut dst = vec![T::zero(); b_sz * c * dst_h * dst_w];
        let src_h_idxs = (0..dst_h)
            .map(|h_idx| usize::min(src_h - 1, (h_idx as f64 * scale_h) as usize))
            .collect::<Vec<_>>();
        let src_w_idxs = (0..dst_w)
            .map(|w_idx| usize::min(src_w - 1, (w_idx as f64 * scale_w) as usize))
            .collect::<Vec<_>>();
        for b_idx in 0..b_sz {
            let dst = &mut dst[b_idx * c * dst_h * dst_w..];
            let src_index = src_index + b_idx * stride[0];
            for c_idx in 0..c {
                let dst = &mut dst[c_idx * dst_h * dst_w..];
                let src_index = src_index + c_idx * stride[1];
                for (h_idx, src_h_idx) in src_h_idxs.iter().enumerate() {
                    for (w_idx, src_w_idx) in src_w_idxs.iter().enumerate() {
                        let src_index = src_index + src_h_idx * stride_h + src_w_idx * stride_w;
                        dst[h_idx * dst_w + w_idx] = src[src_index]
                    }
                }
            }
        }
        Ok(dst)
    }
}

struct Gather<'a, I: IntDType> {
    ids: &'a [I],
    ids_l: &'a Layout,
    dim: usize,
}

impl<'a, I: IntDType> Map1 for Gather<'a, I> {
    fn f<T: WithDType>(&self, src: &[T], src_l: &Layout) -> Result<Vec<T>> {
        let ids = match self.ids_l.contiguous_offsets() {
            Some((a, b)) => &self.ids[a..b],
            None => Err(Error::RequiresContiguous { op: "gather" }.bt())?,
        };
        let src = match src_l.contiguous_offsets() {
            Some((a, b)) => &src[a..b],
            None => Err(Error::RequiresContiguous { op: "gather" }.bt())?,
        };
        let dim = self.dim;
        let ids_dims = self.ids_l.dims();
        let src_dims = src_l.dims();
        let dst_len: usize = ids_dims.iter().product();
        let dst_left_len: usize = ids_dims[..dim].iter().product();
        let dst_dim_len = ids_dims[dim];
        let dst_right_len: usize = ids_dims[dim + 1..].iter().product();

        let src_dim_len = src_dims[dim];
        let src_right_len: usize = src_dims[dim + 1..].iter().product();

        let mut dst = vec![T::zero(); dst_len];
        for left_i in 0..dst_left_len {
            let start_src_idx = left_i * src_right_len * src_dim_len;
            let start_dst_idx = left_i * dst_right_len * dst_dim_len;
            for i in 0..dst_dim_len {
                let start_dst_idx = start_dst_idx + i * dst_right_len;
                for right_i in 0..dst_right_len {
                    let dst_idx = start_dst_idx + right_i;
                    let index = ids[dst_idx].as_usize();
                    if index >= src_dim_len {
                        Err(Error::InvalidIndex {
                            index,
                            size: src_dim_len,
                            op: "gather",
                        }
                        .bt())?
                    }
                    let src_idx = start_src_idx + index * src_right_len + right_i;
                    dst[dst_idx] = src[src_idx]
                }
            }
        }
        Ok(dst)
    }
}

struct IndexSelect<'a, T: IntDType> {
    ids: &'a [T],
    ids_l: &'a Layout,
    dim: usize,
}

impl<'a, I: IntDType> Map1 for IndexSelect<'a, I> {
    fn f<T: WithDType>(&self, src: &[T], layout: &Layout) -> Result<Vec<T>> {
        let src = match layout.contiguous_offsets() {
            Some((a, b)) => &src[a..b],
            None => Err(Error::RequiresContiguous { op: "index-select" }.bt())?,
        };
        let dim = self.dim;
        let n_ids = match self.ids_l.dims() {
            [n_ids] => *n_ids,
            d => Err(Error::UnexpectedNumberOfDims {
                expected: 1,
                got: d.len(),
                shape: self.ids_l.shape().clone(),
            }
            .bt())?,
        };
        let stride_ids = self.ids_l.stride()[0];
        let mut dst_dims = layout.dims().to_vec();
        let src_dim = dst_dims[dim];
        dst_dims[dim] = n_ids;
        let dst_len: usize = dst_dims.iter().product();
        let left_len: usize = dst_dims[..dim].iter().product();
        let right_len: usize = dst_dims[dim + 1..].iter().product();
        let mut dst = vec![T::zero(); dst_len];
        for left_i in 0..left_len {
            let start_src_idx = left_i * right_len * src_dim;
            let start_dst_idx = left_i * right_len * n_ids;
            for i in 0..n_ids {
                let index = self.ids[self.ids_l.start_offset() + stride_ids * i].as_usize();
                if index >= src_dim {
                    Err(Error::InvalidIndex {
                        index,
                        size: src_dim,
                        op: "index-select",
                    }
                    .bt())?
                }
                let start_src_idx = start_src_idx + index * right_len;
                let start_dst_idx = start_dst_idx + i * right_len;
                dst[start_dst_idx..start_dst_idx + right_len]
                    .copy_from_slice(&src[start_src_idx..start_src_idx + right_len])
            }
        }
        Ok(dst)
    }
}

struct ScatterAdd<'a, I: IntDType> {
    ids: &'a [I],
    ids_l: &'a Layout,
    dim: usize,
}

impl<'a, I: IntDType> Map2 for ScatterAdd<'a, I> {
    const OP: &'static str = "scatter-add";
    fn f<T: WithDType>(&self, v1: &[T], l1: &Layout, src: &[T], src_l: &Layout) -> Result<Vec<T>> {
        let dst_len = l1.shape().elem_count();
        let mut dst = vec![T::zero(); dst_len];
        copy_strided_src_(v1, &mut dst, 0, l1);
        let src = match src_l.contiguous_offsets() {
            None => Err(Error::RequiresContiguous { op: "scatter-add" }.bt())?,
            Some((o1, o2)) => &src[o1..o2],
        };

        let dim = self.dim;
        let ids_dims = self.ids_l.dims();
        let dst_dims = l1.dims();
        let dst_dim_len = dst_dims[dim];
        let dst_right_len: usize = dst_dims[dim + 1..].iter().product();

        let ids_left_len: usize = ids_dims[..dim].iter().product();
        let ids_dim_len = ids_dims[dim];
        let ids_right_len: usize = ids_dims[dim + 1..].iter().product();

        let ids = match self.ids_l.contiguous_offsets() {
            Some((a, b)) => &self.ids[a..b],
            None => Err(Error::RequiresContiguous { op: "gather" }.bt())?,
        };
        for left_i in 0..ids_left_len {
            let start_ids_idx = left_i * ids_right_len * ids_dim_len;
            let start_dst_idx = left_i * dst_right_len * dst_dim_len;
            for i in 0..ids_dim_len {
                let start_ids_idx = start_ids_idx + i * ids_right_len;
                for right_i in 0..dst_right_len {
                    let ids_idx = start_ids_idx + right_i;
                    let index = ids[ids_idx].as_usize();
                    if index >= dst_dim_len {
                        Err(Error::InvalidIndex {
                            index,
                            size: dst_dim_len,
                            op: "gather",
                        }
                        .bt())?
                    }
                    let dst_idx = start_dst_idx + index * dst_right_len + right_i;
                    dst[dst_idx] += src[ids_idx]
                }
            }
        }

        Ok(dst)
    }
}

struct IndexAdd<'a, I: IntDType> {
    ids: &'a [I],
    dim: usize,
}

impl<'a, I: IntDType> Map2 for IndexAdd<'a, I> {
    const OP: &'static str = "index-add";
    // https://pytorch.org/docs/stable/generated/torch.Tensor.index_add_.html#torch.Tensor.index_add_
    // v1, l1 -> self
    fn f<T: WithDType>(&self, v1: &[T], l1: &Layout, src: &[T], src_l: &Layout) -> Result<Vec<T>> {
        let dst_len = l1.shape().elem_count();
        let mut dst = vec![T::zero(); dst_len];
        copy_strided_src_(v1, &mut dst, 0, l1);
        let src = match src_l.contiguous_offsets() {
            None => Err(Error::RequiresContiguous { op: "index-add" }.bt())?,
            Some((o1, o2)) => &src[o1..o2],
        };
        let dim = self.dim;
        let max_idx = l1.dims()[dim];
        let pre_dim = src_l.dims()[..dim].iter().product::<usize>();
        let src_dim_sz = src_l.dims()[dim];
        let post_dim = src_l.dims()[dim + 1..].iter().product::<usize>();
        if dim == 0 {
            for (src_idx, dst_idx) in self.ids.iter().enumerate() {
                let dst_idx = dst_idx.as_usize();
                if dst_idx >= max_idx {
                    Err(Error::InvalidIndex {
                        index: dst_idx,
                        op: "index-add",
                        size: max_idx,
                    })?
                }
                let src_idx = src_idx * post_dim;
                let dst_idx = dst_idx * post_dim;
                let src = &src[src_idx..src_idx + post_dim];
                let dst = &mut dst[dst_idx..dst_idx + post_dim];
                for (d, &s) in dst.iter_mut().zip(src.iter()) {
                    *d += s
                }
            }
        } else {
            for (src_idx, dst_idx) in self.ids.iter().enumerate() {
                let dst_idx = dst_idx.as_usize();
                if dst_idx >= max_idx {
                    Err(Error::InvalidIndex {
                        index: dst_idx,
                        op: "index-add",
                        size: max_idx,
                    })?
                }
                for pre_i in 0..pre_dim {
                    let pre_src_i = (pre_i * src_dim_sz + src_idx) * post_dim;
                    let pre_dst_i = (pre_i * max_idx + dst_idx) * post_dim;
                    let src = &src[pre_src_i..pre_src_i + post_dim];
                    let dst = &mut dst[pre_dst_i..pre_dst_i + post_dim];
                    for (d, &s) in dst.iter_mut().zip(src.iter()) {
                        *d += s
                    }
                }
            }
        }
        Ok(dst)
    }
}

fn copy_strided_src_<T: Copy>(src: &[T], dst: &mut [T], dst_offset: usize, src_l: &Layout) {
    match src_l.strided_blocks() {
        crate::StridedBlocks::SingleBlock { start_offset, len } => {
            let to_copy = (dst.len() - dst_offset).min(len);
            dst[dst_offset..dst_offset + to_copy]
                .copy_from_slice(&src[start_offset..start_offset + to_copy])
        }
        crate::StridedBlocks::MultipleBlocks {
            block_start_index,
            block_len: 1,
        } => {
            for (dst_index, src_index) in block_start_index.enumerate() {
                let dst_index = dst_index + dst_offset;
                if dst_index >= dst.len() {
                    break;
                }
                dst[dst_index] = src[src_index]
            }
        }
        crate::StridedBlocks::MultipleBlocks {
            block_start_index,
            block_len,
        } => {
            let mut dst_index = dst_offset;
            for src_index in block_start_index {
                let next_dst_index = dst_index + block_len;
                if dst_index >= dst.len() {
                    break;
                }
                let to_copy = usize::min(block_len, dst.len() - dst_index);
                dst[dst_index..dst_index + to_copy]
                    .copy_from_slice(&src[src_index..src_index + to_copy]);
                dst_index = next_dst_index
            }
        }
    }
}

struct Conv1D<'a>(&'a crate::conv::ParamsConv1D);

impl<'a> Map2 for Conv1D<'a> {
    const OP: &'static str = "conv1d";
    fn f<T: WithDType>(&self, inp: &[T], inp_l: &Layout, k: &[T], k_l: &Layout) -> Result<Vec<T>> {
        let p = self.0;
        let inp = &inp[inp_l.start_offset()..];
        let k = &k[k_l.start_offset()..];
        let (inp_s0, inp_s1, inp_s2) = crate::shape::dims3(inp_l.stride())?;
        let (k_s0, k_s1, k_s2) = crate::shape::dims3(k_l.stride())?;
        let l_out = p.l_out();
        let dst_elems = p.c_out * l_out * p.b_size;
        // The output shape is [b_size, c_out, l_out]
        let dst = vec![T::zero(); dst_elems];

        // TODO: Avoid making this copy if `inp` already has the appropriate layout.
        let mut inp_cont = vec![T::zero(); p.b_size * p.c_in * p.l_in];
        for b_idx in 0..p.b_size {
            for src_l in 0..p.l_in {
                for src_c_idx in 0..p.c_in {
                    let inp_idx = b_idx * inp_s0 + src_c_idx * inp_s1 + src_l * inp_s2;
                    inp_cont[b_idx * p.l_in * p.c_in + src_l * p.c_in + src_c_idx] = inp[inp_idx]
                }
            }
        }

        for offset in 0..p.k_size {
            (0..p.c_out).into_par_iter().for_each(|dst_c_idx| {
                let dst_idx = dst_c_idx * l_out;
                let k_cont = (0..p.c_in)
                    .map(|c_in_idx| k[dst_c_idx * k_s0 + c_in_idx * k_s1 + offset * k_s2])
                    .collect::<Vec<_>>();
                for b_idx in 0..p.b_size {
                    let dst_idx = dst_idx + b_idx * p.c_out * l_out;
                    for dst_l in 0..l_out {
                        let dst_idx = dst_idx + dst_l;
                        let src_l = p.stride * dst_l + offset * p.dilation;
                        if src_l < p.padding || src_l >= p.padding + p.l_in {
                            continue;
                        }
                        let src_l = src_l - p.padding;
                        let inp_cont = &inp_cont[b_idx * p.l_in * p.c_in + src_l * p.c_in..];
                        assert!(inp_cont.len() >= p.c_in);
                        assert!(k_cont.len() >= p.c_in);
                        let mut d = T::zero();
                        unsafe { T::vec_dot(inp_cont.as_ptr(), k_cont.as_ptr(), &mut d, p.c_in) }
                        let dst_p = dst.as_ptr();
                        // Safety: dst_idx are uniques per dst_c_idx which is used to parallelise
                        // the different tasks so no two threads can try to write at the same
                        // location.
                        unsafe {
                            let ptr = dst_p.add(dst_idx) as *mut T;
                            *ptr += d
                        }
                    }
                }
            })
        }
        Ok(dst)
    }
}

struct Im2Col1D {
    l_k: usize,
    stride: usize,
    dilation: usize,
    padding: usize,
}

impl Im2Col1D {
    fn l_out(&self, l: usize) -> usize {
        (l + 2 * self.padding - self.dilation * (self.l_k - 1) - 1) / self.stride + 1
    }
}

impl Map1 for Im2Col1D {
    fn f<T: WithDType>(&self, vs: &[T], layout: &Layout) -> Result<Vec<T>> {
        let &Self {
            l_k,
            stride,
            dilation,
            padding,
        } = self;
        let (b, c, l) = layout.shape().dims3()?;
        let l_out = self.l_out(l);
        let src = &vs[layout.start_offset()..];
        let mut dst = vec![T::zero(); b * l_out * c * l_k];
        let (src_s0, src_s1, src_s2) = {
            let s = layout.stride();
            (s[0], s[1], s[2])
        };
        // TODO: provide specialized kernels for the common use cases.
        // - l_k = 1
        // - padding = 0
        // - stride = 1
        // - dilation = 1
        for b_idx in 0..b {
            let src_idx = b_idx * src_s0;
            let dst_idx = b_idx * l_out * c * l_k;
            for l_idx in 0..l_out {
                let dst_idx = dst_idx + l_idx * c * l_k;
                for c_idx in 0..c {
                    let dst_idx = dst_idx + c_idx * l_k;
                    let src_idx = c_idx * src_s1 + src_idx;
                    for l_k_idx in 0..l_k {
                        let src_l = l_idx * stride + l_k_idx * dilation;
                        if padding != 0 && (src_l < padding || src_l >= l + padding) {
                            continue;
                        }
                        let src_l = src_l - padding;
                        let src_idx = src_idx + src_l * src_s2;
                        let dst_idx = dst_idx + l_k_idx;
                        dst[dst_idx] = src[src_idx]
                    }
                }
            }
        }
        Ok(dst)
    }
}

struct Im2Col {
    h_k: usize,
    w_k: usize,
    stride: usize,
    dilation: usize,
    padding: usize,
}

impl Im2Col {
    fn hw_out(&self, h: usize, w: usize) -> (usize, usize) {
        let h_out = (h + 2 * self.padding - self.dilation * (self.h_k - 1) - 1) / self.stride + 1;
        let w_out = (w + 2 * self.padding - self.dilation * (self.w_k - 1) - 1) / self.stride + 1;
        (h_out, w_out)
    }
}

impl Map1 for Im2Col {
    fn f<T: WithDType>(&self, vs: &[T], layout: &Layout) -> Result<Vec<T>> {
        let &Self {
            h_k,
            w_k,
            stride,
            dilation,
            padding,
        } = self;
        let (b, c, h, w) = layout.shape().dims4()?;
        let (h_out, w_out) = self.hw_out(h, w);
        let src = &vs[layout.start_offset()..];
        let mut dst = vec![T::zero(); b * h_out * w_out * c * h_k * w_k];
        let (src_s0, src_s1, src_s2, src_s3) = {
            let s = layout.stride();
            (s[0], s[1], s[2], s[3])
        };
        // TODO: provide specialized kernels for the common use cases.
        // - h_k = w_k = 1
        // - padding = 0
        // - stride = 1
        // - dilation = 1
        for b_idx in 0..b {
            let src_idx = b_idx * src_s0;
            let dst_idx = b_idx * h_out * w_out * c * h_k * w_k;
            for h_idx in 0..h_out {
                let dst_idx = dst_idx + h_idx * w_out * c * h_k * w_k;
                for w_idx in 0..w_out {
                    let dst_idx = dst_idx + w_idx * c * h_k * w_k;
                    for c_idx in 0..c {
                        let dst_idx = dst_idx + c_idx * h_k * w_k;
                        let src_idx = c_idx * src_s1 + src_idx;
                        for h_k_idx in 0..h_k {
                            let src_h = h_idx * stride + h_k_idx * dilation;
                            if padding != 0 && (src_h < padding || src_h >= h + padding) {
                                continue;
                            }
                            let src_h = src_h - padding;
                            let src_idx = src_idx + src_h * src_s2;
                            let dst_idx = dst_idx + h_k_idx * w_k;
                            for w_k_idx in 0..w_k {
                                let src_w = w_idx * stride + w_k_idx * dilation;
                                if padding != 0 && (src_w < padding || src_w >= w + padding) {
                                    continue;
                                }
                                let src_w = src_w - padding;
                                let src_idx = src_idx + src_w * src_s3;
                                let dst_idx = dst_idx + w_k_idx;
                                dst[dst_idx] = src[src_idx]
                            }
                        }
                    }
                }
            }
        }
        Ok(dst)
    }
}

struct ConvTranspose1D<'a>(&'a crate::conv::ParamsConvTranspose1D);

impl<'a> Map2 for ConvTranspose1D<'a> {
    const OP: &'static str = "conv_transpose1d";
    fn f<T: WithDType>(&self, inp: &[T], inp_l: &Layout, k: &[T], k_l: &Layout) -> Result<Vec<T>> {
        let p = self.0;
        let inp = &inp[inp_l.start_offset()..];
        let k = &k[k_l.start_offset()..];
        let (inp_s0, inp_s1, inp_s2) = crate::shape::dims3(inp_l.stride())?;
        let (k_s0, k_s1, k_s2) = crate::shape::dims3(k_l.stride())?;
        let l_out = p.l_out();

        // Output shape: [b_size, c_out, l_out].
        let dst_elems = p.c_out * l_out * p.b_size;
        let dst = vec![T::zero(); dst_elems];
        let dst_s0 = p.c_out * l_out;
        let dst_s1 = l_out;
        let dst_s2 = 1;

        // TODO: Avoid making this copy if `inp` already has the appropriate layout.
        let mut inp_cont = vec![T::zero(); p.b_size * p.c_in * p.l_in];
        let cont_s0 = p.l_in * p.c_in;
        let cont_s1 = p.c_in;
        for b_idx in 0..p.b_size {
            for l_idx in 0..p.l_in {
                for c_idx in 0..p.c_in {
                    let src_idx = b_idx * inp_s0 + c_idx * inp_s1 + l_idx * inp_s2;
                    let dst_idx = b_idx * cont_s0 + l_idx * cont_s1 + c_idx;
                    inp_cont[dst_idx] = inp[src_idx]
                }
            }
        }

        for k_idx in 0..p.k_size {
            (0..p.c_out).into_par_iter().for_each(|dst_c_idx| {
                let k_cont = (0..p.c_in)
                    .map(|c_in_idx| k[c_in_idx * k_s0 + dst_c_idx * k_s1 + k_idx * k_s2])
                    .collect::<Vec<_>>();
                for b_idx in 0..p.b_size {
                    for l_idx in 0..p.l_in {
                        let out_idx = l_idx * p.stride + k_idx * p.dilation;
                        if out_idx < p.padding {
                            continue;
                        }
                        let out_idx = out_idx - p.padding;
                        if out_idx < l_out {
                            let inp_cont = &inp_cont[b_idx * cont_s0 + l_idx * cont_s1..];
                            let dst_idx = b_idx * dst_s0 + out_idx * dst_s2 + dst_c_idx * dst_s1;
                            let mut d = T::zero();
                            unsafe {
                                T::vec_dot(inp_cont.as_ptr(), k_cont.as_ptr(), &mut d, p.c_in)
                            }
                            let dst_p = dst.as_ptr();
                            // Safety: dst_idx are uniques per dst_c_idx which is used to
                            // parallelise the different tasks so no two threads can try to
                            // write at the same location.
                            unsafe {
                                let ptr = dst_p.add(dst_idx) as *mut T;
                                *ptr += d
                            }
                        }
                    }
                }
            })
        }
        Ok(dst)
    }
}

struct Conv2D<'a>(&'a crate::conv::ParamsConv2D);

impl<'a> Map2 for Conv2D<'a> {
    const OP: &'static str = "conv2d";
    fn f<T: WithDType>(&self, inp: &[T], inp_l: &Layout, k: &[T], k_l: &Layout) -> Result<Vec<T>> {
        let p = self.0;
        let inp = &inp[inp_l.start_offset()..];
        let (inp_s0, inp_s1, inp_s2, inp_s3) = crate::shape::dims4(inp_l.stride())?;
        let k = &k[k_l.start_offset()..];
        let (k_s0, k_s1, k_s2, k_s3) = crate::shape::dims4(k_l.stride())?;
        let (out_h, out_w) = (p.out_h(), p.out_w());

        // Output shape: [b_size, c_out, out_h, out_w].
        let dst = vec![T::zero(); p.b_size * p.c_out * out_h * out_w];

        // TODO: Avoid making this copy if `inp` already has the appropriate layout.
        let mut inp_cont = vec![T::zero(); p.b_size * p.c_in * p.i_h * p.i_w];
        let cont_s0 = p.i_h * p.i_w * p.c_in;
        let cont_s1 = p.i_w * p.c_in;
        let cont_s2 = p.c_in;
        for b_idx in 0..p.b_size {
            for h_idx in 0..p.i_h {
                for w_idx in 0..p.i_w {
                    for c_idx in 0..p.c_in {
                        let src_idx =
                            b_idx * inp_s0 + c_idx * inp_s1 + h_idx * inp_s2 + w_idx * inp_s3;
                        let dst_idx = b_idx * cont_s0 + h_idx * cont_s1 + w_idx * cont_s2 + c_idx;
                        inp_cont[dst_idx] = inp[src_idx]
                    }
                }
            }
        }

        for offset_h in 0..p.k_h {
            for offset_w in 0..p.k_w {
                (0..p.c_out).into_par_iter().for_each(|dst_c_idx| {
                    let dst_idx = dst_c_idx * out_w * out_h;
                    let k_cont = (0..p.c_in)
                        .map(|c_in_idx| {
                            k[dst_c_idx * k_s0
                                + c_in_idx * k_s1
                                + offset_h * k_s2
                                + offset_w * k_s3]
                        })
                        .collect::<Vec<_>>();
                    for b_idx in 0..p.b_size {
                        let dst_idx = dst_idx + b_idx * p.c_out * out_h * out_w;
                        for dst_h in 0..out_h {
                            let dst_idx = dst_idx + dst_h * out_w;
                            let src_h = p.stride * dst_h + offset_h * p.dilation;
                            if src_h < p.padding || src_h >= p.i_h + p.padding {
                                continue;
                            }
                            let src_h = src_h - p.padding;
                            for dst_w in 0..out_w {
                                let dst_idx = dst_idx + dst_w;
                                let src_w = p.stride * dst_w + offset_w * p.dilation;
                                if src_w < p.padding || src_w >= p.i_w + p.padding {
                                    continue;
                                }
                                let src_w = src_w - p.padding;
                                let inp_cont = &inp_cont
                                    [b_idx * cont_s0 + src_h * cont_s1 + src_w * cont_s2..];
                                assert!(inp_cont.len() >= p.c_in);
                                assert!(k_cont.len() >= p.c_in);
                                let mut d = T::zero();
                                unsafe {
                                    T::vec_dot(inp_cont.as_ptr(), k_cont.as_ptr(), &mut d, p.c_in)
                                }
                                let dst_p = dst.as_ptr();
                                // Safety: dst_idx are uniques per dst_c_idx which is used to parallelise
                                // the different tasks so no two threads can try to write at the same
                                // location.
                                unsafe {
                                    let ptr = dst_p.add(dst_idx) as *mut T;
                                    *ptr += d
                                }
                            }
                        }
                    }
                });
            }
        }

        Ok(dst)
    }
}

struct ConvTranspose2D<'a>(&'a crate::conv::ParamsConvTranspose2D);

impl<'a> Map2 for ConvTranspose2D<'a> {
    const OP: &'static str = "conv_transpose2d";
    fn f<T: WithDType>(&self, inp: &[T], inp_l: &Layout, k: &[T], k_l: &Layout) -> Result<Vec<T>> {
        let p = self.0;
        let inp = &inp[inp_l.start_offset()..];
        let (inp_s0, inp_s1, inp_s2, inp_s3) = crate::shape::dims4(inp_l.stride())?;
        let k = &k[k_l.start_offset()..];
        let (k_s0, k_s1, k_s2, k_s3) = crate::shape::dims4(k_l.stride())?;
        let (out_h, out_w) = (p.out_h(), p.out_w());

        // Output shape: [b_size, c_out, out_h, out_w].
        let dst = vec![T::zero(); p.b_size * p.c_out * out_h * out_w];
        let dst_s0 = p.c_out * out_h * out_w;
        let dst_s1 = out_h * out_w;
        let dst_s2 = out_w;
        let dst_s3 = 1;

        // TODO: Avoid making this copy if `inp` already has the appropriate layout.
        let mut inp_cont = vec![T::zero(); p.b_size * p.c_in * p.i_h * p.i_w];
        let cont_s0 = p.i_h * p.i_w * p.c_in;
        let cont_s1 = p.i_w * p.c_in;
        let cont_s2 = p.c_in;
        for b_idx in 0..p.b_size {
            for h_idx in 0..p.i_h {
                for w_idx in 0..p.i_w {
                    for c_idx in 0..p.c_in {
                        let src_idx =
                            b_idx * inp_s0 + c_idx * inp_s1 + h_idx * inp_s2 + w_idx * inp_s3;
                        let dst_idx = b_idx * cont_s0 + h_idx * cont_s1 + w_idx * cont_s2 + c_idx;
                        inp_cont[dst_idx] = inp[src_idx]
                    }
                }
            }
        }

        for k_y in 0..p.k_h {
            for k_x in 0..p.k_w {
                (0..p.c_out).into_par_iter().for_each(|dst_c_idx| {
                    let k_cont = (0..p.c_in)
                        .map(|c_in_idx| {
                            k[c_in_idx * k_s0 + dst_c_idx * k_s1 + k_y * k_s2 + k_x * k_s3]
                        })
                        .collect::<Vec<_>>();
                    for b_idx in 0..p.b_size {
                        for inp_y in 0..p.i_h {
                            for inp_x in 0..p.i_w {
                                let out_x = inp_x * p.stride + k_x * p.dilation;
                                let out_y = inp_y * p.stride + k_y * p.dilation;
                                if out_x < p.padding || out_y < p.padding {
                                    continue;
                                }
                                let out_x = out_x - p.padding;
                                let out_y = out_y - p.padding;
                                if out_x < out_w && out_y < out_h {
                                    let inp_cont = &inp_cont
                                        [b_idx * cont_s0 + inp_y * cont_s1 + inp_x * cont_s2..];
                                    let dst_idx = b_idx * dst_s0
                                        + out_y * dst_s2
                                        + out_x * dst_s3
                                        + dst_c_idx * dst_s1;
                                    let mut d = T::zero();
                                    unsafe {
                                        T::vec_dot(
                                            inp_cont.as_ptr(),
                                            k_cont.as_ptr(),
                                            &mut d,
                                            p.c_in,
                                        )
                                    }
                                    let dst_p = dst.as_ptr();
                                    // Safety: dst_idx are uniques per dst_c_idx which is used to
                                    // parallelise the different tasks so no two threads can try to
                                    // write at the same location.
                                    unsafe {
                                        let ptr = dst_p.add(dst_idx) as *mut T;
                                        *ptr += d
                                    }
                                }
                            }
                        }
                    }
                })
            }
        }
        Ok(dst)
    }
}

struct MatMul((usize, usize, usize, usize));

impl MatMul {
    fn striding_error(&self, lhs_l: &Layout, rhs_l: &Layout, msg: &'static str) -> Error {
        Error::MatMulUnexpectedStriding(Box::new(crate::error::MatMulUnexpectedStriding {
            lhs_l: lhs_l.clone(),
            rhs_l: rhs_l.clone(),
            bmnk: self.0,
            msg,
        }))
        .bt()
    }
}

impl Map2 for MatMul {
    const OP: &'static str = "mat_mul";

    #[cfg(all(not(feature = "mkl"), not(feature = "accelerate")))]
    fn f<T: 'static + WithDType + num_traits::Num + Copy>(
        &self,
        lhs: &[T],
        lhs_l: &Layout,
        rhs: &[T],
        rhs_l: &Layout,
    ) -> Result<Vec<T>> {
        use gemm::{gemm, Parallelism};

        match T::DTYPE {
            DType::F16 | DType::F32 | DType::F64 => {}
            _ => Err(Error::UnsupportedDTypeForOp(T::DTYPE, "matmul").bt())?,
        }

        let (b, m, n, k) = self.0;
        let lhs = &lhs[lhs_l.start_offset()..];
        let rhs = &rhs[rhs_l.start_offset()..];

        let lhs_stride = lhs_l.stride();
        let rhs_stride = rhs_l.stride();
        let rank = lhs_stride.len();
        let lhs_cs = lhs_stride[rank - 1];
        let lhs_rs = lhs_stride[rank - 2];

        let rhs_cs = rhs_stride[rank - 1];
        let rhs_rs = rhs_stride[rank - 2];

        let a_skip: usize = match lhs_stride[..rank - 2] {
            [s1, stride] if s1 == stride * lhs_l.dims()[1] => stride,
            [stride] => stride,
            [] => m * k,
            _ => Err(self.striding_error(lhs_l, rhs_l, "non-contiguous lhs"))?,
        };
        let b_skip: usize = match rhs_stride[..rank - 2] {
            [s1, stride] if s1 == stride * rhs_l.dims()[1] => stride,
            [stride] => stride,
            [] => n * k,
            _ => Err(self.striding_error(lhs_l, rhs_l, "non-contiguous rhs"))?,
        };
        let c_skip: usize = m * n;

        let dst_shape: Shape = (m, n).into();
        let dst_strides = dst_shape.stride_contiguous();
        let dst_rs = dst_strides[0];
        let dst_cs = dst_strides[1];

        let mut dst = vec![T::zero(); b * m * n];
        let num_threads = crate::utils::get_num_threads();
        let parallelism = if num_threads > 1 {
            Parallelism::Rayon(num_threads)
        } else {
            Parallelism::None
        };
        for step in 0..b {
            let lhs_p = &lhs[step * a_skip..];
            let rhs_p = &rhs[step * b_skip..];
            let dst_p = &mut dst[step * c_skip..];
            unsafe {
                gemm(
                    /* m: usize = */ m,
                    /* n: usize = */ n,
                    /* k: usize = */ k,
                    /* dst: *mut T = */ dst_p.as_mut_ptr(),
                    /* dst_cs: isize = */ dst_cs as isize,
                    /* dst_rs: isize = */ dst_rs as isize,
                    /* read_dst: bool = */ false,
                    /* lhs: *const T = */ lhs_p.as_ptr(),
                    /* lhs_cs: isize = */ lhs_cs as isize,
                    /* lhs_rs: isize = */ lhs_rs as isize,
                    /* rhs: *const T = */ rhs_p.as_ptr(),
                    /* rhs_cs: isize = */ rhs_cs as isize,
                    /* rhs_rs: isize = */ rhs_rs as isize,
                    /* alpha: T = */ T::zero(),
                    /* beta: T = */ T::one(),
                    /* conj_dst: bool = */ false,
                    /* conj_lhs: bool = */ false,
                    /* conj_rhs: bool = */ false,
                    parallelism,
                )
            }
        }
        Ok(dst)
    }

    #[cfg(feature = "accelerate")]
    fn f<T: 'static + WithDType + num_traits::Num + Copy>(
        &self,
        lhs: &[T],
        lhs_l: &Layout,
        rhs: &[T],
        rhs_l: &Layout,
    ) -> Result<Vec<T>> {
        let (b, m, n, k) = self.0;
        let lhs = &lhs[lhs_l.start_offset()..];
        let rhs = &rhs[rhs_l.start_offset()..];

        let lhs_stride = lhs_l.stride();
        let rhs_stride = rhs_l.stride();
        let rank = lhs_stride.len();

        let a_skip: usize = match lhs_stride[..rank - 2] {
            [s1, stride] if s1 == stride * lhs_l.dims()[1] => stride,
            [stride] => stride,
            [] => m * k,
            _ => Err(self.striding_error(lhs_l, rhs_l, "non-contiguous lhs"))?,
        };
        let b_skip: usize = match rhs_stride[..rank - 2] {
            [s1, stride] if s1 == stride * rhs_l.dims()[1] => stride,
            [stride] => stride,
            [] => n * k,
            _ => Err(self.striding_error(lhs_l, rhs_l, "non-contiguous rhs"))?,
        };
        let c_skip: usize = m * n;

        let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
        let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
        let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
        let lhs_m2 = lhs_stride[lhs_stride.len() - 2];

        let (lda, transa) = if rhs_m1 == 1 && rhs_m2 == n {
            (n as i32, b'N')
        } else if rhs_m1 == k && rhs_m2 == 1 {
            (k as i32, b'T')
        } else {
            Err(self.striding_error(lhs_l, rhs_l, "non-contiguous rhs"))?
        };
        // The b tensor has dims batching, m, k (lhs)
        let (ldb, transb) = if lhs_m1 == 1 && lhs_m2 == k {
            (k as i32, b'N')
        } else if lhs_m1 == m && lhs_m2 == 1 {
            (m as i32, b'T')
        } else {
            Err(self.striding_error(lhs_l, rhs_l, "non-contiguous lhs"))?
        };

        let mut dst = vec![T::zero(); b * m * n];
        match T::DTYPE {
            DType::F16 => {
                crate::bail!("the accelerate backend does not support f16 matmul")
            }
            DType::F32 => {
                for step in 0..b {
                    let lhs_p = &lhs[step * a_skip..];
                    let rhs_p = &rhs[step * b_skip..];
                    let dst_p = &mut dst[step * c_skip..];
                    unsafe {
                        let a = rhs_p.as_ptr() as *const f32;
                        let b = lhs_p.as_ptr() as *const f32;
                        let c = dst_p.as_mut_ptr() as *mut f32;
                        let a = std::slice::from_raw_parts(a, a_skip);
                        let b = std::slice::from_raw_parts(b, b_skip);
                        let c = std::slice::from_raw_parts_mut(c, c_skip);
                        crate::accelerate::sgemm(
                            transa, transb, /* m= */ n as i32, /* n= */ m as i32,
                            /* k= */ k as i32, /* alpha= */ 1., /* a= */ a,
                            /* lda= */ lda, /* b= */ b, /* ldb= */ ldb,
                            /* beta= */ 0., /* c= */ c, /* ldc= */ n as i32,
                        )
                    }
                }
            }
            DType::F64 => {
                for step in 0..b {
                    let lhs_p = &lhs[step * a_skip..];
                    let rhs_p = &rhs[step * b_skip..];
                    let dst_p = &mut dst[step * c_skip..];
                    unsafe {
                        let a = rhs_p.as_ptr() as *const f64;
                        let b = lhs_p.as_ptr() as *const f64;
                        let c = dst_p.as_mut_ptr() as *mut f64;
                        let a = std::slice::from_raw_parts(a, a_skip);
                        let b = std::slice::from_raw_parts(b, b_skip);
                        let c = std::slice::from_raw_parts_mut(c, c_skip);
                        crate::accelerate::dgemm(
                            transa, transb, /* m= */ n as i32, /* n= */ m as i32,
                            /* k= */ k as i32, /* alpha= */ 1., /* a= */ a,
                            /* lda= */ lda, /* b= */ b, /* ldb= */ ldb,
                            /* beta= */ 0., /* c= */ c, /* ldc= */ n as i32,
                        )
                    }
                }
            }
            dtype => Err(Error::UnsupportedDTypeForOp(dtype, "matmul").bt())?,
        }
        Ok(dst)
    }

    #[cfg(feature = "mkl")]
    fn f<T: 'static + WithDType + num_traits::Num + Copy>(
        &self,
        lhs: &[T],
        lhs_l: &Layout,
        rhs: &[T],
        rhs_l: &Layout,
    ) -> Result<Vec<T>> {
        let (b, m, n, k) = self.0;
        let lhs = &lhs[lhs_l.start_offset()..];
        let rhs = &rhs[rhs_l.start_offset()..];

        let lhs_stride = lhs_l.stride();
        let rhs_stride = rhs_l.stride();
        let rank = lhs_stride.len();

        let a_skip: usize = match lhs_stride[..rank - 2] {
            [s1, stride] if s1 == stride * lhs_l.dims()[1] => stride,
            [stride] => stride,
            [] => m * k,
            _ => Err(self.striding_error(lhs_l, rhs_l, "non-contiguous lhs"))?,
        };
        let b_skip: usize = match rhs_stride[..rank - 2] {
            [s1, stride] if s1 == stride * rhs_l.dims()[1] => stride,
            [stride] => stride,
            [] => n * k,
            _ => Err(self.striding_error(lhs_l, rhs_l, "non-contiguous rhs"))?,
        };
        let c_skip: usize = m * n;

        let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
        let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
        let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
        let lhs_m2 = lhs_stride[lhs_stride.len() - 2];

        let (lda, transa) = if rhs_m1 == 1 && rhs_m2 == n {
            (n as i32, b'N')
        } else if rhs_m1 == k && rhs_m2 == 1 {
            (k as i32, b'T')
        } else {
            Err(self.striding_error(lhs_l, rhs_l, "non-contiguous rhs"))?
        };
        // The b tensor has dims batching, m, k (lhs)
        let (ldb, transb) = if lhs_m1 == 1 && lhs_m2 == k {
            (k as i32, b'N')
        } else if lhs_m1 == m && lhs_m2 == 1 {
            (m as i32, b'T')
        } else {
            Err(self.striding_error(lhs_l, rhs_l, "non-contiguous lhs"))?
        };

        let mut dst = vec![T::zero(); b * m * n];
        match T::DTYPE {
            DType::F16 => {
                for step in 0..b {
                    let lhs_p = &lhs[step * a_skip..];
                    let rhs_p = &rhs[step * b_skip..];
                    let dst_p = &mut dst[step * c_skip..];
                    unsafe {
                        let a = rhs_p.as_ptr() as *const f16;
                        let b = lhs_p.as_ptr() as *const f16;
                        let c = dst_p.as_mut_ptr() as *mut f16;
                        let a = std::slice::from_raw_parts(a, a_skip);
                        let b = std::slice::from_raw_parts(b, b_skip);
                        let c = std::slice::from_raw_parts_mut(c, c_skip);
                        crate::mkl::hgemm(
                            transa,
                            transb,
                            /* m= */ n as i32,
                            /* n= */ m as i32,
                            /* k= */ k as i32,
                            /* alpha= */ f16::ONE,
                            /* a= */ a,
                            /* lda= */ lda,
                            /* b= */ b,
                            /* ldb= */ ldb,
                            /* beta= */ f16::ZERO,
                            /* c= */ c,
                            /* ldc= */ n as i32,
                        )
                    }
                }
            }
            DType::F32 => {
                for step in 0..b {
                    let lhs_p = &lhs[step * a_skip..];
                    let rhs_p = &rhs[step * b_skip..];
                    let dst_p = &mut dst[step * c_skip..];
                    unsafe {
                        let a = rhs_p.as_ptr() as *const f32;
                        let b = lhs_p.as_ptr() as *const f32;
                        let c = dst_p.as_mut_ptr() as *mut f32;
                        let a = std::slice::from_raw_parts(a, a_skip);
                        let b = std::slice::from_raw_parts(b, b_skip);
                        let c = std::slice::from_raw_parts_mut(c, c_skip);
                        crate::mkl::sgemm(
                            transa, transb, /* m= */ n as i32, /* n= */ m as i32,
                            /* k= */ k as i32, /* alpha= */ 1., /* a= */ a,
                            /* lda= */ lda, /* b= */ b, /* ldb= */ ldb,
                            /* beta= */ 0., /* c= */ c, /* ldc= */ n as i32,
                        )
                    }
                }
            }
            DType::F64 => {
                for step in 0..b {
                    let lhs_p = &lhs[step * a_skip..];
                    let rhs_p = &rhs[step * b_skip..];
                    let dst_p = &mut dst[step * c_skip..];
                    unsafe {
                        let a = rhs_p.as_ptr() as *const f64;
                        let b = lhs_p.as_ptr() as *const f64;
                        let c = dst_p.as_mut_ptr() as *mut f64;
                        let a = std::slice::from_raw_parts(a, a_skip);
                        let b = std::slice::from_raw_parts(b, b_skip);
                        let c = std::slice::from_raw_parts_mut(c, c_skip);
                        crate::mkl::dgemm(
                            transa, transb, /* m= */ n as i32, /* n= */ m as i32,
                            /* k= */ k as i32, /* alpha= */ 1., /* a= */ a,
                            /* lda= */ lda, /* b= */ b, /* ldb= */ ldb,
                            /* beta= */ 0., /* c= */ c, /* ldc= */ n as i32,
                        )
                    }
                }
            }
            dtype => Err(Error::UnsupportedDTypeForOp(dtype, "matmul").bt())?,
        }
        Ok(dst)
    }
}

fn elu<T: num_traits::Float>(v: T, alpha: T) -> T {
    if v.is_sign_positive() {
        v
    } else {
        (v.exp() - T::one()) * alpha
    }
}

impl CpuStorage {
    pub fn as_slice<D: WithDType>(&self) -> Result<&[D]> {
        D::cpu_storage_as_slice(self)
    }

    pub fn concat(storages: &[CpuStorage]) -> Result<CpuStorage> {
        let storage0 = &storages[0];
        let s = match storage0 {
            Self::U8(_) => {
                let storages = storages
                    .iter()
                    .map(|s| match s {
                        Self::U8(s) => Ok(s.as_slice()),
                        _ => crate::bail!("dtype mismatch"),
                    })
                    .collect::<Result<Vec<_>>>()?
                    .concat();
                Self::U8(storages)
            }
            Self::U32(_) => {
                let storages = storages
                    .iter()
                    .map(|s| match s {
                        Self::U32(s) => Ok(s.as_slice()),
                        _ => crate::bail!("dtype mismatch"),
                    })
                    .collect::<Result<Vec<_>>>()?
                    .concat();
                Self::U32(storages)
            }
            Self::I64(_) => {
                let storages = storages
                    .iter()
                    .map(|s| match s {
                        Self::I64(s) => Ok(s.as_slice()),
                        _ => crate::bail!("dtype mismatch"),
                    })
                    .collect::<Result<Vec<_>>>()?
                    .concat();
                Self::I64(storages)
            }
            Self::BF16(_) => {
                let storages = storages
                    .iter()
                    .map(|s| match s {
                        Self::BF16(s) => Ok(s.as_slice()),
                        _ => crate::bail!("dtype mismatch"),
                    })
                    .collect::<Result<Vec<_>>>()?
                    .concat();
                Self::BF16(storages)
            }
            Self::F16(_) => {
                let storages = storages
                    .iter()
                    .map(|s| match s {
                        Self::F16(s) => Ok(s.as_slice()),
                        _ => crate::bail!("dtype mismatch"),
                    })
                    .collect::<Result<Vec<_>>>()?
                    .concat();
                Self::F16(storages)
            }
            Self::F32(_) => {
                let storages = storages
                    .iter()
                    .map(|s| match s {
                        Self::F32(s) => Ok(s.as_slice()),
                        _ => crate::bail!("dtype mismatch"),
                    })
                    .collect::<Result<Vec<_>>>()?
                    .concat();
                Self::F32(storages)
            }
            Self::F64(_) => {
                let storages = storages
                    .iter()
                    .map(|s| match s {
                        Self::F64(s) => Ok(s.as_slice()),
                        _ => crate::bail!("dtype mismatch"),
                    })
                    .collect::<Result<Vec<_>>>()?
                    .concat();
                Self::F64(storages)
            }
        };
        Ok(s)
    }
}

impl BackendStorage for CpuStorage {
    type Device = CpuDevice;

    fn dtype(&self) -> DType {
        match self {
            Self::U8(_) => DType::U8,
            Self::U32(_) => DType::U32,
            Self::I64(_) => DType::I64,
            Self::BF16(_) => DType::BF16,
            Self::F16(_) => DType::F16,
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
        }
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        // TODO: find a way around the quadratic number of cases below.
        match (self, dtype) {
            (Self::U8(storage), DType::BF16) => {
                let data = unary_map(storage, layout, |v| bf16::from_f32(v as f32));
                Ok(Self::BF16(data))
            }
            (Self::U32(storage), DType::BF16) => {
                let data = unary_map(storage, layout, |v| bf16::from_f32(v as f32));
                Ok(Self::BF16(data))
            }
            (Self::I64(storage), DType::BF16) => {
                let data = unary_map(storage, layout, |v| bf16::from_f32(v as f32));
                Ok(Self::BF16(data))
            }
            (Self::BF16(storage), DType::BF16) => {
                let data = unary_map(storage, layout, |v| v);
                Ok(Self::BF16(data))
            }
            (Self::F16(storage), DType::BF16) => {
                let data = unary_map(storage, layout, |v| bf16::from_f32(v.to_f32()));
                Ok(Self::BF16(data))
            }
            (Self::F32(storage), DType::BF16) => {
                let data = unary_map(storage, layout, bf16::from_f32);
                Ok(Self::BF16(data))
            }
            (Self::F64(storage), DType::BF16) => {
                let data = unary_map(storage, layout, bf16::from_f64);
                Ok(Self::BF16(data))
            }
            (Self::U8(storage), DType::F16) => {
                let data = unary_map(storage, layout, |v| f16::from_f32(v as f32));
                Ok(Self::F16(data))
            }
            (Self::U32(storage), DType::F16) => {
                let data = unary_map(storage, layout, |v| f16::from_f32(v as f32));
                Ok(Self::F16(data))
            }
            (Self::I64(storage), DType::F16) => {
                let data = unary_map(storage, layout, |v| f16::from_f32(v as f32));
                Ok(Self::F16(data))
            }
            (Self::BF16(storage), DType::F16) => {
                let data = unary_map(storage, layout, |v| f16::from_f32(v.to_f32()));
                Ok(Self::F16(data))
            }
            (Self::F16(storage), DType::F16) => {
                let data = unary_map(storage, layout, |v| v);
                Ok(Self::F16(data))
            }
            (Self::F32(storage), DType::F16) => {
                let data = unary_map(storage, layout, f16::from_f32);
                Ok(Self::F16(data))
            }
            (Self::F64(storage), DType::F16) => {
                let data = unary_map(storage, layout, f16::from_f64);
                Ok(Self::F16(data))
            }
            (Self::U8(storage), DType::F32) => {
                let data = unary_map(storage, layout, |v| v as f32);
                Ok(Self::F32(data))
            }
            (Self::U32(storage), DType::F32) => {
                let data = unary_map(storage, layout, |v| v as f32);
                Ok(Self::F32(data))
            }
            (Self::I64(storage), DType::F32) => {
                let data = unary_map(storage, layout, |v| v as f32);
                Ok(Self::F32(data))
            }
            (Self::BF16(storage), DType::F32) => {
                let data = unary_map(storage, layout, |v| v.to_f32());
                Ok(Self::F32(data))
            }
            (Self::F16(storage), DType::F32) => {
                let data = unary_map(storage, layout, |v| v.to_f32());
                Ok(Self::F32(data))
            }
            (Self::F32(storage), DType::F32) => {
                let data = unary_map(storage, layout, |v| v);
                Ok(Self::F32(data))
            }
            (Self::F64(storage), DType::F32) => {
                let data = unary_map(storage, layout, |v| v as f32);
                Ok(Self::F32(data))
            }
            (Self::U8(storage), DType::U8) => {
                let data = unary_map(storage, layout, |v| v);
                Ok(Self::U8(data))
            }
            (Self::BF16(storage), DType::U8) => {
                let data = unary_map(storage, layout, |v| v.to_f32() as u8);
                Ok(Self::U8(data))
            }
            (Self::F16(storage), DType::U8) => {
                let data = unary_map(storage, layout, |v| v.to_f32() as u8);
                Ok(Self::U8(data))
            }
            (Self::F32(storage), DType::U8) => {
                let data = unary_map(storage, layout, |v| v as u8);
                Ok(Self::U8(data))
            }
            (Self::F64(storage), DType::U8) => {
                let data = unary_map(storage, layout, |v| v as u8);
                Ok(Self::U8(data))
            }
            (Self::U32(storage), DType::U8) => {
                let data = unary_map(storage, layout, |v| v as u8);
                Ok(Self::U8(data))
            }
            (Self::I64(storage), DType::U8) => {
                let data = unary_map(storage, layout, |v| v as u8);
                Ok(Self::U8(data))
            }
            (Self::U8(storage), DType::U32) => {
                let data = unary_map(storage, layout, |v| v as u32);
                Ok(Self::U32(data))
            }
            (Self::U32(storage), DType::U32) => {
                let data = unary_map(storage, layout, |v| v);
                Ok(Self::U32(data))
            }
            (Self::I64(storage), DType::U32) => {
                let data = unary_map(storage, layout, |v| v as u32);
                Ok(Self::U32(data))
            }
            (Self::BF16(storage), DType::U32) => {
                let data = unary_map(storage, layout, |v| v.to_f32() as u32);
                Ok(Self::U32(data))
            }
            (Self::F16(storage), DType::U32) => {
                let data = unary_map(storage, layout, |v| v.to_f32() as u32);
                Ok(Self::U32(data))
            }
            (Self::F32(storage), DType::U32) => {
                let data = unary_map(storage, layout, |v| v as u32);
                Ok(Self::U32(data))
            }
            (Self::F64(storage), DType::U32) => {
                let data = unary_map(storage, layout, |v| v as u32);
                Ok(Self::U32(data))
            }
            (Self::U8(storage), DType::I64) => {
                let data = unary_map(storage, layout, |v| v as i64);
                Ok(Self::I64(data))
            }
            (Self::U32(storage), DType::I64) => {
                let data = unary_map(storage, layout, |v| v as i64);
                Ok(Self::I64(data))
            }
            (Self::I64(storage), DType::I64) => {
                let data = unary_map(storage, layout, |v| v);
                Ok(Self::I64(data))
            }
            (Self::BF16(storage), DType::I64) => {
                let data = unary_map(storage, layout, |v| v.to_f32() as i64);
                Ok(Self::I64(data))
            }
            (Self::F16(storage), DType::I64) => {
                let data = unary_map(storage, layout, |v| v.to_f32() as i64);
                Ok(Self::I64(data))
            }
            (Self::F32(storage), DType::I64) => {
                let data = unary_map(storage, layout, |v| v as i64);
                Ok(Self::I64(data))
            }
            (Self::F64(storage), DType::I64) => {
                let data = unary_map(storage, layout, |v| v as i64);
                Ok(Self::I64(data))
            }
            (Self::U8(storage), DType::F64) => {
                let data = unary_map(storage, layout, |v| v as f64);
                Ok(Self::F64(data))
            }
            (Self::U32(storage), DType::F64) => {
                let data = unary_map(storage, layout, |v| v as f64);
                Ok(Self::F64(data))
            }
            (Self::I64(storage), DType::F64) => {
                let data = unary_map(storage, layout, |v| v as f64);
                Ok(Self::F64(data))
            }
            (Self::BF16(storage), DType::F64) => {
                let data = unary_map(storage, layout, |v| v.to_f64());
                Ok(Self::F64(data))
            }
            (Self::F16(storage), DType::F64) => {
                let data = unary_map(storage, layout, |v| v.to_f64());
                Ok(Self::F64(data))
            }
            (Self::F32(storage), DType::F64) => {
                let data = unary_map(storage, layout, |v| v as f64);
                Ok(Self::F64(data))
            }
            (Self::F64(storage), DType::F64) => {
                let data = unary_map(storage, layout, |v| v);
                Ok(Self::F64(data))
            }
        }
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, reduce_dims: &[usize]) -> Result<Self> {
        match op {
            ReduceOp::Sum => {
                let src_dims = layout.dims();
                let mut dst_dims = src_dims.to_vec();
                for &dim in reduce_dims.iter() {
                    dst_dims[dim] = 1;
                }
                let dst_shape = Shape::from(dst_dims);
                let mut reduce_dims = reduce_dims.to_vec();
                // Sort the reduce_dims as they have to be processed from left to right when converting the
                // indexes.
                reduce_dims.sort();
                let reduce_dims_and_stride: Vec<_> = reduce_dims
                    .iter()
                    .map(|&d| (src_dims[d], src_dims[d + 1..].iter().product::<usize>()))
                    .collect();
                ReduceSum {
                    dst_shape: &dst_shape,
                    reduce_dims: &reduce_dims,
                    reduce_dims_and_stride,
                }
                .map(self, layout)
            }
            ReduceOp::Min | ReduceOp::ArgMin | ReduceOp::Max | ReduceOp::ArgMax => {
                let reduce_dim_index = match reduce_dims {
                    [reduce_dim_index] => *reduce_dim_index,
                    _ => {
                        let op = match op {
                            ReduceOp::Min => "min",
                            ReduceOp::ArgMin => "argmin",
                            ReduceOp::Max => "max",
                            ReduceOp::ArgMax => "argmax",
                            _ => unreachable!(),
                        };
                        let dims = reduce_dims.to_vec();
                        Err(Error::OnlySingleDimension { op, dims })?
                    }
                };
                let (use_min, return_index) = match op {
                    ReduceOp::Min => (true, false),
                    ReduceOp::ArgMin => (true, true),
                    ReduceOp::Max => (false, false),
                    ReduceOp::ArgMax => (false, true),
                    _ => unreachable!(),
                };
                ReduceIndex {
                    reduce_dim_index,
                    use_min,
                    return_index,
                }
                .map(self, layout)
            }
        }
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        Cmp(op).map(self, lhs_l, rhs, rhs_l)
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        Affine(mul, add).map(self, layout)
    }

    fn avg_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        AvgPool2D(kernel_size, stride).map(self, layout)
    }

    fn max_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        MaxPool2D(kernel_size, stride).map(self, layout)
    }

    fn upsample_nearest1d(&self, layout: &Layout, sz: usize) -> Result<Self> {
        UpsampleNearest1D(sz).map(self, layout)
    }

    fn upsample_nearest2d(&self, layout: &Layout, h: usize, w: usize) -> Result<Self> {
        UpsampleNearest2D(h, w).map(self, layout)
    }

    fn powf(&self, layout: &Layout, e: f64) -> Result<Self> {
        use num_traits::Float;
        // TODO: Have some generic map for functions that apply on num_traits::Float elements.
        match self {
            Self::BF16(storage) => {
                let data = unary_map(storage, layout, |v| v.powf(bf16::from_f64(e)));
                Ok(Self::BF16(data))
            }
            Self::F16(storage) => {
                let data = unary_map(storage, layout, |v| v.powf(f16::from_f64(e)));
                Ok(Self::F16(data))
            }
            Self::F32(storage) => {
                let data = unary_map(storage, layout, |v| v.powf(e as f32));
                Ok(Self::F32(data))
            }
            Self::F64(storage) => {
                let data = unary_map(storage, layout, |v| v.powf(e));
                Ok(Self::F64(data))
            }
            Self::U8(_) => Err(Error::UnsupportedDTypeForOp(DType::U8, "elu").bt()),
            Self::U32(_) => Err(Error::UnsupportedDTypeForOp(DType::U32, "elu").bt()),
            Self::I64(_) => Err(Error::UnsupportedDTypeForOp(DType::I64, "elu").bt()),
        }
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        // TODO: Have some generic map for functions that apply on num_traits::Float elements.
        match self {
            Self::BF16(storage) => {
                let data = unary_map(storage, layout, |v| elu(v, bf16::from_f64(alpha)));
                Ok(Self::BF16(data))
            }
            Self::F16(storage) => {
                let data = unary_map(storage, layout, |v| elu(v, f16::from_f64(alpha)));
                Ok(Self::F16(data))
            }
            Self::F32(storage) => {
                let data = unary_map(storage, layout, |v| elu(v, f32::from_f64(alpha)));
                Ok(Self::F32(data))
            }
            Self::F64(storage) => {
                let data = unary_map(storage, layout, |v| elu(v, alpha));
                Ok(Self::F64(data))
            }
            Self::U8(_) => Err(Error::UnsupportedDTypeForOp(DType::U8, "elu").bt()),
            Self::U32(_) => Err(Error::UnsupportedDTypeForOp(DType::U32, "elu").bt()),
            Self::I64(_) => Err(Error::UnsupportedDTypeForOp(DType::I64, "elu").bt()),
        }
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        match self {
            Self::BF16(storage) => {
                if B::BF16_VEC {
                    let data = unary_map_vec(storage, layout, B::bf16, B::bf16_vec);
                    Ok(Self::BF16(data))
                } else {
                    let data = unary_map(storage, layout, B::bf16);
                    Ok(Self::BF16(data))
                }
            }
            Self::F16(storage) => {
                if B::F16_VEC {
                    let data = unary_map_vec(storage, layout, B::f16, B::f16_vec);
                    Ok(Self::F16(data))
                } else {
                    let data = unary_map(storage, layout, B::f16);
                    Ok(Self::F16(data))
                }
            }
            Self::F32(storage) => {
                if B::F32_VEC {
                    let data = unary_map_vec(storage, layout, B::f32, B::f32_vec);
                    Ok(Self::F32(data))
                } else {
                    let data = unary_map(storage, layout, B::f32);
                    Ok(Self::F32(data))
                }
            }
            Self::F64(storage) => {
                if B::F64_VEC {
                    let data = unary_map_vec(storage, layout, B::f64, B::f64_vec);
                    Ok(Self::F64(data))
                } else {
                    let data = unary_map(storage, layout, B::f64);
                    Ok(Self::F64(data))
                }
            }
            Self::U8(storage) => {
                let data = unary_map(storage, layout, B::u8);
                Ok(Self::U8(data))
            }
            Self::U32(storage) => {
                let data = unary_map(storage, layout, B::u32);
                Ok(Self::U32(data))
            }
            Self::I64(storage) => {
                let data = unary_map(storage, layout, B::i64);
                Ok(Self::I64(data))
            }
        }
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        match (self, rhs) {
            (Self::BF16(lhs), Self::BF16(rhs)) => {
                let data = if B::BF16_VEC {
                    binary_map_vec(lhs_l, rhs_l, lhs, rhs, B::bf16, B::bf16_vec)
                } else {
                    binary_map(lhs_l, rhs_l, lhs, rhs, B::bf16)
                };
                Ok(Self::BF16(data))
            }
            (Self::F16(lhs), Self::F16(rhs)) => {
                let data = if B::F16_VEC {
                    binary_map_vec(lhs_l, rhs_l, lhs, rhs, B::f16, B::f16_vec)
                } else {
                    binary_map(lhs_l, rhs_l, lhs, rhs, B::f16)
                };
                Ok(Self::F16(data))
            }
            (Self::F32(lhs), Self::F32(rhs)) => {
                let data = if B::F32_VEC {
                    binary_map_vec(lhs_l, rhs_l, lhs, rhs, B::f32, B::f32_vec)
                } else {
                    binary_map(lhs_l, rhs_l, lhs, rhs, B::f32)
                };
                Ok(Self::F32(data))
            }
            (Self::F64(lhs), Self::F64(rhs)) => {
                let data = if B::F64_VEC {
                    binary_map_vec(lhs_l, rhs_l, lhs, rhs, B::f64, B::f64_vec)
                } else {
                    binary_map(lhs_l, rhs_l, lhs, rhs, B::f64)
                };
                Ok(Self::F64(data))
            }
            (Self::U32(lhs), Self::U32(rhs)) => {
                let data = if B::U32_VEC {
                    binary_map_vec(lhs_l, rhs_l, lhs, rhs, B::u32, B::u32_vec)
                } else {
                    binary_map(lhs_l, rhs_l, lhs, rhs, B::u32)
                };
                Ok(Self::U32(data))
            }
            (Self::I64(lhs), Self::I64(rhs)) => {
                let data = if B::I64_VEC {
                    binary_map_vec(lhs_l, rhs_l, lhs, rhs, B::i64, B::i64_vec)
                } else {
                    binary_map(lhs_l, rhs_l, lhs, rhs, B::i64)
                };
                Ok(Self::I64(data))
            }
            (Self::U8(lhs), Self::U8(rhs)) => {
                let data = if B::U8_VEC {
                    binary_map_vec(lhs_l, rhs_l, lhs, rhs, B::u8, B::u8_vec)
                } else {
                    binary_map(lhs_l, rhs_l, lhs, rhs, B::u8)
                };
                Ok(Self::U8(data))
            }
            _ => {
                // This should be covered by the dtype check above.
                Err(Error::DTypeMismatchBinaryOp {
                    lhs: self.dtype(),
                    rhs: rhs.dtype(),
                    op: B::NAME,
                }
                .bt())
            }
        }
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        match (self, dst) {
            (Self::U8(src), Self::U8(dst)) => copy_strided_src_(src, dst, dst_offset, src_l),
            (Self::U32(src), Self::U32(dst)) => copy_strided_src_(src, dst, dst_offset, src_l),
            (Self::I64(src), Self::I64(dst)) => copy_strided_src_(src, dst, dst_offset, src_l),
            (Self::BF16(src), Self::BF16(dst)) => copy_strided_src_(src, dst, dst_offset, src_l),
            (Self::F16(src), Self::F16(dst)) => copy_strided_src_(src, dst, dst_offset, src_l),
            (Self::F32(src), Self::F32(dst)) => copy_strided_src_(src, dst, dst_offset, src_l),
            (Self::F64(src), Self::F64(dst)) => copy_strided_src_(src, dst, dst_offset, src_l),
            (_, dst) => {
                // This should be covered by the dtype check above.
                return Err(Error::DTypeMismatchBinaryOp {
                    lhs: self.dtype(),
                    rhs: dst.dtype(),
                    op: "copy_strided",
                }
                .bt());
            }
        }
        Ok(())
    }

    fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        t_l: &Layout,
        f: &Self,
        f_l: &Layout,
    ) -> Result<Self> {
        match self {
            Self::U8(pred) => WCond(pred, layout).map(t, t_l, f, f_l),
            Self::U32(pred) => WCond(pred, layout).map(t, t_l, f, f_l),
            Self::I64(pred) => WCond(pred, layout).map(t, t_l, f, f_l),
            _ => Err(Error::UnsupportedDTypeForOp(self.dtype(), "where-cond")),
        }
    }

    fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        if !USE_IM2COL_CONV1D {
            return Conv1D(params).map(self, l, kernel, kernel_l);
        }
        let op = Im2Col1D {
            l_k: params.k_size,
            padding: params.padding,
            stride: params.stride,
            dilation: params.dilation,
        };
        let col = op.map(self, l)?;
        let b = params.b_size;
        let n = params.c_out;
        let l_out = params.l_out();
        let k = op.l_k * params.c_in;
        let m = l_out;
        let col_l = Layout::contiguous((b, m, k));
        let res = if kernel_l.is_contiguous() {
            let kernel_l = Layout::contiguous_with_offset((1, n, k), kernel_l.start_offset())
                .transpose(1, 2)?
                .broadcast_as((b, k, n))?;
            col.matmul(kernel, (b, m, n, k), &col_l, &kernel_l)?
        } else {
            // Make the kernel contiguous if not already the case.
            let mut kernel_c = self.device().zeros_impl(kernel_l.shape(), kernel.dtype())?;
            kernel.copy_strided_src(&mut kernel_c, 0, kernel_l)?;
            let kernel_l = Layout::contiguous_with_offset((1, n, k), kernel_l.start_offset())
                .transpose(1, 2)?
                .broadcast_as((b, k, n))?;
            col.matmul(kernel, (b, m, n, k), &col_l, &kernel_l)?
        };
        let res_l = Layout::contiguous((b, l_out, params.c_out)).transpose(1, 2)?;
        let mut res_t = self.device().zeros_impl(res_l.shape(), res.dtype())?;
        res.copy_strided_src(&mut res_t, 0, &res_l)?;
        Ok(res_t)
    }

    fn conv_transpose1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        ConvTranspose1D(params).map(self, l, kernel, kernel_l)
    }

    fn conv2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        if !USE_IM2COL_CONV2D {
            return Conv2D(params).map(self, l, kernel, kernel_l);
        }
        let op = Im2Col {
            h_k: params.k_h,
            w_k: params.k_w,
            padding: params.padding,
            stride: params.stride,
            dilation: params.dilation,
        };
        let col = op.map(self, l)?;
        let b = params.b_size;
        let n = params.c_out;
        let (h_out, w_out) = (params.out_h(), params.out_w());
        let k = op.h_k * op.w_k * params.c_in;
        let m = h_out * w_out;
        let col_l = Layout::contiguous((b, m, k));
        let res = if kernel_l.is_contiguous() {
            let kernel_l = Layout::contiguous_with_offset((1, n, k), kernel_l.start_offset())
                .transpose(1, 2)?
                .broadcast_as((b, k, n))?;
            col.matmul(kernel, (b, m, n, k), &col_l, &kernel_l)?
        } else {
            // Make the kernel contiguous if not already the case.
            let mut kernel_c = self.device().zeros_impl(kernel_l.shape(), kernel.dtype())?;
            kernel.copy_strided_src(&mut kernel_c, 0, kernel_l)?;
            let kernel_l = Layout::contiguous_with_offset((1, n, k), kernel_l.start_offset())
                .transpose(1, 2)?
                .broadcast_as((b, k, n))?;
            col.matmul(kernel, (b, m, n, k), &col_l, &kernel_l)?
        };
        let res_l = Layout::contiguous((b, h_out, w_out, params.c_out))
            .transpose(1, 2)?
            .transpose(1, 3)?;
        let mut res_t = self.device().zeros_impl(res_l.shape(), res.dtype())?;
        res.copy_strided_src(&mut res_t, 0, &res_l)?;
        Ok(res_t)
    }

    fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        ConvTranspose2D(params).map(self, l, kernel, kernel_l)
    }

    fn index_select(&self, ids: &Self, l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        match ids {
            Self::U8(ids) => IndexSelect { ids, ids_l, dim }.map(self, l),
            Self::U32(ids) => IndexSelect { ids, ids_l, dim }.map(self, l),
            Self::I64(ids) => IndexSelect { ids, ids_l, dim }.map(self, l),
            _ => Err(Error::UnsupportedDTypeForOp(self.dtype(), "index-select").bt()),
        }
    }

    fn gather(&self, l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        match ids {
            Self::U8(ids) => Gather { ids, ids_l, dim }.map(self, l),
            Self::U32(ids) => Gather { ids, ids_l, dim }.map(self, l),
            Self::I64(ids) => Gather { ids, ids_l, dim }.map(self, l),
            _ => Err(Error::UnsupportedDTypeForOp(self.dtype(), "gather").bt()),
        }
    }

    fn scatter_add(
        &self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        match ids {
            Self::U8(ids) => ScatterAdd { ids, ids_l, dim }.map(self, l, src, src_l),
            Self::U32(ids) => ScatterAdd { ids, ids_l, dim }.map(self, l, src, src_l),
            Self::I64(ids) => ScatterAdd { ids, ids_l, dim }.map(self, l, src, src_l),
            _ => Err(Error::UnsupportedDTypeForOp(self.dtype(), "scatter-add").bt()),
        }
    }

    fn index_add(
        &self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        match ids {
            Self::U8(ids) => {
                let ids = match ids_l.contiguous_offsets() {
                    Some((a, b)) => &ids[a..b],
                    None => Err(Error::RequiresContiguous { op: "index-add" }.bt())?,
                };
                IndexAdd { ids, dim }.map(self, l, src, src_l)
            }
            Self::U32(ids) => {
                let ids = match ids_l.contiguous_offsets() {
                    Some((a, b)) => &ids[a..b],
                    None => Err(Error::RequiresContiguous { op: "index-add" }.bt())?,
                };
                IndexAdd { ids, dim }.map(self, l, src, src_l)
            }
            Self::I64(ids) => {
                let ids = match ids_l.contiguous_offsets() {
                    Some((a, b)) => &ids[a..b],
                    None => Err(Error::RequiresContiguous { op: "index-add" }.bt())?,
                };
                IndexAdd { ids, dim }.map(self, l, src, src_l)
            }
            _ => Err(Error::UnsupportedDTypeForOp(self.dtype(), "index-add").bt()),
        }
    }

    fn matmul(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        MatMul(bmnk).map(self, lhs_l, rhs, rhs_l)
    }

    fn device(&self) -> &Self::Device {
        &CpuDevice
    }

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        Ok(self.clone())
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        Ok(self.clone())
    }
}

impl BackendDevice for CpuDevice {
    type Storage = CpuStorage;

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Cpu
    }

    fn same_device(&self, _: &Self) -> bool {
        true
    }

    fn storage_from_cpu_storage(&self, s: &CpuStorage) -> Result<Self::Storage> {
        Ok(s.clone())
    }

    fn new(_: usize) -> Result<Self> {
        Ok(Self)
    }

    fn set_seed(&self, _seed: u64) -> Result<()> {
        crate::bail!("cannot seed the CPU rng with set_seed")
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, min: f64, max: f64) -> Result<CpuStorage> {
        use rand::prelude::*;

        let elem_count = shape.elem_count();
        let mut rng = rand::thread_rng();
        match dtype {
            DType::U8 | DType::U32 | DType::I64 => {
                Err(Error::UnsupportedDTypeForOp(dtype, "rand_uniform").bt())
            }
            DType::BF16 => {
                let mut data = Vec::with_capacity(elem_count);
                let uniform =
                    rand::distributions::Uniform::new(bf16::from_f64(min), bf16::from_f64(max));
                for _i in 0..elem_count {
                    data.push(rng.sample::<bf16, _>(uniform))
                }
                Ok(CpuStorage::BF16(data))
            }
            DType::F16 => {
                let mut data = Vec::with_capacity(elem_count);
                let uniform =
                    rand::distributions::Uniform::new(f16::from_f64(min), f16::from_f64(max));
                for _i in 0..elem_count {
                    data.push(rng.sample::<f16, _>(uniform))
                }
                Ok(CpuStorage::F16(data))
            }
            DType::F32 => {
                let mut data = Vec::with_capacity(elem_count);
                let uniform = rand::distributions::Uniform::new(min as f32, max as f32);
                for _i in 0..elem_count {
                    data.push(rng.sample::<f32, _>(uniform))
                }
                Ok(CpuStorage::F32(data))
            }
            DType::F64 => {
                let mut data = Vec::with_capacity(elem_count);
                let uniform = rand::distributions::Uniform::new(min, max);
                for _i in 0..elem_count {
                    data.push(rng.sample::<f64, _>(uniform))
                }
                Ok(CpuStorage::F64(data))
            }
        }
    }

    fn rand_normal(&self, shape: &Shape, dtype: DType, mean: f64, std: f64) -> Result<CpuStorage> {
        use rand::prelude::*;

        let elem_count = shape.elem_count();
        let mut rng = rand::thread_rng();
        match dtype {
            DType::U8 | DType::U32 | DType::I64 => {
                Err(Error::UnsupportedDTypeForOp(dtype, "rand_normal").bt())
            }
            DType::BF16 => {
                let mut data = Vec::with_capacity(elem_count);
                let normal = rand_distr::Normal::new(bf16::from_f64(mean), bf16::from_f64(std))
                    .map_err(Error::wrap)?;
                for _i in 0..elem_count {
                    data.push(normal.sample(&mut rng))
                }
                Ok(CpuStorage::BF16(data))
            }
            DType::F16 => {
                let mut data = Vec::with_capacity(elem_count);
                let normal = rand_distr::Normal::new(f16::from_f64(mean), f16::from_f64(std))
                    .map_err(Error::wrap)?;
                for _i in 0..elem_count {
                    data.push(normal.sample(&mut rng))
                }
                Ok(CpuStorage::F16(data))
            }
            DType::F32 => {
                let mut data = Vec::with_capacity(elem_count);
                let normal =
                    rand_distr::Normal::new(mean as f32, std as f32).map_err(Error::wrap)?;
                for _i in 0..elem_count {
                    data.push(normal.sample(&mut rng))
                }
                Ok(CpuStorage::F32(data))
            }
            DType::F64 => {
                let mut data = Vec::with_capacity(elem_count);
                let normal = rand_distr::Normal::new(mean, std).map_err(Error::wrap)?;
                for _i in 0..elem_count {
                    data.push(normal.sample(&mut rng))
                }
                Ok(CpuStorage::F64(data))
            }
        }
    }

    fn ones_impl(&self, shape: &Shape, dtype: DType) -> Result<CpuStorage> {
        let elem_count = shape.elem_count();
        let storage = match dtype {
            DType::U8 => CpuStorage::U8(vec![1u8; elem_count]),
            DType::U32 => CpuStorage::U32(vec![1u32; elem_count]),
            DType::I64 => CpuStorage::I64(vec![1i64; elem_count]),
            DType::BF16 => CpuStorage::BF16(vec![bf16::ONE; elem_count]),
            DType::F16 => CpuStorage::F16(vec![f16::ONE; elem_count]),
            DType::F32 => CpuStorage::F32(vec![1f32; elem_count]),
            DType::F64 => CpuStorage::F64(vec![1f64; elem_count]),
        };
        Ok(storage)
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<CpuStorage> {
        let elem_count = shape.elem_count();
        let storage = match dtype {
            DType::U8 => CpuStorage::U8(vec![0u8; elem_count]),
            DType::U32 => CpuStorage::U32(vec![0u32; elem_count]),
            DType::I64 => CpuStorage::I64(vec![0i64; elem_count]),
            DType::BF16 => CpuStorage::BF16(vec![bf16::ZERO; elem_count]),
            DType::F16 => CpuStorage::F16(vec![f16::ZERO; elem_count]),
            DType::F32 => CpuStorage::F32(vec![0f32; elem_count]),
            DType::F64 => CpuStorage::F64(vec![0f64; elem_count]),
        };
        Ok(storage)
    }
}

#[macro_export]
macro_rules! map_dtype {
    ($name:expr, $storage:ident, $fn:expr, ($($dtypes:ident),+)) => {
        match $storage {
            $(CpuStorage::$dtypes(__e) => CpuStorage::$dtypes($fn(__e)),)*
            s => Err(Error::UnsupportedDTypeForOp(s.dtype(), $name).bt())?,
        }
    };
}
