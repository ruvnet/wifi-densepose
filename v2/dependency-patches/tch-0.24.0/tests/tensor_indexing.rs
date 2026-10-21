use tch::{Device, Kind, Tensor};
use tch::{IndexOp, NewAxis};

mod test_utils;
use test_utils::*;

#[test]
fn integer_index() {
    let opt = (Kind::Float, Device::Cpu);

    let tensor = Tensor::arange_start(0, 2 * 3, opt).view([2, 3]);
    let result = tensor.i(1);
    assert_eq!(result.size(), &[3]);
    assert_eq!(vec_i64_from(&result), &[3, 4, 5]);

    let tensor = Tensor::arange_start(0, 2 * 3, opt).view([2, 3]);
    let result = tensor.i((.., 2));
    assert_eq!(result.size(), &[2]);
    assert_eq!(vec_i64_from(&result), &[2, 5]);

    let result = tensor.i((.., -2));
    assert_eq!(result.size(), &[2]);
    assert_eq!(vec_i64_from(&result), &[1, 4]);
}

#[test]
fn range_index() {
    let opt = (Kind::Float, Device::Cpu);

    // Range
    let tensor = Tensor::arange_start(0, 4 * 3, opt).view([4, 3]);
    let result = tensor.i(1..3);
    assert_eq!(result.size(), &[2, 3]);
    assert_eq!(vec_i64_from(&result), &[3, 4, 5, 6, 7, 8]);

    // RangeFull
    let tensor = Tensor::arange_start(0, 2 * 3, opt).view([2, 3]);
    let result = tensor.i(..);
    assert_eq!(result.size(), &[2, 3]);
    assert_eq!(vec_i64_from(&result), &[0, 1, 2, 3, 4, 5]);

    // RangeFrom
    let tensor = Tensor::arange_start(0, 4 * 3, opt).view([4, 3]);
    let result = tensor.i(2..);
    assert_eq!(result.size(), &[2, 3]);
    assert_eq!(vec_i64_from(&result), &[6, 7, 8, 9, 10, 11]);

    // RangeTo
    let tensor = Tensor::arange_start(0, 4 * 3, opt).view([4, 3]);
    let result = tensor.i(..2);
    assert_eq!(result.size(), &[2, 3]);
    assert_eq!(vec_i64_from(&result), &[0, 1, 2, 3, 4, 5]);

    // RangeInclusive
    let tensor = Tensor::arange_start(0, 4 * 3, opt).view([4, 3]);
    let result = tensor.i(1..=2);
    assert_eq!(result.size(), &[2, 3]);
    assert_eq!(vec_i64_from(&result), &[3, 4, 5, 6, 7, 8]);

    // RangeTo
    let tensor = Tensor::arange_start(0, 4 * 3, opt).view([4, 3]);
    let result = tensor.i(..1);
    assert_eq!(result.size(), &[1, 3]);
    assert_eq!(vec_i64_from(&result), &[0, 1, 2]);

    // RangeToInclusive
    let tensor = Tensor::arange_start(0, 4 * 3, opt).view([4, 3]);
    let result = tensor.i(..=1);
    assert_eq!(result.size(), &[2, 3]);
    assert_eq!(vec_i64_from(&result), &[0, 1, 2, 3, 4, 5]);

    // Negative Cases
    //
    // Rust considers both `0..0` and `1..0` to be valid, empty ranges.
    let tensor = Tensor::arange_start(0, 4 * 3, opt).view([4, 3]);
    let result = tensor.i(0..0);
    assert_eq!(result.size(), &[0, 3]);
    assert_eq!(vec_i64_from(&result), &[] as &[i64]);

    // Clippy doesn't like explicitly reversed ranges, so let's pretend the bounds were computed.
    let (start, end) = (1, 0);
    let tensor = Tensor::arange_start(0, 4 * 3, opt).view([4, 3]);
    let result = tensor.i(start..end);
    assert_eq!(result.size(), &[0, 3]);
    assert_eq!(vec_i64_from(&result), &[] as &[i64]);
}

#[test]
fn slice_index() {
    let opt = (Kind::Float, Device::Cpu);

    let tensor = Tensor::arange_start(0, 6 * 2, opt).view([6, 2]);
    let index: &[_] = &[1, 3, 5];
    let result = tensor.i(index);
    assert_eq!(result.size(), &[3, 2]);
    assert_eq!(vec_i64_from(&result), &[2, 3, 6, 7, 10, 11]);

    let tensor = Tensor::arange_start(0, 3 * 4, opt).view([3, 4]);
    let index: &[_] = &[3, 0];
    let result = tensor.i((.., index));
    assert_eq!(result.size(), &[3, 2]);
    assert_eq!(vec_i64_from(&result), &[3, 0, 7, 4, 11, 8]);
}

#[test]
fn new_index() {
    let opt = (Kind::Float, Device::Cpu);

    let tensor = Tensor::arange_start(0, 2 * 3, opt).view([2, 3]);
    let result = tensor.i((NewAxis,));
    assert_eq!(result.size(), &[1, 2, 3]);
    assert_eq!(vec_i64_from(&result), &[0, 1, 2, 3, 4, 5]);

    let tensor = Tensor::arange_start(0, 2 * 3, opt).view([2, 3]);
    let result = tensor.i((.., NewAxis));
    assert_eq!(result.size(), &[2, 1, 3]);
    assert_eq!(vec_i64_from(&result), &[0, 1, 2, 3, 4, 5]);

    let tensor = Tensor::arange_start(0, 2 * 3, opt).view([2, 3]);
    let result = tensor.i((.., .., NewAxis));
    assert_eq!(result.size(), &[2, 3, 1]);
    assert_eq!(vec_i64_from(&result), &[0, 1, 2, 3, 4, 5]);
}

#[cfg(target_os = "linux")]
#[test]
fn complex_index() {
    let opt = (Kind::Float, Device::Cpu);

    let tensor = Tensor::arange_start(0, 2 * 3 * 5 * 7, opt).view([2, 3, 5, 7]);
    let result = tensor.i((1, 1..2, vec![2, 3, 0].as_slice(), NewAxis, 3..));
    assert_eq!(result.size(), &[1, 3, 1, 4]);
    assert_eq!(
        vec_i64_from(&result),
        &[157, 158, 159, 160, 164, 165, 166, 167, 143, 144, 145, 146]
    );
}

#[test]
fn index_3d() {
    let values: Vec<i64> = (0..24).collect();
    let tensor = tch::Tensor::from_slice(&values).view((2, 3, 4));
    assert_eq!(vec_i64_from(&tensor.i((0, 0, 0))), &[0]);
    assert_eq!(vec_i64_from(&tensor.i((1, 0, 0))), &[12]);
    assert_eq!(vec_i64_from(&tensor.i((0..2, 0, 0))), &[0, 12]);
}

#[test]
fn tensor_index() {
    let t = Tensor::arange(6, (Kind::Int64, Device::Cpu)).view((2, 3));
    let rows_select = Tensor::from_slice(&[0i64, 1, 0]);
    let column_select = Tensor::from_slice(&[1i64, 2, 2]);

    let selected = t.index(&[Some(rows_select), Some(column_select)]);
    assert_eq!(selected.size(), &[3]);
    assert_eq!(vec_i64_from(&selected), &[1, 5, 2]);
}

#[test]
fn tensor_index2() {
    let t = Tensor::arange(400, (Kind::Int64, Device::Cpu)).view((2, 2, 10, 10));

    let selected = t.index(&[
        Some(Tensor::from(0i64)),
        Some(Tensor::from(1i64)),
        Some(Tensor::from_slice(&[1i64, 2, 3])),
        Some(Tensor::from_slice(&[5i64, 6, 7])),
    ]);
    assert_eq!(selected.size(), &[3]);
    // 115 = 0 * 200 + 1 * 100 + 1 * 10 + 5
    // 126 = 0 * 200 + 1 * 100 + 2 * 10 + 6
    // 137 = 0 * 200 + 1 * 100 + 3 * 10 + 7
    assert_eq!(vec_i64_from(&selected), &[115, 126, 137]);
}

#[test]
fn tensor_multi_index() {
    let t = Tensor::arange(6, (Kind::Int64, Device::Cpu)).view((2, 3));

    let select_1 = Tensor::from_slice(&[0i64, 1, 0]);
    let select_2 = Tensor::from_slice(&[1i64, 0, 0]);
    let select_final = Tensor::stack(&[select_1, select_2], 0);
    assert_eq!(select_final.size(), &[2, 3]);

    let selected = t.index(&[Some(select_final)]); // index only rows
    assert_eq!(selected.size(), &[2, 3, 3]);
    assert_eq!(vec_i64_from(&selected), &[0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 0, 1, 2]);
    // after flattening
}

#[test]
fn tensor_put() {
    let t = Tensor::arange(6, (Kind::Int64, Device::Cpu)).view((2, 3));
    let rows_select = Tensor::from_slice(&[0i64, 1, 0]);
    let column_select = Tensor::from_slice(&[1i64, 2, 2]);
    let values = Tensor::from_slice(&[10i64, 12, 24]);

    let updated = t.index_put(&[Some(rows_select), Some(column_select)], &values, false);
    assert_eq!(vec_i64_from(&updated), &[0i64, 10, 24, 3, 4, 12]); // after flattening
}

#[test]
fn indexing_doc() {
    let tensor = Tensor::from_slice(&[1, 2, 3, 4, 5, 6]).view((2, 3));
    let t = tensor.i(1);
    assert_eq!(vec_i64_from(&t), [4, 5, 6]);
    let t = tensor.i((.., -2));
    assert_eq!(vec_i64_from(&t), [2, 5]);

    let tensor = Tensor::from_slice(&[1, 2, 3, 4, 5, 6]).view((2, 3));
    let t = tensor.i((.., 1..));
    assert_eq!(t.size(), [2, 2]);
    assert_eq!(vec_i64_from(&t.contiguous().view(-1)), [2, 3, 5, 6]);
    let t = tensor.i((..1, ..));
    assert_eq!(t.size(), [1, 3]);
    assert_eq!(vec_i64_from(&t.contiguous().view(-1)), [1, 2, 3]);
    let t = tensor.i((.., 1..2));
    assert_eq!(t.size(), [2, 1]);
    assert_eq!(vec_i64_from(&t.contiguous().view(-1)), [2, 5]);
    let t = tensor.i((.., 1..=2));
    assert_eq!(t.size(), [2, 2]);
    assert_eq!(vec_i64_from(&t.contiguous().view(-1)), [2, 3, 5, 6]);

    let tensor = Tensor::from_slice(&[1, 2, 3, 4, 5, 6]).view((2, 3));
    let t = tensor.i((NewAxis,));
    assert_eq!(t.size(), &[1, 2, 3]);
    let t = tensor.i((.., .., NewAxis));
    assert_eq!(t.size(), &[2, 3, 1]);
}
