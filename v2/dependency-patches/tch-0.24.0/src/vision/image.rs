//! Utility functions to manipulate images.
use crate::wrappers::image::{load_hwc, load_hwc_from_mem, resize_hwc, save_hwc};
use crate::{Device, TchError, Tensor};
use std::io;
use std::path::Path;

pub(crate) fn hwc_to_chw(tensor: &Tensor) -> Tensor {
    tensor.permute([2, 0, 1])
}

pub(crate) fn chw_to_hwc(tensor: &Tensor) -> Tensor {
    tensor.permute([1, 2, 0])
}

/// Loads an image from a file.
///
/// On success returns a tensor of shape [channel, height, width].
pub fn load<T: AsRef<Path>>(path: T) -> Result<Tensor, TchError> {
    let tensor = load_hwc(path)?;
    Ok(hwc_to_chw(&tensor))
}

/// Loads an image from memory.
///
/// On success returns a tensor of shape [channel, height, width].
pub fn load_from_memory(img_data: &[u8]) -> Result<Tensor, TchError> {
    let tensor = load_hwc_from_mem(img_data)?;
    Ok(hwc_to_chw(&tensor))
}

/// Saves an image to a file.
///
/// This expects as input a tensor of shape [channel, height, width].
/// The image format is based on the filename suffix, supported suffixes
/// are jpg, png, tga, and bmp.
/// The tensor input should be of kind UInt8 with values ranging from
/// 0 to 255.
pub fn save<T: AsRef<Path>>(t: &Tensor, path: T) -> Result<(), TchError> {
    let t = t.to_kind(crate::Kind::Uint8);
    match t.size().as_slice() {
        [1, _, _, _] => save_hwc(&chw_to_hwc(&t.squeeze_dim(0)).to_device(Device::Cpu), path),
        [_, _, _] => save_hwc(&chw_to_hwc(&t).to_device(Device::Cpu), path),
        sz => Err(TchError::FileFormat(format!("unexpected size for image tensor {sz:?}"))),
    }
}

/// Resizes an image.
///
/// This expects as input a tensor of shape [channel, height, width] and returns
/// a tensor of shape [channel, out_h, out_w].
pub fn resize(t: &Tensor, out_w: i64, out_h: i64) -> Result<Tensor, TchError> {
    Ok(hwc_to_chw(&resize_hwc(&chw_to_hwc(t), out_w, out_h)?))
}

pub fn resize_preserve_aspect_ratio_hwc(
    t: &Tensor,
    out_w: i64,
    out_h: i64,
) -> Result<Tensor, TchError> {
    let tensor_size = t.size();
    let (w, h) = (tensor_size[0], tensor_size[1]);
    if w * out_h == h * out_w {
        Ok(hwc_to_chw(&resize_hwc(t, out_w, out_h)?))
    } else {
        let (resize_w, resize_h) = {
            let ratio_w = out_w as f64 / w as f64;
            let ratio_h = out_h as f64 / h as f64;
            let ratio = ratio_w.max(ratio_h);
            ((ratio * h as f64) as i64, (ratio * w as f64) as i64)
        };
        let resize_w = i64::max(resize_w, out_w);
        let resize_h = i64::max(resize_h, out_h);
        let t = hwc_to_chw(&resize_hwc(t, resize_w, resize_h)?);
        let t = if resize_w == out_w { t } else { t.f_narrow(2, (resize_w - out_w) / 2, out_w)? };
        let t = if resize_h == out_h { t } else { t.f_narrow(1, (resize_h - out_h) / 2, out_h)? };
        Ok(t)
    }
}

/// Resize an image, preserve the aspect ratio by taking a center crop.
///
/// This expects as input a tensor of shape [channel, height, width] and returns
pub fn resize_preserve_aspect_ratio(
    t: &Tensor,
    out_w: i64,
    out_h: i64,
) -> Result<Tensor, TchError> {
    resize_preserve_aspect_ratio_hwc(&chw_to_hwc(t), out_w, out_h)
}

/// Loads and resize an image, preserve the aspect ratio by taking a center crop.
pub fn load_and_resize<T: AsRef<Path>>(
    path: T,
    out_w: i64,
    out_h: i64,
) -> Result<Tensor, TchError> {
    let tensor = load_hwc(path)?;
    resize_preserve_aspect_ratio_hwc(&tensor, out_w, out_h)
}

/// Loads and resize an image from memory, preserve the aspect ratio by taking a center crop.
pub fn load_and_resize_from_memory(
    img_data: &[u8],
    out_w: i64,
    out_h: i64,
) -> Result<Tensor, TchError> {
    let tensor = load_hwc_from_mem(img_data)?;
    resize_preserve_aspect_ratio_hwc(&tensor, out_w, out_h)
}

fn visit_dirs(dir: &Path, files: &mut Vec<std::fs::DirEntry>) -> Result<(), TchError> {
    if dir.is_dir() {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                visit_dirs(&path, files)?;
            } else if entry
                .file_name()
                .to_str()
                .is_some_and(|s| s.ends_with(".png") || s.ends_with(".jpg"))
            {
                files.push(entry);
            }
        }
    }
    Ok(())
}

/// Loads all the images in a directory.
pub fn load_dir<T: AsRef<Path>>(path: T, out_w: i64, out_h: i64) -> Result<Tensor, TchError> {
    let mut files: Vec<std::fs::DirEntry> = vec![];
    visit_dirs(path.as_ref(), &mut files)?;
    if files.is_empty() {
        return Err(TchError::Io(io::Error::new(
            io::ErrorKind::NotFound,
            format!("no image found in {:?}", path.as_ref(),),
        )));
    }
    let v: Vec<_> = files
        .iter()
        // Silently discard image errors.
        .filter_map(|x| load_and_resize(x.path(), out_w, out_h).ok())
        .collect();
    Ok(Tensor::stack(&v, 0))
}
