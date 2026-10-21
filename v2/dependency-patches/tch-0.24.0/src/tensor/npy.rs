//! Numpy support for tensors.
//!
//! This module implements the support for reading and writing `.npy` and `.npz` files. The file
//! format spec can be found at:
//! <https://docs.scipy.org/doc/numpy-1.14.2/neps/npy-format.html>.
use crate::{Kind, TchError, Tensor};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::Path;

const NPY_MAGIC_STRING: &[u8] = b"\x93NUMPY";
const NPY_SUFFIX: &str = ".npy";

fn read_header<R: Read>(reader: &mut R) -> Result<String, TchError> {
    let mut magic_string = vec![0u8; NPY_MAGIC_STRING.len()];
    reader.read_exact(&mut magic_string)?;
    if magic_string != NPY_MAGIC_STRING {
        return Err(TchError::FileFormat("magic string mismatch".to_string()));
    }
    let mut version = [0u8; 2];
    reader.read_exact(&mut version)?;
    let header_len_len = match version[0] {
        1 => 2,
        2 => 4,
        otherwise => return Err(TchError::FileFormat(format!("unsupported version {otherwise}"))),
    };
    let mut header_len = vec![0u8; header_len_len];
    reader.read_exact(&mut header_len)?;
    let header_len = header_len.iter().rev().fold(0_usize, |acc, &v| 256 * acc + v as usize);
    let mut header = vec![0u8; header_len];
    reader.read_exact(&mut header)?;
    Ok(String::from_utf8_lossy(&header).to_string())
}

#[derive(Debug, PartialEq)]
struct Header {
    descr: Kind,
    fortran_order: bool,
    shape: Vec<i64>,
}

impl Header {
    fn to_string(&self) -> Result<String, TchError> {
        let fortran_order = if self.fortran_order { "True" } else { "False" };
        let mut shape = self.shape.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
        let descr = match self.descr {
            Kind::Half => "f2",
            Kind::Float => "f4",
            Kind::Double => "f8",
            Kind::Int => "i4",
            Kind::Int64 => "i8",
            Kind::Int16 => "i2",
            Kind::Int8 => "i1",
            Kind::Uint8 => "u1",
            descr => return Err(TchError::FileFormat(format!("unsupported kind {descr:?}"))),
        };
        if !shape.is_empty() {
            shape.push(',')
        }
        Ok(format!(
            "{{'descr': '<{descr}', 'fortran_order': {fortran_order}, 'shape': ({shape}), }}"
        ))
    }

    // Hacky parser for the npy header, a typical example would be:
    // {'descr': '<f8', 'fortran_order': False, 'shape': (128,), }
    fn parse(header: &str) -> Result<Header, TchError> {
        let header =
            header.trim_matches(|c: char| c == '{' || c == '}' || c == ',' || c.is_whitespace());

        let mut parts: Vec<String> = vec![];
        let mut start_index = 0usize;
        let mut cnt_parenthesis = 0i64;
        for (index, c) in header.char_indices() {
            match c {
                '(' => cnt_parenthesis += 1,
                ')' => cnt_parenthesis -= 1,
                ',' => {
                    if cnt_parenthesis == 0 {
                        parts.push(header[start_index..index].to_owned());
                        start_index = index + 1;
                    }
                }
                _ => {}
            }
        }
        parts.push(header[start_index..].to_owned());
        let mut part_map: HashMap<String, String> = HashMap::new();
        for part in parts.iter() {
            let part = part.trim();
            if !part.is_empty() {
                match part.split(':').collect::<Vec<_>>().as_slice() {
                    [key, value] => {
                        let key = key.trim_matches(|c: char| c == '\'' || c.is_whitespace());
                        let value = value.trim_matches(|c: char| c == '\'' || c.is_whitespace());
                        let _ = part_map.insert(key.to_owned(), value.to_owned());
                    }
                    _ => {
                        return Err(TchError::FileFormat(format!(
                            "unable to parse header {header}"
                        )))
                    }
                }
            }
        }
        let fortran_order = match part_map.get("fortran_order") {
            None => false,
            Some(fortran_order) => match fortran_order.as_ref() {
                "False" => false,
                "True" => true,
                _ => {
                    return Err(TchError::FileFormat(format!(
                        "unknown fortran_order {fortran_order}"
                    )))
                }
            },
        };
        let descr = match part_map.get("descr") {
            None => return Err(TchError::FileFormat("no descr in header".to_string())),
            Some(descr) => {
                if descr.is_empty() {
                    return Err(TchError::FileFormat("empty descr".to_string()));
                }
                if descr.starts_with('>') {
                    return Err(TchError::FileFormat(format!("little-endian descr {descr}")));
                }
                // the only supported types in tensor are:
                //     float64, float32, float16,
                //     complex64, complex128,
                //     int64, int32, int16, int8,
                //     uint8, and bool.
                match descr.trim_matches(|c: char| c == '=' || c == '<' || c == '|') {
                    "e" | "f2" => Kind::Half,
                    "f" | "f4" => Kind::Float,
                    "d" | "f8" => Kind::Double,
                    "i" | "i4" => Kind::Int,
                    "q" | "i8" => Kind::Int64,
                    "h" | "i2" => Kind::Int16,
                    "b" | "i1" => Kind::Int8,
                    "B" | "u1" => Kind::Uint8,
                    "?" | "b1" => Kind::Bool,
                    "F" | "F4" | "c8" => Kind::ComplexFloat,
                    "D" | "F8" | "c16" => Kind::ComplexDouble,
                    descr => {
                        return Err(TchError::FileFormat(format!("unrecognized descr {descr}")))
                    }
                }
            }
        };
        let shape = match part_map.get("shape") {
            None => return Err(TchError::FileFormat("no shape in header".to_string())),
            Some(shape) => {
                let shape = shape.trim_matches(|c: char| c == '(' || c == ')' || c == ',');
                if shape.is_empty() {
                    vec![]
                } else {
                    shape
                        .split(',')
                        .map(|v| v.trim().parse::<i64>())
                        .collect::<Result<Vec<_>, _>>()?
                }
            }
        };
        Ok(Header { descr, fortran_order, shape })
    }
}

impl crate::Tensor {
    /// Reads a npy file and return the stored tensor.
    pub fn read_npy<T: AsRef<Path>>(path: T) -> Result<Tensor, TchError> {
        let mut reader = File::open(path.as_ref())?;
        let header = read_header(&mut reader)?;
        let header = Header::parse(&header)?;
        if header.fortran_order {
            return Err(TchError::FileFormat("fortran order not supported".to_string()));
        }
        let mut data: Vec<u8> = vec![];
        reader.read_to_end(&mut data)?;
        Tensor::f_from_data_size(&data, &header.shape, header.descr)
    }

    /// Reads a npz file and returns some named tensors.
    pub fn read_npz<T: AsRef<Path>>(path: T) -> Result<Vec<(String, Tensor)>, TchError> {
        let zip_reader = BufReader::new(File::open(path.as_ref())?);
        let mut zip = zip::ZipArchive::new(zip_reader)?;
        let mut result = vec![];
        for i in 0..zip.len() {
            let mut reader = zip.by_index(i).unwrap();
            let name = {
                let name = reader.name()?;
                name.strip_suffix(NPY_SUFFIX).unwrap_or(&name).to_owned()
            };
            let header = read_header(&mut reader)?;
            let header = Header::parse(&header)?;
            if header.fortran_order {
                return Err(TchError::FileFormat("fortran order not supported".to_string()));
            }
            let mut data: Vec<u8> = vec![];
            reader.read_to_end(&mut data)?;
            let tensor = Tensor::f_from_data_size(&data, &header.shape, header.descr)?;
            result.push((name, tensor))
        }
        Ok(result)
    }

    fn write<T: Write>(&self, f: &mut T) -> Result<(), TchError> {
        f.write_all(NPY_MAGIC_STRING)?;
        f.write_all(&[1u8, 0u8])?;
        let kind = self.f_kind()?;
        let header = Header { descr: kind, fortran_order: false, shape: self.size() };
        let mut header = header.to_string()?;
        let pad = 16 - (NPY_MAGIC_STRING.len() + 5 + header.len()) % 16;
        for _ in 0..pad % 16 {
            header.push(' ')
        }
        header.push('\n');
        f.write_all(&[(header.len() % 256) as u8, (header.len() / 256) as u8])?;
        f.write_all(header.as_bytes())?;
        let numel = self.numel();
        let mut content = vec![0u8; numel * kind.elt_size_in_bytes()];
        self.f_copy_data_u8(&mut content, numel)?;
        f.write_all(&content)?;
        Ok(())
    }

    /// Writes a tensor in the npy format so that it can be read using python.
    pub fn write_npy<T: AsRef<Path>>(&self, path: T) -> Result<(), TchError> {
        let mut f = File::create(path.as_ref())?;
        self.write(&mut f)
    }

    pub fn write_npz<S: AsRef<str>, T: AsRef<Tensor>, P: AsRef<Path>>(
        ts: &[(S, T)],
        path: P,
    ) -> Result<(), TchError> {
        let mut zip = zip::ZipWriter::new(File::create(path.as_ref())?);
        let options =
            zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);

        for (name, tensor) in ts.iter() {
            zip.start_file(format!("{}.npy", name.as_ref()), options)?;
            tensor.as_ref().write(&mut zip)?
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::Header;

    #[test]
    fn parse() {
        let h = "{'descr': '<f8', 'fortran_order': False, 'shape': (128,), }";
        assert_eq!(
            Header::parse(h).unwrap(),
            Header { descr: crate::Kind::Double, fortran_order: false, shape: vec![128] }
        );
        let h = "{'descr': '<f4', 'fortran_order': True, 'shape': (256,1,128), }";
        let h = Header::parse(h).unwrap();
        assert_eq!(
            h,
            Header { descr: crate::Kind::Float, fortran_order: true, shape: vec![256, 1, 128] }
        );
        assert_eq!(
            h.to_string().unwrap(),
            "{'descr': '<f4', 'fortran_order': True, 'shape': (256,1,128,), }"
        );

        let h = Header { descr: crate::Kind::Int64, fortran_order: false, shape: vec![] };
        assert_eq!(
            h.to_string().unwrap(),
            "{'descr': '<i8', 'fortran_order': False, 'shape': (), }"
        );
    }
}
