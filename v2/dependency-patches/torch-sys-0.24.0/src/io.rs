use libc::{c_void, size_t};
use std::io::{Read, Seek, SeekFrom, Write};

pub trait ReadStream: Read + Seek {}

#[no_mangle]
extern "C" fn tch_write_stream_destructor(stream_ptr: *mut c_void) {
    unsafe {
        let boxed_stream: Box<Box<dyn Write>> = Box::from_raw(stream_ptr as *mut Box<dyn Write>);
        drop(boxed_stream)
    }
}

#[no_mangle]
extern "C" fn tch_write_stream_write(
    stream_ptr: *mut c_void,
    buf: *const u8,
    size: size_t,
    out_size: *mut size_t,
) -> bool {
    unsafe {
        let boxed_stream = stream_ptr as *mut Box<dyn Write>;
        let buffer = std::slice::from_raw_parts(buf, size);
        match (*boxed_stream).write(buffer) {
            Ok(x) => {
                *out_size = x;
                true
            }
            Err(_) => false,
        }
    }
}

#[no_mangle]
extern "C" fn tch_read_stream_destructor(stream_ptr: *mut c_void) {
    unsafe {
        let boxed_stream: Box<Box<dyn ReadStream>> =
            Box::from_raw(stream_ptr as *mut Box<dyn ReadStream>);
        drop(boxed_stream)
    }
}

#[no_mangle]
extern "C" fn tch_read_stream_stream_position(stream_ptr: *mut c_void, current: *mut u64) -> bool {
    unsafe {
        let boxed_stream = stream_ptr as *mut Box<dyn ReadStream>;
        match (*boxed_stream).stream_position() {
            Ok(ret) => {
                *current = ret;
                true
            }
            Err(_) => false,
        }
    }
}

#[no_mangle]
extern "C" fn tch_read_stream_seek_start(
    stream_ptr: *mut c_void,
    pos: u64,
    new_pos: *mut u64,
) -> bool {
    unsafe {
        let boxed_stream = stream_ptr as *mut Box<dyn ReadStream>;
        match (*boxed_stream).seek(SeekFrom::Start(pos)) {
            Ok(ret) => {
                *new_pos = ret;
                true
            }
            Err(_) => false,
        }
    }
}

#[no_mangle]
extern "C" fn tch_read_stream_seek_end(
    stream_ptr: *mut c_void,
    pos: i64,
    new_pos: *mut u64,
) -> bool {
    unsafe {
        let boxed_stream = stream_ptr as *mut Box<dyn ReadStream>;
        match (*boxed_stream).seek(SeekFrom::End(pos)) {
            Ok(ret) => {
                *new_pos = ret;
                true
            }
            Err(_) => false,
        }
    }
}

#[no_mangle]
extern "C" fn tch_read_stream_read(
    stream_ptr: *mut c_void,
    buf: *mut u8,
    size: size_t,
    read_size: *mut size_t,
) -> bool {
    unsafe {
        let boxed_stream = stream_ptr as *mut Box<dyn ReadStream>;
        let buffer = std::slice::from_raw_parts_mut(buf, size);
        match (*boxed_stream).read(buffer) {
            Ok(ret) => {
                *read_size = ret;
                true
            }
            Err(_) => false,
        }
    }
}
