use std::io::{Read, Result, Seek, SeekFrom};
use torch_sys::io::ReadStream;

pub struct ReadSeekAdapter<T> {
    inner: T,
}

impl<T: Read + Seek> ReadSeekAdapter<T> {
    pub fn new(inner: T) -> Self {
        Self { inner }
    }
}

impl<T: Read> Read for ReadSeekAdapter<T> {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        self.inner.read(buf)
    }
}

impl<T: Seek> Seek for ReadSeekAdapter<T> {
    fn seek(&mut self, pos: SeekFrom) -> Result<u64> {
        self.inner.seek(pos)
    }
}

impl<T: Read + Seek> ReadStream for ReadSeekAdapter<T> {}
