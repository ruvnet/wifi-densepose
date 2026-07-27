//! Bounded HAP TLV8 primitives and fail-closed pairing responses.

use std::collections::BTreeMap;

use crate::error::HapError;

pub const TLV_METHOD: u8 = 0x00;
pub const TLV_IDENTIFIER: u8 = 0x01;
pub const TLV_PUBLIC_KEY: u8 = 0x03;
pub const TLV_STATE: u8 = 0x06;
pub const TLV_ERROR: u8 = 0x07;
pub const TLV_SIGNATURE: u8 = 0x0a;

pub const TLV_ERROR_AUTHENTICATION: u8 = 0x02;
pub const TLV_ERROR_UNAVAILABLE: u8 = 0x06;
pub const TLV_ERROR_BUSY: u8 = 0x07;

const MAX_TLV_BYTES: usize = 4096;
const MAX_TLV_TYPES: usize = 32;

/// Parsed TLV8 values. Repeated fragments of a type are concatenated.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Tlv8 {
    values: BTreeMap<u8, Vec<u8>>,
}

impl Tlv8 {
    pub fn parse(input: &[u8]) -> Result<Self, HapError> {
        if input.len() > MAX_TLV_BYTES {
            return Err(HapError::Protocol("TLV8 body exceeds 4096 bytes".into()));
        }
        let mut values: BTreeMap<u8, Vec<u8>> = BTreeMap::new();
        let mut offset = 0usize;
        while offset < input.len() {
            if input.len() - offset < 2 {
                return Err(HapError::Protocol("truncated TLV8 header".into()));
            }
            let kind = input[offset];
            let len = input[offset + 1] as usize;
            offset += 2;
            let end = offset
                .checked_add(len)
                .filter(|end| *end <= input.len())
                .ok_or_else(|| HapError::Protocol("truncated TLV8 value".into()))?;
            if !values.contains_key(&kind) && values.len() == MAX_TLV_TYPES {
                return Err(HapError::Protocol("too many TLV8 types".into()));
            }
            values
                .entry(kind)
                .or_default()
                .extend_from_slice(&input[offset..end]);
            offset = end;
        }
        Ok(Self { values })
    }

    pub fn get(&self, kind: u8) -> Option<&[u8]> {
        self.values.get(&kind).map(Vec::as_slice)
    }

    pub fn byte(&self, kind: u8) -> Option<u8> {
        let value = self.get(kind)?;
        (value.len() == 1).then_some(value[0])
    }

    pub fn insert(&mut self, kind: u8, value: impl Into<Vec<u8>>) {
        self.values.insert(kind, value.into());
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut encoded = Vec::new();
        for (&kind, value) in &self.values {
            if value.is_empty() {
                encoded.extend_from_slice(&[kind, 0]);
                continue;
            }
            for chunk in value.chunks(u8::MAX as usize) {
                encoded.push(kind);
                encoded.push(chunk.len() as u8);
                encoded.extend_from_slice(chunk);
            }
        }
        encoded
    }
}

/// Standards-shaped response used while cryptographic pairing phases are
/// unavailable. It is a real TLV8 error, never a success-shaped placeholder.
pub fn pairing_unavailable_response(request: &[u8]) -> Result<Vec<u8>, HapError> {
    let request = Tlv8::parse(request)?;
    let request_state = request
        .byte(TLV_STATE)
        .ok_or_else(|| HapError::Protocol("pairing request lacks one-byte State".into()))?;
    let response_state = request_state.saturating_add(1);
    let mut response = Tlv8::default();
    response.insert(TLV_STATE, vec![response_state]);
    response.insert(TLV_ERROR, vec![TLV_ERROR_UNAVAILABLE]);
    Ok(response.encode())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tlv8_roundtrip_supports_fragmented_values() {
        let mut tlv = Tlv8::default();
        tlv.insert(TLV_PUBLIC_KEY, vec![3; 300]);
        tlv.insert(TLV_STATE, vec![1]);
        let encoded = tlv.encode();
        let decoded = Tlv8::parse(&encoded).unwrap();
        assert_eq!(decoded, tlv);
    }

    #[test]
    fn malformed_or_oversized_tlv_is_rejected() {
        assert!(Tlv8::parse(&[TLV_STATE]).is_err());
        assert!(Tlv8::parse(&[TLV_STATE, 2, 1]).is_err());
        assert!(Tlv8::parse(&vec![0; MAX_TLV_BYTES + 1]).is_err());
    }

    #[test]
    fn unavailable_pairing_is_an_explicit_error_not_success() {
        let response = pairing_unavailable_response(&[TLV_STATE, 1, 1]).unwrap();
        let decoded = Tlv8::parse(&response).unwrap();
        assert_eq!(decoded.byte(TLV_STATE), Some(2));
        assert_eq!(decoded.byte(TLV_ERROR), Some(TLV_ERROR_UNAVAILABLE));
    }
}
