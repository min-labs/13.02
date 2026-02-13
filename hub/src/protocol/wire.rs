// M13 HUB — WIRE PROTOCOL
// Canonical definitions for the M13 on-wire format.
// All header structs are #[repr(C, packed)] for zero-copy cast from UMEM.
// All constants are the single source of truth — no duplication.

use std::mem;
use bytemuck::{Pod, Zeroable};

// ============================================================================
// WIRE CONSTANTS
// ============================================================================

/// IEEE 802.1 Local Experimental EtherType for M13 raw Ethernet frames.
pub const ETH_P_M13: u16 = 0x88B5;
/// Wire protocol magic byte. Stored in M13Header.signature[0].
pub const M13_WIRE_MAGIC: u8 = 0xD1;
/// Wire protocol version. Phase 1 = 0x01. Stored in M13Header.signature[1].
pub const M13_WIRE_VERSION: u8 = 0x01;

// M13 header flags (single-byte bitfield at M13Header.flags)
pub const FLAG_CONTROL: u8   = 0x80;
pub const FLAG_FEEDBACK: u8  = 0x40;
pub const FLAG_TUNNEL: u8    = 0x20;
pub const FLAG_ECN: u8       = 0x10;  // Receiver signals congestion
pub const FLAG_FIN: u8       = 0x08;  // Graceful close signal
// FLAG_FEC (0x04) reserved for RLNC — not yet implemented
pub const FLAG_HANDSHAKE: u8 = 0x02;  // PQC handshake control
pub const FLAG_FRAGMENT: u8  = 0x01;  // Fragmented message

// PQC handshake sub-types (first byte of handshake payload)
pub const HS_CLIENT_HELLO: u8 = 0x01;
pub const HS_SERVER_HELLO: u8 = 0x02;
pub const HS_FINISHED: u8     = 0x03;

// AEAD nonce direction bytes (prevents reflection attacks)
pub const DIR_HUB_TO_NODE: u8 = 0x00;
#[allow(dead_code)]
pub const DIR_NODE_TO_HUB: u8 = 0x01;

// Session limits
pub const REKEY_FRAME_LIMIT: u64 = 1u64 << 32;
pub const REKEY_TIME_LIMIT_NS: u64 = 3_600_000_000_000; // 1 hour

// ============================================================================
// WIRE HEADERS
// ============================================================================

/// IEEE 802.3 Ethernet header. 14 bytes on wire: dst(6) + src(6) + ethertype(2).
#[repr(C, packed)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct EthernetHeader {
    pub dst: [u8; 6],
    pub src: [u8; 6],
    pub ethertype: u16,
}

/// M13 wire protocol header. 48 bytes. Carried after EthernetHeader.
/// signature[0]=magic(0xD1), signature[1]=version(0x01), [2]=encrypted flag.
/// Encrypted region: bytes [32..48] (seq_id, flags, payload_len, padding).
#[repr(C, packed)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct M13Header {
    pub signature: [u8; 32],
    pub seq_id: u64,
    pub flags: u8,
    pub payload_len: u32,
    pub padding: [u8; 3],
}
const _: () = assert!(mem::size_of::<M13Header>() == 48);

/// Feedback frame payload v2. 40 bytes. Carried after M13Header with flags=0xC0.
/// Wire: EthernetHeader(14) + M13Header(48) + FeedbackFrame(40) = 102 bytes.
/// v2 adds loss_count (exact gap count from RxBitmap) and nack_bitmap (64-bit
/// per-packet loss map for RLNC retransmission decisions).
#[repr(C, packed)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct FeedbackFrame {
    pub highest_seq: u64,
    pub rx_timestamp_ns: u64,
    pub delivered: u32,
    pub delivered_time_ns: u64,
    pub loss_count: u32,
    pub nack_bitmap: u64,
}
const _: () = assert!(mem::size_of::<FeedbackFrame>() == 40);

// ============================================================================
// DERIVED CONSTANTS
// ============================================================================

pub const ETH_HDR_SIZE: usize = mem::size_of::<EthernetHeader>();
pub const M13_HDR_SIZE: usize = mem::size_of::<M13Header>();
pub const FEEDBACK_FRAME_LEN: u32 =
    (ETH_HDR_SIZE + M13_HDR_SIZE + mem::size_of::<FeedbackFrame>()) as u32;

// Fragment sub-header. 8 bytes, prepended to payload when FLAG_FRAGMENT set.
#[repr(C, packed)]
#[derive(Copy, Clone)]
pub struct FragHeader {
    pub frag_msg_id: u16,
    pub frag_index: u8,
    pub frag_total: u8,
    pub frag_offset: u16,
    pub frag_len: u16,
}
pub const FRAG_HDR_SIZE: usize = 8;
const _: () = assert!(mem::size_of::<FragHeader>() == FRAG_HDR_SIZE);
