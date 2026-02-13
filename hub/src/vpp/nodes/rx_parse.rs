// M13 HUB — VPP NODE: RX PARSE
// First node in the RX graph. Receives raw AF_XDP frames, validates M13 headers,
// extracts metadata into PacketDesc, and classifies into branches.
//
// This node performs:
// 1. Magic/version validation
// 2. M13 header extraction (flags, seq_id, payload_len)
// 3. Peer lookup (creates PeerAddr from src_ip:src_port or src_mac)
// 4. Route to next node based on flags:
//    - FLAG_CONTROL|FLAG_FEEDBACK → Feedback
//    - FLAG_HANDSHAKE → Handshake
//    - FLAG_TUNNEL (encrypted) → AeadDecrypt
//    - FLAG_TUNNEL (cleartext) → ClassifyRoute
//    - else → Drop
//
// Dual-loop pattern: prefetch next 4 packets while processing current 4.

use crate::vpp::{PacketVector, PacketDesc, Disposition, NextNode};
use crate::protocol::wire::*;
use crate::protocol::peer::{PeerAddr, PeerTable};
use crate::engine::clock::{prefetch_read_l1, TscCal, rdtsc_ns};

/// Context for RX parse node. Passed by the graph executor.
pub struct RxParseCtx<'a> {
    pub peer_table: &'a mut PeerTable,
    pub cal: &'a TscCal,
    /// If true, frames are wrapped in UDP (hub mode). If false, raw L2 (air-gapped).
    pub udp_mode: bool,
}

/// Parse a vector of raw AF_XDP frames.
/// Fills `out` vector with parsed PacketDesc and `disp` with routing decisions.
#[inline]
pub fn rx_parse(
    input: &PacketVector,
    disp: &mut Disposition,
    ctx: &mut RxParseCtx<'_>,
) {
    let now_ns = rdtsc_ns(ctx.cal);
    let n = input.len;
    let mut i = 0;

    // ---- Dual-loop: main body (4 packets per iteration) ----
    while i + 4 <= n {
        // Prefetch next 4 packet data from UMEM
        if i + 8 <= n {
            for k in 4..8 {
                let idx = i + k;
                if input.descs[idx].addr != 0 {
                    unsafe { prefetch_read_l1(input.descs[idx].addr as *const u8); }
                }
            }
        }

        // Process current 4
        for j in 0..4 {
            disp.next[i + j] = parse_one(&input.descs[i + j], ctx, now_ns);
        }
        i += 4;
    }

    // ---- Remainder loop ----
    while i < n {
        disp.next[i] = parse_one(&input.descs[i], ctx, now_ns);
        i += 1;
    }
}

/// Parse a single packet. Returns the next node for this packet.
#[inline(always)]
fn parse_one(
    desc: &PacketDesc,
    _ctx: &mut RxParseCtx<'_>,
    _now_ns: u64,
) -> NextNode {
    if desc.len < (ETH_HDR_SIZE + M13_HDR_SIZE) as u32 {
        return NextNode::Drop; // Runt frame
    }

    let frame = unsafe { std::slice::from_raw_parts(desc.addr as *const u8, desc.len as usize) };
    let m13_off = desc.m13_offset as usize;

    // Validate magic and version
    if frame.len() < m13_off + M13_HDR_SIZE { return NextNode::Drop; }
    if frame[m13_off] != M13_WIRE_MAGIC || frame[m13_off + 1] != M13_WIRE_VERSION {
        return NextNode::Drop;
    }

    // Extract M13 header fields
    let flags = frame[m13_off + 40]; // offset 40 within M13Header = byte 54 in ETH+M13
    let _seq_bytes = &frame[m13_off + 32..m13_off + 40];

    // Route based on flags
    if flags & FLAG_CONTROL != 0 {
        if flags & FLAG_FEEDBACK != 0 {
            return NextNode::Feedback;
        }
        if flags & FLAG_HANDSHAKE != 0 {
            return NextNode::Handshake;
        }
        return NextNode::Drop; // Unknown control frame
    }

    if flags & FLAG_TUNNEL != 0 {
        // Data frame: check if encrypted (signature[2] == 0x01)
        let crypto_ver = if m13_off + 2 < frame.len() { frame[m13_off + 2] } else { 0 };
        if crypto_ver == 0x01 {
            return NextNode::AeadDecrypt;
        } else {
            return NextNode::ClassifyRoute; // Cleartext tunnel data
        }
    }

    if flags & FLAG_HANDSHAKE != 0 {
        return NextNode::Handshake;
    }

    NextNode::Drop // Unknown frame type
}
