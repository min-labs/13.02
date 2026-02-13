// M13 HUB — VPP NODE: AEAD OPERATIONS
// Vectorized AES-256-GCM encrypt/decrypt.
// Processes entire batch at each stage to keep AES-NI pipeline saturated.
//
// Key insight: AES-NI latency is ~7 cycles per round, throughput is ~1 byte/cycle.
// Processing packets one-at-a-time wastes 90% of the pipeline.
// Batch processing (dual-loop with prefetch) saturates the pipeline.
//
// Decrypt path: RxParse → [this node] → ClassifyRoute
// Encrypt path: TunRead → [this node] → TxEnqueue

use ring::aead;
use crate::vpp::{PacketVector, Disposition, NextNode, VECTOR_SIZE};
use crate::protocol::wire::*;
use crate::protocol::peer::PeerTable;
use crate::engine::clock::prefetch_read_l1;

/// Vectorized AEAD decrypt. Processes the entire batch.
/// On success: routes to ClassifyRoute. On failure: routes to Drop.
#[inline]
pub fn aead_decrypt_vector(
    input: &mut PacketVector,
    disp: &mut Disposition,
    peer_table: &PeerTable,
) {
    let n = input.len;
    let mut i = 0;

    // Dual-loop: 4 packets per iteration
    while i + 4 <= n {
        // Prefetch next 4 packet payloads
        if i + 8 <= n {
            for k in 4..8 {
                let idx = i + k;
                if input.descs[idx].addr != 0 {
                    unsafe { prefetch_read_l1(input.descs[idx].addr as *const u8); }
                }
            }
        }

        for j in 0..4 {
            disp.next[i + j] = decrypt_one(&mut input.descs[i + j], peer_table);
        }
        i += 4;
    }

    // Remainder
    while i < n {
        disp.next[i] = decrypt_one(&mut input.descs[i], peer_table);
        i += 1;
    }
}

/// Decrypt a single packet in-place.
#[inline(always)]
fn decrypt_one(
    desc: &mut crate::vpp::PacketDesc,
    peer_table: &PeerTable,
) -> NextNode {
    let peer_idx = desc.peer_idx as usize;
    if peer_idx >= crate::protocol::peer::MAX_PEERS {
        return NextNode::Drop;
    }

    // Get the peer's cached AEAD cipher
    let cipher = match &peer_table.ciphers[peer_idx] {
        Some(c) => c,
        None => return NextNode::Drop, // No session key
    };

    let frame = unsafe {
        std::slice::from_raw_parts_mut(desc.addr as *mut u8, desc.len as usize)
    };
    let m13_off = desc.m13_offset as usize;

    // AEAD nonce direction: Hub receives direction 0x00, our direction is 0x01
    if crate::crypto::aead::open_frame(frame, cipher, 0x01, m13_off) {
        NextNode::ClassifyRoute
    } else {
        NextNode::Drop // AEAD authentication failed
    }
}

/// Vectorized AEAD encrypt. Processes the entire batch.
/// Called on the TUN→wire TX path.
#[inline]
pub fn aead_encrypt_vector(
    input: &mut PacketVector,
    disp: &mut Disposition,
    peer_table: &mut PeerTable,
) {
    let n = input.len;
    let mut i = 0;

    while i + 4 <= n {
        if i + 8 <= n {
            for k in 4..8 {
                let idx = i + k;
                if input.descs[idx].addr != 0 {
                    unsafe { prefetch_read_l1(input.descs[idx].addr as *const u8); }
                }
            }
        }

        for j in 0..4 {
            disp.next[i + j] = encrypt_one(&mut input.descs[i + j], peer_table);
        }
        i += 4;
    }

    while i < n {
        disp.next[i] = encrypt_one(&mut input.descs[i], peer_table);
        i += 1;
    }
}

/// Encrypt a single packet in-place.
#[inline(always)]
fn encrypt_one(
    desc: &mut crate::vpp::PacketDesc,
    peer_table: &mut PeerTable,
) -> NextNode {
    let peer_idx = desc.peer_idx as usize;
    if peer_idx >= crate::protocol::peer::MAX_PEERS {
        return NextNode::Drop;
    }

    let cipher = match &peer_table.ciphers[peer_idx] {
        Some(c) => c,
        None => return NextNode::Drop,
    };

    let seq = peer_table.slots[peer_idx].next_seq();
    let frame = unsafe {
        std::slice::from_raw_parts_mut(desc.addr as *mut u8, desc.len as usize)
    };
    let m13_off = desc.m13_offset as usize;

    // Hub encrypts with direction 0x01
    crate::crypto::aead::seal_frame(frame, cipher, seq, 0x01, m13_off);
    desc.seq_id = seq;
    peer_table.slots[peer_idx].frame_count += 1;

    NextNode::TxEnqueue
}
