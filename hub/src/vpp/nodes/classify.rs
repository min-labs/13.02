// M13 HUB — VPP NODE: CLASSIFY & ROUTE
// Post-decrypt classification. Determines where decrypted frames should go:
// - Tunnel data → TunWrite (inject into TUN interface for kernel routing)
// - Feedback → FeedbackProcess (BBR telemetry)
// - Fragment → reassemble, then re-classify
//
// Dual-loop pattern with prefetch.

use crate::vpp::{PacketVector, Disposition, NextNode};
use crate::protocol::wire::*;
use crate::engine::clock::prefetch_read_l1;

/// Classify decrypted packets and route to next node.
#[inline]
pub fn classify_route(
    input: &PacketVector,
    disp: &mut Disposition,
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
            disp.next[i + j] = classify_one(&input.descs[i + j]);
        }
        i += 4;
    }
    while i < n {
        disp.next[i] = classify_one(&input.descs[i]);
        i += 1;
    }
}

#[inline(always)]
fn classify_one(desc: &crate::vpp::PacketDesc) -> NextNode {
    let flags = desc.flags;

    // After decryption, tunnel data goes to TUN write
    if flags & FLAG_TUNNEL != 0 {
        return NextNode::TunWrite;
    }

    // Control frames after decrypt (e.g., encrypted feedback)
    if flags & FLAG_CONTROL != 0 {
        if flags & FLAG_FEEDBACK != 0 {
            return NextNode::Feedback;
        }
        return NextNode::Drop;
    }

    // Handshake frames post-decrypt
    if flags & FLAG_HANDSHAKE != 0 {
        return NextNode::Handshake;
    }

    // Fragment flag — needs reassembly
    if flags & FLAG_FRAGMENT != 0 {
        return NextNode::Handshake; // Fragments are always handshake in current protocol
    }

    NextNode::Drop
}
