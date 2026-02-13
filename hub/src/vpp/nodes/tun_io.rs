// M13 HUB — VPP NODE: TUN I/O
// Handles reading from and writing to the TUN interface.
//
// TunWrite: takes decrypted, classified packets and injects their payload
//           into the TUN device for kernel IP routing.
//
// TunRead:  reads IP packets from TUN (kernel → userspace), wraps them in
//           M13 frames, and routes to AeadEncrypt for wire transmission.
//
// Both paths use vectorized processing with dual-loop prefetch.

use crate::vpp::{PacketVector, PacketDesc, Disposition, NextNode};
use crate::protocol::wire::*;
use crate::engine::clock::prefetch_read_l1;

/// Write decrypted tunnel packets to the TUN fd.
/// Each packet's payload (after M13 header) is an IP packet that goes into the TUN.
#[inline]
pub fn tun_write_vector(
    input: &PacketVector,
    disp: &mut Disposition,
    tun_fd: i32,
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
            disp.next[i + j] = write_one_tun(&input.descs[i + j], tun_fd);
        }
        i += 4;
    }
    while i < n {
        disp.next[i] = write_one_tun(&input.descs[i], tun_fd);
        i += 1;
    }
}

#[inline(always)]
fn write_one_tun(desc: &PacketDesc, tun_fd: i32) -> NextNode {
    let m13_off = desc.m13_offset as usize;
    let payload_start = m13_off + M13_HDR_SIZE;
    let payload_len = desc.payload_len as usize;

    if payload_len == 0 || payload_start + payload_len > desc.len as usize {
        return NextNode::Consumed; // Invalid or empty payload
    }

    let payload_ptr = unsafe { (desc.addr as *const u8).add(payload_start) };
    let written = unsafe {
        libc::write(tun_fd, payload_ptr as *const libc::c_void, payload_len)
    };

    if written < 0 {
        // EAGAIN/EWOULDBLOCK is expected under pressure — not an error
        return NextNode::Consumed;
    }

    NextNode::Consumed // Packet delivered to kernel
}

/// Read IP packets from TUN and build M13 frames for wire TX.
/// Returns the number of packets read and placed into `output`.
#[inline]
pub fn tun_read_batch(
    output: &mut PacketVector,
    tun_fd: i32,
    frame_alloc: &mut dyn FnMut() -> Option<(u64, *mut u8)>,
    src_mac: &[u8; 6],
    _now_ns: u64,
) -> usize {
    let mut count = 0;

    while !output.is_full() {
        // Allocate a UMEM frame for this packet
        let (addr, frame_ptr) = match frame_alloc() {
            Some(f) => f,
            None => break, // No free frames
        };

        let payload_ptr = unsafe { frame_ptr.add(ETH_HDR_SIZE + M13_HDR_SIZE) };
        let max_payload = 4096 - ETH_HDR_SIZE - M13_HDR_SIZE;

        let n = unsafe {
            libc::read(tun_fd, payload_ptr as *mut libc::c_void, max_payload)
        };

        if n <= 0 {
            // Return the frame — nothing to read
            // Caller must handle frame deallocation
            break;
        }

        let payload_len = n as usize;
        let frame_len = ETH_HDR_SIZE + M13_HDR_SIZE + payload_len;

        // Build Ethernet header
        let frame = unsafe { std::slice::from_raw_parts_mut(frame_ptr, frame_len) };
        frame[0..6].copy_from_slice(&[0xFF; 6]); // dst: broadcast (will be overwritten)
        frame[6..12].copy_from_slice(src_mac);
        frame[12] = (ETH_P_M13 >> 8) as u8;
        frame[13] = (ETH_P_M13 & 0xFF) as u8;

        // Build M13 header
        frame[14] = M13_WIRE_MAGIC;
        frame[15] = M13_WIRE_VERSION;
        frame[54] = FLAG_TUNNEL;
        frame[55..59].copy_from_slice(&(payload_len as u32).to_le_bytes());

        let mut desc = PacketDesc::EMPTY;
        desc.addr = addr;
        desc.len = frame_len as u32;
        desc.m13_offset = ETH_HDR_SIZE as u16;
        desc.flags = FLAG_TUNNEL;
        desc.payload_len = payload_len as u32;

        // Extract destination IP for peer lookup (first 20 bytes = IP header)
        if payload_len >= 20 {
            let ip_hdr = unsafe { std::slice::from_raw_parts(payload_ptr, 20) };
            desc.src_ip.copy_from_slice(&ip_hdr[16..20]); // dst IP = peer's tunnel IP
        }

        output.push(desc);
        count += 1;
    }

    count
}
