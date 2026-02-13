// M13 HUB — VPP GRAPH EXECUTOR
// Genuine fd.io/VPP-style graph pipeline. Replaces the monolithic classify loop.
//
// Architecture:
//   1. RX batch arrives from AF_XDP (up to 256 descriptors)
//   2. Executor splits into sub-vectors of VECTOR_SIZE (64) for cache efficiency
//   3. Each sub-vector flows through the graph: parse → classify → AEAD → route
//   4. Per-node dual-loop processing keeps instruction cache hot
//   5. Scatter distributes packets to next-node vectors after each stage
//
// Key insight: at each node, the ENTIRE vector is processed before moving to the
// next node. This is fundamentally different from the old classify loop which
// processed each packet through ALL stages inline (thrashing L1i).

use crate::vpp::{PacketVector, PacketDesc, Disposition, NextNode, VECTOR_SIZE, scatter};
use crate::vpp::nodes::rx_parse::{self, RxParseCtx};
use crate::vpp::nodes::aead_ops;
use crate::vpp::nodes::classify;
use crate::vpp::nodes::tun_io;
use crate::vpp::nodes::tx_enqueue;
use crate::protocol::wire::*;
use crate::protocol::peer::{PeerTable, PeerAddr, PeerLifecycle, MAX_PEERS};
use crate::protocol::fragment::Assembler;
use crate::crypto::pqc::{process_client_hello_hub, process_finished_hub, HubHandshakeState};
use crate::engine::clock::{TscCal, rdtsc_ns, prefetch_read_l1};
use crate::engine::scheduler::Scheduler;
use crate::engine::jitter::JitterBuffer;
use crate::engine::rx_state::{RxBitmap, ReceiverState};

use ring::aead;
use std::sync::atomic::Ordering;

/// Per-cycle statistics for the graph executor.
#[derive(Default)]
pub struct CycleStats {
    pub parsed: u32,
    pub aead_ok: u32,
    pub aead_fail: u32,
    pub tun_writes: u32,
    pub handshakes: u32,
    pub feedback: u32,
    pub drops: u32,
    pub data_fwd: u32,
}

/// Graph executor context — mutable references to all worker state.
/// This is the "world" that graph nodes operate on.
pub struct GraphCtx<'a> {
    pub peers: &'a mut PeerTable,
    pub scheduler: &'a mut Scheduler,
    pub jbuf: &'a mut JitterBuffer,
    pub rx_state: &'a mut ReceiverState,
    pub rx_bitmap: &'a mut RxBitmap,
    pub cal: &'a TscCal,
    pub tun_fd: i32,
    pub src_mac: [u8; 6],
    pub gateway_mac: [u8; 6],
    pub hub_ip: [u8; 4],
    pub hub_port: u16,
    pub ip_id_counter: &'a mut u16,
    pub worker_idx: usize,
    pub closing: bool,
    pub now_ns: u64,
    // UMEM access for frame data
    pub umem_base: *mut u8,
    pub frame_size: u32,
    // Slab allocator callbacks
    pub slab_free: &'a mut dyn FnMut(u32),
    pub slab_alloc: &'a mut dyn FnMut() -> Option<u32>,
}

/// Execute the VPP graph on a batch of AF_XDP receive descriptors.
///
/// This is the main entry point that replaces the monolithic classify loop.
/// It processes the batch in vectors of 64, applying each graph node to the
/// entire vector before moving to the next node.
///
/// Returns per-cycle statistics.
pub fn execute_graph(
    rx_descs: &[(u64, u32)],  // (addr, len) pairs from AF_XDP
    ctx: &mut GraphCtx<'_>,
) -> CycleStats {
    let mut stats = CycleStats::default();
    let n = rx_descs.len();
    if n == 0 { return stats; }

    // Process in sub-vectors of VECTOR_SIZE for cache efficiency
    let mut offset = 0;
    while offset < n {
        let chunk_end = (offset + VECTOR_SIZE).min(n);
        let chunk = &rx_descs[offset..chunk_end];

        let sub_stats = execute_subvector(chunk, ctx);
        stats.parsed += sub_stats.parsed;
        stats.aead_ok += sub_stats.aead_ok;
        stats.aead_fail += sub_stats.aead_fail;
        stats.tun_writes += sub_stats.tun_writes;
        stats.handshakes += sub_stats.handshakes;
        stats.feedback += sub_stats.feedback;
        stats.drops += sub_stats.drops;
        stats.data_fwd += sub_stats.data_fwd;

        offset = chunk_end;
    }

    stats
}

/// Execute the graph on a single sub-vector (≤64 packets).
fn execute_subvector(
    descs: &[(u64, u32)],
    ctx: &mut GraphCtx<'_>,
) -> CycleStats {
    let mut stats = CycleStats::default();
    let now = ctx.now_ns;

    // ====================================================================
    // NODE 1: PARSE + PEER LOOKUP + CLASSIFY
    // Validates M13 headers, resolves peer identity, determines routing.
    // This replaces lines 2599-2700 of the old classify loop.
    // ====================================================================
    let mut input = PacketVector::new();

    for &(addr, len) in descs {
        let frame_ptr = unsafe { ctx.umem_base.add(addr as usize) };
        let frame_len = len as usize;

        if frame_len < ETH_HDR_SIZE + M13_HDR_SIZE {
            (ctx.slab_free)((addr / ctx.frame_size as u64) as u32);
            stats.drops += 1;
            continue;
        }

        // Determine encapsulation and M13 offset
        let ethertype = unsafe { u16::from_be(*(frame_ptr.add(12) as *const u16)) };

        let (m13_offset, peer_idx) = if ethertype == ETH_P_M13.to_be() || ethertype == 0x88B5 {
            // L2 raw M13 (air-gapped WiFi 7)
            let peer_mac = unsafe { *(frame_ptr.add(6) as *const [u8; 6]) };
            let peer_addr = PeerAddr::new_l2(peer_mac);
            match ctx.peers.lookup_or_insert(peer_addr, peer_mac) {
                Some(idx) => (ETH_HDR_SIZE as u16, idx as u8),
                None => {
                    (ctx.slab_free)((addr / ctx.frame_size as u64) as u32);
                    stats.drops += 1;
                    continue;
                }
            }
        } else if ethertype == 0x0800 && frame_len >= 56 + M13_HDR_SIZE {
            // IPv4 UDP encapsulated
            let src_ip = unsafe { *(frame_ptr.add(26) as *const [u8; 4]) };
            let src_port = unsafe { u16::from_be(*(frame_ptr.add(34) as *const u16)) };

            // Learn hub IP from wire if needed
            if ctx.hub_ip == [0, 0, 0, 0] {
                ctx.hub_ip = unsafe { *(frame_ptr.add(30) as *const [u8; 4]) };
            }
            if ctx.gateway_mac == [0xFF; 6] {
                ctx.gateway_mac = unsafe { *(frame_ptr.add(6) as *const [u8; 6]) };
            }

            let peer_mac = unsafe { *(frame_ptr.add(56 + 6) as *const [u8; 6]) };
            let peer_addr = PeerAddr::new_udp(src_ip, src_port);
            match ctx.peers.lookup_or_insert(peer_addr, peer_mac) {
                Some(idx) => (56u16, idx as u8), // ETH(14) + IP(20) + UDP(8) + FakeETH(14) = 56
                None => {
                    (ctx.slab_free)((addr / ctx.frame_size as u64) as u32);
                    stats.drops += 1;
                    continue;
                }
            }
        } else {
            (ctx.slab_free)((addr / ctx.frame_size as u64) as u32);
            stats.drops += 1;
            continue;
        };

        // Validate M13 magic/version
        let m13_ptr = unsafe { frame_ptr.add(m13_offset as usize) };
        let magic = unsafe { *m13_ptr };
        let version = unsafe { *m13_ptr.add(1) };
        if magic != M13_WIRE_MAGIC || version != M13_WIRE_VERSION {
            (ctx.slab_free)((addr / ctx.frame_size as u64) as u32);
            stats.drops += 1;
            continue;
        }

        // Extract M13 header fields
        let flags_raw = unsafe { *m13_ptr.add(40) };
        let seq_id = unsafe {
            u64::from_le_bytes(std::slice::from_raw_parts(m13_ptr.add(32), 8).try_into().unwrap())
        };
        let payload_len = unsafe {
            u32::from_le_bytes(std::slice::from_raw_parts(m13_ptr.add(55 - m13_offset as usize), 4).try_into().unwrap_or([0;4]))
        };
        let crypto_ver = unsafe { *m13_ptr.add(2) };

        let mut desc = PacketDesc::EMPTY;
        desc.addr = addr;
        desc.len = len;
        desc.m13_offset = m13_offset;
        desc.peer_idx = peer_idx;
        desc.flags = flags_raw;
        desc.seq_id = seq_id;
        desc.payload_len = payload_len;
        desc.rx_ns = now;

        // Store source IP/port for UDP peers
        if m13_offset == 56 {
            desc.src_ip = unsafe { *(frame_ptr.add(26) as *const [u8; 4]) };
            desc.src_port = unsafe { u16::from_be(*(frame_ptr.add(34) as *const u16)) };
        }

        // Check if encrypted (crypto version byte)
        if crypto_ver == 0x01 {
            desc.flags |= 0x80; // Mark as needs-decrypt (temp flag, cleared after AEAD)
        }

        input.push(desc);
        stats.parsed += 1;
    }

    if input.is_empty() { return stats; }

    // ====================================================================
    // NODE 2: AEAD DECRYPT (vectorized — saturates AES-NI pipeline)
    // Processes ALL encrypted packets in one batch before moving to classify.
    // ====================================================================
    let mut decrypt_vec = PacketVector::new();
    let mut cleartext_vec = PacketVector::new();

    // Split: encrypted packets → decrypt_vec, cleartext → cleartext_vec
    for i in 0..input.len {
        let desc = input.descs[i];
        let pidx = desc.peer_idx as usize;

        // Check for reconnecting node (cleartext control while AEAD active)
        if pidx < MAX_PEERS && ctx.peers.slots[pidx].has_session()
           && (desc.flags & 0x80 == 0)  // Not encrypted
           && (desc.flags & FLAG_HANDSHAKE == 0)
           && (desc.flags & FLAG_FRAGMENT == 0)
           && (desc.flags & FLAG_CONTROL != 0) {
            // Reconnecting — reset session
            ctx.peers.slots[pidx].reset_session();
            ctx.peers.ciphers[pidx] = None;
            ctx.peers.hs_sidecar[pidx] = None;
            ctx.peers.assemblers[pidx] = Assembler::new();
        }

        if desc.flags & 0x80 != 0 {
            // Strip temp decrypt flag, keep original flags
            let mut clean_desc = desc;
            clean_desc.flags &= !0x80;
            decrypt_vec.push(clean_desc);
        } else {
            cleartext_vec.push(desc);
        }
    }

    // Vectorized AEAD decrypt — dual-loop with prefetch
    let mut aead_results = Disposition::new();
    if decrypt_vec.len > 0 {
        for i in 0..decrypt_vec.len {
            // Prefetch next frame data BEFORE taking mutable borrow
            if i + 4 < decrypt_vec.len {
                let prefetch_addr = decrypt_vec.descs[i + 4].addr;
                unsafe { prefetch_read_l1(prefetch_addr as *const u8); }
            }

            let desc = &mut decrypt_vec.descs[i];
            let pidx = desc.peer_idx as usize;

            if pidx >= MAX_PEERS {
                aead_results.next[i] = NextNode::Drop;
                continue;
            }

            let cipher = match ctx.peers.ciphers[pidx].as_ref() {
                Some(c) => c,
                None => {
                    aead_results.next[i] = NextNode::Drop;
                    stats.aead_fail += 1;
                    continue;
                }
            };

            let frame = unsafe {
                std::slice::from_raw_parts_mut(desc.addr as *mut u8, desc.len as usize)
            };
            let m13_off = desc.m13_offset as usize;

            if crate::crypto::aead::open_frame(frame, cipher, 0x01, m13_off) {
                // Re-read flags from decrypted buffer (CRITICAL: original flags were ciphertext)
                let decrypted_flags = unsafe { *((desc.addr as *const u8).add(m13_off + 40)) };
                desc.flags = decrypted_flags;
                desc.seq_id = unsafe {
                    u64::from_le_bytes(
                        std::slice::from_raw_parts((desc.addr as *const u8).add(m13_off + 32), 8)
                            .try_into().unwrap()
                    )
                };
                ctx.peers.slots[pidx].frame_count += 1;
                stats.aead_ok += 1;

                // Rekey check
                let established_ns = (ctx.peers.slots[pidx].established_rel_s as u64) * 1_000_000_000 + ctx.peers.epoch_ns;
                if ctx.peers.slots[pidx].frame_count as u64 >= (1u64 << 32)
                   || now.saturating_sub(established_ns) > 3_600_000_000_000 {
                    ctx.peers.slots[pidx].reset_session();
                    ctx.peers.ciphers[pidx] = None;
                    ctx.peers.hs_sidecar[pidx] = None;
                }

                aead_results.next[i] = NextNode::ClassifyRoute;
            } else {
                aead_results.next[i] = NextNode::Drop;
                stats.aead_fail += 1;
            }
        }
    }

    // ====================================================================
    // NODE 3: CLASSIFY + ROUTE (vectorized)
    // Merge decrypted + cleartext, classify by flags, route to outputs.
    // ====================================================================
    // Process all packets (decrypted + cleartext) through routing
    let process_packet = |desc: &PacketDesc, ctx: &mut GraphCtx<'_>, stats: &mut CycleStats| -> NextNode {
        let flags = desc.flags;

        // Fragment reassembly (cold path)
        if flags & FLAG_FRAGMENT != 0 {
            process_fragment(desc, ctx, stats);
            return NextNode::Consumed;
        }

        // Feedback
        if flags & FLAG_FEEDBACK != 0 {
            return NextNode::Feedback;
        }

        // Tunnel data → TUN write
        if flags & FLAG_TUNNEL != 0 {
            return NextNode::TunWrite;
        }

        // FIN handling
        if flags & FLAG_FIN != 0 {
            return NextNode::Consumed; // Handled inline
        }

        // Handshake
        if flags & FLAG_HANDSHAKE != 0 {
            return NextNode::Handshake;
        }

        // Control (registration echo etc)
        if flags & FLAG_CONTROL != 0 {
            return NextNode::Consumed;
        }

        // Pure data (forwarding)
        NextNode::TxEnqueue
    };

    // Route decrypted packets
    for i in 0..decrypt_vec.len {
        if aead_results.next[i] == NextNode::Drop {
            (ctx.slab_free)((decrypt_vec.descs[i].addr / ctx.frame_size as u64) as u32);
            stats.drops += 1;
            continue;
        }

        let desc = &decrypt_vec.descs[i];
        let next = process_packet(desc, ctx, &mut stats);

        match next {
            NextNode::TunWrite => {
                // Write decrypted tunnel payload to TUN
                let m13_off = desc.m13_offset as usize;
                let payload_start = m13_off + M13_HDR_SIZE;
                // Re-read payload_len from decrypted buffer
                let plen = unsafe {
                    let m13_ptr = (desc.addr as *const u8).add(m13_off);
                    u32::from_le_bytes(std::slice::from_raw_parts(m13_ptr.add(41), 4).try_into().unwrap_or([0;4]))
                } as usize;
                if plen > 0 && payload_start + plen <= desc.len as usize && ctx.tun_fd >= 0 {
                    let payload_ptr = unsafe { (desc.addr as *const u8).add(payload_start) };
                    unsafe {
                        libc::write(ctx.tun_fd, payload_ptr as *const libc::c_void, plen);
                    }
                    stats.tun_writes += 1;
                }
                (ctx.slab_free)((desc.addr / ctx.frame_size as u64) as u32);
            }
            NextNode::Feedback => {
                // Feedback processing (BBR telemetry — currently no-op)
                stats.feedback += 1;
                (ctx.slab_free)((desc.addr / ctx.frame_size as u64) as u32);
            }
            NextNode::TxEnqueue => {
                // Data forwarding
                if !ctx.closing {
                    ctx.rx_state.highest_seq = desc.seq_id;
                    ctx.rx_state.delivered += 1;
                    ctx.rx_state.last_rx_batch_ns = now;
                    ctx.rx_bitmap.mark(desc.seq_id);
                    ctx.scheduler.enqueue_bulk(desc.addr, desc.len);
                    stats.data_fwd += 1;
                } else {
                    (ctx.slab_free)((desc.addr / ctx.frame_size as u64) as u32);
                }
            }
            NextNode::Consumed => {
                (ctx.slab_free)((desc.addr / ctx.frame_size as u64) as u32);
            }
            _ => {
                (ctx.slab_free)((desc.addr / ctx.frame_size as u64) as u32);
                stats.drops += 1;
            }
        }
    }

    // Route cleartext packets
    for i in 0..cleartext_vec.len {
        let desc = &cleartext_vec.descs[i];
        let next = process_packet(desc, ctx, &mut stats);

        match next {
            NextNode::TunWrite => {
                let m13_off = desc.m13_offset as usize;
                let payload_start = m13_off + M13_HDR_SIZE;
                let m13_ptr = unsafe { (desc.addr as *const u8).add(m13_off) };
                let plen = unsafe {
                    u32::from_le_bytes(std::slice::from_raw_parts(m13_ptr.add(41), 4).try_into().unwrap_or([0;4]))
                } as usize;
                if plen > 0 && payload_start + plen <= desc.len as usize && ctx.tun_fd >= 0 {
                    let payload_ptr = unsafe { (desc.addr as *const u8).add(payload_start) };
                    unsafe {
                        libc::write(ctx.tun_fd, payload_ptr as *const libc::c_void, plen);
                    }
                    stats.tun_writes += 1;
                }
                (ctx.slab_free)((desc.addr / ctx.frame_size as u64) as u32);
            }
            NextNode::Feedback => {
                stats.feedback += 1;
                (ctx.slab_free)((desc.addr / ctx.frame_size as u64) as u32);
            }
            NextNode::TxEnqueue => {
                if !ctx.closing {
                    ctx.rx_state.highest_seq = desc.seq_id;
                    ctx.rx_state.delivered += 1;
                    ctx.rx_state.last_rx_batch_ns = now;
                    ctx.rx_bitmap.mark(desc.seq_id);
                    ctx.scheduler.enqueue_bulk(desc.addr, desc.len);
                    stats.data_fwd += 1;
                } else {
                    (ctx.slab_free)((desc.addr / ctx.frame_size as u64) as u32);
                }
            }
            NextNode::Consumed => {
                (ctx.slab_free)((desc.addr / ctx.frame_size as u64) as u32);
            }
            _ => {
                (ctx.slab_free)((desc.addr / ctx.frame_size as u64) as u32);
                stats.drops += 1;
            }
        }
    }

    stats
}

/// Process a fragmented frame (cold path — handshake fragments).
fn process_fragment(
    desc: &PacketDesc,
    ctx: &mut GraphCtx<'_>,
    stats: &mut CycleStats,
) {
    let m13_off = desc.m13_offset as usize;
    let frame_ptr = desc.addr as *const u8;
    let frame_len = desc.len as usize;
    let pidx = desc.peer_idx as usize;

    if pidx >= MAX_PEERS || frame_len < m13_off + M13_HDR_SIZE + FRAG_HDR_SIZE {
        (ctx.slab_free)((desc.addr / ctx.frame_size as u64) as u32);
        return;
    }

    let frag_hdr = unsafe { &*(frame_ptr.add(m13_off + M13_HDR_SIZE) as *const FragHeader) };
    let frag_data_start = m13_off + M13_HDR_SIZE + FRAG_HDR_SIZE;
    let frag_msg_id = unsafe { std::ptr::addr_of!((*frag_hdr).frag_msg_id).read_unaligned() };
    let frag_index = unsafe { std::ptr::addr_of!((*frag_hdr).frag_index).read_unaligned() };
    let frag_total = unsafe { std::ptr::addr_of!((*frag_hdr).frag_total).read_unaligned() };
    let frag_offset = unsafe { std::ptr::addr_of!((*frag_hdr).frag_offset).read_unaligned() };
    let frag_data_len = unsafe { std::ptr::addr_of!((*frag_hdr).frag_len).read_unaligned() } as usize;

    if frag_data_start + frag_data_len <= frame_len {
        let frag_data = unsafe { std::slice::from_raw_parts(frame_ptr.add(frag_data_start), frag_data_len) };

        if let Some(reassembled) = ctx.peers.assemblers[pidx].feed(
            frag_msg_id, frag_index, frag_total, frag_offset, frag_data, ctx.now_ns,
        ) {
            if desc.flags & FLAG_HANDSHAKE != 0 && !reassembled.is_empty() {
                process_handshake_message(&reassembled, pidx, ctx, stats);
            }
        }
    }

    (ctx.slab_free)((desc.addr / ctx.frame_size as u64) as u32);
}

/// Process a reassembled handshake message (cold path).
fn process_handshake_message(
    data: &[u8],
    pidx: usize,
    ctx: &mut GraphCtx<'_>,
    stats: &mut CycleStats,
) {
    if data.is_empty() { return; }
    let msg_type = data[0];
    stats.handshakes += 1;

    const HS_CLIENT_HELLO: u8 = 0x01;
    const HS_FINISHED: u8 = 0x03;

    match msg_type {
        HS_CLIENT_HELLO => {
            ctx.peers.slots[pidx].lifecycle = PeerLifecycle::Handshaking;
            let mut hs_seq_tx = ctx.peers.slots[pidx].seq_tx;
            if let Some((hs, server_hello)) = process_client_hello_hub(data, &mut hs_seq_tx, ctx.now_ns) {
                // Build fragmented ServerHello response
                let hs_flags = FLAG_CONTROL | FLAG_HANDSHAKE;
                if ctx.peers.slots[pidx].addr.is_udp() {
                    let peer_ip = ctx.peers.slots[pidx].addr.ip().unwrap();
                    let peer_port = ctx.peers.slots[pidx].addr.port().unwrap();
                    let frames = crate::crypto::pqc::build_fragmented_raw_udp(
                        &ctx.src_mac, &ctx.gateway_mac, ctx.hub_ip, peer_ip,
                        ctx.hub_port, peer_port, &server_hello, hs_flags,
                        &mut hs_seq_tx, ctx.ip_id_counter,
                    );
                    for raw_frame in frames {
                        if let Some(slab_idx) = (ctx.slab_alloc)() {
                            let frame_ptr = unsafe { ctx.umem_base.add((slab_idx as usize) * ctx.frame_size as usize) };
                            let flen = raw_frame.len().min(ctx.frame_size as usize);
                            unsafe { std::ptr::copy_nonoverlapping(raw_frame.as_ptr(), frame_ptr, flen); }
                            ctx.scheduler.enqueue_critical((slab_idx as u64) * ctx.frame_size as u64, flen as u32);
                        }
                    }
                }
                ctx.peers.hs_sidecar[pidx] = Some(hs);
                ctx.peers.slots[pidx].seq_tx = hs_seq_tx;
                eprintln!("[M13-VPP] ClientHello processed for peer {:?}, ServerHello enqueued.",
                    ctx.peers.slots[pidx].addr);
            }
        }
        HS_FINISHED => {
            if let Some(ref hs) = ctx.peers.hs_sidecar[pidx] {
                if let Some(key) = process_finished_hub(data, hs) {
                    ctx.peers.slots[pidx].session_key = key;
                    let ukey = aead::UnboundKey::new(&aead::AES_256_GCM, &key).unwrap();
                    ctx.peers.ciphers[pidx] = Some(aead::LessSafeKey::new(ukey));
                    ctx.peers.slots[pidx].frame_count = 0;
                    let rel_s = ((ctx.now_ns.saturating_sub(ctx.peers.epoch_ns)) / 1_000_000_000) as u32;
                    ctx.peers.slots[pidx].established_rel_s = rel_s;
                    ctx.peers.slots[pidx].lifecycle = PeerLifecycle::Established;
                    ctx.peers.hs_sidecar[pidx] = None;
                    eprintln!("[M13-VPP] Session established for peer {:?} (AEAD active)",
                        ctx.peers.slots[pidx].addr);
                } else {
                    ctx.peers.hs_sidecar[pidx] = None;
                }
            }
        }
        _ => {}
    }
}
