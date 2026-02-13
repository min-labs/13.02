// M13 HUB — VPP GRAPH FRAMEWORK
// Genuine Vector Packet Processing: graph-based pipeline where packets flow
// through nodes in vectors. Each node processes the entire batch before
// handing off, keeping instruction caches hot and saturating AES-NI pipelines.
//
// Architecture: fd.io/VPP-style graph topology.
// Key insight: batch N packets at each stage, not 1 packet through all stages.
//
// Graph topology:
//   RxParse → [control branch]  → ClassifyRoute → TunIO
//           → [data branch]     → AeadDecrypt → ClassifyRoute → TunIO
//           → [handshake branch] → HandshakeProcess
//           → [feedback branch]  → FeedbackProcess
//
// TunRead → AeadEncrypt → TxEnqueue
//
// Each node implements the GraphNode trait with a process() method that takes
// a PacketVector (batch of packet descriptors) and returns output vectors for
// the next nodes in the graph.

pub mod nodes;
pub mod executor;

use std::fmt;

// ============================================================================
// PACKET VECTOR — The fundamental VPP data structure
// ============================================================================

/// Maximum batch size for vector processing. Must be power of 2.
/// 64 packets is the sweet spot: fits in L1 data cache (64 * 48B = 3KB)
/// while providing enough amortization for AEAD and syscall overhead.
pub const VECTOR_SIZE: usize = 64;

/// Descriptor for one packet in a vector. Minimal metadata for zero-copy routing.
/// 48 bytes — exactly 3/4 of a cache line. Aligned to 8 bytes.
#[repr(C, align(8))]
#[derive(Clone, Copy)]
pub struct PacketDesc {
    /// UMEM frame address (for AF_XDP zero-copy)
    pub addr: u64,
    /// Frame length in bytes
    pub len: u32,
    /// Offset to M13 header within the frame (ETH_HDR_SIZE for L2, RAW_HDR_LEN for UDP)
    pub m13_offset: u16,
    /// Flags extracted from M13Header.flags
    pub flags: u8,
    /// Peer slot index (from PeerTable lookup, 0xFF = unknown)
    pub peer_idx: u8,
    /// Sequence ID from M13Header
    pub seq_id: u64,
    /// Payload length from M13Header
    pub payload_len: u32,
    /// RX timestamp (rdtsc_ns)
    pub rx_ns: u64,
    /// Source IP (for UDP peers)
    pub src_ip: [u8; 4],
    /// Source port (for UDP peers)
    pub src_port: u16,
    /// Padding for alignment
    _pad: [u8; 2],
}

impl PacketDesc {
    pub const EMPTY: Self = PacketDesc {
        addr: 0, len: 0, m13_offset: 0, flags: 0, peer_idx: 0xFF,
        seq_id: 0, payload_len: 0, rx_ns: 0,
        src_ip: [0; 4], src_port: 0, _pad: [0; 2],
    };
}

/// Vector of packet descriptors. Stack-allocated, fixed capacity.
/// This is the fundamental unit of work in the VPP graph.
pub struct PacketVector {
    pub descs: [PacketDesc; VECTOR_SIZE],
    pub len: usize,
}

impl PacketVector {
    #[inline(always)]
    pub fn new() -> Self {
        PacketVector {
            descs: [PacketDesc::EMPTY; VECTOR_SIZE],
            len: 0,
        }
    }

    #[inline(always)]
    pub fn push(&mut self, desc: PacketDesc) -> bool {
        if self.len < VECTOR_SIZE {
            self.descs[self.len] = desc;
            self.len += 1;
            true
        } else {
            false
        }
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    pub fn is_full(&self) -> bool {
        self.len >= VECTOR_SIZE
    }
}

// ============================================================================
// GRAPH NODE TRAIT — Each processing stage implements this
// ============================================================================

/// Output disposition for a packet after graph node processing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum NextNode {
    /// Drop the packet (free UMEM frame)
    Drop = 0,
    /// Send to AEAD decrypt
    AeadDecrypt = 1,
    /// Send to classify/route
    ClassifyRoute = 2,
    /// Send to TUN write
    TunWrite = 3,
    /// Send to AEAD encrypt (TUN read → wire TX)
    AeadEncrypt = 4,
    /// Send to TX enqueue (scheduler)
    TxEnqueue = 5,
    /// Send to handshake processor (cold path)
    Handshake = 6,
    /// Send to feedback processor
    Feedback = 7,
    /// Packet consumed by this node (no forwarding needed)
    Consumed = 8,
}

impl fmt::Display for NextNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NextNode::Drop => write!(f, "drop"),
            NextNode::AeadDecrypt => write!(f, "aead-decrypt"),
            NextNode::ClassifyRoute => write!(f, "classify-route"),
            NextNode::TunWrite => write!(f, "tun-write"),
            NextNode::AeadEncrypt => write!(f, "aead-encrypt"),
            NextNode::TxEnqueue => write!(f, "tx-enqueue"),
            NextNode::Handshake => write!(f, "handshake"),
            NextNode::Feedback => write!(f, "feedback"),
            NextNode::Consumed => write!(f, "consumed"),
        }
    }
}

/// Per-packet disposition array. Maps each packet in the input vector
/// to its next graph node.
pub struct Disposition {
    pub next: [NextNode; VECTOR_SIZE],
}

impl Disposition {
    #[inline(always)]
    pub fn new() -> Self {
        Disposition { next: [NextNode::Drop; VECTOR_SIZE] }
    }
}

// ============================================================================
// GRAPH EXECUTOR — Runs the VPP pipeline
// ============================================================================

/// Statistics for a single graph node execution.
#[derive(Default, Clone, Copy)]
pub struct NodeStats {
    pub packets_in: u64,
    pub packets_out: u64,
    pub drops: u64,
    pub cycles: u64,
}

/// Graph-wide statistics.
pub struct GraphStats {
    pub rx_parse: NodeStats,
    pub aead_decrypt: NodeStats,
    pub classify_route: NodeStats,
    pub tun_write: NodeStats,
    pub tun_read: NodeStats,
    pub aead_encrypt: NodeStats,
    pub tx_enqueue: NodeStats,
    pub handshake: NodeStats,
    pub feedback: NodeStats,
    pub vectors_processed: u64,
}

impl GraphStats {
    pub fn new() -> Self {
        GraphStats {
            rx_parse: NodeStats::default(),
            aead_decrypt: NodeStats::default(),
            classify_route: NodeStats::default(),
            tun_write: NodeStats::default(),
            tun_read: NodeStats::default(),
            aead_encrypt: NodeStats::default(),
            tx_enqueue: NodeStats::default(),
            handshake: NodeStats::default(),
            feedback: NodeStats::default(),
            vectors_processed: 0,
        }
    }
}

/// Scatter a source vector into multiple output vectors based on disposition.
/// This is the core routing mechanism between graph nodes.
/// Uses the dual-loop pattern: process 4 packets at a time with prefetch.
#[inline]
pub fn scatter(
    src: &PacketVector,
    disp: &Disposition,
    decrypt_out: &mut PacketVector,
    classify_out: &mut PacketVector,
    tun_out: &mut PacketVector,
    encrypt_out: &mut PacketVector,
    tx_out: &mut PacketVector,
    handshake_out: &mut PacketVector,
    feedback_out: &mut PacketVector,
    drop_out: &mut PacketVector,
) {
    // Dual-loop pattern: process 4 at a time with lookahead prefetch
    let n = src.len;
    let mut i = 0;

    // Main loop: 4 packets per iteration
    while i + 4 <= n {
        // Prefetch next 4 descriptors (if available)
        if i + 8 <= n {
            unsafe {
                let base = src.descs.as_ptr().add(i + 4);
                crate::engine::clock::prefetch_read_l1(base as *const u8);
                crate::engine::clock::prefetch_read_l1(base.add(1) as *const u8);
                crate::engine::clock::prefetch_read_l1(base.add(2) as *const u8);
                crate::engine::clock::prefetch_read_l1(base.add(3) as *const u8);
            }
        }

        // Process current 4
        for j in 0..4 {
            let idx = i + j;
            let desc = &src.descs[idx];
            match disp.next[idx] {
                NextNode::AeadDecrypt => { decrypt_out.push(*desc); }
                NextNode::ClassifyRoute => { classify_out.push(*desc); }
                NextNode::TunWrite => { tun_out.push(*desc); }
                NextNode::AeadEncrypt => { encrypt_out.push(*desc); }
                NextNode::TxEnqueue => { tx_out.push(*desc); }
                NextNode::Handshake => { handshake_out.push(*desc); }
                NextNode::Feedback => { feedback_out.push(*desc); }
                NextNode::Drop => { drop_out.push(*desc); }
                NextNode::Consumed => { /* Node consumed it, no forwarding */ }
            }
        }
        i += 4;
    }

    // Remainder loop
    while i < n {
        let desc = &src.descs[i];
        match disp.next[i] {
            NextNode::AeadDecrypt => { decrypt_out.push(*desc); }
            NextNode::ClassifyRoute => { classify_out.push(*desc); }
            NextNode::TunWrite => { tun_out.push(*desc); }
            NextNode::AeadEncrypt => { encrypt_out.push(*desc); }
            NextNode::TxEnqueue => { tx_out.push(*desc); }
            NextNode::Handshake => { handshake_out.push(*desc); }
            NextNode::Feedback => { feedback_out.push(*desc); }
            NextNode::Drop => { drop_out.push(*desc); }
            NextNode::Consumed => {}
        }
        i += 1;
    }
}
