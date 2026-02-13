// M13 HUB — PEER TABLE
// DPDK/VPP-style flat array with FNV-1a linear probing.
// Cache-aligned: each PeerSlot is exactly 1 cache line (64 bytes).
// Zero heap allocation on hot path. O(1) amortized lookup.

use ring::aead;
use crate::protocol::fragment::Assembler;
use crate::crypto::pqc::HubHandshakeState;

pub const MAX_PEERS: usize = 256;
pub const TUNNEL_SUBNET: [u8; 4] = [10, 13, 0, 0];

/// 6-byte peer identity. Natural key for UDP peers behind NAT.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PeerAddr {
    Empty,
    Udp { ip: [u8; 4], port: u16 },
    L2 { mac: [u8; 6] },
}

impl PeerAddr {
    pub const EMPTY: PeerAddr = PeerAddr::Empty;

    #[inline(always)]
    pub fn new_udp(ip: [u8; 4], port: u16) -> Self { PeerAddr::Udp { ip, port } }
    #[inline(always)]
    pub fn new_l2(mac: [u8; 6]) -> Self { PeerAddr::L2 { mac } }
    #[inline(always)]
    pub fn ip(&self) -> Option<[u8; 4]> {
        match self { PeerAddr::Udp { ip, .. } => Some(*ip), _ => None }
    }
    #[inline(always)]
    pub fn port(&self) -> Option<u16> {
        match self { PeerAddr::Udp { port, .. } => Some(*port), _ => None }
    }
    #[inline(always)]
    pub fn is_udp(&self) -> bool { matches!(self, PeerAddr::Udp { .. }) }

    /// FNV-1a hash. 6 bytes for UDP (ip+port), 6 bytes for L2 (mac).
    #[inline(always)]
    pub fn hash(&self) -> usize {
        let mut h: u64 = 0xcbf29ce484222325;
        let bytes: [u8; 6] = match self {
            PeerAddr::Udp { ip, port } => [
                ip[0], ip[1], ip[2], ip[3],
                (*port & 0xFF) as u8, (*port >> 8) as u8,
            ],
            PeerAddr::L2 { mac } => *mac,
            PeerAddr::Empty => return 0,
        };
        let mut i = 0;
        while i < 6 {
            h ^= bytes[i] as u64;
            h = h.wrapping_mul(0x100000001b3);
            i += 1;
        }
        h as usize
    }
}

impl std::fmt::Debug for PeerAddr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PeerAddr::Empty => write!(f, "<empty>"),
            PeerAddr::Udp { ip, port } => write!(f, "{}.{}.{}.{}:{}", ip[0], ip[1], ip[2], ip[3], port),
            PeerAddr::L2 { mac } => write!(f, "{:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}",
                mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum PeerLifecycle {
    Empty = 0,
    Registered = 1,
    Handshaking = 2,
    Established = 3,
}

/// Per-peer state. Exactly 1 cache line (64 bytes) for zero false sharing.
#[repr(C, align(64))]
pub struct PeerSlot {
    pub addr: PeerAddr,
    pub lifecycle: PeerLifecycle,
    pub tunnel_ip_idx: u8,
    pub session_key: [u8; 32],
    pub seq_tx: u64,
    pub frame_count: u32,
    pub established_rel_s: u32,
    pub mac: [u8; 6],
}

impl PeerSlot {
    pub const EMPTY: PeerSlot = PeerSlot {
        addr: PeerAddr::EMPTY, lifecycle: PeerLifecycle::Empty,
        tunnel_ip_idx: 0, session_key: [0u8; 32], seq_tx: 0,
        frame_count: 0, established_rel_s: 0, mac: [0xFF; 6],
    };
    #[inline(always)] pub fn is_empty(&self) -> bool { self.lifecycle == PeerLifecycle::Empty }
    #[inline(always)] pub fn has_session(&self) -> bool { self.session_key != [0u8; 32] }
    #[inline(always)]
    pub fn next_seq(&mut self) -> u64 {
        let s = self.seq_tx; self.seq_tx = s.wrapping_add(1); s
    }
    pub fn reset_session(&mut self) {
        self.session_key = [0u8; 32]; self.seq_tx = 0;
        self.frame_count = 0; self.established_rel_s = 0;
        self.lifecycle = PeerLifecycle::Registered;
    }
}

/// Multi-tenant peer table. Single-threaded (owned by one worker).
pub struct PeerTable {
    pub slots: Box<[PeerSlot; MAX_PEERS]>,
    pub hs_sidecar: Vec<Option<HubHandshakeState>>,
    pub ciphers: Vec<Option<aead::LessSafeKey>>,
    pub assemblers: Vec<Assembler>,
    pub count: u16,
    tunnel_ip_bitmap: [u64; 4],
    pub epoch_ns: u64,
}

impl PeerTable {
    pub fn new(epoch_ns: u64) -> Self {
        let mut slots_vec: Vec<PeerSlot> = Vec::with_capacity(MAX_PEERS);
        for _ in 0..MAX_PEERS { slots_vec.push(PeerSlot::EMPTY); }
        let slots_array: Box<[PeerSlot; MAX_PEERS]> = match slots_vec.into_boxed_slice().try_into() {
            Ok(arr) => arr, Err(_) => unreachable!(),
        };
        let mut hs_sidecar = Vec::with_capacity(MAX_PEERS);
        let mut ciphers: Vec<Option<aead::LessSafeKey>> = Vec::with_capacity(MAX_PEERS);
        let mut assemblers = Vec::with_capacity(MAX_PEERS);
        for _ in 0..MAX_PEERS {
            hs_sidecar.push(None); ciphers.push(None);
            assemblers.push(Assembler::new());
        }
        let mut tunnel_ip_bitmap = [0u64; 4];
        tunnel_ip_bitmap[0] = 0b11;
        PeerTable {
            slots: slots_array, hs_sidecar, ciphers, assemblers,
            count: 0, tunnel_ip_bitmap, epoch_ns,
        }
    }

    #[inline(always)]
    pub fn lookup(&self, addr: PeerAddr) -> Option<usize> {
        let mut idx = addr.hash() & (MAX_PEERS - 1);
        for _ in 0..MAX_PEERS {
            if self.slots[idx].addr == addr && !self.slots[idx].is_empty() { return Some(idx); }
            idx = (idx + 1) & (MAX_PEERS - 1);
        }
        None
    }

    pub fn lookup_or_insert(&mut self, addr: PeerAddr, mac: [u8; 6]) -> Option<usize> {
        let mut idx = addr.hash() & (MAX_PEERS - 1);
        let mut first_empty: Option<usize> = None;
        for _ in 0..MAX_PEERS {
            if self.slots[idx].addr == addr && !self.slots[idx].is_empty() {
                self.slots[idx].mac = mac;
                return Some(idx);
            }
            if self.slots[idx].is_empty() && first_empty.is_none() {
                first_empty = Some(idx); break;
            }
            idx = (idx + 1) & (MAX_PEERS - 1);
        }
        let slot_idx = first_empty?;
        if self.count as usize >= MAX_PEERS - 1 { return None; }
        // Stale peer eviction: same IP, different port
        for i in 0..MAX_PEERS {
            if i == slot_idx { continue; }
            if !self.slots[i].is_empty() && self.slots[i].addr.ip() == addr.ip() && self.slots[i].addr != addr {
                eprintln!("[M13-PEERS] Stale peer {:?} in slot {} — evicting.", self.slots[i].addr, i);
                self.evict(i);
            }
        }
        let tunnel_idx = self.alloc_tunnel_ip()?;
        self.slots[slot_idx] = PeerSlot {
            addr, lifecycle: PeerLifecycle::Registered, tunnel_ip_idx: tunnel_idx,
            session_key: [0u8; 32], seq_tx: 0, frame_count: 0, established_rel_s: 0, mac,
        };
        self.hs_sidecar[slot_idx] = None;
        self.assemblers[slot_idx] = Assembler::new();
        self.count += 1;
        eprintln!("[M13-PEERS] New peer {:?} → slot {} tunnel_ip=10.13.0.{} (total: {})",
            addr, slot_idx, tunnel_idx, self.count);
        Some(slot_idx)
    }

    pub fn evict(&mut self, idx: usize) {
        if idx >= MAX_PEERS || self.slots[idx].is_empty() { return; }
        let tip = self.slots[idx].tunnel_ip_idx;
        self.free_tunnel_ip(tip);
        eprintln!("[M13-PEERS] Evicted peer {:?} from slot {} (tunnel_ip=10.13.0.{})",
            self.slots[idx].addr, idx, tip);
        self.slots[idx] = PeerSlot::EMPTY;
        self.hs_sidecar[idx] = None; self.ciphers[idx] = None;
        self.assemblers[idx] = Assembler::new();
        if self.count > 0 { self.count -= 1; }
    }

    pub fn lookup_by_tunnel_ip(&self, dst_ip: [u8; 4]) -> Option<usize> {
        if dst_ip[0] != TUNNEL_SUBNET[0] || dst_ip[1] != TUNNEL_SUBNET[1]
           || dst_ip[2] != TUNNEL_SUBNET[2] { return None; }
        let target_idx = dst_ip[3];
        for i in 0..MAX_PEERS {
            if !self.slots[i].is_empty()
               && self.slots[i].tunnel_ip_idx == target_idx
               && self.slots[i].has_session() { return Some(i); }
        }
        None
    }

    pub fn alloc_tunnel_ip(&mut self) -> Option<u8> {
        for word_idx in 0..4u8 {
            let word = self.tunnel_ip_bitmap[word_idx as usize];
            if word == u64::MAX { continue; }
            let bit = (!word).trailing_zeros() as u8;
            let global_idx = word_idx * 64 + bit;
            if global_idx >= 255 { continue; }
            self.tunnel_ip_bitmap[word_idx as usize] |= 1u64 << bit;
            return Some(global_idx);
        }
        None
    }

    pub fn free_tunnel_ip(&mut self, idx: u8) {
        let word_idx = (idx / 64) as usize;
        let bit = idx % 64;
        self.tunnel_ip_bitmap[word_idx] &= !(1u64 << bit);
    }

    pub fn gc(&mut self, _now_ns: u64) {
        // Future: time-based eviction for stale peers
    }
}
