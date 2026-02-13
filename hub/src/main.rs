// M13 HUB - LOGIC KERNEL (REV 6.1)
// Sprint 6.1: Fragmentation Engine + Hexdump (bilateral with Node)
// Sprint 5.1: The Executive (Thread Pinning, Multi-Core)
// Sprint 5.2: Typestate FSM (Compile-Time Protocol State)
// Sprint 5.3: The Graph Pipeline (I/O Vector Batching)
// Sprint 5.4: The Latency Floor (Adaptive Batching)
// Sprint 5.5: Software Prefetching (D-Cache Warming)
// Sprint 5.6: The Interrupt Fence (IRQ Affinity)
// Sprint 5.7: The Isochronous Scheduler (Hierarchical DWRR)
// Sprint 5.8: The Feedback Channel (ACK Frame)
// Sprint 5.9: BBRv3 Congestion Control (Model-Based Rate Pacing)
// Sprint 5.10: Deterministic Jitter Buffer (RFC 3550 EWMA, Adaptive D_buf)
// Sprint 5.11: Diagnostic Error Codes (Structured Exit, Zero-Alloc Fatal)
//   - Windowed max filter for BtlBw (Kathleen Nichols algorithm, 10-round window)
//   - Expiring min for RTprop (10-second window, triggers ProbeRTT)
//   - Full BBRv3 state machine: Startup → Drain → ProbeBW (Down/Cruise/Refill/Up) → ProbeRTT
//   - Token bucket pacing at batch level (no per-packet busy-wait, preserves batching)
//   - cwnd = BDP × cwnd_gain (caps inflight, replaces static HW_FILL_MAX when active)
//   - All gains from Google's BBRv3: Startup 2.77x, Drain 0.36x, ProbeUp 1.25x, ProbeDown 0.9x
//   - Fixed-point arithmetic (integer rationals, no f64)
//   - Feedback-first pipeline: control frames processed BEFORE generation for zero-cycle latency
mod datapath;
use crate::datapath::{M13_WIRE_MAGIC, M13_WIRE_VERSION, 
    FixedSlab, Engine, Telemetry, EthernetHeader, M13Header, FeedbackFrame,
    ETH_P_M13, ZeroCopyTx, TxPath, BpfSteersman, MAX_WORKERS, FRAME_SIZE, UMEM_SIZE,
    // Structured exits — single source of truth for error codes
    fatal,
    E_NO_ISOLATED_CORES, E_AFFINITY_FAIL, E_AFFINITY_VERIFY,
};
use std::mem;
use std::sync::atomic::{AtomicBool, Ordering};
use std::fs::{OpenOptions, File};
use std::io::{Write, Read, Seek, SeekFrom, BufReader, BufRead};

use std::time::Duration;
use std::os::unix::io::AsRawFd;
use std::process::Command;

use std::marker::PhantomData;
use std::collections::HashMap;

// ============================================================================
// VPP LIBRARY IMPORTS — canonical source of truth for all extracted modules
// ============================================================================
use m13_hub::protocol::wire::{FLAG_CONTROL, FLAG_FEEDBACK, FLAG_TUNNEL, FLAG_ECN, FLAG_FIN,
    FLAG_HANDSHAKE, FLAG_FRAGMENT, HS_CLIENT_HELLO, HS_SERVER_HELLO, HS_FINISHED,
    DIR_HUB_TO_NODE, REKEY_FRAME_LIMIT, REKEY_TIME_LIMIT_NS,
    FragHeader, FRAG_HDR_SIZE};
use m13_hub::protocol::peer::{PeerTable, PeerAddr, PeerSlot, PeerLifecycle, MAX_PEERS, TUNNEL_SUBNET};
use m13_hub::protocol::fragment::Assembler;
use m13_hub::crypto::aead::{seal_frame, open_frame};
use m13_hub::crypto::pqc::{HubHandshakeState, process_client_hello_hub, process_finished_hub,
    build_fragmented_raw_udp as build_fragmented_raw_udp_nohex};
use m13_hub::engine::clock::{TscCal, calibrate_tsc, rdtsc_ns, clock_ns, prefetch_read_l1};
use m13_hub::engine::scheduler::{Scheduler, TxSubmit, TxCounter, TxDesc,
    TX_RING_SIZE, HW_FILL_MAX};
use m13_hub::engine::jitter::{JitterBuffer, measure_epsilon_proc, JBUF_CAPACITY};
use m13_hub::engine::rx_state::{RxBitmap, ReceiverState, FEEDBACK_INTERVAL_PKTS, FEEDBACK_RTT_DEFAULT_NS};
use m13_hub::tunnel::net::{build_raw_udp_frame, ip_checksum, detect_mac, resolve_gateway_mac,
    get_interface_ip, RAW_HDR_LEN, IP_HDR_LEN, UDP_HDR_LEN};

use sha2::Sha512;
use hkdf::Hkdf;

use ring::aead;

const SLAB_DEPTH: usize = 8192;
const GRAPH_BATCH: usize = 256;
const ETH_HDR_SIZE: usize = mem::size_of::<EthernetHeader>();
const M13_HDR_SIZE: usize = mem::size_of::<M13Header>();
const FEEDBACK_FRAME_LEN: u32 = (ETH_HDR_SIZE + M13_HDR_SIZE + mem::size_of::<FeedbackFrame>()) as u32;
const DEADLINE_NS: u64 = 50_000;
const PREFETCH_DIST: usize = 4;
const SEQ_WINDOW: usize = 131_072; // 2^17
const _: () = assert!(SEQ_WINDOW & (SEQ_WINDOW - 1) == 0);

/// Bridge: ZeroCopyTx (datapath.rs TxPath) → lib Scheduler's TxSubmit trait.
impl TxSubmit for ZeroCopyTx {
    fn available_slots(&mut self) -> u32 { TxPath::available_slots(self) }
    fn stage_tx_addr(&mut self, addr: u64, len: u32) { TxPath::stage_tx_addr(self, addr, len) }
    fn commit_tx(&mut self) { TxPath::commit_tx(self) }
    fn kick_tx(&mut self) { TxPath::kick_tx(self) }
}


/// Build fragmented handshake frames as raw ETH+IP+UDP packets for AF_XDP TX.
/// Returns Vec of ready-to-transmit raw frames (no kernel socket involvement).
/// Each frame: ETH(14) + IP(20) + UDP(8) + M13(48) + FragHdr(8) + chunk = 98 + chunk
fn build_fragmented_raw_udp(
    src_mac: &[u8; 6], gw_mac: &[u8; 6],
    hub_ip: [u8; 4], peer_ip: [u8; 4],
    hub_port: u16, peer_port: u16,
    payload: &[u8],
    flags: u8,
    seq: &mut u64,
    ip_id_base: &mut u16,
    hexdump: &mut HexdumpState,
    cal: &TscCal,
) -> Vec<Vec<u8>> {
    // MTU 1500 - IP(20) - UDP(8) = 1472 bytes max UDP payload
    // M13 frame overhead: ETH(14) + M13(48) + FRAG(8) = 70 bytes
    // But inside the UDP payload, we have: ETH_FAKE(14) + M13(48) + FRAG(8) = 70 bytes of M13 overhead
    // Max handshake data per fragment: 1472 - 70 = 1402 bytes
    let max_chunk = 1402;
    let total = (payload.len() + max_chunk - 1) / max_chunk;
    let msg_id = (*seq & 0xFFFF) as u16;
    let mut frames = Vec::with_capacity(total);

    for i in 0..total {
        let offset = i * max_chunk;
        let chunk_len = (payload.len() - offset).min(max_chunk);

        // Build the M13 fragment payload (what goes inside the UDP payload)
        let m13_flen = ETH_HDR_SIZE + M13_HDR_SIZE + FRAG_HDR_SIZE + chunk_len;
        let mut m13_frame = vec![0u8; m13_flen];
        m13_frame[0..6].copy_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
        m13_frame[6..12].copy_from_slice(src_mac);
        m13_frame[12] = (ETH_P_M13 >> 8) as u8;
        m13_frame[13] = (ETH_P_M13 & 0xFF) as u8;
        m13_frame[14] = M13_WIRE_MAGIC; m13_frame[15] = M13_WIRE_VERSION;
        m13_frame[46..54].copy_from_slice(&seq.to_le_bytes());
        m13_frame[54] = flags | FLAG_FRAGMENT;
        let fh = ETH_HDR_SIZE + M13_HDR_SIZE;
        m13_frame[fh..fh+2].copy_from_slice(&msg_id.to_le_bytes());
        m13_frame[fh+2] = i as u8; m13_frame[fh+3] = total as u8;
        m13_frame[fh+4..fh+6].copy_from_slice(&(offset as u16).to_le_bytes());
        m13_frame[fh+6..fh+8].copy_from_slice(&(chunk_len as u16).to_le_bytes());
        let dp = fh + FRAG_HDR_SIZE;
        m13_frame[dp..dp+chunk_len].copy_from_slice(&payload[offset..offset+chunk_len]);

        // Wrap in raw ETH+IP+UDP
        let raw_len = RAW_HDR_LEN + m13_flen;
        let mut raw = vec![0u8; raw_len];
        let flen = build_raw_udp_frame(
            &mut raw, src_mac, gw_mac, hub_ip, peer_ip,
            hub_port, peer_port, *ip_id_base, &m13_frame,
        );
        *ip_id_base = ip_id_base.wrapping_add(1);
        hexdump.dump_tx(raw.as_ptr(), flen, rdtsc_ns(cal));
        frames.push(raw);
        *seq += 1;
    }
    frames
}


// ============================================================================
// SPRINT 6.1: HEXDUMP ENGINE (all 4 capture points, rate-limited)
// ============================================================================
const HEXDUMP_INTERVAL_NS: u64 = 100_000_000; // 100ms = 10/sec max

struct HexdumpState { enabled: bool, last_tx_ns: u64 }
impl HexdumpState {
    fn new(enabled: bool) -> Self { HexdumpState { enabled, last_tx_ns: 0 } }
    fn dump_tx(&mut self, frame: *const u8, len: usize, now_ns: u64) {
        if !self.enabled { return; }
        if now_ns.saturating_sub(self.last_tx_ns) < HEXDUMP_INTERVAL_NS { return; }
        self.last_tx_ns = now_ns;
        dump_frame("[HUB-TX]", frame, len);
    }
}

fn dump_frame(label: &str, frame: *const u8, len: usize) {
    let cap = len.min(80);
    let data = unsafe { std::slice::from_raw_parts(frame, cap) };
    let (seq, flags) = if cap >= ETH_HDR_SIZE + M13_HDR_SIZE {
        let m13 = unsafe { &*(frame.add(ETH_HDR_SIZE) as *const M13Header) };
        (m13.seq_id, m13.flags)
    } else { (0, 0) };
    let dst = if cap >= 6 { format!("{:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}",
        data[0], data[1], data[2], data[3], data[4], data[5]) } else { "?".into() };
    let src = if cap >= 12 { format!("{:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}",
        data[6], data[7], data[8], data[9], data[10], data[11]) } else { "?".into() };
    eprintln!("{} seq={} flags=0x{:02X} len={} dst={} src={}", label, seq, flags, len, dst, src);
    if cap >= 14 {
        eprint!("  [00..14] ETH  |"); for i in 0..14 { eprint!(" {:02X}", data[i]); } eprintln!();
    }
    if cap >= 16 { eprint!("  [14..16] MAGIC|"); eprint!(" {:02X} {:02X}", data[14], data[15]); eprintln!(); }
    if cap >= 18 {
        eprint!("  [16..18] CRYPT|"); eprint!(" {:02X} {:02X}", data[16], data[17]);
        eprintln!("  (crypto_ver=0x{:02X}={})", data[16], if data[16] == 0 { "cleartext" } else { "encrypted" });
    }
    if cap >= 34 { eprint!("  [18..34] MAC  |"); for i in 18..34 { eprint!(" {:02X}", data[i]); } eprintln!(); }
    if cap >= 46 { eprint!("  [34..46] NONCE|"); for i in 34..46 { eprint!(" {:02X}", data[i]); } eprintln!(); }
    if cap >= 54 {
        eprint!("  [46..54] SEQ  |"); for i in 46..54 { eprint!(" {:02X}", data[i]); }
        eprintln!("  (LE: seq_id={})", seq);
    }
    if cap >= 55 { eprintln!("  [54..55] FLAGS| {:02X}", data[54]); }
    if cap >= 59 {
        let plen = if cap >= ETH_HDR_SIZE + M13_HDR_SIZE {
            let m13 = unsafe { &*(frame.add(ETH_HDR_SIZE) as *const M13Header) }; m13.payload_len
        } else { 0 };
        eprint!("  [55..59] PLEN |"); for i in 55..59 { eprint!(" {:02X}", data[i]); }
        eprintln!("  (LE: payload_len={})", plen);
    }
    if cap >= 62 { eprint!("  [59..62] PAD  |"); for i in 59..62 { eprint!(" {:02X}", data[i]); } eprintln!(); }
}

/// Process-wide shutdown flag. Set by SIGTERM/SIGINT handler.
/// Checked at the top of every worker loop iteration.
/// Ordering::Relaxed is correct — monotonic flag (once true, never false).
static SHUTDOWN: AtomicBool = AtomicBool::new(false);

extern "C" fn signal_handler(_sig: i32) {
    SHUTDOWN.store(true, Ordering::Relaxed);
}

/// Read the hardware MAC address of a network interface from sysfs.
/// Returns the 6-byte MAC or a locally-administered fallback if sysfs is unavailable.

fn main() {
    let args: Vec<String> = std::env::args().collect();
    unsafe {
        libc::signal(libc::SIGTERM, signal_handler as *const () as libc::sighandler_t);
        libc::signal(libc::SIGINT, signal_handler as *const () as libc::sighandler_t);
    }

    // Panic hook: guarantee cleanup even on unwinding crash
    let panic_if_name = args.get(1).cloned().unwrap_or_else(|| "veth0".to_string());
    std::panic::set_hook(Box::new(move |info| {
        eprintln!("[M13-HUB] PANIC: {}", info);
        nuke_cleanup_hub(&panic_if_name);
        std::process::exit(1);
    }));

    if args.iter().any(|a| a == "--monitor") {
        run_monitor();
        return;
    }
    let mut if_name = "veth0".to_string();
    let mut single_queue: Option<i32> = None;
    let mut hexdump_mode = false;
    let mut tunnel_mode = false;
    let mut listen_port: Option<u16> = Some(443); // Default: UDP/443 (blends with QUIC)
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--monitor" => { run_monitor(); return; }
            "--hexdump" => { hexdump_mode = true; }
            "--tunnel" => { tunnel_mode = true; }
            "--port" | "--listen" => {
                i += 1;
                if i < args.len() {
                    listen_port = Some(match args[i].parse() {
                        Ok(p) => p,
                        Err(_) => fatal(E_AFFINITY_FAIL, "Invalid port number"),
                    });
                }
            }
            "--single-queue" => {
                i += 1;
                if i < args.len() {
                    single_queue = Some(match args[i].parse() {
                        Ok(v) => v,
                        Err(_) => fatal(E_AFFINITY_FAIL, "Invalid queue ID argument"),
                    });
                }
            }
            "-i" | "--iface" => {
                i += 1;
                if i < args.len() { if_name = args[i].clone(); }
            }
            other => {
                if !other.starts_with("--") { if_name = other.to_string(); }
            }
        }
        i += 1;
    }
    if hexdump_mode {
        std::env::set_var("M13_HEXDUMP", "1");
    }
    // Hub always listens. CLI --port takes precedence, then env var, then default 443.
    // If M13_LISTEN_PORT is already set in the environment, don't clobber it with the default.
    let effective_port = match listen_port {
        Some(p) => {
            // CLI arg was explicitly given OR we have the default.
            // Check if user provided --port/--listen on the command line
            let cli_port_given = args.iter().any(|a| a == "--port" || a == "--listen");
            if cli_port_given {
                // Explicit CLI port always wins
                std::env::set_var("M13_LISTEN_PORT", p.to_string());
                p
            } else if let Ok(env_p) = std::env::var("M13_LISTEN_PORT") {
                // Env var already set, respect it
                env_p.parse::<u16>().unwrap_or(p)
            } else {
                // Neither CLI nor env var: use default
                std::env::set_var("M13_LISTEN_PORT", p.to_string());
                p
            }
        }
        None => {
            if let Ok(env_p) = std::env::var("M13_LISTEN_PORT") {
                env_p.parse::<u16>().unwrap_or(443)
            } else {
                443
            }
        }
    };
    let _ = effective_port; // suppress unused warning
    run_executive(&if_name, single_queue, tunnel_mode);
}

// ============================================================================
// TYPESTATE FSM — Compile-time protocol state. No BBR pollution.
// ============================================================================
pub struct Listening;
pub struct Established;
#[repr(C)]
pub struct Peer<S> { mac: [u8; 6], seq_tx: u64, _state: PhantomData<S> }
impl Peer<Listening> {
    pub fn new() -> Self { Self { mac: [0u8; 6], seq_tx: 0, _state: PhantomData } }
    pub fn accept_registration(self, peer_mac: [u8; 6]) -> Peer<Established> {
        Peer { mac: peer_mac, seq_tx: 0, _state: PhantomData }
    }
}
impl Peer<Established> {
    #[inline(always)] pub fn next_seq(&mut self) -> u64 { let s = self.seq_tx; self.seq_tx = s.wrapping_add(1); s }
    #[inline(always)] pub fn mac(&self) -> &[u8; 6] { &self.mac }
}

// produce_data_frame: removed (was only used by the old UDP kernel socket path)

// ============================================================================
// MULTI-TENANT PEER TABLE — Cache-aligned, O(1) linear probing.
// Design: DPDK/VPP-style flat array. Zero heap allocation on hot path.
// At N≤256 peers, the entire table fits in L2 cache (~16KB for 64B slots).
// ============================================================================

/// Maximum concurrent peers. Must be power of 2 for mask-based indexing.

#[inline(always)]
fn produce_feedback_frame(
    frame_ptr: *mut u8, dst_mac: &[u8; 6], src_mac: &[u8; 6],
    rx_state: &mut ReceiverState, rx_bitmap: &mut RxBitmap, now_ns: u64,
    jbuf_len: usize,
) {
    // Drain loss accumulator and NACK bitmap from RxBitmap
    let (loss_count, nack_bitmap) = rx_bitmap.drain_losses();
    // ECN decision: mark if jitter buffer > 75% or any loss detected.
    // This gives the sender advance warning of congestion before overflow.
    let congested = jbuf_len > JBUF_CAPACITY * 3 / 4 || loss_count > 0;
    unsafe {
        let eth = &mut *(frame_ptr as *mut EthernetHeader);
        eth.dst = *dst_mac; eth.src = *src_mac; eth.ethertype = ETH_P_M13.to_be();
        let m13 = &mut *(frame_ptr.add(ETH_HDR_SIZE) as *mut M13Header);
        m13.signature = [0; 32];
        m13.signature[0] = M13_WIRE_MAGIC;
        m13.signature[1] = M13_WIRE_VERSION;
        m13.seq_id = 0;
        m13.flags = FLAG_CONTROL | FLAG_FEEDBACK | if congested { FLAG_ECN } else { 0 };
        m13.payload_len = mem::size_of::<FeedbackFrame>() as u32;
        m13.padding = [0; 3];
        let fb = &mut *(frame_ptr.add(ETH_HDR_SIZE + M13_HDR_SIZE) as *mut FeedbackFrame);
        fb.highest_seq = rx_state.highest_seq;
        fb.rx_timestamp_ns = rx_state.last_rx_batch_ns;
        fb.delivered = rx_state.delivered;
        fb.delivered_time_ns = now_ns;
        fb.loss_count = loss_count;
        fb.nack_bitmap = nack_bitmap;
    }
    rx_state.delivered = 0;
    rx_state.last_feedback_ns = now_ns;
}

/// Send `count` redundant FIN or FIN-ACK frames wrapped in raw UDP.
/// Used for multi-tenant peers connected via UDP (not L2).
/// Constructs the M13 FIN payload, then wraps it in ETH+IP+UDP via build_raw_udp_frame.
#[inline(never)]
fn send_fin_burst_udp(
    slab: &mut FixedSlab, engine: &Engine<ZeroCopyTx>,
    scheduler: &mut Scheduler,
    src_mac: &[u8; 6], gateway_mac: &[u8; 6],
    hub_ip: [u8; 4], peer_ip: [u8; 4],
    hub_port: u16, peer_port: u16,
    ip_id: &mut u16,
    final_seq: u64, fin_ack: bool, count: usize,
) -> usize {
    let mut sent = 0;
    // Build the M13 FIN payload (62 bytes: ETH(14) + M13(48))
    let mut fin_m13 = [0u8; 62];
    fin_m13[0..6].copy_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]); // dst MAC (broadcast, inner)
    fin_m13[6..12].copy_from_slice(src_mac);
    fin_m13[12] = (ETH_P_M13 >> 8) as u8;
    fin_m13[13] = (ETH_P_M13 & 0xFF) as u8;
    fin_m13[14] = M13_WIRE_MAGIC;
    fin_m13[15] = M13_WIRE_VERSION;
    fin_m13[46..54].copy_from_slice(&final_seq.to_le_bytes()); // seq_id
    fin_m13[54] = FLAG_CONTROL | FLAG_FIN | if fin_ack { FLAG_FEEDBACK } else { 0 };
    // payload_len = 0, padding = 0 (already zeroed)

    for _ in 0..count {
        if let Some(idx) = slab.alloc() {
            let frame_ptr = unsafe { engine.umem_base().add((idx as usize) * FRAME_SIZE as usize) };
            let total_len;
            unsafe {
                let buf = std::slice::from_raw_parts_mut(frame_ptr, FRAME_SIZE as usize);
                total_len = build_raw_udp_frame(
                    buf, src_mac, gateway_mac,
                    hub_ip, peer_ip, hub_port, peer_port,
                    *ip_id, &fin_m13,
                );
                *ip_id = ip_id.wrapping_add(1);
            }
            scheduler.enqueue_critical((idx as u64) * FRAME_SIZE as u64, total_len as u32);
            sent += 1;
        }
    }
    sent
}

/// Send a raw L2 M13 frame (EtherType 0x88B5) into UMEM for AF_XDP TX.
/// For air-gapped WiFi 7 drones — no IP/UDP encapsulation.
/// `m13_payload` is the complete M13 frame: ETH(14) + M13(48) [+ data].
/// The ETH dst_mac in m13_payload is overwritten with peer_mac.
#[inline(never)]
fn send_l2_frame(
    slab: &mut FixedSlab, engine: &Engine<ZeroCopyTx>,
    scheduler: &mut Scheduler,
    peer_mac: &[u8; 6],
    m13_payload: &[u8],
) -> bool {
    if let Some(idx) = slab.alloc() {
        let frame_ptr = unsafe { engine.umem_base().add((idx as usize) * FRAME_SIZE as usize) };
        let flen = m13_payload.len().min(FRAME_SIZE as usize);
        unsafe {
            let buf = std::slice::from_raw_parts_mut(frame_ptr, FRAME_SIZE as usize);
            buf[..flen].copy_from_slice(&m13_payload[..flen]);
            // Overwrite dst MAC with peer's actual MAC
            buf[0..6].copy_from_slice(peer_mac);
        }
        scheduler.enqueue_critical((idx as u64) * FRAME_SIZE as u64, flen as u32);
        true
    } else {
        false
    }
}

/// Send `count` redundant FIN or FIN-ACK frames via raw L2 (EtherType 0x88B5).
/// Used for air-gapped WiFi 7 drones connected via raw Ethernet.
#[inline(never)]
fn send_fin_burst_l2(
    slab: &mut FixedSlab, engine: &Engine<ZeroCopyTx>,
    scheduler: &mut Scheduler,
    src_mac: &[u8; 6], peer_mac: &[u8; 6],
    final_seq: u64, fin_ack: bool, count: usize,
) -> usize {
    let mut sent = 0;
    let mut fin_m13 = [0u8; 62];
    fin_m13[0..6].copy_from_slice(peer_mac);
    fin_m13[6..12].copy_from_slice(src_mac);
    fin_m13[12] = (ETH_P_M13 >> 8) as u8;
    fin_m13[13] = (ETH_P_M13 & 0xFF) as u8;
    fin_m13[14] = M13_WIRE_MAGIC;
    fin_m13[15] = M13_WIRE_VERSION;
    fin_m13[46..54].copy_from_slice(&final_seq.to_le_bytes());
    fin_m13[54] = FLAG_CONTROL | FLAG_FIN | if fin_ack { FLAG_FEEDBACK } else { 0 };

    for _ in 0..count {
        if let Some(idx) = slab.alloc() {
            let frame_ptr = unsafe { engine.umem_base().add((idx as usize) * FRAME_SIZE as usize) };
            unsafe {
                let buf = std::slice::from_raw_parts_mut(frame_ptr, FRAME_SIZE as usize);
                buf[..62].copy_from_slice(&fin_m13);
            }
            scheduler.enqueue_critical((idx as u64) * FRAME_SIZE as u64, 62);
            sent += 1;
        }
    }
    sent
}

/// Build fragmented handshake frames as raw L2 packets (EtherType 0x88B5) for AF_XDP TX.
/// Returns Vec of ready-to-transmit raw L2 frames.
/// Each frame: ETH(14) + M13(48) + FragHdr(8) + chunk = 70 + chunk
fn build_fragmented_l2(
    src_mac: &[u8; 6], peer_mac: &[u8; 6],
    payload: &[u8],
    flags: u8,
    seq: &mut u64,
    hexdump: &mut HexdumpState,
    cal: &TscCal,
) -> Vec<Vec<u8>> {
    // L2 MTU: 1500 bytes. Frame: ETH(14) + M13(48) + FRAG(8) + chunk = 70 + chunk
    // Max chunk = 1500 - 70 = 1430 (slightly larger than UDP since no IP/UDP overhead)
    let max_chunk = 1430;
    let total = (payload.len() + max_chunk - 1) / max_chunk;
    let msg_id = (*seq & 0xFFFF) as u16;
    let mut frames = Vec::with_capacity(total);

    for i in 0..total {
        let offset = i * max_chunk;
        let chunk_len = (payload.len() - offset).min(max_chunk);

        let m13_flen = ETH_HDR_SIZE + M13_HDR_SIZE + FRAG_HDR_SIZE + chunk_len;
        let mut m13_frame = vec![0u8; m13_flen];
        m13_frame[0..6].copy_from_slice(peer_mac);
        m13_frame[6..12].copy_from_slice(src_mac);
        m13_frame[12] = (ETH_P_M13 >> 8) as u8;
        m13_frame[13] = (ETH_P_M13 & 0xFF) as u8;
        m13_frame[14] = M13_WIRE_MAGIC; m13_frame[15] = M13_WIRE_VERSION;
        m13_frame[46..54].copy_from_slice(&seq.to_le_bytes());
        m13_frame[54] = flags | FLAG_FRAGMENT;
        let fh = ETH_HDR_SIZE + M13_HDR_SIZE;
        m13_frame[fh..fh+2].copy_from_slice(&msg_id.to_le_bytes());
        m13_frame[fh+2] = i as u8; m13_frame[fh+3] = total as u8;
        m13_frame[fh+4..fh+6].copy_from_slice(&(offset as u16).to_le_bytes());
        m13_frame[fh+6..fh+8].copy_from_slice(&(chunk_len as u16).to_le_bytes());
        let dp = fh + FRAG_HDR_SIZE;
        m13_frame[dp..dp+chunk_len].copy_from_slice(&payload[offset..offset+chunk_len]);

        hexdump.dump_tx(m13_frame.as_ptr(), m13_flen, rdtsc_ns(cal));
        frames.push(m13_frame);
        *seq += 1;
    }
    frames
}

// ============================================================================
// THE EXECUTIVE (Boot Calibration Sequence)
// ============================================================================

fn run_executive(if_name: &str, single_queue: Option<i32>, tunnel: bool) {
    // Register signal handlers before spawning workers.
    // signal() is async-signal-safe. Handler sets AtomicBool — one CPU instruction.
    unsafe {
        libc::signal(libc::SIGTERM, signal_handler as *const () as libc::sighandler_t);
        libc::signal(libc::SIGINT, signal_handler as *const () as libc::sighandler_t);
    }

    // === AUTO-CLEANUP: kill stale hub, detach XDP, allocate hugepages ===
    eprintln!("[M13-EXEC] Pre-flight cleanup...");
    // Kill any previous m13-hub (SIGKILL, exclude ourselves)
    let my_pid = std::process::id();
    if let Ok(output) = std::process::Command::new("pgrep").arg("m13-hub").output() {
        if output.status.success() {
            let pids = String::from_utf8_lossy(&output.stdout);
            for pid_str in pids.lines() {
                if let Ok(pid) = pid_str.trim().parse::<u32>() {
                    if pid != my_pid {
                        unsafe { libc::kill(pid as i32, 9); }
                    }
                }
            }
        }
    }
    // Detach any stale XDP programs from the interface
    let _ = std::process::Command::new("ip").args(&["link", "set", if_name, "xdp", "off"]).output();
    let _ = std::process::Command::new("ip").args(&["link", "set", if_name, "xdpgeneric", "off"]).output();

    // CRITICAL: Collapse NIC to single queue so ALL traffic hits queue 0 (where AF_XDP binds).
    // Without this, RSS distributes UDP/443 across multiple queues → AF_XDP never sees packets
    // on queues != 0 → silent RX drop. This is the #1 failure mode for pure AF_XDP.
    let sq_result = std::process::Command::new("ethtool")
        .args(&["-L", if_name, "combined", "1"])
        .output();
    match sq_result {
        Ok(ref o) if o.status.success() => {
            eprintln!("[M13-EXEC] NIC {} collapsed to single queue (ethtool -L combined 1).", if_name);
        }
        Ok(ref o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            eprintln!("[M13-EXEC] WARNING: ethtool -L combined 1 failed: {}. UDP/443 may miss AF_XDP queue.", stderr.trim());
        }
        Err(e) => {
            eprintln!("[M13-EXEC] WARNING: ethtool not found: {}. Ensure NIC has 1 queue manually.", e);
        }
    }

    // Allocate hugepages: workers × UMEM_SIZE / 2MB per hugepage + headroom
    let hp_worker_count = match single_queue { Some(_) => 1, None => MAX_WORKERS };
    let hugepages_needed = (hp_worker_count * (UMEM_SIZE + 16 * 1024 * 1024)) / (2 * 1024 * 1024);
    let _ = std::fs::write("/proc/sys/vm/nr_hugepages", format!("{}\n", hugepages_needed));
    if let Ok(hp) = std::fs::read_to_string("/proc/sys/vm/nr_hugepages") {
        eprintln!("[M13-EXEC] Hugepages: {} allocated", hp.trim());
    }
    eprintln!("[M13-EXEC] Pre-flight cleanup complete.");

    // Sprint 5.17: TSC calibration. Must happen before workers spawn.
    // Workers receive a copy of the calibration data (immutable, per-worker).
    let tsc_cal = calibrate_tsc();

    lock_pmu();
    fence_interrupts();
    let isolated_cores = discover_isolated_cores();
    if isolated_cores.is_empty() {
        fatal(E_NO_ISOLATED_CORES, "No isolated cores. Boot with isolcpus=... or set M13_MOCK_CMDLINE");
    }
    // Detect UDP/Internet mode early — affects worker count and BPF decisions.
    let udp_mode = std::env::var("M13_LISTEN_PORT").is_ok();
    // In UDP/Internet mode: single worker only. Only worker 0 has the UDP socket,
    // and without BPF/XDP there's no point spawning workers that busy-poll a mock engine.
    let worker_count = if udp_mode {
        eprintln!("[M13-EXEC] UDP mode: single worker (worker 0 owns UDP socket).");
        1
    } else {
        match single_queue { Some(_) => 1, None => isolated_cores.len().min(MAX_WORKERS) }
    };
    eprintln!("[M13-EXEC] Discovered {} isolated core(s): {:?}. Spawning {} worker(s).",
        isolated_cores.len(), &isolated_cores[..worker_count], worker_count);
    // BPF Steersman: always attach. It steers both raw L2 M13 (ethertype 0x88B5)
    // AND IPv4 UDP port 443 (Internet mode) into AF_XDP. SSH/ARP/etc pass to kernel.
    let steersman = BpfSteersman::load_and_attach(if_name);
    let map_fd = if let Some(ref s) = steersman {
        let fd = s.map_fd();
        eprintln!("[M13-EXEC] BPF Steersman attached to {} [{}]. map_fd={}", if_name, s.attach_mode, fd);
        fd
    } else {
        eprintln!("[M13-EXEC] WARNING: BPF/XDP failed. Running in UDP-ONLY fallback mode.");
        -1
    };
    
    // Sprint 6.3: TUN Interface
    let tun_ref = if tunnel {
        let t = create_tun("m13tun0");
        if t.is_some() { setup_nat(); }
        t
    } else { None };

    let mut handles = Vec::with_capacity(worker_count);
    for worker_idx in 0..worker_count {
        let core_id = isolated_cores[worker_idx];
        let queue_id = match single_queue { Some(q) => q, None => worker_idx as i32 };
        let iface = if_name.to_string();
        let cal = tsc_cal; // Copy for this worker (TscCal is Copy)
        let tun = tun_ref.as_ref().and_then(|f| f.try_clone().ok());
        
        let handle = std::thread::Builder::new()
            .name(format!("m13-w{}", worker_idx)).stack_size(2 * 1024 * 1024)
            .spawn(move || { worker_entry(worker_idx, core_id, queue_id, &iface, map_fd, cal, tun); })
            .unwrap_or_else(|_| fatal(E_AFFINITY_FAIL, "Thread spawn failed"));
        handles.push(handle);
    }

    eprintln!("[M13-EXEC] Engine operational. Workers running.");

    for h in handles { let _ = h.join(); }
    drop(steersman);

    // Post-worker cleanup: nuke everything
    nuke_cleanup_hub(if_name);
    eprintln!("[M13-EXEC] All workers stopped. XDP detached. Clean exit.");
}

// ============================================================================
// INTERRUPT FENCE — pin NIC IRQs away from isolated cores
// ============================================================================
fn fence_interrupts() {
    if std::env::var("M13_MOCK_CMDLINE").is_ok() { return; }

    // DPDK/SPDK pattern: move IRQs OFF isolated (dataplane) cores only.
    // Compute mask of all non-isolated cores. IRQs can run on ANY of them.
    let isolated = discover_isolated_cores();
    if isolated.is_empty() { return; }

    // Get total CPU count
    let nproc = match std::fs::read_to_string("/sys/devices/system/cpu/present") {
        Ok(s) => {
            // Format: "0-N" or "0"
            let parts: Vec<&str> = s.trim().split('-').collect();
            match parts.last() {
                Some(n) => n.parse::<usize>().unwrap_or(0) + 1,
                None => 1,
            }
        }
        Err(_) => { eprintln!("[M13-EXEC] WARNING: Cannot read CPU topology, skipping IRQ fence"); return; }
    };

    // Build hex affinity mask: all cores EXCEPT isolated ones
    // e.g. 8 cores, isolated=[2,3] → mask = 0b11110011 = "f3"
    let mut mask_bits = vec![0u8; (nproc + 7) / 8];
    for cpu in 0..nproc {
        if !isolated.contains(&cpu) {
            mask_bits[cpu / 8] |= 1 << (cpu % 8);
        }
    }
    // Convert to hex string (MSByte first for /proc/irq format)
    let mask_hex: String = mask_bits.iter().rev()
        .map(|b| format!("{:02x}", b))
        .collect::<String>()
        .trim_start_matches('0')
        .to_string();
    let mask_str = if mask_hex.is_empty() { "1".to_string() } else { mask_hex };

    eprintln!("[M13-EXEC] IRQ fence mask: 0x{} (isolating cores {:?} from interrupts)", mask_str, isolated);

    // Set default SMP affinity for new IRQs
    let _ = std::fs::write("/proc/irq/default_smp_affinity", format!("{}\n", mask_str));

    if let Ok(output) = std::process::Command::new("pgrep").arg("irqbalance").output() {
        if output.status.success() {
            eprintln!("[M13-EXEC] WARNING: irqbalance is running. It will fight the IRQ fence.");
            eprintln!("[M13-EXEC] WARNING: Run 'systemctl stop irqbalance' for optimal performance.");
        }
    }

    let irq_dir = match std::fs::read_dir("/proc/irq") {
        Ok(d) => d, Err(_) => { eprintln!("[M13-EXEC] WARNING: Cannot read /proc/irq, skipping IRQ fence"); return; }
    };
    let (mut fenced, mut skipped) = (0u32, 0u32);
    for entry in irq_dir {
        let entry = match entry { Ok(e) => e, Err(_) => continue };
        let name = entry.file_name();
        let name_str = match name.to_str() { Some(s) => s, None => continue };
        if !name_str.bytes().next().map_or(false, |b| b.is_ascii_digit()) { continue; }
        // Write hex mask to smp_affinity (not affinity_list — hex is more reliable)
        let affinity_path = format!("/proc/irq/{}/smp_affinity", name_str);
        match std::fs::write(&affinity_path, format!("{}\n", mask_str)) {
            Ok(_) => { fenced += 1; }
            Err(_) => { skipped += 1; }
        }
    }
    eprintln!("[M13-EXEC] Interrupt Fence: {} IRQs moved to housekeeping cores, {} immovable", fenced, skipped);
}

// ============================================================================
// TUNNELING & NAT
// ============================================================================
const IFF_TUN: i16 = 0x0001;
const IFF_NO_PI: i16 = 0x1000;
const TUNSETIFF: u64 = 0x400454ca;

#[repr(C)]
struct ifreq_tun {
    ifr_name: [u8; 16],
    ifr_flags: i16,
}

fn create_tun(name: &str) -> Option<std::fs::File> {
    let tun_path = "/dev/net/tun";
    let file = match OpenOptions::new().read(true).write(true).open(tun_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[M13-TUN] Failed to open {}: {}", tun_path, e);
            return None;
        }
    };

    let mut req = ifreq_tun {
        ifr_name: [0; 16],
        ifr_flags: IFF_TUN | IFF_NO_PI,
    };
    
    let name_bytes = name.as_bytes();
    if name_bytes.len() > 15 {
        eprintln!("[M13-TUN] Interface name too long");
        return None;
    }
    for (i, b) in name_bytes.iter().enumerate() {
        req.ifr_name[i] = *b;
    }

    unsafe {
        if libc::ioctl(file.as_raw_fd(), TUNSETIFF, &req) < 0 {
            eprintln!("[M13-TUN] ioctl(TUNSETIFF) failed");
            return None;
        }
         // Set non-blocking
        let fd = file.as_raw_fd();
        let flags = libc::fcntl(fd, libc::F_GETFL);
        if flags >= 0 {
             libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK);
        }
    }
    
    // Set interface up and assign IP (Hub = 10.13.0.1/24)
    let _ = Command::new("ip").args(&["link", "set", "dev", name, "up"]).output();
    let _ = Command::new("ip").args(&["addr", "add", "10.13.0.1/24", "dev", name]).output();
    // MTU 1400: max safe for raw frame path (1500-104=1396), with 4B margin
    let _ = Command::new("ip").args(&["link", "set", "dev", name, "mtu", "1400"]).output();
    // txqueuelen 1000: prevent kernel TUN queue drops at burst rates
    let _ = Command::new("ip").args(&["link", "set", "dev", name, "txqueuelen", "1000"]).output();

    eprintln!("[M13-TUN] Created tunnel interface {} (10.13.0.1/24, MTU 1400)", name);
    Some(file)
}

fn setup_nat() {
    eprintln!("[M13-NAT] Enabling NAT + TCP BDP tuning...");
    // Enable forwarding
    let _ = Command::new("sysctl").args(&["-w", "net.ipv4.ip_forward=1"]).output();

    // === TCP BDP tuning ===
    // Tunnel adds RTT (~100-200ms). TCP throughput = window / RTT.
    // Default rmem_max=208KB → max 8Mbps at 200ms. Raise to 16MB → 640Mbps ceiling.
    let _ = Command::new("sysctl").args(&["-w", "net.core.rmem_max=16777216"]).output();
    let _ = Command::new("sysctl").args(&["-w", "net.core.wmem_max=16777216"]).output();
    let _ = Command::new("sysctl").args(&["-w", "net.ipv4.tcp_rmem=4096 1048576 16777216"]).output();
    let _ = Command::new("sysctl").args(&["-w", "net.ipv4.tcp_wmem=4096 1048576 16777216"]).output();
    // Keep cwnd warm between bursts (don't reset after idle)
    let _ = Command::new("sysctl").args(&["-w", "net.ipv4.tcp_slow_start_after_idle=0"]).output();
    // Larger NIC backlog for burst absorption
    let _ = Command::new("sysctl").args(&["-w", "net.core.netdev_max_backlog=10000"]).output();
    // Enable TCP window scaling (usually default, ensure it's on)
    let _ = Command::new("sysctl").args(&["-w", "net.ipv4.tcp_window_scaling=1"]).output();

    // TUN qdisc: fq (fair queueing) — absorbs bursts + paces TCP.
    // noqueue drops instantly when reader can't keep up; fq is production-grade.
    let _ = Command::new("tc").args(&["qdisc", "replace", "dev", "m13tun0", "root", "fq"]).output();

    // Sprint 4: Restrict MASQUERADE to exclude m13tun0 — prevents double-NAT
    // on return traffic through the tunnel interface.
    let _ = Command::new("iptables").args(&["-t", "nat", "-A", "POSTROUTING", "!", "-o", "m13tun0", "-j", "MASQUERADE"]).output();
    
    // TCP MSS clamping: TUN MTU=1400, so MSS must be ≤ 1360 (1400-40).
    // Without this, TCP negotiates MSS=1460 (based on remote 1500 MTU),
    // causing PMTUD blackhole → throughput collapses to ~5 Mbps.
    let _ = Command::new("iptables").args(&["-A", "FORWARD", "-p", "tcp", "--tcp-flags", "SYN,RST", "SYN", "-j", "TCPMSS", "--clamp-mss-to-pmtu"]).output();

    // Allow forwarding between interfaces
    let _ = Command::new("iptables").args(&["-A", "FORWARD", "-m", "conntrack", "--ctstate", "RELATED,ESTABLISHED", "-j", "ACCEPT"]).output();
    let _ = Command::new("iptables").args(&["-A", "FORWARD", "-i", "m13tun0", "-j", "ACCEPT"]).output();
    let _ = Command::new("iptables").args(&["-A", "FORWARD", "-o", "m13tun0", "-j", "ACCEPT"]).output();
}

/// Nuclear cleanup: tear down ALL Hub state — NAT, iptables, TUN, XDP.
/// Safe to call multiple times (idempotent). Safe from panic hook.
fn nuke_cleanup_hub(if_name: &str) {
    eprintln!("[M13-NUKE] Tearing down all Hub state...");

    // 1. Remove iptables NAT and FORWARD rules
    let _ = Command::new("iptables").args(&["-t", "nat", "-D", "POSTROUTING", "!", "-o", "m13tun0", "-j", "MASQUERADE"]).output();
    let _ = Command::new("iptables").args(&["-D", "FORWARD", "-p", "tcp", "--tcp-flags", "SYN,RST", "SYN", "-j", "TCPMSS", "--clamp-mss-to-pmtu"]).output();
    let _ = Command::new("iptables").args(&["-D", "FORWARD", "-m", "conntrack", "--ctstate", "RELATED,ESTABLISHED", "-j", "ACCEPT"]).output();
    let _ = Command::new("iptables").args(&["-D", "FORWARD", "-o", "m13tun0", "-j", "ACCEPT"]).output();
    let _ = Command::new("iptables").args(&["-D", "FORWARD", "-i", "m13tun0", "-j", "ACCEPT"]).output();

    // 2. Destroy TUN interface
    let _ = Command::new("ip").args(&["link", "del", "m13tun0"]).output();

    // 3. Detach XDP from interface
    let _ = Command::new("ip").args(&["link", "set", if_name, "xdp", "off"]).output();
    let _ = Command::new("ip").args(&["link", "set", if_name, "xdpgeneric", "off"]).output();

    eprintln!("[M13-NUKE] ✓ All Hub state destroyed.");
}

// ============================================================================
// WORKER ENTRY — Feedback-first BBRv3 paced pipeline
// Stage order: Classify → Feedback Processing → Feedback Generation →
//              Token Refill → Enqueue+Generate (paced) → Schedule
// ============================================================================
fn worker_entry(worker_idx: usize, core_id: usize, queue_id: i32, if_name: &str, bpf_map_fd: i32, cal: TscCal, mut tun: Option<std::fs::File>) {
    pin_to_core(core_id);
    verify_affinity(core_id);
    let stats = Telemetry::map_worker(worker_idx, true);
    stats.pid.value.store(unsafe { libc::syscall(libc::SYS_gettid) } as u32, Ordering::Relaxed);
    let mut engine = Engine::<ZeroCopyTx>::new_zerocopy(if_name, queue_id, bpf_map_fd);
    eprintln!("[M13-W{}] Datapath: {}", worker_idx, engine.xdp_mode);
    let mut slab = FixedSlab::new(SLAB_DEPTH);
    let mut scheduler = Scheduler::new();
    let mut rx_state = ReceiverState::new();
    let mut rx_bitmap = RxBitmap::new();
    let hexdump_enabled = std::env::var("M13_HEXDUMP").is_ok();
    let mut hexdump = HexdumpState::new(hexdump_enabled);
    let mut gc_counter: u64 = 0;
    let iface_mac = detect_mac(if_name);
    let src_mac = iface_mac;

    // Sprint S1: Multi-tenant peer table replaces all flat per-peer state.
    let mut peers = PeerTable::new(rdtsc_ns(&cal));
    eprintln!("[M13-W{}] PeerTable: {} slots, tunnel subnet 10.13.0.0/24", worker_idx, MAX_PEERS);

    // === PURE AF_XDP: No kernel socket. Raw UDP TX via AF_XDP ring. ===
    let hub_port: u16 = std::env::var("M13_LISTEN_PORT").ok()
        .and_then(|p| p.parse::<u16>().ok()).unwrap_or(443);
    let mut hub_ip: [u8; 4] = get_interface_ip(if_name).unwrap_or_else(|| {
        eprintln!("[M13-W{}] Hub IP not on interface — will learn from first inbound packet.", worker_idx);
        [0, 0, 0, 0]
    });
    let (mut gateway_mac, _gw_ip) = resolve_gateway_mac(if_name).unwrap_or_else(|| {
        eprintln!("[M13-W{}] Gateway MAC not in ARP — will learn from first inbound packet.", worker_idx);
        ([0xFF; 6], [0, 0, 0, 0])
    });
    let mut ip_id_counter: u16 = (worker_idx as u16).wrapping_mul(10000);
    eprintln!("[M13-W{}] AF_XDP Pure Mode: hub={}:{} gw_mac={:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
        worker_idx,
        hub_ip[0], hub_ip[1],
        gateway_mac[0], gateway_mac[1], gateway_mac[2],
        gateway_mac[3], gateway_mac[4], gateway_mac[5]);

    let mut udp_rx_count: u64 = 0;
    let mut udp_tx_count: u64 = 0;
    let mut tun_write_count: u64 = 0;
    let mut tun_read_count: u64 = 0;
    let mut aead_ok_count: u64 = 0;
    let mut aead_fail_count: u64 = 0;
    let mut last_hub_report_ns: u64 = 0;


    for i in 0..SLAB_DEPTH {
        let fp = engine.get_frame_ptr(i as u32);
        unsafe {
            let eth = &mut *(fp as *mut EthernetHeader);
            let m13 = &mut *(fp.add(ETH_HDR_SIZE) as *mut M13Header);
            eth.dst = [0xFF; 6]; // broadcast — real dst set per-frame
            eth.src = src_mac;
            eth.ethertype = ETH_P_M13.to_be();
            m13.signature = [0; 32];
            m13.signature[0] = M13_WIRE_MAGIC;
            m13.signature[1] = M13_WIRE_VERSION;
            m13.seq_id = 0; m13.flags = 0;
            m13.payload_len = 0; m13.padding = [0; 3];
        }
    }
    engine.refill_rx_full(&mut slab);
    let umem = engine.umem_base();

    // Measure ε_proc (processing jitter floor) and create jitter buffer
    let epsilon_ns = measure_epsilon_proc(&cal);
    let mut jbuf = JitterBuffer::new(epsilon_ns);
    eprintln!("[M13-W{}] ACTIVE. Pipeline: Graph({}) Deadline: {}us Prefetch: {} HW_Fill: {}/{} \
              SeqWin: {} Feedback: every {} pkts \
              JBuf: {}entries D_buf={}us ε={}us",
        worker_idx, GRAPH_BATCH, DEADLINE_NS / 1000, PREFETCH_DIST, HW_FILL_MAX, TX_RING_SIZE,
        SEQ_WINDOW, FEEDBACK_INTERVAL_PKTS,
        JBUF_CAPACITY, jbuf.depth_ns / 1000, epsilon_ns / 1000);

    let mut rx_batch: [libbpf_sys::xdp_desc; GRAPH_BATCH] = unsafe { mem::zeroed() };
    let mut data_indices = [0u16; GRAPH_BATCH];
    let mut ctrl_indices = [0u16; GRAPH_BATCH];
    let crit_indices = [0u16; GRAPH_BATCH];

    // Sprint 5.21: Graceful close state
    let mut closing = false;
    let mut fin_deadline_ns: u64 = 0;

    // === Sprint 4b: Hub header template for TUN→AF_XDP path ===
    // Pre-stamp static M13 header bytes once. Copy per-packet via single memcpy
    // instead of 8 individual writes + fill(0). bytes 16..62 already 0.
    let mut m13_hdr_template = [0u8; 62];
    m13_hdr_template[0..6].copy_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
    m13_hdr_template[6..12].copy_from_slice(&src_mac);
    m13_hdr_template[12] = (ETH_P_M13 >> 8) as u8;
    m13_hdr_template[13] = (ETH_P_M13 & 0xFF) as u8;
    m13_hdr_template[14] = M13_WIRE_MAGIC;
    m13_hdr_template[15] = M13_WIRE_VERSION;

    loop {
        // Sprint 5.21: Graceful close protocol.
        // On SHUTDOWN: send 3x FIN, then keep looping (RX only) until FIN-ACK or deadline.
        if SHUTDOWN.load(Ordering::Relaxed) && !closing {
            closing = true;
            let rtprop: u64 = 10_000_000;
            fin_deadline_ns = rdtsc_ns(&cal) + (rtprop.saturating_mul(5).max(10_000_000).min(100_000_000));
            // Send FIN to all established peers via raw UDP
            let mut fin_total = 0usize;
            for pi in 0..MAX_PEERS {
                if peers.slots[pi].lifecycle == PeerLifecycle::Established {
                    let sent = if peers.slots[pi].addr.is_udp() {
                        send_fin_burst_udp(
                            &mut slab, &engine, &mut scheduler,
                            &src_mac, &gateway_mac,
                            hub_ip, peers.slots[pi].addr.ip().unwrap(),
                            hub_port, peers.slots[pi].addr.port().unwrap(),
                            &mut ip_id_counter,
                            peers.slots[pi].seq_tx, false, 3,
                        )
                    } else {
                        send_fin_burst_l2(
                            &mut slab, &engine, &mut scheduler,
                            &src_mac, &peers.slots[pi].mac,
                            peers.slots[pi].seq_tx, false, 3,
                        )
                    };
                    fin_total += sent;
                }
            }
            eprintln!("[M13-W{}] FIN sent to {} peers ({}x total). Deadline={}ms.",
                worker_idx, peers.count, fin_total,
                (fin_deadline_ns.saturating_sub(rdtsc_ns(&cal))) / 1_000_000);
        }
        if closing && rdtsc_ns(&cal) >= fin_deadline_ns {
            eprintln!("[M13-W{}] FIN deadline expired. Force-closing.", worker_idx);
            break;
        }
        let now = rdtsc_ns(&cal);
        stats.cycles.value.fetch_add(1, Ordering::Relaxed);
        engine.recycle_tx(&mut slab);
        engine.refill_rx(&mut slab);

        // Stage -1 REMOVED: All RX is now pure AF_XDP (Stage 1 classify).
        // No kernel socket recv_from. Eliminates dual-receiver contention on UDP/443.

        // === SPRINT S2: TUN TX — Zero-copy drain 64/tick + per-peer routing ===
        // Only worker 0 reads TUN to avoid contention/reordering.
        // Speculative slab alloc → read TUN directly into UMEM → route → encapsulate.
        // No intermediate tun_buf — BoringTun/Cloudflare pattern.
        if worker_idx == 0 {
            if let Some(ref mut tun_file) = tun {
                for _tun_batch in 0..256u32 {
                    // Speculative slab alloc — read directly into UMEM payload region
                    let idx = match slab.alloc() {
                        Some(i) => i,
                        None => break, // slab exhausted
                    };
                    let frame_ptr = unsafe { engine.umem_base().add((idx as usize) * FRAME_SIZE as usize) };
                    // TUN payload goes at M13 offset 62 (ETH_HDR_SIZE + M13_HDR_SIZE)
                    let payload_offset = ETH_HDR_SIZE + M13_HDR_SIZE; // 62
                    let tun_slice = unsafe { std::slice::from_raw_parts_mut(frame_ptr.add(payload_offset), 1500) };
                    match tun_file.read(tun_slice) {
                        Ok(n) if n > 0 => {
                            // Route by destination IP in the TUN packet (already in UMEM).
                            // IPv4: dst_ip at offset 16..20 in the IP header.
                            if n < 20 {
                                slab.free(idx);
                                continue; // too short for IP header
                            }
                            // Read dst_ip from UMEM (already there — zero extra copy)
                            let dst_ip = unsafe {
                                [*tun_slice.get_unchecked(16), *tun_slice.get_unchecked(17),
                                 *tun_slice.get_unchecked(18), *tun_slice.get_unchecked(19)]
                            };

                            // Find the peer that owns this tunnel IP.
                            let peer_idx = match peers.lookup_by_tunnel_ip(dst_ip) {
                                Some(idx) => idx,
                                None => {
                                    // Fallback: if only 1 established peer, send to them.
                                    let mut fallback_idx: Option<usize> = None;
                                    let mut established_count = 0u16;
                                    for pi in 0..MAX_PEERS {
                                        if peers.slots[pi].lifecycle == PeerLifecycle::Established {
                                            if fallback_idx.is_none() { fallback_idx = Some(pi); }
                                            established_count += 1;
                                        }
                                    }
                                    if established_count == 1 {
                                        fallback_idx.unwrap()
                                    } else {
                                        slab.free(idx);
                                        continue; // No peer or ambiguous — drop
                                    }
                                }
                            };

                            let peer_slot = &mut peers.slots[peer_idx];
                            if !peer_slot.has_session() {
                                slab.free(idx);
                                continue;
                            }

                            let peer_ip = peer_slot.addr.ip().unwrap_or([0;4]);
                            let peer_port = peer_slot.addr.port().unwrap_or(0);
                            let m13_flen = ETH_HDR_SIZE + M13_HDR_SIZE + n;
                            unsafe {
                                let m13_buf = std::slice::from_raw_parts_mut(frame_ptr, FRAME_SIZE as usize);
                                // Sprint 4b: Copy pre-built header template (46 static bytes)
                                // then stamp only variable fields (seq, flags, payload_len).
                                m13_buf[0..46].copy_from_slice(&m13_hdr_template[0..46]);
                                let udp_seq = peer_slot.next_seq();
                                m13_buf[46..54].copy_from_slice(&udp_seq.to_le_bytes());
                                m13_buf[54] = FLAG_TUNNEL;
                                m13_buf[55..59].copy_from_slice(&(n as u32).to_le_bytes());
                                m13_buf[59..62].copy_from_slice(&m13_hdr_template[59..62]);
                                // Payload already at [62..62+n] — no copy needed
                                if let Some(ref cipher) = peers.ciphers[peer_idx] {
                                    seal_frame(&mut m13_buf[..m13_flen], cipher, udp_seq, DIR_HUB_TO_NODE, ETH_HDR_SIZE);
                                }

                                let total_len = RAW_HDR_LEN + m13_flen;
                                if total_len <= FRAME_SIZE as usize {
                                    std::ptr::copy(frame_ptr, frame_ptr.add(RAW_HDR_LEN), m13_flen);
                                    let raw_frame = std::slice::from_raw_parts_mut(frame_ptr, total_len);
                                    // ETH header
                                    raw_frame[0..6].copy_from_slice(&gateway_mac);
                                    raw_frame[6..12].copy_from_slice(&src_mac);
                                    raw_frame[12..14].copy_from_slice(&0x0800u16.to_be_bytes());
                                    // IP header
                                    let ip_total = (IP_HDR_LEN + UDP_HDR_LEN + m13_flen) as u16;
                                    let ip = &mut raw_frame[14..34];
                                    ip[0] = 0x45; ip[1] = 0x00;
                                    ip[2..4].copy_from_slice(&ip_total.to_be_bytes());
                                    ip[4..6].copy_from_slice(&ip_id_counter.to_be_bytes());
                                    ip_id_counter = ip_id_counter.wrapping_add(1);
                                    ip[6..8].copy_from_slice(&0x4000u16.to_be_bytes());
                                    ip[8] = 64; ip[9] = 17;
                                    ip[10..12].copy_from_slice(&[0, 0]);
                                    ip[12..16].copy_from_slice(&hub_ip);
                                    ip[16..20].copy_from_slice(&peer_ip);
                                    let cksum = ip_checksum(ip);
                                    ip[10..12].copy_from_slice(&cksum.to_be_bytes());
                                    // UDP header
                                    let udp_total = (UDP_HDR_LEN + m13_flen) as u16;
                                    raw_frame[34..36].copy_from_slice(&hub_port.to_be_bytes());
                                    raw_frame[36..38].copy_from_slice(&peer_port.to_be_bytes());
                                    raw_frame[38..40].copy_from_slice(&udp_total.to_be_bytes());
                                    raw_frame[40..42].copy_from_slice(&[0, 0]);

                                    hexdump.dump_tx(raw_frame.as_ptr(), total_len, now);
                                    scheduler.enqueue_bulk((idx as u64) * FRAME_SIZE as u64, total_len as u32);
                                    udp_tx_count += 1;
                                    tun_read_count += 1;
                                    stats.tx_count.value.fetch_add(1, Ordering::Relaxed);
                                } else {
                                    slab.free(idx);
                                }
                            }
                        }
                        _ => {
                            // WouldBlock or EOF — free speculative slab, drain complete
                            slab.free(idx);
                            break;
                        }
                    }
                }
            }
        }
        // === HUB WORKER TELEMETRY (1/sec) ===
        if now.saturating_sub(last_hub_report_ns) > 1_000_000_000 {
            last_hub_report_ns = now;
            let mut established = 0u16;
            for pi in 0..MAX_PEERS {
                if peers.slots[pi].lifecycle == PeerLifecycle::Established { established += 1; }
            }
            eprintln!("[M13-W{}] RX:{} TX:{} TUN_R:{} TUN_W:{} AEAD_OK:{} FAIL:{} Peers:{}/{}",
                worker_idx, udp_rx_count, udp_tx_count,
                tun_read_count, tun_write_count,
                aead_ok_count, aead_fail_count,
                established, peers.count);
        }

        // === STAGE 0: ADAPTIVE BATCH DRAIN ===
        let mut rx_count = engine.poll_rx_batch(&mut rx_batch, &stats);
        if rx_count > 0 && rx_count < GRAPH_BATCH {
            loop {
                engine.recycle_tx(&mut slab); engine.refill_rx(&mut slab);
                let n = engine.poll_rx_batch(&mut rx_batch[rx_count..], &stats);
                rx_count += n;
                if rx_count >= GRAPH_BATCH || rdtsc_ns(&cal) - now >= DEADLINE_NS { break; }
            }
        }

        // === STAGE 0.5: JITTER BUFFER DRAIN ===
        // Release TC_CRITICAL frames whose release time has arrived.
        // Must happen BEFORE classification so buffered frames from previous
        // cycles get scheduled THIS cycle.
        {
            let (rel, _) = jbuf.drain(now, &mut scheduler);
            if rel > 0 {
                // Bridge jitter buffer telemetry (4 Relaxed stores)
                stats.jbuf_depth_us.value.store(jbuf.depth_ns / 1000, Ordering::Relaxed);
                stats.jbuf_jitter_us.value.store(jbuf.estimator.get() / 1000, Ordering::Relaxed);
                stats.jbuf_releases.value.store(jbuf.total_releases, Ordering::Relaxed);
                stats.jbuf_drops.value.store(jbuf.total_drops, Ordering::Relaxed);
            }
        }

        // === STAGE 1: CLASSIFY (3-way split: data / feedback / critical) ===
        // Data (TC_BULK) | Feedback → BBR | TC_CRITICAL → Jitter Buffer
        let rx_batch_ns = if rx_count > 0 { now } else { 0 };
        let (mut data_count, mut ctrl_count, crit_count) = (0usize, 0usize, 0usize);
        // Sprint S2: Deferred TUN write batch — collect during classify, flush after.
        let mut tun_write_indices: [u16; GRAPH_BATCH] = [0; GRAPH_BATCH];
        let mut tun_write_offsets: [u16; GRAPH_BATCH] = [0; GRAPH_BATCH]; // m13_offset per frame
        let mut tun_write_batch: usize = 0;
        for i in 0..rx_count {
            if i + PREFETCH_DIST < rx_count {
                unsafe { prefetch_read_l1(umem.add(rx_batch[i + PREFETCH_DIST].addr as usize + ETH_HDR_SIZE)); }
            }
            // Determine M13 header offset based on encapsulation:
            // - EtherType 0x88B5: raw L2 M13, M13 at offset 14 (ETH_HDR_SIZE)
            // - EtherType 0x0800: IPv4+UDP encapsulated, M13 at offset 42 (14+20+8)
            let frame_base = unsafe { umem.add(rx_batch[i].addr as usize) };
            let frame_len = rx_batch[i].len as usize;
            let ethertype = unsafe { u16::from_be(*(frame_base.add(12) as *const u16)) };

            let m13_offset = if ethertype == 0x88B5 {
                // L2 path: air-gapped WiFi 7 drones. Peer identity = src MAC.
                let peer_mac_wire = unsafe { *(frame_base.add(6) as *const [u8; 6]) };
                let peer_addr = PeerAddr::new_l2(peer_mac_wire);
                let _peer_idx = match peers.lookup_or_insert(peer_addr, peer_mac_wire) {
                    Some(idx) => idx,
                    None => {
                        slab.free((rx_batch[i].addr / FRAME_SIZE as u64) as u32);
                        continue;
                    }
                };
                ETH_HDR_SIZE // 14
            } else if ethertype == 0x0800 && frame_len >= 56 + M13_HDR_SIZE {
                // IPv4 UDP: extract source IP:port as peer identity
                let src_ip = unsafe { *(frame_base.add(26) as *const [u8; 4]) };
                let src_port = unsafe { u16::from_be(*(frame_base.add(34) as *const u16)) };
                // Learn Hub's public IP and gateway MAC from the first inbound packet.
                if hub_ip == [0, 0, 0, 0] {
                    let wire_dst_ip = unsafe { *(frame_base.add(30) as *const [u8; 4]) };
                    hub_ip = wire_dst_ip;
                    eprintln!("[M13-W{}] Learned Hub IP from wire: {}.{}.{}.{}",
                        worker_idx, hub_ip[0], hub_ip[1], hub_ip[2], hub_ip[3]);
                }
                if gateway_mac == [0xFF; 6] {
                    let wire_gw_mac = unsafe { *(frame_base.add(6) as *const [u8; 6]) };
                    gateway_mac = wire_gw_mac;
                    eprintln!("[M13-W{}] Learned Gateway MAC from wire: {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
                        worker_idx, gateway_mac[0], gateway_mac[1], gateway_mac[2],
                        gateway_mac[3], gateway_mac[4], gateway_mac[5]);
                }

                // Sprint S1: Per-peer lookup/insert
                let peer_mac_wire = unsafe { *(frame_base.add(56 + 6) as *const [u8; 6]) }; // M13 ETH src_mac
                let peer_addr = PeerAddr::new_udp(src_ip, src_port);
                let _peer_idx = match peers.lookup_or_insert(peer_addr, peer_mac_wire) {
                    Some(idx) => idx,
                    None => {
                        // Table full — drop
                        slab.free((rx_batch[i].addr / FRAME_SIZE as u64) as u32);
                        continue;
                    }
                };
                // Store peer address for use after classify
                // (src_ip, src_port already extracted above)
                // We'll re-lookup per section below using the frame's src_ip:src_port

                udp_rx_count += 1;
                56 // ETH(14) + IP(20) + UDP(8) + FakeETH(14)
            } else {
                // Unknown encapsulation — drop
                slab.free((rx_batch[i].addr / FRAME_SIZE as u64) as u32);
                continue;
            };

            let m13 = unsafe { &*(frame_base.add(m13_offset) as *const M13Header) };
            // Wire protocol validation: reject frames with wrong magic/version.
            // Defense-in-depth behind BPF EtherType filter.
            if m13.signature[0] != M13_WIRE_MAGIC || m13.signature[1] != M13_WIRE_VERSION {
                slab.free((rx_batch[i].addr / FRAME_SIZE as u64) as u32);
                continue;
            }

            // Sprint S1: Resolve peer index for per-peer state.
            // For L2 frames (0x88B5), no peer table — use legacy path.
            // For UDP (0x0800), peer was inserted during classify above.
            let cur_peer_idx: Option<usize> = if m13_offset == 56 {
                // UDP path: lookup by (src_ip, src_port)
                let src_ip = unsafe { *(frame_base.add(26) as *const [u8; 4]) };
                let src_port = unsafe { u16::from_be(*(frame_base.add(34) as *const u16)) };
                peers.lookup(PeerAddr::new_udp(src_ip, src_port))
            } else {
                // L2 path: lookup by src MAC (air-gapped WiFi 7 drones)
                let src_mac_wire: [u8; 6] = unsafe { *(frame_base.add(6) as *const [u8; 6]) };
                peers.lookup(PeerAddr::new_l2(src_mac_wire))
            };

            // Sprint S1: Detect reconnecting Node during active AEAD session.
            // Per-peer: only reset THIS peer's session, not the global state.
            if let Some(pidx) = cur_peer_idx {
                if peers.slots[pidx].has_session()
                   && m13.signature[2] != 0x01
                   && m13.flags & FLAG_HANDSHAKE == 0 && m13.flags & FLAG_FRAGMENT == 0 {
                    if m13.flags & FLAG_CONTROL != 0 {
                        eprintln!("[M13-W{}] Peer {:?} reconnecting (cleartext CONTROL while AEAD active). Resetting session.",
                            worker_idx, peers.slots[pidx].addr);
                        peers.slots[pidx].reset_session();
                        peers.ciphers[pidx] = None;
                        peers.hs_sidecar[pidx] = None;
                        peers.assemblers[pidx] = Assembler::new();
                        // Send registration echo to this peer
                        let mut echo_m13 = [0u8; 62];
                        echo_m13[0..6].copy_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
                        echo_m13[6..12].copy_from_slice(&src_mac);
                        echo_m13[12] = (ETH_P_M13 >> 8) as u8;
                        echo_m13[13] = (ETH_P_M13 & 0xFF) as u8;
                        echo_m13[14] = M13_WIRE_MAGIC;
                        echo_m13[15] = M13_WIRE_VERSION;
                        echo_m13[54] = FLAG_CONTROL;
                        if peers.slots[pidx].addr.is_udp() {
                            let peer_ip = peers.slots[pidx].addr.ip().unwrap();
                            let peer_port = peers.slots[pidx].addr.port().unwrap();
                            if let Some(idx) = slab.alloc() {
                                let frame_ptr = unsafe { engine.umem_base().add((idx as usize) * FRAME_SIZE as usize) };
                                let total_len = RAW_HDR_LEN + 62;
                                unsafe {
                                    let buf = std::slice::from_raw_parts_mut(frame_ptr, FRAME_SIZE as usize);
                                    build_raw_udp_frame(
                                        buf, &src_mac, &gateway_mac,
                                        hub_ip, peer_ip, hub_port, peer_port,
                                        ip_id_counter, &echo_m13,
                                    );
                                    ip_id_counter = ip_id_counter.wrapping_add(1);
                                }
                                scheduler.enqueue_critical((idx as u64) * FRAME_SIZE as u64, total_len as u32);
                                udp_tx_count += 1;
                            }
                        } else {
                            if send_l2_frame(&mut slab, &engine, &mut scheduler, &peers.slots[pidx].mac, &echo_m13) {
                                udp_tx_count += 1;
                            }
                        }
                    }
                    slab.free((rx_batch[i].addr / FRAME_SIZE as u64) as u32);
                    continue;
                }
            }

            // Sprint S1: AEAD verification on encrypted frames (per-peer session key)
            if m13.signature[2] == 0x01 {
                if let Some(pidx) = cur_peer_idx {
                    let cipher = match peers.ciphers[pidx].as_ref() {
                        Some(c) => c,
                        None => {
                            // Peer has no cached cipher (session not established)
                            aead_fail_count += 1;
                            slab.free((rx_batch[i].addr / FRAME_SIZE as u64) as u32);
                            continue;
                        }
                    };
                    let frame_ptr = unsafe { umem.add(rx_batch[i].addr as usize) };
                    let frame_len = rx_batch[i].len as usize;
                    let frame_mut = unsafe { std::slice::from_raw_parts_mut(frame_ptr, frame_len) };
                    if !open_frame(frame_mut, cipher, DIR_HUB_TO_NODE, m13_offset) {
                        stats.auth_fail.value.fetch_add(1, Ordering::Relaxed);
                        aead_fail_count += 1;
                        slab.free((rx_batch[i].addr / FRAME_SIZE as u64) as u32);
                        continue;
                    }
                    let dec_seq = u64::from_le_bytes(
                        frame_mut[m13_offset + 32..m13_offset + 40].try_into().unwrap()
                    );
                    let _ = dec_seq;
                    peers.slots[pidx].frame_count += 1;
                    aead_ok_count += 1;
                    stats.decrypt_ok.value.fetch_add(1, Ordering::Relaxed);

                    // Per-peer rekey check
                    let established_ns = (peers.slots[pidx].established_rel_s as u64) * 1_000_000_000 + peers.epoch_ns;
                    if peers.slots[pidx].frame_count as u64 >= REKEY_FRAME_LIMIT
                       || now.saturating_sub(established_ns) > REKEY_TIME_LIMIT_NS {
                        eprintln!("[M13-HUB-PQC] Peer {:?} rekey threshold (frames={} time={}s)",
                            peers.slots[pidx].addr, peers.slots[pidx].frame_count,
                            now.saturating_sub(established_ns) / 1_000_000_000);
                        peers.slots[pidx].reset_session();
                        peers.ciphers[pidx] = None;
                        peers.hs_sidecar[pidx] = None;
                    }
                } else {
                    // AEAD frame from unknown peer — drop
                    aead_fail_count += 1;
                    slab.free((rx_batch[i].addr / FRAME_SIZE as u64) as u32);
                    continue;
                }
            }

            // CRITICAL: Re-read flags from decrypted buffer.
            // `m13.flags` is an immutable &M13Header reference created BEFORE open_frame()
            // decrypted the buffer in-place. Since flags (offset +40) is in the encrypted
            // region (offset +32..+48), the original reference holds ciphertext garbage.
            // LLVM may also cache the stale value due to noalias. All flag-based routing
            // below MUST use this freshly-read decrypted value.
            let frame_base = unsafe { umem.add(rx_batch[i].addr as usize) };
            let flags = unsafe { *frame_base.add(m13_offset + 40) };

            // Sprint S1: Fragment reassembly on cold path (per-peer assembler)
            if flags & FLAG_FRAGMENT != 0 {
                let frame_ptr = unsafe { umem.add(rx_batch[i].addr as usize) };
                let frame_len = rx_batch[i].len as usize;
                if frame_len >= m13_offset + M13_HDR_SIZE + FRAG_HDR_SIZE {
                    let frag_hdr = unsafe { &*(frame_ptr.add(m13_offset + M13_HDR_SIZE) as *const FragHeader) };
                    let frag_data_start = m13_offset + M13_HDR_SIZE + FRAG_HDR_SIZE;
                    let frag_msg_id = unsafe { std::ptr::addr_of!((*frag_hdr).frag_msg_id).read_unaligned() };
                    let frag_index = unsafe { std::ptr::addr_of!((*frag_hdr).frag_index).read_unaligned() };
                    let frag_total = unsafe { std::ptr::addr_of!((*frag_hdr).frag_total).read_unaligned() };
                    let frag_offset = unsafe { std::ptr::addr_of!((*frag_hdr).frag_offset).read_unaligned() };
                    let frag_data_len = unsafe { std::ptr::addr_of!((*frag_hdr).frag_len).read_unaligned() } as usize;
                    if frag_data_start + frag_data_len <= frame_len {
                        let frag_data = unsafe { std::slice::from_raw_parts(frame_ptr.add(frag_data_start), frag_data_len) };
                        // Use per-peer assembler
                        let asm_idx = cur_peer_idx.unwrap_or(0);
                        if let Some(reassembled) = peers.assemblers[asm_idx].feed(
                            frag_msg_id, frag_index, frag_total,
                            frag_offset, frag_data, now,
                        ) {
                            // Sprint S1: Route handshake messages to PQC processor (per-peer)
                            if flags & FLAG_HANDSHAKE != 0 && !reassembled.is_empty() {
                                let msg_type = reassembled[0];
                                eprintln!("[M13-W{}] Reassembled handshake type=0x{:02X} len={} peer_idx={}",
                                    worker_idx, msg_type, reassembled.len(), asm_idx);
                                if let Some(pidx) = cur_peer_idx {
                                    let mut hs_seq_tx: u64 = peers.slots[pidx].seq_tx;
                                    match msg_type {
                                        HS_CLIENT_HELLO => {
                                            peers.slots[pidx].lifecycle = PeerLifecycle::Handshaking;
                                            if let Some((hs, server_hello)) = process_client_hello_hub(
                                                &reassembled, &mut hs_seq_tx, now,
                                            ) {
                                                // Frame ServerHello: UDP or L2
                                                let hs_flags = FLAG_CONTROL | FLAG_HANDSHAKE;
                                                let frames = if peers.slots[pidx].addr.is_udp() {
                                                    let peer_ip = peers.slots[pidx].addr.ip().unwrap();
                                                    let peer_port = peers.slots[pidx].addr.port().unwrap();
                                                    build_fragmented_raw_udp(
                                                        &src_mac, &gateway_mac, hub_ip, peer_ip,
                                                        hub_port, peer_port, &server_hello, hs_flags,
                                                        &mut hs_seq_tx, &mut ip_id_counter,
                                                        &mut hexdump, &cal,
                                                    )
                                                } else {
                                                    let peer_mac = &peers.slots[pidx].mac;
                                                    build_fragmented_l2(
                                                        &src_mac, peer_mac, &server_hello, hs_flags,
                                                        &mut hs_seq_tx, &mut hexdump, &cal,
                                                    )
                                                };
                                                for raw_frame in frames {
                                                    if let Some(idx) = slab.alloc() {
                                                        let frame_ptr = unsafe { engine.umem_base().add((idx as usize) * FRAME_SIZE as usize) };
                                                        let flen = raw_frame.len().min(FRAME_SIZE as usize);
                                                        unsafe { std::ptr::copy_nonoverlapping(raw_frame.as_ptr(), frame_ptr, flen); }
                                                        scheduler.enqueue_critical((idx as u64) * FRAME_SIZE as u64, flen as u32);
                                                        udp_tx_count += 1;
                                                    }
                                                }
                                                peers.hs_sidecar[pidx] = Some(hs);
                                                peers.slots[pidx].seq_tx = hs_seq_tx;
                                                eprintln!("[M13-HUB-PQC] ClientHello processed for peer {:?}, ServerHello enqueued.",
                                                    peers.slots[pidx].addr);
                                            }
                                        }
                                        HS_FINISHED => {
                                            if let Some(ref hs) = peers.hs_sidecar[pidx] {
                                                if let Some(key) = process_finished_hub(&reassembled, hs) {
                                                    peers.slots[pidx].session_key = key;
                                                    let ukey = aead::UnboundKey::new(&aead::AES_256_GCM, &key).unwrap();
                                                    peers.ciphers[pidx] = Some(aead::LessSafeKey::new(ukey));
                                                    peers.slots[pidx].frame_count = 0;
                                                    let rel_s = ((now.saturating_sub(peers.epoch_ns)) / 1_000_000_000) as u32;
                                                    peers.slots[pidx].established_rel_s = rel_s;
                                                    peers.slots[pidx].lifecycle = PeerLifecycle::Established;
                                                    peers.hs_sidecar[pidx] = None;
                                                    stats.handshake_ok.value.fetch_add(1, Ordering::Relaxed);
                                                    eprintln!("[M13-HUB-PQC] → Session established for peer {:?} (AEAD active)",
                                                        peers.slots[pidx].addr);
                                                } else {
                                                    peers.hs_sidecar[pidx] = None;
                                                }
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                            } else {
                                eprintln!("[M13-W{}] Reassembled msg_id={} total_len={}",
                                    worker_idx, frag_msg_id, reassembled.len());
                            }
                        }
                    }
                }
                slab.free((rx_batch[i].addr / FRAME_SIZE as u64) as u32);
                continue;
            }
            if flags & FLAG_FEEDBACK != 0 {
                ctrl_indices[ctrl_count] = i as u16; ctrl_count += 1;
            } else if flags & FLAG_TUNNEL != 0 {
                // Sprint S2: Defer TUN write to after classify loop (keep hot classify cache-tight).
                // Store rx_batch index + m13_offset (varies per encapsulation type).
                tun_write_indices[tun_write_batch] = i as u16;
                tun_write_offsets[tun_write_batch] = m13_offset as u16;
                tun_write_batch += 1;
            } else if flags & FLAG_CONTROL != 0 {
                // Sprint 5.21: Check for FIN/FIN-ACK
                if flags & FLAG_FIN != 0 {
                    if closing {
                        eprintln!("[M13-W{}] FIN-ACK received. Graceful close complete.", worker_idx);
                        slab.free((rx_batch[i].addr / FRAME_SIZE as u64) as u32);
                        fin_deadline_ns = 0;
                        continue;
                    } else {
                        // FIN from peer — send FIN-ACK, then evict
                        if let Some(pidx) = cur_peer_idx {
                            eprintln!("[M13-W{}] FIN received from peer {:?}. Sending FIN-ACK.",
                                worker_idx, peers.slots[pidx].addr);
                            if peers.slots[pidx].addr.is_udp() {
                                send_fin_burst_udp(
                                    &mut slab, &engine, &mut scheduler,
                                    &src_mac, &gateway_mac,
                                    hub_ip, peers.slots[pidx].addr.ip().unwrap(),
                                    hub_port, peers.slots[pidx].addr.port().unwrap(),
                                    &mut ip_id_counter,
                                    peers.slots[pidx].seq_tx, true, 3,
                                );
                            } else {
                                send_fin_burst_l2(
                                    &mut slab, &engine, &mut scheduler,
                                    &src_mac, &peers.slots[pidx].mac,
                                    peers.slots[pidx].seq_tx, true, 3,
                                );
                            }
                            peers.evict(pidx);
                        }
                        slab.free((rx_batch[i].addr / FRAME_SIZE as u64) as u32);
                        continue;
                    }
                }
                // Per-peer registration echo: only if peer has no AEAD session yet
                if flags & FLAG_HANDSHAKE == 0 && flags & FLAG_FRAGMENT == 0 {
                    if let Some(pidx) = cur_peer_idx {
                        if !peers.slots[pidx].has_session() {
                            let mut echo_m13 = [0u8; 62];
                            echo_m13[0..6].copy_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
                            echo_m13[6..12].copy_from_slice(&src_mac);
                            echo_m13[12] = (ETH_P_M13 >> 8) as u8;
                            echo_m13[13] = (ETH_P_M13 & 0xFF) as u8;
                            echo_m13[14] = M13_WIRE_MAGIC;
                            echo_m13[15] = M13_WIRE_VERSION;
                            echo_m13[54] = FLAG_CONTROL;
                            if peers.slots[pidx].addr.is_udp() {
                                let peer_ip = peers.slots[pidx].addr.ip().unwrap();
                                let peer_port = peers.slots[pidx].addr.port().unwrap();
                                if let Some(idx) = slab.alloc() {
                                    let frame_ptr = unsafe { engine.umem_base().add((idx as usize) * FRAME_SIZE as usize) };
                                    let total_len = RAW_HDR_LEN + 62;
                                    unsafe {
                                        let buf = std::slice::from_raw_parts_mut(frame_ptr, FRAME_SIZE as usize);
                                        build_raw_udp_frame(
                                            buf, &src_mac, &gateway_mac,
                                            hub_ip, peer_ip, hub_port, peer_port,
                                            ip_id_counter, &echo_m13,
                                        );
                                        ip_id_counter = ip_id_counter.wrapping_add(1);
                                    }
                                    scheduler.enqueue_critical((idx as u64) * FRAME_SIZE as u64, total_len as u32);
                                    udp_tx_count += 1;
                                }
                            } else {
                                if send_l2_frame(&mut slab, &engine, &mut scheduler, &peers.slots[pidx].mac, &echo_m13) {
                                    udp_tx_count += 1;
                                }
                            }
                        }
                    }
                }
                slab.free((rx_batch[i].addr / FRAME_SIZE as u64) as u32);
            } else {
                data_indices[data_count] = i as u16; data_count += 1;
                rx_state.highest_seq = m13.seq_id;
                rx_state.delivered += 1;
                rx_state.last_rx_batch_ns = rx_batch_ns;
                rx_bitmap.mark(m13.seq_id);
            }
        }
        // === SPRINT S2: DEFERRED TUN WRITE BATCH ===
        // All tunnel frames collected during classify are written + freed here.
        // Keeps the hot classify loop free of TUN write() syscalls.
        if tun_write_batch > 0 {
            if let Some(ref mut tun_file) = tun {
                for ti in 0..tun_write_batch {
                    let idx = tun_write_indices[ti] as usize;
                    let m13_off = tun_write_offsets[ti] as usize;
                    let frame_ptr = unsafe { umem.add(rx_batch[idx].addr as usize) };
                    let frame_len = rx_batch[idx].len as usize;
                    let m13 = unsafe { &*(frame_ptr.add(m13_off) as *const M13Header) };
                    let plen = m13.payload_len as usize;
                    let start = m13_off + M13_HDR_SIZE;
                    if start + plen <= frame_len {
                        let payload = unsafe { std::slice::from_raw_parts(frame_ptr.add(start), plen) };
                        let _ = tun_file.write(payload);
                        tun_write_count += 1;
                    }
                    slab.free((rx_batch[idx].addr / FRAME_SIZE as u64) as u32);
                }
            } else {
                // No TUN — just free the slab slots
                for ti in 0..tun_write_batch {
                    let idx = tun_write_indices[ti] as usize;
                    slab.free((rx_batch[idx].addr / FRAME_SIZE as u64) as u32);
                }
            }
        }

        // === STAGE 2: FEEDBACK PROCESSING — BBR ===
        let _ctrl_now_ns = if ctrl_count > 0 { now } else { 0 };
        for i in 0..ctrl_count {
            let d = &rx_batch[ctrl_indices[i] as usize];
            let frame = unsafe { umem.add(d.addr as usize) as *const u8 };
            let flags = unsafe { (*(frame.add(ETH_HDR_SIZE) as *const M13Header)).flags };
            if flags & FLAG_FEEDBACK != 0 && d.len >= FEEDBACK_FRAME_LEN {
                let fb = unsafe { &*(frame.add(ETH_HDR_SIZE + M13_HDR_SIZE) as *const FeedbackFrame) };
                // BBRv3 removed — feedback frames are received but not processed
                let _ = (fb, flags);
            }
            slab.free((d.addr / FRAME_SIZE as u64) as u32);
        }

        // === STAGE 2.5: TC_CRITICAL → JITTER BUFFER INSERT ===
        for i in 0..crit_count {
            let d = &rx_batch[crit_indices[i] as usize];
            let m13 = unsafe { &*(umem.add(d.addr as usize + ETH_HDR_SIZE) as *const M13Header) };
            // Update RFC 3550 EWMA jitter + adaptive D_buf
            jbuf.update_jitter(rx_batch_ns, m13.seq_id);
            // Insert into jitter buffer (slab NOT freed — frame held in UMEM)
            // If overflow drops oldest, free its slab slot to prevent UMEM leak
            if let Some(dropped_addr) = jbuf.insert(d.addr, d.len, rx_batch_ns) {
                slab.free((dropped_addr / FRAME_SIZE as u64) as u32);
            }
        }

        // === STAGE 3: FEEDBACK GENERATION ===
        let rtt_est = FEEDBACK_RTT_DEFAULT_NS;
        if rx_state.needs_feedback(rx_batch_ns, rtt_est) {
            if let Some(idx) = slab.alloc() {
                let frame_ptr = unsafe { umem.add((idx as usize) * FRAME_SIZE as usize) };
                // Feedback frames use broadcast dst MAC (L2 only, not per-peer routed)
                let bcast_mac = [0xFF; 6];
                produce_feedback_frame(frame_ptr, &bcast_mac, &src_mac, &mut rx_state, &mut rx_bitmap, rx_batch_ns, jbuf.tail - jbuf.head);
                scheduler.enqueue_critical((idx as u64) * FRAME_SIZE as u64, FEEDBACK_FRAME_LEN);
            }
        }

        // === STAGE 5: ENQUEUE DATA (full-rate, no pacing) ===
        // Sprint 5.21: When closing, don't send new data — only process RX + FIN.
        if !closing {
            let tx_budget = scheduler.budget(TxPath::available_slots(&mut engine.tx_path) as usize, HW_FILL_MAX);
            let forward_count = data_count.min(tx_budget);
            for i in 0..forward_count {
                let d = &rx_batch[data_indices[i] as usize];
                scheduler.enqueue_bulk(d.addr, d.len);
            }
            for i in forward_count..data_count {
                slab.free((rx_batch[data_indices[i] as usize].addr / FRAME_SIZE as u64) as u32);
                stats.drops.value.fetch_add(1, Ordering::Relaxed);
            }

            // (BBRv3 probe generator removed — data only comes from TUN or forwarded RX)
        } else {
            // Closing: free all received data frames (no forwarding)
            for i in 0..data_count {
                slab.free((rx_batch[data_indices[i] as usize].addr / FRAME_SIZE as u64) as u32);
            }
        }

        // === STAGE 6: SCHEDULE (critical bypasses pacing) ===
        let tx_counter = TxCounter::new();
        scheduler.schedule(&mut engine.tx_path, &tx_counter, usize::MAX);
        stats.tx_count.value.fetch_add(tx_counter.value.load(Ordering::Relaxed), Ordering::Relaxed);



        // Sprint S1: Periodic peer table GC + per-peer assembler GC
        gc_counter += 1;
        if gc_counter % 10000 == 0 {
            peers.gc(now);
            for asm in peers.assemblers.iter_mut() { asm.gc(now); }
        }
    }

    // === GRACEFUL SHUTDOWN CLEANUP ===
    while jbuf.head < jbuf.tail {
        let slot = jbuf.head & (JBUF_CAPACITY - 1);
        slab.free((jbuf.entries[slot].addr / FRAME_SIZE as u64) as u32);
        jbuf.head += 1;
    }
    eprintln!("[M13-W{}] Shutdown complete. Slab: {}/{} free. UDP TX:{} RX:{} TUN_R:{} TUN_W:{} AEAD_OK:{} FAIL:{} Peers:{}",
        worker_idx, slab.available(), SLAB_DEPTH, udp_tx_count, udp_rx_count,
        tun_read_count, tun_write_count, aead_ok_count, aead_fail_count, peers.count);
}

// ============================================================================
// UTILS (unchanged)
// ============================================================================
fn discover_isolated_cores() -> Vec<usize> {
    if let Ok(mock) = std::env::var("M13_MOCK_CMDLINE") {
        if let Some(part) = mock.split_whitespace().find(|p| p.starts_with("isolcpus=")) {
            return parse_cpu_list(part.strip_prefix("isolcpus=").unwrap_or(""));
        }
        return Vec::new();
    }
    match std::fs::read_to_string("/sys/devices/system/cpu/isolated") {
        Ok(s) => parse_cpu_list(s.trim()), Err(_) => Vec::new(),
    }
}
fn parse_cpu_list(list: &str) -> Vec<usize> {
    let mut cores = Vec::new();
    if list.is_empty() { return cores; }
    for part in list.split(',') {
        if part.contains('-') {
            let r: Vec<&str> = part.split('-').collect();
            if r.len() == 2 {
                let s: usize = match r[0].parse() {
                    Ok(v) => v,
                    Err(_) => fatal(E_NO_ISOLATED_CORES, "Invalid CPU range in isolcpus"),
                };
                let e: usize = match r[1].parse() {
                    Ok(v) => v,
                    Err(_) => fatal(E_NO_ISOLATED_CORES, "Invalid CPU range in isolcpus"),
                };
                for i in s..=e { cores.push(i); }
            }
        } else if let Ok(id) = part.parse::<usize>() { cores.push(id); }
    }
    cores.sort(); cores.dedup(); cores
}
fn pin_to_core(core_id: usize) {
    unsafe {
        let mut cpuset: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(core_id, &mut cpuset);
        if libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &cpuset) != 0 {
            fatal(E_AFFINITY_FAIL, "sched_setaffinity failed");
        }
    }
}
fn verify_affinity(expected_core: usize) {
    if std::env::var("M13_MOCK_CMDLINE").is_ok() { return; }
    let tid = unsafe { libc::syscall(libc::SYS_gettid) };
    let path = format!("/proc/self/task/{}/status", tid);
    let file = match File::open(&path) {
        Ok(f) => f, Err(_) => match File::open("/proc/self/status") {
            Ok(f) => f, Err(_) => fatal(E_AFFINITY_VERIFY, "Cannot open status file"),
        }
    };
    for line in BufReader::new(file).lines() {
        if let Ok(l) = line {
            if l.starts_with("Cpus_allowed_list:") {
                let mask = l.split_whitespace().last().unwrap_or("");
                if mask != expected_core.to_string() {
                    fatal(E_AFFINITY_VERIFY, "Core affinity mismatch");
                }
                return;
            }
        }
    }
    fatal(E_AFFINITY_VERIFY, "Could not verify affinity");
}
fn lock_pmu() {
    if std::env::var("M13_MOCK_CMDLINE").is_ok() { return; }
    let mut file = match OpenOptions::new().read(true).write(true).open("/dev/cpu_dma_latency") {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[M13-EXEC] WARNING: PMU lock failed (open): {}. Continuing.", e);
            return;
        }
    };
    if file.write_all(&0i32.to_ne_bytes()).is_err() {
        eprintln!("[M13-EXEC] WARNING: PMU lock failed (write). Continuing.");
        return;
    }
    if file.seek(SeekFrom::Start(0)).is_err() { 
        eprintln!("[M13-EXEC] WARNING: PMU lock failed (seek). Continuing.");
        return;
    }
    let mut buf = [0u8; 4];
    if file.read_exact(&mut buf).is_err() || i32::from_ne_bytes(buf) != 0 {
        eprintln!("[M13-EXEC] WARNING: PMU lock rejected (read!=0). Continuing.");
        return;
    }
    eprintln!("[M13-EXEC] PMU Locked: max_latency=0us (C0 only)");
    mem::forget(file);
}

// ============================================================================
// MONITOR
// ============================================================================
fn run_monitor() {
    eprintln!("[M13-MONITOR] Scanning for active workers...");
    let mut workers = Vec::new();
    for i in 0..MAX_WORKERS {
        if let Some(t) = Telemetry::try_map_worker(i) { workers.push(t); } else { break; }
    }
    if workers.is_empty() {
        eprintln!("[M13-MONITOR] No workers found. Waiting...");
        while workers.is_empty() {
            if let Some(t) = Telemetry::try_map_worker(0) { workers.push(t); break; }
            std::thread::sleep(Duration::from_millis(500));
        }
    }
    eprintln!("[M13-MONITOR] Attached to {} worker(s).", workers.len());
    eprintln!("---------------------------------------------------------------------");
    let mut last_tx = vec![0u64; workers.len()];
    let mut tids = vec![0u32; workers.len()];
    loop {
        let (mut ttx, mut trx, mut _td, mut tpps) = (0u64, 0u64, 0u64, 0u64);
        let mut cs = String::new();
        for (i, w) in workers.iter().enumerate() {
            let tx = w.tx_count.value.load(Ordering::Relaxed);
            let rx = w.rx_count.value.load(Ordering::Relaxed);
            let d = w.drops.value.load(Ordering::Relaxed);
            let pps = tx - last_tx[i]; last_tx[i] = tx;
            ttx += tx; trx += rx; _td += d; tpps += pps;
            if tids[i] == 0 { tids[i] = w.pid.value.load(Ordering::Relaxed); }
            if tids[i] != 0 {
                let (v, n) = read_ctxt_switches(tids[i]);
                if i > 0 { cs.push('|'); }
                cs.push_str(&format!("W{}:{}/{}", i, v, n));
            }
        }
        // BBR state from worker 0
        // Jitter buffer state from worker 0
        let jb_depth = workers[0].jbuf_depth_us.value.load(Ordering::Relaxed);
        let jb_jitter = workers[0].jbuf_jitter_us.value.load(Ordering::Relaxed);
        let jb_rel = workers[0].jbuf_releases.value.load(Ordering::Relaxed);
        let jb_drop = workers[0].jbuf_drops.value.load(Ordering::Relaxed);
        eprint!("\r[TELEM] TX:{:<12} RX:{:<12} PPS:{:<10} JB:{}us/{}us R:{} D:{} CTX:[{}]   ",
            ttx, trx, tpps, jb_depth, jb_jitter, jb_rel, jb_drop, cs);
        std::thread::sleep(Duration::from_secs(1));
    }
}
fn read_ctxt_switches(tid: u32) -> (u64, u64) {
    let path = format!("/proc/{}/status", tid);
    if let Ok(file) = File::open(&path) {
        let (mut v, mut n) = (0u64, 0u64);
        for line in BufReader::new(file).lines() {
            if let Ok(l) = line {
                if l.starts_with("voluntary_ctxt_switches:") {
                    // Intentional unwrap_or(0): monitor graceful degradation
                    v = l.split_whitespace().nth(1).unwrap_or("0").parse().unwrap_or(0);
                } else if l.starts_with("nonvoluntary_ctxt_switches:") {
                    // Intentional unwrap_or(0): monitor graceful degradation
                    n = l.split_whitespace().nth(1).unwrap_or("0").parse().unwrap_or(0);
                }
            }
        }
        (v, n)
    } else { (0, 0) }
}
