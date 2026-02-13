/* M13 NODE - DATAPATH KERNEL (REV 6.1)
 * Sprint 6.1: Node AF_XDP Engine, FixedSlab, BPF Steersman, Telemetry
 * Zero-share: completely independent from Hub datapath.
 * Tailored for drone leaf: 4096 frames (vs Hub's 8192), subset telemetry.
 */
use libbpf_sys::{
    xsk_umem__create, xsk_socket__create, xsk_umem_config, xsk_socket_config,
    xsk_ring_prod, xsk_ring_cons, xdp_desc,
    XSK_LIBBPF_FLAGS__INHIBIT_PROG_LOAD,
    bpf_object, bpf_object__open_mem, bpf_object__load, bpf_object__find_program_by_name,
    bpf_program__fd, bpf_object__find_map_by_name, bpf_map__fd,
    bpf_map_update_elem, bpf_set_link_xdp_fd,
    XDP_FLAGS_SKB_MODE, XDP_FLAGS_DRV_MODE, XDP_FLAGS_UPDATE_IF_NOEXIST
};
use libc::{
    mmap, munmap, shm_open, ftruncate, close, ioctl, socket, setsockopt, getsockopt,
    MAP_PRIVATE, MAP_ANONYMOUS, MAP_HUGETLB, MAP_POPULATE, MAP_SHARED,
    PROT_READ, PROT_WRITE, MAP_FAILED, O_CREAT, O_RDWR, S_IRUSR, S_IWUSR,
    c_void, off_t, c_char, AF_INET, SOCK_DGRAM, SOL_SOCKET, MSG_DONTWAIT, sendto,
    SOL_XDP, setrlimit, rlimit, RLIMIT_MEMLOCK, RLIM_INFINITY
};
use std::ptr;
use std::mem;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering, fence};
use std::ffi::CString;
use bytemuck::{Pod, Zeroable};

/// Maximum concurrent worker threads. Node = single worker (one WiFi NIC).
#[allow(dead_code)] pub const MAX_WORKERS: usize = 1;
/// UMEM region size (512 MB — drone has less RAM than Hub's 1 GB).
pub const UMEM_SIZE: usize = 512 * 1024 * 1024;
/// UMEM frame size. Each frame holds one Ethernet packet + headroom.
pub const FRAME_SIZE: u32 = 4096;
pub const SHM_NAME_PREFIX: &str = "/m13_node_telem_";
pub const SO_BUSY_POLL: i32 = 46;
pub const XDP_MMAP_OFFSETS: i32 = 1;
pub const XDP_RING_NEED_WAKEUP: u32 = 1;

// ============================================================================
// DIAGNOSTIC ERROR CODES (Node-specific)
// Convention: 0x10-0x1F = Boot, 0x20-0x2F = Runtime, 0x40-0x4F = Link
// ============================================================================
pub const E_NO_ISOLATED_CORES: i32  = 0x10;
pub const E_AFFINITY_FAIL: i32      = 0x11;
#[allow(dead_code)] pub const E_PMU_LOCK_FAIL: i32      = 0x12;
#[allow(dead_code)] pub const E_AFFINITY_VERIFY: i32    = 0x13;
pub const E_UMEM_ALLOC_FAIL: i32    = 0x14;
pub const E_XSK_BIND_FAIL: i32      = 0x15;
pub const E_RING_SIZE_FAIL: i32     = 0x16;
pub const E_BPF_LOAD_FAIL: i32      = 0x17;
pub const E_SHM_MAP_FAIL: i32       = 0x18;
#[allow(dead_code)] pub const E_IRQ_FENCE_FAIL: i32     = 0x19;
#[allow(dead_code)] pub const E_SLAB_EXHAUSTION: i32     = 0x22;
#[allow(dead_code)] pub const E_PEER_TIMEOUT: i32        = 0x40;
#[allow(dead_code)] pub const E_REGISTRATION_FAIL: i32   = 0x41;

/// Structured fatal exit. Identical semantics to Hub version — writev atomicity.
#[inline(never)]
pub fn fatal(code: i32, msg: &str) -> ! {
    let prefix = b"[M13-NODE FATAL 0x";
    let hex = [
        b"0123456789ABCDEF"[((code >> 4) & 0xF) as usize],
        b"0123456789ABCDEF"[(code & 0xF) as usize],
    ];
    let suffix = b"] ";
    let newline = b"\n";
    let iov = [
        libc::iovec { iov_base: prefix.as_ptr() as *mut _, iov_len: prefix.len() },
        libc::iovec { iov_base: hex.as_ptr() as *mut _, iov_len: 2 },
        libc::iovec { iov_base: suffix.as_ptr() as *mut _, iov_len: suffix.len() },
        libc::iovec { iov_base: msg.as_ptr() as *mut _, iov_len: msg.len() },
        libc::iovec { iov_base: newline.as_ptr() as *mut _, iov_len: 1 },
    ];
    unsafe { libc::writev(2, iov.as_ptr(), 5); }
    std::process::exit(code);
}

#[allow(non_upper_case_globals)]
#[allow(non_camel_case_types)]
#[allow(non_snake_case)]
#[allow(dead_code)]
mod bindings { include!(concat!(env!("OUT_DIR"), "/bindings.rs")); }
use bindings::{ethtool_ringparam, ifreq, SIOCETHTOOL, ETHTOOL_GRINGPARAM};

const BPF_OBJ_BYTES: &[u8] = include_bytes!(env!("BPF_OBJECT_PATH"));

/// IEEE 802.1 Local Experimental EtherType for M13 raw Ethernet frames.
pub const ETH_P_M13: u16 = 0x88B5;
/// Wire protocol magic byte. Stored in M13Header.signature[0].
pub const M13_WIRE_MAGIC: u8 = 0xD1;
/// Wire protocol version. Phase 1 = 0x01.
pub const M13_WIRE_VERSION: u8 = 0x01;

/// IEEE 802.3 Ethernet header. 14 bytes on wire: dst(6) + src(6) + ethertype(2).
#[repr(C, packed)] #[derive(Copy, Clone, Pod, Zeroable)]
pub struct EthernetHeader { pub dst: [u8; 6], pub src: [u8; 6], pub ethertype: u16 }

/// M13 wire protocol header. 48 bytes. Carried after EthernetHeader.
/// signature[0]=magic(0xD1), signature[1]=version(0x01), [2..32]=reserved(Phase 2 crypto).
#[repr(C, packed)] #[derive(Copy, Clone, Pod, Zeroable)]
pub struct M13Header {
    pub signature: [u8; 32], pub seq_id: u64, pub flags: u8,
    pub payload_len: u32, pub padding: [u8; 3],
}
const _: () = assert!(std::mem::size_of::<M13Header>() == 48);

// ============================================================================
// TELEMETRY (per-worker sharded shared memory — Node subset, no jitter buffer)
// ============================================================================
#[repr(align(128))] pub struct CachePadded<T> { pub value: T }

/// Node telemetry — subset of Hub's. No BBR, no jitter buffer metrics.
#[repr(C)]
pub struct Telemetry {
    pub tx_count: CachePadded<AtomicU64>, pub rx_count: CachePadded<AtomicU64>,
    pub drops: CachePadded<AtomicU64>, pub cycles: CachePadded<AtomicU64>,
    pub pid: CachePadded<AtomicU32>,
    pub node_state: CachePadded<AtomicU32>,  // NodeState as u32 (0=Disconnected,1=Registering,2=Established)
}

pub struct TelemetryPtr(*mut Telemetry);
unsafe impl Send for TelemetryPtr {}
impl std::ops::Deref for TelemetryPtr {
    type Target = Telemetry;
    fn deref(&self) -> &Telemetry { unsafe { &*self.0 } }
}

impl Telemetry {
    pub fn map_worker(worker_idx: usize, is_owner: bool) -> TelemetryPtr {
        let name = format!("{}{}", SHM_NAME_PREFIX, worker_idx);
        match Self::map_named(&name, is_owner) {
            Some(t) => t,
            None => fatal(E_SHM_MAP_FAIL, "Telemetry shm map failed"),
        }
    }
    #[allow(dead_code)] pub fn try_map_worker(worker_idx: usize) -> Option<TelemetryPtr> {
        let name = format!("{}{}", SHM_NAME_PREFIX, worker_idx);
        Self::map_named(&name, false)
    }
    fn map_named(name: &str, is_owner: bool) -> Option<TelemetryPtr> {
        unsafe {
            let c_name = match CString::new(name) {
                Ok(c) => c,
                Err(_) => fatal(E_SHM_MAP_FAIL, "SHM name contains null byte"),
            };
            let mut fd = shm_open(c_name.as_ptr(), O_RDWR, 0);
            if is_owner {
                if fd < 0 { fd = shm_open(c_name.as_ptr(), O_CREAT | O_RDWR, S_IRUSR | S_IWUSR); }
                if fd < 0 { return None; }
                if ftruncate(fd, mem::size_of::<Telemetry>() as off_t) != 0 {
                    close(fd);
                    return None;
                }
            } else if fd < 0 { return None; }
            let ptr = mmap(ptr::null_mut(), mem::size_of::<Telemetry>(), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            close(fd);
            if ptr == MAP_FAILED { return None; }
            if is_owner { ptr::write_bytes(ptr, 0, mem::size_of::<Telemetry>()); }
            Some(TelemetryPtr(ptr as *mut Telemetry))
        }
    }
}

// ============================================================================
// SLAB ALLOCATOR — 4096 frames for Node (vs Hub's 8192)
// ============================================================================
#[repr(align(64))]
pub struct FixedSlab { stack: Box<[u32]>, top: usize, capacity: usize }
impl FixedSlab {
    pub fn new(capacity: usize) -> Self {
        let mut vec = Vec::with_capacity(capacity);
        for i in 0..capacity { vec.push(i as u32); }
        FixedSlab { stack: vec.into_boxed_slice(), top: capacity, capacity }
    }
    #[inline(always)] pub fn alloc(&mut self) -> Option<u32> {
        if self.top == 0 { return None; }
        self.top -= 1; unsafe { Some(*self.stack.get_unchecked(self.top)) }
    }
    #[inline(always)] pub fn free(&mut self, idx: u32) {
        if self.top < self.capacity { unsafe { *self.stack.get_unchecked_mut(self.top) = idx; } self.top += 1; }
    }
    #[inline(always)] pub fn available(&self) -> usize { self.top }
}

// ============================================================================
// TX PATH TRAIT
// ============================================================================
#[allow(dead_code)]
pub trait TxPath {
    fn available_slots(&mut self) -> u32;
    fn stage_tx(&mut self, frame_idx: u32, len: u32);
    fn stage_tx_addr(&mut self, addr: u64, len: u32);
    fn commit_tx(&mut self);
    fn kick_tx(&mut self);
}

pub struct ZeroCopyTx { tx: RingProd, sock_fd: i32 }
impl TxPath for ZeroCopyTx {
    #[inline(always)] fn available_slots(&mut self) -> u32 { unsafe { self.tx.available() } }
    #[inline(always)] fn stage_tx(&mut self, frame_idx: u32, len: u32) { unsafe { self.tx.stage(frame_idx, len) } }
    #[inline(always)] fn stage_tx_addr(&mut self, addr: u64, len: u32) { unsafe { self.tx.stage_addr_desc(addr, len) } }
    #[inline(always)] fn commit_tx(&mut self) { unsafe { self.tx.commit() } }
    #[inline(always)] fn kick_tx(&mut self) { unsafe { if self.tx.needs_wakeup() { sendto(self.sock_fd, ptr::null(), 0, MSG_DONTWAIT, ptr::null(), 0); } } }
}

// ============================================================================
// ENGINE (Owns UMEM, XSK socket, all rings)
// ============================================================================
pub struct Engine<T: TxPath> {
    umem_area: *mut u8,
    #[allow(dead_code)] _umem_handle: *mut libbpf_sys::xsk_umem,
    #[allow(dead_code)] sock_handle: *mut libbpf_sys::xsk_socket,
    cq: RingCons, rx: RingCons, fq: RingProd,
    pub tx_path: T,
}
unsafe impl<T: TxPath> Send for Engine<T> {}

impl Engine<ZeroCopyTx> {
    pub fn new_zerocopy(if_name: &str, queue_id: i32, bpf_map_fd: i32) -> Self {
        check_nic_limits(if_name);
        let is_sim = std::env::var("M13_SIMULATION").is_ok();
        let mut flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE;
        if !is_sim { flags |= MAP_HUGETLB; }
        let umem_area = unsafe { mmap(ptr::null_mut(), UMEM_SIZE, PROT_READ | PROT_WRITE, flags, -1, 0) };
        if umem_area == MAP_FAILED { fatal(E_UMEM_ALLOC_FAIL, "UMEM mmap failed (check hugepages)"); }
        let umem_cfg = xsk_umem_config { fill_size: 4096, comp_size: 4096, frame_size: FRAME_SIZE, frame_headroom: 0, flags: 0 };
        let mut umem_handle: *mut libbpf_sys::xsk_umem = ptr::null_mut();
        let mut fq_def: xsk_ring_prod = unsafe { mem::zeroed() };
        let mut cq_def: xsk_ring_cons = unsafe { mem::zeroed() };
        let ret = unsafe { xsk_umem__create(&mut umem_handle, umem_area as *mut c_void, UMEM_SIZE as u64, &mut fq_def, &mut cq_def, &umem_cfg) };
        if ret != 0 { fatal(E_UMEM_ALLOC_FAIL, "xsk_umem__create failed"); }
        let bind_flags = if is_sim { 1 << 1 } else { 1 << 2 };
        let mut sock_cfg: xsk_socket_config = unsafe { mem::zeroed() };
        sock_cfg.rx_size = 2048; sock_cfg.tx_size = 2048;
        sock_cfg.libbpf_flags = XSK_LIBBPF_FLAGS__INHIBIT_PROG_LOAD;
        sock_cfg.xdp_flags = 0; sock_cfg.bind_flags = bind_flags as u16;
        let mut sock_handle: *mut libbpf_sys::xsk_socket = ptr::null_mut();
        let mut rx_def: xsk_ring_cons = unsafe { mem::zeroed() };
        let mut tx_def: xsk_ring_prod = unsafe { mem::zeroed() };
        let c_ifname = match CString::new(if_name) {
            Ok(c) => c,
            Err(_) => fatal(E_XSK_BIND_FAIL, "Interface name contains null byte"),
        };
        let ret = unsafe { xsk_socket__create(&mut sock_handle, c_ifname.as_ptr(), queue_id as u32, umem_handle, &mut rx_def, &mut tx_def, &sock_cfg) };
        if ret != 0 { fatal(E_XSK_BIND_FAIL, "xsk_socket__create failed"); }
        let sock_fd = unsafe { libbpf_sys::xsk_socket__fd(sock_handle) };
        unsafe {
            let key = queue_id; let val = sock_fd;
            let ret = bpf_map_update_elem(bpf_map_fd, &key as *const _ as *const c_void, &val as *const _ as *const c_void, 0);
            if ret != 0 { fatal(E_XSK_BIND_FAIL, "BPF map update failed (xsks_map)"); }
        }
        let poll_us: i32 = 50;
        let ret = unsafe { setsockopt(sock_fd, SOL_SOCKET, SO_BUSY_POLL, &poll_us as *const _ as *const c_void, 4) };
        if ret != 0 {
            unsafe { libc::write(2, b"[M13-NODE-WARN] SO_BUSY_POLL not supported\n".as_ptr() as _, 43); }
        }
        let mut offsets = XdpMmapOffsets::default();
        let mut optlen = mem::size_of::<XdpMmapOffsets>() as u32;
        let ret = unsafe { getsockopt(sock_fd, SOL_XDP, XDP_MMAP_OFFSETS, &mut offsets as *mut _ as *mut c_void, &mut optlen) };
        if ret != 0 { fatal(E_XSK_BIND_FAIL, "getsockopt XDP_MMAP_OFFSETS failed"); }
        unsafe {
            let tx_flags = (tx_def.producer as *mut u8).sub(offsets.tx.producer as usize).add(offsets.tx.flags as usize) as *mut u32;
            let fq_flags = (fq_def.producer as *mut u8).sub(offsets.fr.producer as usize).add(offsets.fr.flags as usize) as *mut u32;
            let tx_strategy = ZeroCopyTx { tx: RingProd::new(&tx_def, tx_flags), sock_fd };
            let rx_ring = RingCons::new(&rx_def);
            let fq_ring = RingProd::new(&fq_def, fq_flags);
            let cq_ring = RingCons::new(&cq_def);
            Engine { umem_area: umem_area as *mut u8, _umem_handle: umem_handle, sock_handle, cq: cq_ring, rx: rx_ring, fq: fq_ring, tx_path: tx_strategy }
        }
    }

    pub fn get_frame_ptr(&self, idx: u32) -> *mut u8 { unsafe { self.umem_area.add((idx * FRAME_SIZE) as usize) } }
    #[inline(always)] pub fn umem_base(&self) -> *mut u8 { self.umem_area }

    pub fn recycle_tx(&mut self, allocator: &mut FixedSlab) -> usize { unsafe { self.cq.consume_addr(allocator) } }

    pub fn refill_rx(&mut self, allocator: &mut FixedSlab) {
        let count = unsafe { self.fq.available() } as usize;
        let batch = std::cmp::min(count, 16);
        if batch > 0 { self.refill_internal(allocator, batch); }
    }
    pub fn refill_rx_full(&mut self, allocator: &mut FixedSlab) {
        let count = unsafe { self.fq.available() } as usize;
        if count > 0 { self.refill_internal(allocator, count); }
    }
    fn refill_internal(&mut self, allocator: &mut FixedSlab, count: usize) {
        unsafe {
            let mut added = 0;
            for _ in 0..count {
                if let Some(idx) = allocator.alloc() {
                    self.fq.stage_addr((idx as u64) * (FRAME_SIZE as u64));
                    added += 1;
                } else { break; }
            }
            if added > 0 { self.fq.commit(); }
        }
    }

    #[inline(always)]
    pub fn poll_rx_batch(&mut self, out: &mut [xdp_desc], stats: &Telemetry) -> usize {
        unsafe { self.rx.consume_batch(out, out.len(), stats) }
    }
}

impl<T: TxPath> Drop for Engine<T> {
    fn drop(&mut self) { unsafe { munmap(self.umem_area as *mut c_void, UMEM_SIZE); } }
}

// ============================================================================
// RING OPERATIONS (Lock-free SPSC with explicit memory barriers)
// ============================================================================
struct RingProd { producer: *mut u32, consumer: *mut u32, ring: *mut c_void, flags: *mut u32, mask: u32, cached_cons: u32, local_prod: u32 }
struct RingCons { producer: *mut u32, consumer: *mut u32, ring: *mut c_void, mask: u32 }

impl RingProd {
    unsafe fn new(r: *const xsk_ring_prod, flags: *mut u32) -> Self {
        let prod_ptr = (*r).producer as *mut AtomicU32;
        let init_prod = (*prod_ptr).load(Ordering::Relaxed);
        RingProd { producer: (*r).producer, consumer: (*r).consumer, ring: (*r).ring, flags, mask: (*r).mask as u32, cached_cons: 0, local_prod: init_prod }
    }
    #[inline(always)] unsafe fn needs_wakeup(&self) -> bool { ptr::read_volatile(self.flags) & XDP_RING_NEED_WAKEUP != 0 }
    #[inline(always)] unsafe fn available(&mut self) -> u32 {
        self.cached_cons = (*(self.consumer as *mut AtomicU32)).load(Ordering::Acquire);
        (self.mask + 1) - (self.local_prod - self.cached_cons)
    }
    #[inline(always)] unsafe fn stage(&mut self, frame_idx: u32, len: u32) {
        let desc = (self.ring as *mut xdp_desc).offset((self.local_prod & self.mask) as isize);
        (*desc).addr = (frame_idx as u64) * FRAME_SIZE as u64; (*desc).len = len; (*desc).options = 0;
        self.local_prod = self.local_prod.wrapping_add(1);
    }
    #[inline(always)] unsafe fn stage_addr(&mut self, addr: u64) {
        let ptr = (self.ring as *mut u64).offset((self.local_prod & self.mask) as isize);
        *ptr = addr;
        self.local_prod = self.local_prod.wrapping_add(1);
    }
    #[inline(always)] #[allow(dead_code)] unsafe fn stage_addr_desc(&mut self, addr: u64, len: u32) {
        let desc = (self.ring as *mut xdp_desc).offset((self.local_prod & self.mask) as isize);
        (*desc).addr = addr; (*desc).len = len; (*desc).options = 0;
        self.local_prod = self.local_prod.wrapping_add(1);
    }
    #[inline(always)] unsafe fn commit(&mut self) {
        let prod_ptr = self.producer as *mut AtomicU32;
        fence(Ordering::Release);
        (*prod_ptr).store(self.local_prod, Ordering::Relaxed);
    }
}

impl RingCons {
    unsafe fn new(r: *const xsk_ring_cons) -> Self {
        RingCons { producer: (*r).producer, consumer: (*r).consumer, ring: (*r).ring, mask: (*r).mask as u32 }
    }
    #[inline(always)] unsafe fn consume_addr(&mut self, allocator: &mut FixedSlab) -> usize {
        let prod_ptr = self.producer as *mut AtomicU32;
        let cons_ptr = self.consumer as *mut AtomicU32;
        let cons_val = (*cons_ptr).load(Ordering::Relaxed);
        let prod_val = (*prod_ptr).load(Ordering::Relaxed);
        fence(Ordering::Acquire);
        let available = prod_val.wrapping_sub(cons_val);
        if available == 0 { return 0; }
        let addr_arr = self.ring as *mut u64;
        for i in 0..available {
            let addr = *addr_arr.offset(((cons_val + i) & self.mask) as isize);
            allocator.free((addr / FRAME_SIZE as u64) as u32);
        }
        (*cons_ptr).store(cons_val.wrapping_add(available), Ordering::Release);
        available as usize
    }
    #[inline(always)] unsafe fn consume_batch(&mut self, out: &mut [xdp_desc], limit: usize, stats: &Telemetry) -> usize {
        let prod_ptr = self.producer as *mut AtomicU32;
        let cons_ptr = self.consumer as *mut AtomicU32;
        let cons_val = (*cons_ptr).load(Ordering::Relaxed);
        let prod_val = (*prod_ptr).load(Ordering::Relaxed);
        fence(Ordering::Acquire);
        let available = prod_val.wrapping_sub(cons_val) as usize;
        if available == 0 { return 0; }
        let count = available.min(limit);
        let desc_arr = self.ring as *const xdp_desc;
        for i in 0..count {
            out[i] = *desc_arr.add((cons_val.wrapping_add(i as u32) & self.mask) as usize);
        }
        (*cons_ptr).store(cons_val.wrapping_add(count as u32), Ordering::Release);
        stats.rx_count.value.fetch_add(count as u64, Ordering::Relaxed);
        count
    }
}

fn check_nic_limits(if_name: &str) {
    let fd = unsafe { socket(AF_INET, SOCK_DGRAM, 0) };
    if fd < 0 { fatal(E_RING_SIZE_FAIL, "Failed to open probe socket"); }
    let mut gring: ethtool_ringparam = unsafe { mem::zeroed() };
    gring.cmd = ETHTOOL_GRINGPARAM;
    let mut ifr: ifreq = unsafe { mem::zeroed() };
    if if_name.len() >= 16 { fatal(E_XSK_BIND_FAIL, "Interface name exceeds IFNAMSIZ"); }
    unsafe {
        ptr::copy_nonoverlapping(if_name.as_ptr() as *const c_char, ifr.ifr_ifrn.ifrn_name.as_mut_ptr(), if_name.len());
        ifr.ifr_ifru.ifru_data = &mut gring as *mut _ as *mut c_void;
    }
    let ret = unsafe { ioctl(fd, SIOCETHTOOL as u64, &mut ifr) };
    unsafe { close(fd); }
    if ret != 0 && std::env::var("M13_SIMULATION").is_err() {
        fatal(E_RING_SIZE_FAIL, "SIOCETHTOOL ioctl failed");
    }
    if gring.tx_max_pending == 0 && std::env::var("M13_SIMULATION").is_err() { fatal(E_RING_SIZE_FAIL, "SIOCETHTOOL query returned zero"); }
    if gring.rx_max_pending > 0 && 2048 > gring.rx_max_pending { fatal(E_RING_SIZE_FAIL, "NIC RX ring too small for 2048"); }
    if gring.tx_max_pending > 0 && 2048 > gring.tx_max_pending { fatal(E_RING_SIZE_FAIL, "NIC TX ring too small for 2048"); }
}

// ============================================================================
// BPF STEERSMAN (Node-independent binary from Hub)
// ============================================================================
pub struct BpfSteersman { #[allow(dead_code)] obj: *mut bpf_object, map_fd: i32, if_index: i32 }
unsafe impl Send for BpfSteersman {}
impl BpfSteersman {
    pub fn load_and_attach(if_name: &str) -> Self {
        unsafe {
            let needed = (UMEM_SIZE + 16 * 1024 * 1024) as u64;
            let rlim = rlimit { rlim_cur: needed, rlim_max: needed };
            if setrlimit(RLIMIT_MEMLOCK, &rlim) != 0 {
                let rlim = rlimit { rlim_cur: RLIM_INFINITY, rlim_max: RLIM_INFINITY };
                setrlimit(RLIMIT_MEMLOCK, &rlim);
            }
        }
        let c_ifname = match CString::new(if_name) {
            Ok(c) => c,
            Err(_) => fatal(E_BPF_LOAD_FAIL, "Interface name contains null byte"),
        };
        let if_index = unsafe { libc::if_nametoindex(c_ifname.as_ptr()) } as i32;
        if if_index == 0 { fatal(E_BPF_LOAD_FAIL, "Interface not found"); }
        unsafe {
            let mut opts: libbpf_sys::bpf_object_open_opts = mem::zeroed();
            opts.sz = mem::size_of::<libbpf_sys::bpf_object_open_opts>() as u64;
            let obj = bpf_object__open_mem(BPF_OBJ_BYTES.as_ptr() as *const c_void, BPF_OBJ_BYTES.len() as u64, &opts);
            if obj.is_null() { fatal(E_BPF_LOAD_FAIL, "BPF object open failed"); }
            let ret = bpf_object__load(obj);
            if ret != 0 { fatal(E_BPF_LOAD_FAIL, "BPF object load failed"); }
            let prog_name = match CString::new("m13_steersman") {
                Ok(c) => c,
                Err(_) => fatal(E_BPF_LOAD_FAIL, "BPF prog name null byte"),
            };
            let prog = bpf_object__find_program_by_name(obj, prog_name.as_ptr());
            let prog_fd = bpf_program__fd(prog);
            let map_name = match CString::new("xsks_map") {
                Ok(c) => c,
                Err(_) => fatal(E_BPF_LOAD_FAIL, "BPF map name null byte"),
            };
            let map = bpf_object__find_map_by_name(obj, map_name.as_ptr());
            let map_fd = bpf_map__fd(map);
            let is_sim = std::env::var("M13_SIMULATION").is_ok();
            let mut flags = if is_sim { XDP_FLAGS_SKB_MODE } else { XDP_FLAGS_DRV_MODE };
            flags |= XDP_FLAGS_UPDATE_IF_NOEXIST;
            let mut ret = bpf_set_link_xdp_fd(if_index, prog_fd, flags);
            if ret != 0 && !is_sim {
                flags = XDP_FLAGS_SKB_MODE | XDP_FLAGS_UPDATE_IF_NOEXIST;
                ret = bpf_set_link_xdp_fd(if_index, prog_fd, flags);
            }
            if ret != 0 { fatal(E_BPF_LOAD_FAIL, "BPF XDP attach failed"); }
            BpfSteersman { obj, map_fd, if_index }
        }
    }
    pub fn map_fd(&self) -> i32 { self.map_fd }
}
impl Drop for BpfSteersman { fn drop(&mut self) { unsafe { if self.if_index > 0 { bpf_set_link_xdp_fd(self.if_index, -1, 0); } } } }

#[repr(C)] #[derive(Default, Debug)] struct XdpMmapOffsets { rx: XdpRingOffset, tx: XdpRingOffset, fr: XdpRingOffset, cr: XdpRingOffset }
#[repr(C)] #[derive(Default, Debug)] struct XdpRingOffset { producer: u64, consumer: u64, desc: u64, flags: u64 }
