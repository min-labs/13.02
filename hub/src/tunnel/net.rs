// M13 HUB — NETWORK UTILITIES
// Raw UDP framing, IP checksum, MAC/gateway resolution, NAT setup.

use crate::protocol::wire::*;

pub const IP_HDR_LEN: usize = 20;
pub const UDP_HDR_LEN: usize = 8;
pub const RAW_HDR_LEN: usize = ETH_HDR_SIZE + IP_HDR_LEN + UDP_HDR_LEN; // 42

/// RFC 1071: Internet checksum.
#[inline]
pub fn ip_checksum(data: &[u8]) -> u16 {
    let mut sum: u32 = 0;
    let mut i = 0;
    while i + 1 < data.len() {
        sum += u16::from_be_bytes([data[i], data[i + 1]]) as u32;
        i += 2;
    }
    if i < data.len() { sum += (data[i] as u32) << 8; }
    while sum >> 16 != 0 { sum = (sum & 0xFFFF) + (sum >> 16); }
    !(sum as u16)
}

/// Construct a raw Ethernet + IPv4 + UDP frame in a buffer.
/// Returns total frame length. Ready for AF_XDP TX.
pub fn build_raw_udp_frame(
    buf: &mut [u8], src_mac: &[u8; 6], dst_mac: &[u8; 6],
    src_ip: [u8; 4], dst_ip: [u8; 4],
    src_port: u16, dst_port: u16, ip_id: u16, payload: &[u8],
) -> usize {
    let payload_len = payload.len();
    let udp_len = UDP_HDR_LEN + payload_len;
    let ip_total_len = IP_HDR_LEN + udp_len;
    let frame_len = ETH_HDR_SIZE + ip_total_len;
    debug_assert!(frame_len <= buf.len(), "frame too large for buffer");

    buf[0..6].copy_from_slice(dst_mac);
    buf[6..12].copy_from_slice(src_mac);
    buf[12..14].copy_from_slice(&0x0800u16.to_be_bytes());

    let ip = &mut buf[14..34];
    ip[0] = 0x45; ip[1] = 0x00;
    ip[2..4].copy_from_slice(&(ip_total_len as u16).to_be_bytes());
    ip[4..6].copy_from_slice(&ip_id.to_be_bytes());
    ip[6..8].copy_from_slice(&0x4000u16.to_be_bytes());
    ip[8] = 64; ip[9] = 17;
    ip[10..12].copy_from_slice(&[0, 0]);
    ip[12..16].copy_from_slice(&src_ip);
    ip[16..20].copy_from_slice(&dst_ip);
    let cksum = ip_checksum(ip);
    ip[10..12].copy_from_slice(&cksum.to_be_bytes());

    let udp = &mut buf[34..42];
    udp[0..2].copy_from_slice(&src_port.to_be_bytes());
    udp[2..4].copy_from_slice(&dst_port.to_be_bytes());
    udp[4..6].copy_from_slice(&(udp_len as u16).to_be_bytes());
    udp[6..8].copy_from_slice(&[0, 0]);

    buf[42..42 + payload_len].copy_from_slice(payload);
    frame_len
}

/// Read hardware MAC from sysfs.
pub fn detect_mac(if_name: &str) -> [u8; 6] {
    let path = format!("/sys/class/net/{}/address", if_name);
    if let Ok(contents) = std::fs::read_to_string(&path) {
        let parts: Vec<u8> = contents.trim().split(':')
            .filter_map(|h| u8::from_str_radix(h, 16).ok()).collect();
        if parts.len() == 6 {
            eprintln!("[M13-EXEC] Detected MAC for {}: {}", if_name, contents.trim());
            return [parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]];
        }
    }
    eprintln!("[M13-EXEC] WARNING: Could not read MAC from sysfs ({}), using LAA fallback", path);
    [0x02, 0x00, 0x00, 0x00, 0x00, 0x01]
}

/// Resolve default gateway MAC from /proc/net/route + /proc/net/arp.
pub fn resolve_gateway_mac(if_name: &str) -> Option<([u8; 6], [u8; 4])> {
    let route_data = std::fs::read_to_string("/proc/net/route").ok()?;
    let mut gw_ip_hex: Option<u32> = None;
    for line in route_data.lines().skip(1) {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() >= 3 && fields[0] == if_name && fields[1] == "00000000" {
            gw_ip_hex = u32::from_str_radix(fields[2], 16).ok();
            break;
        }
    }
    let gw_hex = gw_ip_hex?;
    let gw_ip = gw_hex.to_le_bytes();

    let arp_data = std::fs::read_to_string("/proc/net/arp").ok()?;
    let gw_ip_str = format!("{}.{}.{}.{}", gw_ip[0], gw_ip[1], gw_ip[2], gw_ip[3]);
    let try_resolve = |data: &str| -> Option<[u8; 6]> {
        for line in data.lines().skip(1) {
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() >= 6 && fields[0] == gw_ip_str && fields[5] == if_name {
                let mac_parts: Vec<u8> = fields[3].split(':')
                    .filter_map(|h| u8::from_str_radix(h, 16).ok()).collect();
                if mac_parts.len() == 6 {
                    return Some([mac_parts[0], mac_parts[1], mac_parts[2],
                                 mac_parts[3], mac_parts[4], mac_parts[5]]);
                }
            }
        }
        None
    };
    if let Some(mac) = try_resolve(&arp_data) {
        eprintln!("[M13-NET] Gateway: {} MAC: {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x} dev: {}",
            gw_ip_str, mac[0], mac[1], mac[2], mac[3], mac[4], mac[5], if_name);
        return Some((mac, gw_ip));
    }
    eprintln!("[M13-NET] WARNING: Gateway {} not in ARP cache. Pinging...", gw_ip_str);
    let _ = std::process::Command::new("ping")
        .args(["-c", "1", "-W", "1", &gw_ip_str])
        .stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).status();
    if let Ok(arp2) = std::fs::read_to_string("/proc/net/arp") {
        if let Some(mac) = try_resolve(&arp2) {
            eprintln!("[M13-NET] Gateway: {} MAC resolved (after ARP)", gw_ip_str);
            return Some((mac, gw_ip));
        }
    }
    None
}

/// Read IPv4 address of a network interface via ioctl.
pub fn get_interface_ip(if_name: &str) -> Option<[u8; 4]> {
    unsafe {
        let sock = libc::socket(libc::AF_INET, libc::SOCK_DGRAM, 0);
        if sock < 0 { return None; }
        let mut ifr: libc::ifreq = std::mem::zeroed();
        let name_bytes = if_name.as_bytes();
        let copy_len = name_bytes.len().min(libc::IFNAMSIZ - 1);
        std::ptr::copy_nonoverlapping(name_bytes.as_ptr(), ifr.ifr_name.as_mut_ptr() as *mut u8, copy_len);
        if libc::ioctl(sock, libc::SIOCGIFADDR as libc::c_ulong, &mut ifr) < 0 {
            libc::close(sock); return None;
        }
        libc::close(sock);
        let sa = &*(&ifr.ifr_ifru as *const _ as *const libc::sockaddr_in);
        let ip_u32 = sa.sin_addr.s_addr;
        let ip_ne = ip_u32.to_ne_bytes();
        eprintln!("[M13-NET] Interface {} IP: {}.{}.{}.{}", if_name,
            ip_ne[0], ip_ne[1], ip_ne[2], ip_ne[3]);
        Some(ip_ne)
    }
}
