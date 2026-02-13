/* * M13 PHASE I - BUILD ORCHESTRATOR (REV 4.13.1)
 * TARGET: Hybrid (Vitis HLS / Clang BPF)
 * FUNCTION: Generates & Compiles the eBPF Steersman.
 * FIX: Added 'ifreq' to bindgen allowlist.
 */
use std::process::Command;
use std::fs;
use std::env;
use std::path::PathBuf;

// 1. THE EBPF STEERSMAN (Embedded C Source)
const BPF_SOURCE: &str = r#"
#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_endian.h>
#include <linux/if_ether.h>

/* Manual definitions to avoid header dependency hell */
struct iphdr {
    __u8 ihl:4, version:4;
    __u8 tos;
    __be16 tot_len;
    __be16 id;
    __be16 frag_off;
    __u8 ttl;
    __u8 protocol;
    __sum16 check;
    __be32 saddr;
    __be32 daddr;
};
struct udphdr {
    __be16 source;
    __be16 dest;
    __be16 len;
    __sum16 check;
};

struct {
    __uint(type, BPF_MAP_TYPE_XSKMAP);
    __uint(max_entries, 64);
    __uint(key_size, sizeof(int));
    __uint(value_size, sizeof(int));
} xsks_map SEC(".maps");

SEC("xdp")
int m13_steersman(struct xdp_md *ctx) {
    void *data     = (void *)(long)ctx->data;
    void *data_end = (void *)(long)ctx->data_end;

    /* Bounds check: pass (not drop) to avoid killing SSH on malformed pkts. */
    if (data + sizeof(struct ethhdr) > data_end)
        return XDP_PASS;

    struct ethhdr *eth = data;

    /* Path 1: Raw L2 M13 (EtherType 0x88B5) — direct datacenter mode. */
    if (eth->h_proto == bpf_htons(0x88B5))
        return bpf_redirect_map(&xsks_map, ctx->rx_queue_index, XDP_PASS);

    /* Path 2: IPv4 UDP port 443 — Internet/UDP encapsulated M13. */
    if (eth->h_proto == bpf_htons(0x0800)) { // ETH_P_IP
        if (data + sizeof(struct ethhdr) + sizeof(struct iphdr) > data_end)
            return XDP_PASS;

        struct iphdr *ip = data + sizeof(struct ethhdr);
        if (ip->protocol != 17) // IPPROTO_UDP
            return XDP_PASS;

        if (data + sizeof(struct ethhdr) + sizeof(struct iphdr) + sizeof(struct udphdr) > data_end)
            return XDP_PASS;

        struct udphdr *udp = data + sizeof(struct ethhdr) + sizeof(struct iphdr);
        if (udp->dest == bpf_htons(443))
            return bpf_redirect_map(&xsks_map, ctx->rx_queue_index, XDP_PASS);
    }

    /* Everything else (SSH, ARP, IPv6, LLDP, etc.) passes to kernel. */
    return XDP_PASS;
}

char _license[] SEC("license") = "GPL";
"#;

// 2. HLS PLACEHOLDER
const HLS_SOURCE: &str = "#ifndef __SYNTHESIS__\nint main(){return 0;}\n#endif";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    
    let out_dir = PathBuf::from(match env::var("OUT_DIR") {
        Ok(v) => v,
        Err(_) => { eprintln!("[M13-BUILD] OUT_DIR not set"); std::process::exit(1); }
    });
    let bpf_src = out_dir.join("m13_xdp.c");
    let bpf_obj = out_dir.join("m13_xdp.o");

    // A. COMPILE BPF PROGRAM
    if fs::write(&bpf_src, BPF_SOURCE).is_err() {
        eprintln!("[M13-BUILD] Failed to write BPF source");
        std::process::exit(1);
    }

    if Command::new("clang").arg("--version").output().is_ok() {
        println!("cargo:warning=[M13-BUILD] Compiling BPF Steersman...");
        let status = Command::new("clang")
            .arg("-O2")
            .arg("-g")
            .arg("-target").arg("bpf")
            .arg("-c").arg(&bpf_src)
            .arg("-o").arg(&bpf_obj)
            .status();
        let status = match status {
            Ok(s) => s,
            Err(_) => { eprintln!("[M13-BUILD] Failed to execute clang"); std::process::exit(1); }
        };
        
        if !status.success() {
            eprintln!("[M13-BUILD] BPF compilation failed. Install clang/libbpf-dev.");
                std::process::exit(1);
        } else {
            println!("cargo:rustc-env=BPF_OBJECT_PATH={}", bpf_obj.display());
        }
    } else {
        println!("cargo:warning=[M13-BUILD] Clang not found. BPF Steersman skipped (Sim Only).");
    }

    // B. GENERATE KERNEL BINDINGS
    let bindings = bindgen::Builder::default()
        // FIX: Include linux/if.h for ifreq definition
        .header_contents("wrapper.h", "#include <linux/ethtool.h>\n#include <linux/sockios.h>\n#include <linux/if.h>")
        .allowlist_type("ethtool_ringparam")
        // FIX: Explicitly allowlist ifreq
        .allowlist_type("ifreq")
        .allowlist_var("SIOCETHTOOL")
        .allowlist_var("ETHTOOL_GRINGPARAM")
        .derive_default(true) // Ensure Default is derived for easier init
        .generate();
    let bindings = match bindings {
        Ok(b) => b,
        Err(_) => { eprintln!("[M13-BUILD] Unable to generate bindings"); std::process::exit(1); }
    };

    if bindings.write_to_file(out_dir.join("bindings.rs")).is_err() {
        eprintln!("[M13-BUILD] Failed to write bindings");
        std::process::exit(1);
    }
    
    // C. ARTIFACTS
    fs::write(out_dir.join("m13_math_core.cpp"), HLS_SOURCE).ok();
}
