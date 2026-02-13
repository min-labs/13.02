/* M13 NODE - BUILD ORCHESTRATOR (REV 6.1)
 * Compiles eBPF Steersman + generates kernel bindings.
 * Mirror of hub/build.rs — independent binary, same BPF filter structure.
 */
use std::process::Command;
use std::fs;
use std::env;
use std::path::PathBuf;

const BPF_SOURCE: &str = r#"
#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_endian.h>
#include <linux/if_ether.h>

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

    if (data + sizeof(struct ethhdr) > data_end)
        return XDP_DROP;

    struct ethhdr *eth = data;

    /* EtherType filter: only M13 (0x88B5) reaches AF_XDP UMEM. */
    if (eth->h_proto != bpf_htons(0x88B5))
        return XDP_PASS;

    return bpf_redirect_map(&xsks_map, ctx->rx_queue_index, XDP_PASS);
}

char _license[] SEC("license") = "GPL";
"#;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = PathBuf::from(match env::var("OUT_DIR") {
        Ok(v) => v,
        Err(_) => { eprintln!("[M13-NODE-BUILD] OUT_DIR not set"); std::process::exit(1); }
    });
    let bpf_src = out_dir.join("m13_node_xdp.c");
    let bpf_obj = out_dir.join("m13_node_xdp.o");

    // A. COMPILE BPF PROGRAM
    if fs::write(&bpf_src, BPF_SOURCE).is_err() {
        eprintln!("[M13-NODE-BUILD] Failed to write BPF source");
        std::process::exit(1);
    }

    if Command::new("clang").arg("--version").output().is_ok() {
        println!("cargo:warning=[M13-NODE-BUILD] Compiling BPF Steersman...");
        let status = Command::new("clang")
            .arg("-O2")
            .arg("-g")
            .arg("-target").arg("bpf")
            .arg("-c").arg(&bpf_src)
            .arg("-o").arg(&bpf_obj)
            .status();
        let status = match status {
            Ok(s) => s,
            Err(_) => { eprintln!("[M13-NODE-BUILD] Failed to execute clang"); std::process::exit(1); }
        };

        if !status.success() {
            eprintln!("[M13-NODE-BUILD] BPF compilation failed. Install clang/libbpf-dev.");
            std::process::exit(1);
        } else {
            println!("cargo:rustc-env=BPF_OBJECT_PATH={}", bpf_obj.display());
        }
    } else {
        println!("cargo:warning=[M13-NODE-BUILD] Clang not found. BPF Steersman skipped.");
    }

    // B. GENERATE KERNEL BINDINGS
    let bindings = bindgen::Builder::default()
        .header_contents("wrapper.h", "#include <linux/ethtool.h>\n#include <linux/sockios.h>\n#include <linux/if.h>")
        .allowlist_type("ethtool_ringparam")
        .allowlist_type("ifreq")
        .allowlist_var("SIOCETHTOOL")
        .allowlist_var("ETHTOOL_GRINGPARAM")
        .derive_default(true)
        .generate();
    let bindings = match bindings {
        Ok(b) => b,
        Err(_) => { eprintln!("[M13-NODE-BUILD] Unable to generate bindings"); std::process::exit(1); }
    };

    if bindings.write_to_file(out_dir.join("bindings.rs")).is_err() {
        eprintln!("[M13-NODE-BUILD] Failed to write bindings");
        std::process::exit(1);
    }
}
