# M13 PHASE I - OS CONTAINER RECIPE
# TARGET: PetaLinux 2023.2 (Yocto)
# STRICTNESS: Fiduciary / Flight-Grade
# FUNCTION: Configures the "Parasitic Unikernel" Environment

SUMMARY = "M13 High-Frequency Kinetic Edge Fabric"
LICENSE = "CLOSED"

inherit cargo systemd

# 1. SOURCE DEFINITION
SRC_URI = "file://m13"
S = "${WORKDIR}/m13"

# 2. KERNEL BOOT ARGUMENTS (THE SILENCE)
# Physics Compliance:
#   isolcpus=1,2,3    -> BANS Linux Scheduler from Cores 1-3
#   rcu_nocbs=1,2,3   -> BANS RCU Garbage Collection from Cores 1-3
#   nohz_full=1,2,3   -> BANS Timer Interrupts on Cores 1-3
#   hugepagesz=1G     -> Enables 1GB Page support
#   audit=0           -> Disables Audit Logging
APPEND += " isolcpus=1,2,3 rcu_nocbs=1,2,3 nohz_full=1,2,3 hugepagesz=1G default_hugepagesz=1G audit=0"

# 3. INSTALLATION
do_install() {
    install -d ${D}${bindir}
    install -m 0755 ${B}/target/aarch64-unknown-linux-gnu/release/m13-hub ${D}${bindir}/m13-hub
}

# 4. SERVICE DEFINITION
SYSTEMD_SERVICE_${PN} = "m13.service"
