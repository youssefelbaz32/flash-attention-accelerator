# Build the ZUBoard 1CG block design, synthesize, implement, write the bitstream
# and export an XSA.
#
#   vivado -mode batch -source fpga/build_bd.tcl
#   vivado -mode batch -source fpga/build_bd.tcl -tclargs 8 16 16 16 100
#                                                          ^BLK ^N ^D ^DV ^PL clock MHz
#
# The PL clock defaults to 100 MHz. The first build (N=4 D=4 BLK=2) closed at
# about 130 MHz: the critical path runs from one lane's dot-product DSP through
# the cross-lane max and into the online-softmax rebase, 23 logic levels. 150 MHz
# fails timing, and a bitstream that fails timing is not one to debug on a board.
#
# Run fpga/package_ip.tcl first. Run both from the project root.
#
# THE DATA PATH THIS BUILDS:
#   PS DDR --HP0--> AXI DMA MM2S --AXIS--> accelerator --AXIS--> DMA S2MM --HP0--> PS DDR
#   PS LPD --AXI-Lite--> accelerator control registers
#   accelerator irq --> PS pl_ps_irq0[0]

set root [file normalize [file dirname [info script]]/..]
set part xczu1cg-sbva484-1-e

# Optional overrides so a sweep is a loop over this script rather than a GUI session.
set BLK [expr {$argc > 0 ? [lindex $argv 0] : 2}]
set N   [expr {$argc > 1 ? [lindex $argv 1] : 4}]
set D   [expr {$argc > 2 ? [lindex $argv 2] : 4}]
set DV  [expr {$argc > 3 ? [lindex $argv 3] : 4}]
set FCLK [expr {$argc > 4 ? [lindex $argv 4] : 100}]

set tag  "N${N}_D${D}_BLK${BLK}"
set proj $root/fpga/build/$tag

puts "=== building $tag ==="
file delete -force $proj
file mkdir $proj

create_project attn $proj -part $part -force
set_property ip_repo_paths $root/fpga/ip_repo [current_project]
update_ip_catalog -rebuild

create_bd_design "attn_bd"

# ---- the PS ------------------------------------------------------------------
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e ps]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
  -config {apply_board_preset "1"} $ps

# One HP slave port for the DMA to reach DDR, one master for AXI-Lite control,
# PL clock and reset, and one PL-to-PS interrupt.
# The PSU names are indexed, not named after the ports they create:
#   S_AXI_GP2 = S_AXI_HP0_FPD, M_AXI_GP0/GP1 = M_AXI_HPM0/1_FPD,
#   M_AXI_GP2 = M_AXI_HPM0_LPD.
set_property -dict [list \
  CONFIG.PSU__USE__S_AXI_GP2 {1}          \
  CONFIG.PSU__SAXIGP2__DATA_WIDTH {64}    \
  CONFIG.PSU__USE__M_AXI_GP0 {0}          \
  CONFIG.PSU__USE__M_AXI_GP1 {0}          \
  CONFIG.PSU__USE__M_AXI_GP2 {1}          \
  CONFIG.PSU__USE__IRQ0 {1}               \
  CONFIG.PSU__FPGA_PL0_ENABLE {1}         \
  CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $FCLK \
] $ps

# ---- the accelerator ---------------------------------------------------------
set acc [create_bd_cell -type ip -vlnv youssefelhagrasy:accel:attention_axi:1.0 acc]
set_property -dict [list \
  CONFIG.N   $N   \
  CONFIG.D   $D   \
  CONFIG.DV  $DV  \
  CONFIG.BLK $BLK \
] $acc

# ---- the DMA -----------------------------------------------------------------
# Scatter-gather off: one contiguous buffer per direction is all this needs, and
# simple mode is far easier to reason about from PYNQ.
# Width 32 to match the accelerator's stream; the DMA widens to the 64-bit HP
# port internally.
set dma [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma dma]
set_property -dict [list \
  CONFIG.c_include_sg {0}                 \
  CONFIG.c_sg_include_stscntrl_strm {0}   \
  CONFIG.c_include_mm2s_dre {1}           \
  CONFIG.c_include_s2mm_dre {1}           \
  CONFIG.c_m_axi_mm2s_data_width {64}     \
  CONFIG.c_m_axis_mm2s_tdata_width {32}   \
  CONFIG.c_m_axi_s2mm_data_width {64}     \
  CONFIG.c_s_axis_s2mm_tdata_width {32}   \
  CONFIG.c_mm2s_burst_size {64}           \
  CONFIG.c_s2mm_burst_size {64}           \
] $dma

# ---- streams: the only hand connections that matter --------------------------
connect_bd_intf_net [get_bd_intf_pins dma/M_AXIS_MM2S] [get_bd_intf_pins acc/s_axis]
connect_bd_intf_net [get_bd_intf_pins acc/m_axis]      [get_bd_intf_pins dma/S_AXIS_S2MM]

# ---- everything else by automation ------------------------------------------
# Automation builds the interconnects, clocks and resets. Doing this by hand is
# where block designs usually go wrong.
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
  -config [list Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
                Master {/ps/M_AXI_HPM0_LPD} Slave {/dma/S_AXI_LITE} \
                intc_ip {New AXI Interconnect} master_apm {0}] \
  [get_bd_intf_pins dma/S_AXI_LITE]

apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
  -config [list Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
                Master {/ps/M_AXI_HPM0_LPD} Slave {/acc/s_axi_lite} \
                intc_ip {Auto} master_apm {0}] \
  [get_bd_intf_pins acc/s_axi_lite]

# The first DMA master is applied to the PS SLAVE pin (that is how Vivado's own
# generated scripts do it; applied to the master pin it finds no valid slave).
# The second then joins the SmartConnect the first one created.
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
  -config [list Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
                Master {/dma/M_AXI_MM2S} Slave {/ps/S_AXI_HP0_FPD} \
                ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}] \
  [get_bd_intf_pins ps/S_AXI_HP0_FPD]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
  -config [list Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
                Master {/dma/M_AXI_S2MM} Slave {/ps/S_AXI_HP0_FPD} \
                ddr_seg {Auto} intc_ip {Auto} master_apm {0}] \
  [get_bd_intf_pins dma/M_AXI_S2MM]

# ---- interrupts --------------------------------------------------------------
# Two sources (accelerator done, DMA done) so they go through a concat.
set cc [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat irq_concat]
set_property CONFIG.NUM_PORTS {3} $cc
connect_bd_net [get_bd_pins acc/irq]              [get_bd_pins irq_concat/In0]
connect_bd_net [get_bd_pins dma/mm2s_introut]     [get_bd_pins irq_concat/In1]
connect_bd_net [get_bd_pins dma/s2mm_introut]     [get_bd_pins irq_concat/In2]
connect_bd_net [get_bd_pins irq_concat/dout]      [get_bd_pins ps/pl_ps_irq0]

assign_bd_address
validate_bd_design

# Record the address map: PYNQ finds the IP by name, but when something does not
# respond it is much faster to have this written down than to reopen the GUI.
set addr_log $proj/address_map.txt
set fh [open $addr_log w]
foreach seg [get_bd_addr_segs -of_objects [get_bd_addr_spaces]] {
  puts $fh [format "%-40s %-14s %s" $seg \
    [get_property OFFSET $seg] [get_property RANGE $seg]]
}
close $fh
puts "address map -> $addr_log"

make_wrapper -files [get_files attn_bd.bd] -top
add_files -norecurse $proj/attn.gen/sources_1/bd/attn_bd/hdl/attn_bd_wrapper.v
set_property top attn_bd_wrapper [current_fileset]

# ---- build -------------------------------------------------------------------
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1

# ---- report what the sweep actually wants to know ----------------------------
open_run impl_1
set wns [get_property STATS.WNS [get_runs impl_1]]
set util $proj/utilization.txt
report_utilization -file $util
report_timing_summary -file $proj/timing.txt

puts "=== $tag ==="
set period [get_property PERIOD [get_clocks clk_pl_0]]
puts "  clock: [format %.1f [expr {1000.0 / $period}]] MHz"
puts "  WNS  : $wns ns"
puts "  fmax : [format %.1f [expr {1000.0 / ($period - $wns)}]] MHz"
puts "  utilization -> $util"
puts "  timing      -> $proj/timing.txt"

write_hw_platform -fixed -include_bit -force $proj/attn_$tag.xsa
file copy -force $proj/attn.runs/impl_1/attn_bd_wrapper.bit $root/fpga/build/attn_$tag.bit
file copy -force $proj/attn_$tag.xsa $root/fpga/build/attn_$tag.xsa
# PYNQ loads the overlay from the .bit but reads the design (IP names, address
# map, PL clock, interrupts) from a .hwh with the same basename.
file copy -force $proj/attn.gen/sources_1/bd/attn_bd/hw_handoff/attn_bd.hwh $root/fpga/build/attn_$tag.hwh
puts "  bitstream   -> fpga/build/attn_$tag.bit"
puts "  hwh         -> fpga/build/attn_$tag.hwh"
puts "  xsa         -> fpga/build/attn_$tag.xsa"
close_project
