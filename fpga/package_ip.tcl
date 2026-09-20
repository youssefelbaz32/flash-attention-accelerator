# Package attention_axi as a Vivado IP for the ZUBoard 1CG.
#
#   vivado -mode batch -source fpga/package_ip.tcl
#
# Run from the project root. Produces fpga/ip_repo/attention_axi, which
# build_bd.tcl then adds as an IP repository.
#
# The AXI interfaces are INFERRED from the port names in attention_axi.sv
# (s_axi_lite_*, s_axis_*, m_axis_*). That is why those names are what they are:
# rename them and this script stops finding interfaces and you end up mapping
# forty loose scalar ports by hand.

set root      [file normalize [file dirname [info script]]/..]
set part      xczu1cg-sbva484-1-e
set ip_name   attention_axi
set ip_vendor youssefelhagrasy
set ip_lib    accel
set ip_ver    1.0
set stage     $root/fpga/.ip_stage
set repo      $root/fpga/ip_repo

file delete -force $stage
file mkdir $stage

create_project ${ip_name}_pkg $stage -part $part -force

# ---- sources ----------------------------------------------------------------
# Order does not matter for elaboration, but keeping leaf-first makes a missing
# file obvious from where the error appears.
add_files -norecurse [list \
  $root/rtl/dot4.sv \
  $root/rtl/exp_rom.sv \
  $root/rtl/flash_top.sv \
  $root/rtl/axil_regs.sv \
  $root/rtl/attention_axi.sv ]

# The exp LUT is read by $readmemh at elaboration. Vivado resolves a bare
# filename against the sources, so it has to be ADDED to the project, not just
# sitting on disk. Forgetting this gives a ROM full of X and an accelerator that
# returns zeros, with no error message at all.
add_files -norecurse -fileset sources_1 $root/rtl/vectors/exp_lut.hex
set_property file_type {Memory Initialization Files} \
  [get_files $root/rtl/vectors/exp_lut.hex]

set_property top $ip_name [current_fileset]
update_compile_order -fileset sources_1

# Fail here rather than three steps later in block design.
if {[llength [get_msg_config -severity ERROR -count]] > 0} {
  puts "ERROR: elaboration problems before packaging"
}

# ---- package ----------------------------------------------------------------
ipx::package_project -root_dir $repo/$ip_name -vendor $ip_vendor \
  -library $ip_lib -taxonomy /UserIP -import_files -force

set core [ipx::current_core]
set_property name        $ip_name              $core
set_property version     $ip_ver               $core
set_property display_name "Attention Accelerator (online softmax)" $core
set_property description \
  "Single-head scaled dot-product attention with streaming softmax. Q8.8 fixed point. AXI4-Lite control with per-phase cycle counters and a completion interrupt." $core
set_property vendor_display_name "Youssef Elhagrasy" $core

# ---- user-visible parameters ------------------------------------------------
# Exposing these in the IP GUI is what makes the Pareto sweep a rebuild rather
# than an edit: BLK and N become fields in the customisation dialog.
foreach {p disp} {
  N    "Sequence length (N)"
  D    "Head dimension (D)"
  DV   "Value dimension (DV)"
  BLK  "Key block size / MAC lanes (BLK)"
  DW   "Word width (DW)"
  FRAC "Fraction bits (FRAC)"
} {
  set gp [ipgui::get_guiparamspec -name $p -component $core -quiet]
  if {$gp ne ""} {
    set_property display_name $disp $gp
    set_property widget {textEdit} $gp
  }
}

# ---- interrupt -------------------------------------------------------------
# Without this the `irq` port is just a wire and Vivado will not offer to
# connect it to the PS interrupt controller.
ipx::infer_bus_interface irq xilinx.com:signal:interrupt_rtl:1.0 $core
set_property value LEVEL_HIGH [ipx::get_bus_parameters SENSITIVITY \
  -of_objects [ipx::get_bus_interfaces irq -of_objects $core] -quiet] -quiet

# ---- sanity: did the AXI interfaces actually get inferred? ------------------
set want {s_axi_lite s_axis m_axis irq}
foreach bif $want {
  if {[llength [ipx::get_bus_interfaces $bif -of_objects $core -quiet]] == 0} {
    puts "ERROR: interface '$bif' was not inferred. Check the port names in attention_axi.sv."
  } else {
    puts "  inferred interface: $bif"
  }
}

ipx::create_xgui_files $core
ipx::update_checksums  $core
ipx::check_integrity   $core
ipx::save_core         $core

puts "packaged to $repo/$ip_name"
close_project
