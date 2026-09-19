// Shared combinational exp() ROM + index logic.
//
// Factored out because M8 needs the exponential in TWO places -- the score
// exponentials and the online-softmax correction factor -- and duplicating the
// clamp/index arithmetic is how the two quietly stop agreeing.
//
// Contract: x is signed and expected <= 0 (guaranteed by the row-max subtract).
// Returns exp(x) in Q(FRAC), i.e. 2^FRAC == 1.0.
//
// Index math:  idx = (-x) >> EXP_SH, saturated at EXP_N-1.
// Because x <= 0 the table only stores the negative half-line -- half the ROM,
// which is what row_max buys us. The saturation IS the table's floor: anything
// below -EXP_RANGE reads the last entry, which is 0 in Q8.8 anyway.
//
// The table contents come from python/04_rtl_fixed_model.py via $readmemh, so
// there is exactly one definition of exp() in the project.

module exp_rom #(
  parameter int DW        = 16,
  parameter int FRAC      = 8,
  parameter int EXP_N     = 256,
  parameter int EXP_RANGE = 8,
  parameter     EXP_FILE  = "rtl/vectors/exp_lut.hex"
)(
  input  logic signed [DW:0] x,    // DW+1 bits: a difference of two DW values
  output logic        [DW-1:0] e   // Q(FRAC), 2^FRAC == 1.0
);

  localparam int LUTAW    = $clog2(EXP_N);
  localparam int EXP_STEP = (EXP_RANGE << FRAC) / EXP_N;
  localparam int EXP_SH   = $clog2(EXP_STEP);

  initial if (EXP_STEP != (1 << EXP_SH))
    $error("(EXP_RANGE<<FRAC)/EXP_N = %0d is not a power of 2", EXP_STEP);

  logic [DW-1:0] rom [EXP_N];
  initial $readmemh(EXP_FILE, rom);       // Vivado infers a ROM from this

  logic signed [DW:0] xc;
  logic [DW:0] negx, idx_raw;
  logic [LUTAW-1:0] idx;

  // Clamp x <= 0 defensively. A positive x would make -x wrap to a huge
  // unsigned and SILENTLY return exp = 0 rather than failing loudly.
  assign xc      = (x > 0) ? '0 : x;
  assign negx    = (~xc) + 1'b1;
  assign idx_raw = negx >> EXP_SH;
  assign idx     = (idx_raw >= (DW+1)'(EXP_N)) ? LUTAW'(EXP_N - 1)
                                               : idx_raw[LUTAW-1:0];
  assign e       = rom[idx];

endmodule
