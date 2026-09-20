"""
Host driver for the attention accelerator on the ZUBoard 1CG, under PYNQ.

Runs on the A53, not on a laptop. That is the whole reason this is simple: the
bit-exact model in python/04_rtl_fixed_model.py imports directly into the same
process, so the golden model IS the on-board self test. There is no wire format
to agree on, no CRC, no byte assembler, nothing to get subtly wrong between the
two sides.

Register map mirrors rtl/axil_regs.sv. If you change one, change the other, and
the PARAM registers exist so a mismatch is caught immediately rather than
presenting as wrong arithmetic.

Usage on the board:
    sudo python3 09_pynq_driver.py --bit attention.bit
    sudo python3 09_pynq_driver.py --bit attention.bit --sweep
"""

import argparse
import importlib.util
import os
import time

import numpy as np

try:
    from pynq import Overlay, allocate
    HAVE_PYNQ = True
except ImportError:
    HAVE_PYNQ = False


# =============================================================================
# CHUNK 1: register map, mirroring rtl/axil_regs.sv
# -----------------------------------------------------------------------------
class Reg:
    CTRL            = 0x00
    STATUS          = 0x04
    ISR             = 0x08
    CYC_TOTAL       = 0x0C
    CYC_LOAD        = 0x10
    CYC_LOAD_STALL  = 0x14
    CYC_COMPUTE     = 0x18
    CYC_STORE       = 0x1C
    CYC_STORE_STALL = 0x20
    PARAM0          = 0x24
    PARAM1          = 0x28
    PARAM2          = 0x2C
    BUILD_ID        = 0x30

CTRL_START      = 1 << 0
CTRL_IRQ_EN     = 1 << 1
CTRL_SOFT_RESET = 1 << 2

ST_BUSY         = 1 << 0
ST_DONE         = 1 << 1
ST_IRQ_PENDING  = 1 << 2
ST_ERR_OVERRUN  = 1 << 3
ST_ERR_UNDERRUN = 1 << 4

EXPECTED_BUILD_ID = 0xA77E0001


class Profile:
    """One run's worth of cycle counters, plus the ratios worth looking at."""

    def __init__(self, total, load, load_stall, compute, store, store_stall, f_pl):
        self.total, self.load, self.compute, self.store = total, load, compute, store
        self.load_stall, self.store_stall = load_stall, store_stall
        self.f_pl = f_pl

    @property
    def us(self):
        return self.total / self.f_pl * 1e6

    @property
    def load_efficiency(self):
        """Fraction of the load phase that moved data instead of waiting on DMA."""
        return 0.0 if self.load == 0 else 1.0 - self.load_stall / self.load

    def __str__(self):
        unaccounted = self.total - (self.load + self.compute + self.store)
        s = [
            f"  total    {self.total:>10,} cy  ({self.us:8.1f} us @ {self.f_pl/1e6:.0f} MHz)",
            f"  load     {self.load:>10,} cy   of which {self.load_stall:,} stalled "
            f"({100*self.load_efficiency:.1f}% efficient)",
            f"  compute  {self.compute:>10,} cy   {100*self.compute/max(self.total,1):.1f}% of total",
            f"  store    {self.store:>10,} cy   of which {self.store_stall:,} backpressured",
        ]
        if unaccounted > 0:
            # TOTAL is counted independently of the phases on purpose, so a gap
            # here means real cycles nobody is attributing. Worth knowing.
            s.append(f"  unaccounted {unaccounted:>7,} cy  <- cycles in no phase, investigate")
        return "\n".join(s)

    def verdict(self):
        """What the numbers say to do next. This is the point of splitting them."""
        if self.load_efficiency < 0.5:
            return ("DMA-starved: over half the load phase was spent waiting for data. "
                    "Widen the stream or use a longer burst. More MAC lanes will do nothing.")
        if self.compute > 4 * (self.load + self.store):
            return ("Compute-bound: the accelerator dominates. This is where adding "
                    "lanes (BLK) actually pays, and where fmax matters.")
        return ("Balanced: transfer and compute are the same order. Improving either "
                "alone gives well under 2x.")


# =============================================================================
# CHUNK 2: the accelerator
# -----------------------------------------------------------------------------
class AttentionAccel:
    def __init__(self, bitfile, f_pl=150e6, use_irq=True):
        if not HAVE_PYNQ:
            raise RuntimeError("pynq not available; this runs on the board")
        self.ol   = Overlay(bitfile)
        self.regs = self.ol.axil_regs_0          # rename to match your block design
        self.dma  = self.ol.axi_dma_0
        self.f_pl = f_pl
        self.use_irq = use_irq

        bid = self.regs.read(Reg.BUILD_ID)
        if bid != EXPECTED_BUILD_ID:
            raise RuntimeError(
                f"BUILD_ID 0x{bid:08X} != 0x{EXPECTED_BUILD_ID:08X}. "
                "Wrong bitstream, or the register map changed under you.")

        p0, p1, p2 = (self.regs.read(r) for r in (Reg.PARAM0, Reg.PARAM1, Reg.PARAM2))
        self.N   =  p0        & 0xFFFF
        self.D   = (p0 >> 16) & 0xFFFF
        self.DV  =  p1        & 0xFFFF
        self.BLK = (p1 >> 16) & 0xFFFF
        self.DW  =  p2        & 0xFF
        self.FRAC= (p2 >> 8)  & 0xFF
        print(f"bitstream: N={self.N} D={self.D} DV={self.DV} BLK={self.BLK} "
              f"Q{self.DW-self.FRAC}.{self.FRAC}")

        self._in  = allocate(shape=(self.N*self.D*2 + self.N*self.DV,), dtype=np.int16)
        self._out = allocate(shape=(self.N*self.DV,), dtype=np.int16)

    def check_shape(self, N, D, DV):
        """Refuse to run on a shape mismatch. This is the failure that otherwise
        presents as wrong arithmetic and costs a day."""
        if (N, D, DV) != (self.N, self.D, self.DV):
            raise ValueError(
                f"host has N={N} D={D} DV={DV}, bitstream has "
                f"N={self.N} D={self.D} DV={self.DV}. Rebuild one of them.")

    def run(self, Qq, Kq, Vq, timeout=5.0):
        self.check_shape(Qq.shape[0], Qq.shape[1], Vq.shape[1])

        self._in[:] = np.concatenate([Qq.ravel(), Kq.ravel(), Vq.ravel()]).astype(np.int16)

        self.regs.write(Reg.CTRL, CTRL_SOFT_RESET)
        self.regs.write(Reg.CTRL, CTRL_IRQ_EN if self.use_irq else 0)

        # Arm the receive side BEFORE starting, or the first output beats have
        # nowhere to go and the datapath backpressures immediately. That shows
        # up as a large CYC_STORE_STALL, which is at least self-diagnosing.
        self.dma.recvchannel.transfer(self._out)
        self.regs.write(Reg.CTRL, CTRL_START | (CTRL_IRQ_EN if self.use_irq else 0))
        self.dma.sendchannel.transfer(self._in)

        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()
        self._wait_done(timeout)

        st = self.regs.read(Reg.STATUS)
        if st & ST_ERR_OVERRUN:
            raise RuntimeError("overrun: data arrived while the sink was not ready")
        if st & ST_ERR_UNDERRUN:
            raise RuntimeError("underrun: the datapath ran dry before the run ended")

        prof = Profile(*(self.regs.read(r) for r in
                         (Reg.CYC_TOTAL, Reg.CYC_LOAD, Reg.CYC_LOAD_STALL,
                          Reg.CYC_COMPUTE, Reg.CYC_STORE, Reg.CYC_STORE_STALL)),
                       f_pl=self.f_pl)
        return np.array(self._out).reshape(self.N, self.DV).copy(), prof

    def _wait_done(self, timeout):
        """Interrupt if the overlay exposes one, polling otherwise.

        The interrupt is worth having not because polling is slow but because it
        frees the A53 during a long run and, more usefully, it makes a hang look
        like a timeout instead of a spin."""
        if self.use_irq and hasattr(self.regs, "interrupt"):
            import asyncio
            try:
                asyncio.get_event_loop().run_until_complete(
                    asyncio.wait_for(self.regs.interrupt.wait(), timeout))
                self.regs.write(Reg.ISR, 1)          # write-1-to-clear
                return
            except asyncio.TimeoutError:
                raise TimeoutError(f"no completion interrupt within {timeout}s")

        deadline = time.time() + timeout
        while not (self.regs.read(Reg.STATUS) & ST_DONE):
            if time.time() > deadline:
                raise TimeoutError(f"STATUS never reported done within {timeout}s")
        self.regs.write(Reg.ISR, 1)


# =============================================================================
# CHUNK 3: the self test. The golden model runs here, on the board.
# -----------------------------------------------------------------------------
def load_spec(N, D, DV):
    os.environ.update(N=str(N), D=str(D), DV=str(DV))
    path = os.path.join(os.path.dirname(__file__), "04_rtl_fixed_model.py")
    spec = importlib.util.spec_from_file_location("m5", path)
    m5 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m5)
    return m5


def self_test(accel):
    m5 = load_spec(accel.N, accel.D, accel.DV)
    np.random.seed(0)
    Q = np.random.randn(accel.N, accel.D)
    K = np.random.randn(accel.N, accel.D)
    V = np.random.randn(accel.N, accel.DV)
    Qq, Kq, Vq = m5.quantize(Q), m5.quantize(K), m5.quantize(V)

    O_hw, prof = accel.run(Qq, Kq, Vq)

    # The spec here is the ONLINE model, because flash_top is what is in the PL.
    path = os.path.join(os.path.dirname(__file__), "05_online_softmax_model.py")
    spec = importlib.util.spec_from_file_location("m8", path)
    m8 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m8)
    O_spec, _, _, _ = m8.online_attention(Qq, Kq, Vq)

    exact = np.array_equal(O_hw.astype(np.int64), O_spec.astype(np.int64))
    print(f"\nhardware vs spec: {'BIT-EXACT' if exact else 'MISMATCH'}")
    if not exact:
        bad = np.argwhere(O_hw.astype(np.int64) != O_spec)
        print(f"  {len(bad)} of {O_hw.size} differ; first at {tuple(bad[0])}: "
              f"hw={O_hw[tuple(bad[0])]} spec={O_spec[tuple(bad[0])]}")
    print(prof)
    print(f"\n  {prof.verdict()}")
    return exact


def _offline_check():
    """Exercise the reporting math with no board, using the exact counter values
    rtl/tb_axil_regs.sv drives into the hardware. If the numbers the testbench
    produces do not turn into the right summary here, the bug is in this file."""
    p = Profile(total=169, load=40, load_stall=12, compute=100,
                store=25, store_stall=7, f_pl=150e6)
    assert abs(p.load_efficiency - 0.70) < 1e-9, p.load_efficiency
    assert abs(p.us - 169/150e6*1e6) < 1e-9
    print(p)
    print(f"\n  {p.verdict()}")

    starved = Profile(1000, 900, 800, 50, 50, 0, 150e6)
    assert "DMA-starved" in starved.verdict()
    bound = Profile(1000, 50, 0, 900, 50, 0, 150e6)
    assert "Compute-bound" in bound.verdict()
    print("\n  offline reporting checks PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true",
                    help="check the reporting math with no board attached")
    ap.add_argument("--bit", required=False, help="path to the .bit overlay")
    ap.add_argument("--fpl", type=float, default=150e6, help="PL clock in Hz")
    ap.add_argument("--poll", action="store_true", help="poll instead of using the interrupt")
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()

    if args.offline:
        _offline_check()
        raise SystemExit(0)

    accel = AttentionAccel(args.bit, f_pl=args.fpl, use_irq=not args.poll)
    ok = True
    for i in range(args.runs):
        if args.runs > 1:
            print(f"\n--- run {i+1}/{args.runs}")
        ok &= self_test(accel)
    raise SystemExit(0 if ok else 1)
