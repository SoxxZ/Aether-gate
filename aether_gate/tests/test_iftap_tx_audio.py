#
# Aether-gate (SoxxZ fork) — IF-tap / TX / audio-source regression tests.
# Copyright (C) 2026. GPL-3.0-or-later.
#
"""Guards for the fork's additions on top of upstream main:

  * Waterfall subheader uses VitaFrequency encoding (Hz x 2^20) — the decode
    AetherSDR 26.10.2 performs unconditionally (#3449/#4412). Plain-Hz here
    mis-reads to ~0 Hz and AE shows a BLACK waterfall — the bug this fork
    fixed on the FT-600 gate. Do not "simplify" these multipliers away.
  * Kenwood/Yaesu adapter TX: key_tx/unkey_tx -> hamlib PTT through the
    single worker (_apply_ptt), TX-audio drain follows PTT, close() never
    leaves the rig keyed.
  * Audio-source routing: "sdr" (dongle demod) vs "radio" (rig audio via a
    USB audio card, e.g. PICO-Diginterface), with SDR fallback.
  * Panel fragment: adapter sources only appear for adapters that have them.

Run:  python -m aether_gate.tests.test_iftap_tx_audio
"""
import struct
import sys
import types


def test_wf_vitafrequency_encoding():
    from aether_gate.core.engine import wf_packet
    low_hz, binbw_hz = 7062500.0, 375.0
    pkt = wf_packet(1, 0, [0, 1, 2, 3], low_hz, binbw_hz, 1234)
    # FrameLowFreq + BinBandwidth as big-endian int64 VitaFrequency (Hz x 2^20)
    expect = struct.pack(">qq", int(round(low_hz * 1048576.0)),
                         int(round(binbw_hz * 1048576.0)))
    assert expect in pkt, "wf subheader not VitaFrequency-encoded (black waterfall bug)"
    # and the plain-Hz encoding must NOT be what's on the wire
    plain = struct.pack(">qq", int(low_hz), int(binbw_hz))
    assert plain not in pkt, "wf subheader encoded as plain Hz (AE mis-reads)"
    print("ok  wf: FrameLowFreq/BinBandwidth are VitaFrequency (Hz x 2^20)")


class _StubSdr:
    def __init__(self):
        self.closed = False
    def get_audio(self, n_samples, slice_hz=None, mode=None):
        return [1.0] * n_samples
    def close(self):
        self.closed = True


class _StubCapture:
    detail = "stub-capture"
    def __init__(self):
        self.closed = False
    def read(self, n):
        return [0.25] * n
    def close(self):
        self.closed = True


class _StubCtl:
    def __init__(self):
        self.ptt_calls = []
        self.closed = False
    def set_ptt(self, on):
        self.ptt_calls.append(bool(on))
        return True
    def close(self):
        self.closed = True


def _adapter(**kw):
    from aether_gate.adapters.kenwood.adapter import KenwoodAdapter
    a = KenwoodAdapter(model="TS-2000", enable_tx=True, **kw)
    a._sdr = _StubSdr()
    a._ctl = _StubCtl()
    return a


def test_ptt_via_worker():
    a = _adapter()
    assert a._apply_ptt() is False, "idle adapter consumed a PTT target"

    a.key_tx()
    assert a._ptt_target is True
    assert a._apply_ptt() is True, "posted key not consumed by worker"
    assert a._ctl.ptt_calls == [True], a._ctl.ptt_calls
    assert a._ptt_state is True
    assert a._ptt_target is None, "target not cleared after apply"

    a.unkey_tx()
    assert a._apply_ptt() is True
    assert a._ctl.ptt_calls == [True, False]
    assert a._ptt_state is False

    a.close()
    assert a._ctl.ptt_calls[-1] is False, "close() must not leave the rig keyed"
    assert a._sdr.closed, "close() must close the SDR"
    print("ok  tx: key/unkey -> hamlib PTT via worker; close() unkeys")


def test_audio_source_routing():
    a = _adapter()
    assert a.audio_sources() == ("sdr", "radio")
    assert a.get_audio(4) == [1.0] * 4, "sdr source must use the demod"

    cap = _StubCapture()
    a._radio_capture = cap
    a.set_audio_source("radio")
    assert a.get_audio(4) == [0.25] * 4, "radio source must use the capture"

    # dead capture -> fall back to SDR demod (audio never goes silent)
    cap.read = lambda n: None
    assert a.get_audio(4) == [1.0] * 4, "dead capture must fall back to sdr"

    a.set_audio_source("sdr")
    assert a.get_audio(4) == [1.0] * 4
    try:
        a.set_audio_source("nope")
        raise AssertionError("invalid audio source accepted")
    except ValueError:
        pass
    print("ok  audio: sdr/radio routing + fallback + validation")


def test_radio_capture_released_on_switch():
    a = _adapter()
    cap = _StubCapture()
    a._radio_capture = cap                     # as if radio mode ran for a while
    a.set_audio_source("radio")
    assert a._radio_capture is cap, "existing capture must be kept while radio"
    a.set_audio_source("sdr")
    assert cap.closed, "leaving radio must close the arecord (device + child)"
    assert a._radio_capture is None, "capture must be dropped after close"
    # an in-flight read after the switch must not resurrect the capture
    assert a._radio_capture_open() is None, "capture must not reopen while on sdr"
    print("ok  audio: radio capture closed + not resurrected on switch to sdr")


def test_panel_fragment():
    from aether_gate.core.engine import _adapter_audio_fragment
    none_radio = types.SimpleNamespace(adapter=None)
    assert _adapter_audio_fragment(none_radio) == ""

    plain = types.SimpleNamespace(adapter=object())          # no audio_sources()
    assert _adapter_audio_fragment(plain) == ""

    stub = types.SimpleNamespace(audio_sources=lambda: ("sdr", "radio"),
                                 _audio_source="radio")
    frag = _adapter_audio_fragment(types.SimpleNamespace(adapter=stub))
    for needle in ("value=sdr", "value=radio", "checked"):
        assert needle in frag, f"fragment missing {needle!r}"
    assert "value=radio checked" in frag, "current source must render checked"
    print("ok  panel: adapter audio sources rendered (checked = current)")


def _demod_adapter(lock=True, center=47_055_000.0, samp=960_000.0):
    """SoapyAdapter without hardware: demod chain built, no open()/USB."""
    import numpy as np
    from aether_gate.adapters.soapy import SoapyAdapter
    a = SoapyAdapter(center_hz=center, samp_rate=samp,
                     lock_hz=(center if lock else None), model="TEST")
    a._np = np
    a._setup_demod()
    return a


def _tone_blocks(n_blocks, block, f0=1875.0, amp=0.003, sr=960_000.0):
    """Continuous-phase complex64 tone at f0 — 1875 Hz lands on bin 10 of a
    128-sample FFT at 24 kHz, making peak checks exact."""
    import numpy as np
    out, n = [], 0
    for _ in range(n_blocks):
        k = np.arange(n, n + block)
        out.append((amp * np.exp(2j * np.pi * f0 * k / sr)).astype(np.complex64))
        n += block
    return out


def test_demod_fast_path():
    import numpy as np
    a = _demod_adapter()                       # lock mode (IF tap): f_off == 0
    need = 128 * a._decim
    assert a._box_m == 5 and [f[2] for f in a._stage_firs] == [4, 2], \
        (a._box_m, [f[2] for f in a._stage_firs])      # 960k -> box(5) + FIR 4, 2

    # --- DC exactness: constant input -> constant output at gain 1 ----------
    dc = 0.003
    blk = np.full(need, dc, dtype=np.complex64)
    for _ in range(2):                          # warmup states
        a._audio_q.append(blk.copy())
        assert len(a.get_audio(128)) == 128
    a._agc_level = a._agc_target = dc * a._audio_gain   # neutral AGC: scale == 1
    a._audio_q.append(blk.copy())
    out = np.array(a.get_audio(128))
    assert out.shape == (128,) and np.all(np.isfinite(out))
    assert np.max(np.abs(out - dc * a._audio_gain)) < 0.02, \
        f"box/FIR DC gain off: mean={out.mean():.4f} want={dc * a._audio_gain:.4f}"

    # --- NCO skip proof: np.exp must NOT be called in lock mode ------------
    tblock = _tone_blocks(1, need)[0]           # generate BEFORE patching np.exp
    boom = []
    orig = np.exp
    def counting_exp(x, *args, **kwargs):
        boom.append(1)
        return orig(x, *args, **kwargs)
    np.exp = counting_exp
    try:
        a._audio_q.append(tblock)
        got = a.get_audio(128)
    finally:
        np.exp = orig
    assert got is not None and not boom, \
        "NCO ran in lock mode (f_off==0) — exp() must be skipped"

    # --- non-lock mode still uses the NCO ---------------------------------
    b = _demod_adapter(lock=False, center=14_100_000.0)
    b.set_slice(14_101_000.0)                   # slice off centre -> mix required
    nblock = _tone_blocks(1, 128 * b._decim)[0]  # BEFORE patching
    calls = []
    orig2 = np.exp
    def counting_exp2(x, *args, **kwargs):
        calls.append(1)
        return orig2(x, *args, **kwargs)
    np.exp = counting_exp2
    try:
        b._audio_q.append(nblock)
        got2 = b.get_audio(128)
    finally:
        np.exp = orig2
    assert got2 is not None and len(calls) >= 1, "non-lock path must run the NCO"
    print("ok  demod: box DC gain, NCO skipped in lock mode, NCO runs when off-centre")


def test_demod_vs_legacy_chain():
    """The optimized chain (box stage + NCO skip) must demodulate the same
    tone to the same FFT peak as the ORIGINAL all-FIR + cexp chain — catches
    wrong decimation factors, misalignment, polarity/sideband flips."""
    import numpy as np
    a = _demod_adapter()                        # fresh states
    need = 128 * a._decim
    blocks = _tone_blocks(5, need)              # 5 calls: warmup + 4 measured

    # --- legacy reference: original stage chain (FIRs for ALL factors + cexp)
    decim = int(round(a.samp_rate / 24000.0))
    firs = []
    for M in a._factor_decim(decim):
        ntaps = 4 * M + 1
        cutoff = 0.45 / M
        idx = np.arange(ntaps) - (ntaps - 1) / 2.0
        h = np.sinc(2 * cutoff * idx) * np.hamming(ntaps)
        firs.append([h / h.sum(), np.zeros(ntaps - 1, dtype=np.complex128), M])

    def legacy_once(iq_in):
        sig = iq_in.astype(np.complex128)
        k = np.arange(len(sig))                 # f_off == 0 -> exp(0) == 1
        sig = sig * np.exp(1j * (0.0 * k))
        for fir in firs:
            taps, state, M = fir
            x = np.concatenate([state, sig])
            y = np.convolve(x, taps, mode="valid")
            fir[1] = sig[-(len(taps) - 1):]
            sig = y[::M]
        return np.real(sig[:128]) * a._audio_gain

    new_outs, old_outs = [], []
    for i, blk in enumerate(blocks):
        a._audio_q.append(blk)
        o = a.get_audio(128)
        assert o is not None and len(o) == 128 and np.all(np.isfinite(o))
        new_outs.append(np.array(o))
        old_outs.append(legacy_once(blk))
        if i == 1:                              # after warmup: neutralize AGC so the
            # measured output == raw gain, directly comparable to legacy (no AGC)
            a._agc_level = a._agc_target = 0.003 * a._audio_gain / np.sqrt(2.0)

    new, old = np.concatenate(new_outs[3:]), np.concatenate(old_outs[3:])
    X, Y = np.abs(np.fft.rfft(new)), np.abs(np.fft.rfft(old))
    kx, ky = int(np.argmax(X[2:]) + 2), int(np.argmax(Y[2:]) + 2)
    want = int(round(1875.0 * len(new) / 24000.0))   # 1875 Hz in rfft bins of len(new) @24k
    assert kx == ky == want, f"peak bin mismatch: new={kx} old={ky} (want {want})"
    ratio = X[kx] / max(Y[ky], 1e-9)
    assert 0.6 <= ratio <= 1.7, f"tone amplitude off vs legacy: ratio={ratio:.2f}"
    print("ok  demod: optimized chain matches legacy FFT peak + amplitude")


def main():
    for t in (test_wf_vitafrequency_encoding, test_ptt_via_worker,
              test_audio_source_routing, test_radio_capture_released_on_switch,
              test_panel_fragment, test_demod_fast_path,
              test_demod_vs_legacy_chain):
        try:
            t()
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            sys.exit(1)
    print("\nall iftap/tx/audio tests passed")


if __name__ == "__main__":
    main()
