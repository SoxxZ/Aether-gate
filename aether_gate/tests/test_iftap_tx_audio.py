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


def main():
    for t in (test_wf_vitafrequency_encoding, test_ptt_via_worker,
              test_audio_source_routing, test_panel_fragment):
        try:
            t()
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            sys.exit(1)
    print("\nall iftap/tx/audio tests passed")


if __name__ == "__main__":
    main()
