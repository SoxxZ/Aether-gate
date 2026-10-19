#
# Aether-gate - Kenwood (and any hamlib CAT rig) adapter: hamlib control + IF-tap SDR spectrum.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""Bridge a Kenwood (or any hamlib-supported CAT rig) into AetherSDR as a Flex.

The two axes are sourced SEPARATELY and combined here (see RADIO_SUPPORT.md):
  * CONTROL  = hamlib `rigctld` — freq / mode / PTT / S-meter. Vendor-neutral, so
               the SAME adapter serves Kenwood, Yaesu, Elecraft, Icom-USB by just
               changing the rigctld model number. No spectrum/audio over CAT.
  * SPECTRUM = a SoapySDR dongle (the existing SoapyAdapter, provides="iq"), tapped
               off-air / IF / antenna and STEERED to follow the rig's tuned freq.

`provides = "iq"` (delegated to the soapy dongle; the core runs the FFT). This
adapter's own job is the glue: read the rig's freq/mode from hamlib and keep the
dongle centred on the rig (CAT-steer), so AE's panadapter tracks where the rig is
tuned even though the spectrum is the dongle's.

⚠ STATUS: scaffold. Untested end-to-end (needs a Kenwood + a dongle + rigctld
running). The soapy half is HW-proven (RTL-SDR V4 on the Pi5); the hamlib half
is new. The CAT-steer cadence is deliberately gentle (poll the rig ~2-3x/s) —
mirrors the lesson from the IC-9700 (don't flood the rig's CAT).
"""
import struct
import threading
import time

from ..base import RadioAdapter, AdapterCaps, Meters
from ..soapy import SoapyAdapter
from ..audio_io import RadioAudioCapture, AlsaPcmPlayback, find_alsa_card
from ..hamlib.rigctld import Rigctld
from .radios import get as get_kenwood


class KenwoodAdapter(RadioAdapter):
    """hamlib-controlled CAT rig + IF-tap SoapySDR spectrum, presented as a Flex."""

    provides = "iq"          # spectrum/IQ comes from the dongle; core does the FFT

    def __init__(self, model="TS-2000",
                 # hamlib control
                 rigctld_host="127.0.0.1", rigctld_port=4532, hamlib_model=None,
                 # rigctld auto-spawn (if serial_port given, we launch rigctld ourselves;
                 # otherwise we connect to an already-running rigctld at host:port)
                 serial_port=None, serial_baud=4800, rigctld_bin="rigctld",
                 serial_conf="serial_handshake=None,rts_state=ON,dtr_state=ON",
                 # soapy spectrum dongle (defaults suit an RTL-SDR Blog V4)
                 soapy_driver="rtlsdr", soapy_args="", samp_rate=2_040_000,
                 gain_db=40.0, direct_samp=None, agc=False,
                 # IF-tap lock: fix the dongle centre (Hz), never retune it — the
                 # rig's IF is always translated there, so the spectrum block
                 # stays put while only the rig VFO moves.
                 soapy_lock_hz=None,
                 # I/Q invert mirrors the spectrum (RTL 'iq_swap'); iq_correct
                 # enables the standard Soapy auto-correction API.
                 soapy_iq_invert=False, soapy_iq_correct=False,
                 # RX audio source: "sdr" = demodulate the dongle IQ (default);
                 # "radio" = the rig's own RX audio captured from a USB audio
                 # interface (e.g. PICO-Diginterface) — every mode, the rig's
                 # own filters/AGC, and no demod DSP load on this host.
                 audio_source="sdr", radio_audio_in=None, radio_audio_out=None,
                 # AE identity
                 advertise=None, serial="GATEKENW", station="Kenwood-CAT",
                 # TX intent — OFF by default. There is no PTT path wired in the
                 # gate yet (no wants_tx()/hamlib 'T 1' seam), so a real transceiver
                 # is receive+control-only until that lands. Setting this True only
                 # ADVERTISES tx_capable to AE; it does NOT yet key the radio. Wire
                 # it from the setup connection once a tested PTT seam exists.
                 enable_tx=False):
        self.model = model
        row = get_kenwood(model)
        self._row = row
        # advertise: explicit arg > registry row > sensible default
        adv = advertise or (row.advertise if row else "FLEX-6600")
        bands = tuple(b.name for b in row.bands) if row else ()

        self._hamlib_model = hamlib_model or (row.hamlib_model if row else None)
        # rigctld auto-spawn config. serial_conf default carries the TS-450 fix
        # (RTS+DTR asserted, no handshake) — proven necessary 2026-07-02 or hamlib
        # times out even though the radio answers.
        self._serial_port = serial_port
        self._serial_baud = int(serial_baud)
        self._rigctld_bin = rigctld_bin
        self._serial_conf = serial_conf
        self._rigctld_proc = None
        self._ctl = Rigctld(rigctld_host, rigctld_port)

        # RX audio source: "sdr" (dongle demod, default) | "radio" (the rig's
        # own audio, captured from the USB audio card — lazily opened).
        self._audio_source = audio_source if audio_source in ("sdr", "radio") else "sdr"
        self._radio_audio_in = radio_audio_in
        self._radio_audio_out = radio_audio_out
        self._radio_capture = None

        # TX (PTT + TX audio). PTT is applied by the SINGLE hamlib worker (the
        # 4800-baud serial rule) from a target posted by key_tx/unkey_tx; the
        # TX-audio drain runs on its own thread while keyed, feeding AE's
        # dax_tx audio to the rig's DATA input via the USB audio card. NB the
        # rig only routes its rear DATA input in DATA modes (J2B etc.); on
        # phone SSB use the rig's own mic — PTT-only from here.
        self._ptt_target = None              # True/False posted by key/unkey_tx
        self._ptt_state = False              # last state actually applied
        self._tx_audio_source = None         # engine.drain_tx_audio (wired by engine)
        self._tx_thread = None
        self._tx_stop = None
        self._tx_play = None

        # The dongle does the spectrum; start it centred wherever the rig is (updated on open()).
        self._sdr = SoapyAdapter(driver=soapy_driver, device_args=soapy_args,
                                 samp_rate=samp_rate, gain_db=gain_db,
                                 center_hz=14_100_000.0, model=adv,
                                 serial=serial, station=station,
                                 direct_samp=direct_samp, agc=agc,
                                 lock_hz=soapy_lock_hz,
                                 iq_invert=soapy_iq_invert, iq_correct=soapy_iq_correct)

        # bands= advertised to AE (radio-declared-bands). tx_capable reflects the
        # opt-in enable_tx flag, NOT a hardcoded "real transceiver" guess: no PTT is
        # wired yet, so it defaults False (RX+control only, like the icom9700/soapy
        # adapters). Flipping enable_tx True advertises the capability to AE but does
        # not key the rig until a wants_tx()/hamlib-PTT seam is implemented.
        self._enable_tx = bool(enable_tx)
        self.capabilities = AdapterCaps(model=adv, serial=serial, station=station,
                                        tx_capable=self._enable_tx,
                                        min_span_hz=48_000.0, max_span_hz=samp_rate,
                                        bands=bands)

        # rig state (from hamlib), refreshed by the single worker thread
        self._freq_hz = None
        self._mode = None
        self._smeter_db = None
        self._set_freq_target = None    # AE-requested freq (worker applies, coalesced)
        self._set_mode_target = None    # AE-requested mode
        self._ae_drive_t = 0.0          # when AE last drove a set (suppresses dial-read echo)
        self._poll_run = False
        self._poll_thread = None

    # --- rigctld auto-spawn ---------------------------------------------
    def _spawn_rigctld(self):
        """Launch rigctld ourselves for a serial rig, with the serial config
        baked in (so the user doesn't have to remember the RTS/DTR/handshake
        incantation). Only when serial_port was given; else we assume an
        already-running daemon at rigctld_host:port."""
        import subprocess
        cmd = [self._rigctld_bin, "-m", str(self._hamlib_model),
               "-r", self._serial_port, "-s", str(self._serial_baud),
               "-t", str(self._ctl.port)]
        if self._serial_conf:
            cmd += ["--set-conf", self._serial_conf]
        print(f"[kenwood] spawning rigctld: {' '.join(cmd)}", flush=True)
        self._rigctld_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                               stderr=subprocess.STDOUT)
        # wait until rigctld is actually accepting connections (it can take a
        # couple of seconds to open the serial port + bind on a Pi) — poll up to 8s.
        import socket as _sock
        end = time.monotonic() + 8.0
        while time.monotonic() < end:
            time.sleep(0.4)
            if self._rigctld_proc.poll() is not None:
                raise RuntimeError(f"rigctld exited immediately (rc={self._rigctld_proc.returncode}) "
                                   f"— check serial port {self._serial_port} / model {self._hamlib_model}")
            try:
                s = _sock.create_connection(("127.0.0.1", self._ctl.port), 0.5)
                s.close()
                print("[kenwood] rigctld ready", flush=True)
                return
            except OSError:
                continue
        print("[kenwood] WARNING: rigctld not accepting connections after 8s", flush=True)

    # --- lifecycle -------------------------------------------------------
    def open(self):
        # 0. if a serial port was given, spawn rigctld ourselves (with the fix)
        if self._serial_port:
            self._spawn_rigctld()
        # 1. hamlib control up first (so we know where the rig is tuned)
        self._ctl.connect()
        f = self._ctl.get_freq_hz()
        m = self._ctl.get_mode()
        if f:
            self._freq_hz = f
            if self._sdr.lock_hz is None:      # IF-tap lock: dongle centre is fixed,
                self._sdr.center_hz = float(f) # never clobber it with the rig's RF
                self._sdr._slice_hz = float(f)
        if m:
            self._mode = m
        # 2. dongle spectrum up, centred on the rig
        self._sdr.open()
        if m:
            self._sdr.set_mode(m)
        # 3. start the gentle CAT-steer poll
        self._poll_run = True
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True,
                                             name="kenwood-catsteer")
        self._poll_thread.start()

    def close(self):
        self._poll_run = False
        try:                                    # never leave the rig keyed
            self._ctl.set_ptt(False); self._ptt_state = False
        except Exception: pass
        self._stop_tx_audio()
        try: self._sdr.close()
        except Exception: pass
        try:
            if self._radio_capture is not None:
                self._radio_capture.close()
        except Exception: pass
        try: self._ctl.close()
        except Exception: pass
        if self._rigctld_proc:                 # kill our spawned rigctld
            try:
                self._rigctld_proc.terminate()
                self._rigctld_proc.wait(timeout=3)
            except Exception:
                try: self._rigctld_proc.kill()
                except Exception: pass

    # --- the SINGLE hamlib worker: ONLY this thread touches rigctld -------
    # 4800-baud serial can't take contention: the old design had the poll
    # thread AND the control-thread setters both hitting rigctld, corrupting
    # request/reply and dropping the link on rapid AE tunes (the "crash +
    # recover to 14.100" symptom). Now ALL rigctld I/O is serialised through
    # this one loop: it services a pending AE freq/mode SET first (coalesced —
    # only the latest target), else does one READ per tick to follow the dial.
    def _poll_loop(self):
        read_i = 0
        while self._poll_run:
            time.sleep(0.20)
            try:
                # 0. AE keyed/unkeyed? PTT first (safety-critical), and the
                #    single-writer rule means only this thread may touch rigctld.
                if self._apply_ptt():
                    continue
                # 1. AE asked to SET freq? (coalesced: only the latest wins)
                tgt = self._set_freq_target
                if tgt is not None:
                    self._set_freq_target = None
                    if self._ctl.set_freq_hz(tgt):
                        self._freq_hz = tgt
                        self._sdr.set_slice(float(tgt))
                    self._ae_drive_t = time.monotonic()   # suppress dial-read echo briefly
                    continue
                # 2. AE asked to SET mode?
                mtgt = self._set_mode_target
                if mtgt is not None:
                    self._set_mode_target = None
                    ok = self._ctl.set_mode(mtgt)
                    if ok:
                        self._mode = mtgt
                        self._sdr.set_mode(mtgt)
                    continue
                # 3. otherwise READ the rig to follow the dial. Still ONE serial
                #    transaction per tick (the 4800-baud contention rule), but
                #    WEIGHTED: freq gets 2 of every 3 ticks so the dial-sync lag
                #    is ~0.3-0.4s instead of ~0.6-0.9s ("a bit laggy" feedback);
                #    mode + S-meter alternate on the third tick (each ~1.2s —
                #    plenty for a rare mode change / a meter). Hold freq-read
                #    off for 1.5s after an AE set so we don't fight the user.
                read_i += 1
                slot = read_i % 3
                if slot != 2:                                  # 2 of 3 ticks: freq
                    if time.monotonic() - getattr(self, "_ae_drive_t", 0) > 1.5:
                        f = self._ctl.get_freq_hz()
                        if f and abs((self._freq_hz or 0) - f) > 1:
                            self._freq_hz = f
                            self._sdr.set_slice(float(f))
                elif (read_i // 3) % 2 == 0:                   # 3rd tick, alternating: mode
                    if time.monotonic() - getattr(self, "_ae_drive_t", 0) > 1.5:
                        m = self._ctl.get_mode()
                        if m and m != self._mode:
                            self._mode = m
                            self._sdr.set_mode(m)
                else:                                          # ...or S-meter
                    s = self._ctl.get_smeter_db()
                    if s is not None:
                        self._smeter_db = s
            except Exception:
                pass   # a CAT hiccup must not kill the worker

    # --- spectrum (delegated to the dongle) -----------------------------
    def get_iq(self, n, center_hz, span_hz):
        return self._sdr.get_iq(n, center_hz, span_hz)

    def get_audio(self, n_samples, slice_hz=None, mode=None):
        """RX audio for AE from the SELECTED source. "radio" reads the rig's own
        audio off the USB card (slice/mode are the rig's business there — it is
        already demodulating); a dead capture falls back to the SDR demod so the
        stream never goes silent."""
        if self._audio_source == "radio":
            cap = self._radio_capture_open()
            a = cap.read(n_samples) if cap else None
            if a is not None:
                return a
        return self._sdr.get_audio(n_samples, slice_hz=slice_hz, mode=mode) \
            if hasattr(self._sdr, "get_audio") else None

    # --- audio source selection (panel/CLI: sdr | radio) -------------------
    def audio_sources(self):
        return ("sdr", "radio")

    def set_audio_source(self, name):
        if name not in ("sdr", "radio"):
            raise ValueError(f"unknown audio source {name!r} (sdr|radio)")
        if name != self._audio_source:
            self._audio_source = name          # flip FIRST: in-flight reads fall back to SDR
            if name != "radio" and self._radio_capture is not None:
                # Leaving radio: release the arecord child — otherwise it keeps
                # capturing forever, holds the ALSA device (blocks other users)
                # and can linger as an unreaped zombie.
                cap, self._radio_capture = self._radio_capture, None
                try:
                    cap.close()
                except Exception:
                    pass
            print(f"[kenwood] audio source -> {name}", flush=True)

    def _radio_capture_open(self):
        if self._audio_source != "radio":
            return None                        # switched away mid-read: never (re)create
        if self._radio_capture is None:
            dev = self._radio_audio_in or find_alsa_card("pico") or "default"
            self._radio_capture = RadioAudioCapture(device=dev)
            print(f"[kenwood] radio-audio capture on {self._radio_capture.detail}", flush=True)
        return self._radio_capture

    # --- TX: AE's xmit -> hamlib PTT (single worker) + dax_tx -> rig audio --
    def key_tx(self):
        """AE keyed (MOX / xmit 1). The worker thread asserts PTT via hamlib."""
        self._ptt_target = True

    def unkey_tx(self):
        """AE unkeyed. Worker drops PTT; the drain thread fades out + exits."""
        self._ptt_target = False

    def _apply_ptt(self):
        """Worker side: apply a posted PTT target via hamlib (single writer).
        Returns True if a target was consumed (poll tick should skip other
        rig work). TX audio follows PTT: start on key, fade out on unkey."""
        pt = self._ptt_target
        if pt is None:
            return False
        self._ptt_target = None
        if self._ctl.set_ptt(pt):
            self._ptt_state = pt
            if pt:
                self._start_tx_audio()
            else:
                self._stop_tx_audio()
        return True

    def set_tx_audio_source(self, source):
        """Engine seam: source(max_bytes) -> AE TX audio, mono int16 LE @ 24 kHz
        (the dax_tx rate). Called once at wiring."""
        self._tx_audio_source = source

    def _start_tx_audio(self):
        if self._tx_audio_source is None or self._tx_thread is not None:
            return
        dev = self._radio_audio_out or find_alsa_card("pico") or "default"
        play = AlsaPcmPlayback(device=dev)
        if not play.open():
            print(f"[tx-audio] open failed on {dev}: {play.last_error}", flush=True)
            return
        self._tx_play = play
        self._tx_stop = threading.Event()
        self._tx_thread = threading.Thread(target=self._tx_loop, args=(self._tx_stop,),
                                           name="hamlib-tx-audio", daemon=True)
        self._tx_thread.start()
        print(f"[tx-audio] drain START -> {dev}", flush=True)

    def _stop_tx_audio(self):
        ev = self._tx_stop
        if ev is not None:
            ev.set()
        self._tx_thread = None
        self._tx_stop = None

    def _tx_loop(self, stop):
        """Drain AE's dax_tx ring -> aplay while keyed. 20 ms ticks with a
        fade in/out (no key clicks). Latency grace — the lesson from the
        IC-9700 path: AE keys BEFORE the audio lands over the network, so
        stay silent-but-running until real audio arrives, then keep flushing
        until the ring dries (bounded) after unkey."""
        src, play = self._tx_audio_source, self._tx_play
        n = 480                                    # 20 ms @ 24 kHz mono
        fade_chunks = 8                            # ~160 ms fade in/out
        chunk_i = 0
        seen = False
        stopped_at = None
        try:
            while True:
                stopped = stop.is_set()
                try:
                    got = src(n * 2) or b""
                except Exception:
                    got = b""
                if got:
                    seen = True
                    stopped_at = None             # audio still arriving: no fade-out yet
                elif stopped:
                    if not seen:
                        break                     # never got audio: drop quietly
                    if stopped_at is None:
                        stopped_at = time.monotonic()
                    elif time.monotonic() - stopped_at > 1.0:
                        break                     # flushed enough: fade done
                blk = (got + b"\x00" * (n * 2))[:n * 2] if got else b"\x00" * (n * 2)
                g = 1.0
                if not seen:
                    g = 0.0                       # silence until real audio shows
                elif stopped_at is not None:
                    g = max(0.0, 1.0 - (time.monotonic() - stopped_at) / 0.16)
                elif chunk_i < fade_chunks:
                    g = (chunk_i + 1) / fade_chunks
                if g < 1.0:
                    blk = self._scale_pcm(blk, g)
                if not play.write(blk):
                    break                         # sink died (unplugged) — stop
                chunk_i += 1
                time.sleep(0.005)                 # aplay buffers; natural pacing
        finally:
            try:
                play.close()
            except Exception:
                pass
            self._tx_play = None
            print("[tx-audio] drain STOP", flush=True)

    @staticmethod
    def _scale_pcm(blk, g):
        vals = struct.unpack("<%dh" % (len(blk) // 2), blk)
        vals = [max(-32768, min(32767, int(v * g))) for v in vals]
        return struct.pack("<%dh" % len(vals), *vals)

    # --- control (AE -> rig): POST a target; the worker does the hamlib I/O.
    # Non-blocking + coalescing — rapid AE tunes just overwrite the target, so
    # the serial link never backs up (fixes the crash-on-rapid-tune).
    def retune(self, center_hz):
        self._set_freq_target = int(center_hz)
        self._sdr.retune(center_hz)          # dongle centre is cheap + local, do it now

    def set_slice(self, slice_hz):
        self._set_freq_target = int(slice_hz)
        self._sdr.set_slice(slice_hz)        # local demod target, immediate

    def set_mode(self, mode):
        self._set_mode_target = mode
        self._mode = mode                    # optimistic: reflect intent NOW so the
                                             # radio->AE sync doesn't read stale + fight it
        self._ae_drive_t = time.monotonic()  # also guard the freq/mode read loop briefly
        self._sdr.set_mode(mode)             # local, immediate

    def set_span(self, span_hz):
        # dongle span is fixed by sample rate; nothing to push to the rig.
        pass

    # --- readback (rig -> AE) -------------------------------------------
    def initial_center_hz(self):
        return float(self._freq_hz) if self._freq_hz else None

    def initial_mode(self):
        return self._mode

    def radio_freq_hz(self):
        return self._freq_hz

    def radio_mode(self):
        return self._mode

    def receivers(self):
        """Single receiver -> one VFO. The engine's radio->AE sync
        (_sync_receivers) drives slice 0 from this list, so the rig's dial
        (read by the worker) reaches AE. One entry = one slice, no SUB."""
        if not self._freq_hz:
            return []
        return [{"freq_hz": self._freq_hz, "mode": self._mode}]

    def read_meters(self):
        # hamlib S-meter is dB relative to S9 (S9 = -73 dBm on HF).
        if self._smeter_db is None:
            return Meters()
        return Meters(s_meter_dbm=-73.0 + float(self._smeter_db))

    def diagnostics(self):
        return {
            "radio": self.model + " (hamlib CAT + IF-tap SDR)",
            "presented_as": self.capabilities.model,
            "link": {"transport": "hamlib rigctld + SoapySDR dongle",
                     "host": f"rigctld {self._ctl.host}:{self._ctl.port}",
                     "state": "connected" if self._ctl.connected else "down",
                     "hamlib_model": self._hamlib_model,
                     "dongle": self._sdr.driver},
            "vfos": ([{"name": "rig VFO", "freq_hz": self._freq_hz,
                       "mode": self._mode, "selected": True}] if self._freq_hz else []),
            "meters": {"s_meter_dbm": (round(-73.0 + self._smeter_db, 1)
                                       if self._smeter_db is not None else None)},
            "flags": {"cat_control": self._ctl.connected},
            "audio": {"source": self._audio_source,
                      "radio_in": (self._radio_capture.detail
                                   if self._radio_capture else None),
                      "tx_audio": bool(self._tx_audio_source)},
            "ptt": self._ptt_state,
        }
