#
# Aether-gate — ALSA audio I/O for rig-audio bridges (capture + playback).
# Copyright (C) 2026 Nigel Fenton (G0JKN) + contributors. GPL-3.0-or-later.
#
"""Small stdlib-only ALSA helpers (arecord/aplay subprocesses, no bindings).

Two roles, both aimed at a USB audio interface that taps a rig (e.g. the
PICO-Diginterface), which shows up as its own ALSA card:

  * RadioAudioCapture — rig RX audio -> 24 kHz mono floats (the rate the
    engine's remote_audio_rx path expects). `plughw:` lets ALSA do rate/
    channel conversion, so whatever the card natively runs (48k stereo on
    full-speed USB) comes out at 24 kHz mono.
  * AlsaPcmPlayback   — AE TX audio (mono int16 LE @ 24 kHz, the dax_tx
    rate) -> the card, feeding the rig's DATA input. `plughw:` upsamples.

No python-alsaaudio / pyaudio dependency: arecord and aplay ship with
alsa-utils, which any host already has if it has ever recorded anything.
"""
import glob
import shutil
import struct
import subprocess


def find_alsa_card(name_frag="pico"):
    """Return a plughw: device string for the first ALSA card whose id
    contains name_frag (case-insensitive), else None. Scans /proc/asound."""
    frag = (name_frag or "").lower()
    if not frag:
        return None
    for path in sorted(glob.glob("/proc/asound/card*/id")):
        try:
            with open(path, "r", encoding="ascii", errors="ignore") as f:
                cid = f.read().strip()
        except OSError:
            continue
        if frag in cid.lower():
            return f"plughw:CARD={cid},DEV=0"
    return None


class RadioAudioCapture:
    """arecord wrapper: capture the rig's RX audio as 24 kHz mono floats.

    device=None -> auto: the first card whose id looks like a Pico
    Diginterface, else "default". The process is kept alive across reads
    (like the SoapyAdapter's persistent stream — reopening per frame would
    glitch the card); a dead arecord is reopened on the next read()."""

    def __init__(self, device=None, rate=24000, channels=1,
                 frames_per_read=1024, buffer_frames=4096):
        self.device = device
        self.rate = int(rate)
        self.channels = int(channels)
        self.frames_per_read = int(frames_per_read)
        self.buffer_frames = int(buffer_frames)
        self._proc = None
        self._buf = bytearray()
        self.last_error = None

    @staticmethod
    def default_device():
        return find_alsa_card("pico") or "default"

    @property
    def detail(self):
        return self.last_error or (self.device or self.default_device())

    def open(self):
        if self._proc and self._proc.poll() is None:
            return
        if not shutil.which("arecord"):
            self.last_error = "arecord not found (install alsa-utils)"
            return
        dev = self.device or self.default_device()
        cmd = ["arecord", "-q", "-D", dev, "-f", "S16_LE", "-c", str(self.channels),
               "-r", str(self.rate), "--period-size", str(self.frames_per_read),
               "--buffer-size", str(max(self.frames_per_read * 2, self.buffer_frames)),
               "--start-delay", "0", "-t", "raw", "-"]
        try:
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            self.last_error = None
        except Exception as e:
            self._proc = None
            self.last_error = str(e)

    def close(self):
        self._buf.clear()
        p, self._proc = self._proc, None
        if p and p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=1.0)
            except Exception:
                p.kill()

    def read(self, n_samples):
        """n mono float samples in [-1,1], or None (no data yet / capture down)."""
        self.open()
        if not self._proc or not self._proc.stdout:
            return None
        need = int(n_samples) * 2 * self.channels
        try:
            while len(self._buf) < need:
                chunk = self._proc.stdout.read(
                    max(need - len(self._buf), self.frames_per_read * 2 * self.channels))
                if not chunk:
                    err = b""
                    try:
                        err = self._proc.stderr.read(200) if self._proc.stderr else b""
                    except Exception:
                        pass
                    self.last_error = err.decode(errors="replace").strip() or "arecord stopped"
                    self.close()
                    return None
                self._buf.extend(chunk)
            raw = bytes(self._buf[:need])
            del self._buf[:need]
            vals = struct.unpack("<%dh" % (len(raw) // 2), raw)
            if self.channels != 1:
                vals = tuple(int(sum(vals[i:i + self.channels]) / self.channels)
                             for i in range(0, len(vals), self.channels))
            return [max(-1.0, min(1.0, s / 32768.0)) for s in vals[:n_samples]]
        except Exception as e:
            self.last_error = str(e)
            return None


class AlsaPcmPlayback:
    """aplay wrapper: stream mono int16 LE @ 24 kHz to the card (stdin pipe).

    Same plughw trick as the capture: ALSA converts 24 kHz to whatever the
    device runs. Used to feed a rig's DATA input from AE's dax_tx audio."""

    def __init__(self, device=None, rate=24000, channels=1):
        self.device = device
        self.rate = int(rate)
        self.channels = int(channels)
        self._proc = None
        self.last_error = None

    def open(self):
        if self._proc and self._proc.poll() is None:
            return True
        if not shutil.which("aplay"):
            self.last_error = "aplay not found (install alsa-utils)"
            return False
        dev = self.device or find_alsa_card("pico") or "default"
        cmd = ["aplay", "-q", "-D", dev, "-f", "S16_LE", "-c", str(self.channels),
               "-r", str(self.rate), "-t", "raw", "-"]
        try:
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.last_error = None
            return True
        except Exception as e:
            self._proc = None
            self.last_error = str(e)
            return False

    def write(self, pcm_bytes):
        """Push one block. Returns False if the sink died (caller may reopen)."""
        if not self.open() or not self._proc or not self._proc.stdin:
            return False
        try:
            self._proc.stdin.write(pcm_bytes)
            self._proc.stdin.flush()
            return True
        except Exception as e:
            self.last_error = str(e)
            return False

    def close(self):
        p, self._proc = self._proc, None
        if p:
            try:
                if p.stdin:
                    p.stdin.close()
            except Exception:
                pass
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=1.0)
                except Exception:
                    p.kill()
