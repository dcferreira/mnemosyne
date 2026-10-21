"""Opt-in accuracy settings: sequential WhisperX decoding and language from the remote audio."""

import sys
import types

import pytest

from mnemosyne.config import Settings
from mnemosyne.transcription import composed
from mnemosyne.transcription.composed import ComposedEngine
from mnemosyne.transcription.engine import AudioSource
from mnemosyne.transcription.registry import ENGINE_SETTINGS, build_transcriber
from mnemosyne.transcription.transcribers.whisperx import WhisperXTranscriber
from tests.fakes import FakeTranscriber

# ---- language from the remote audio ---------------------------------------------


class DetectingTranscriber(FakeTranscriber):
    """Detects `detected` for every call that is not given a language, like auto-detect."""

    def __init__(self, detected: str = "sl"):
        super().__init__()
        self.detected = detected
        self.last_language: str | None = None

    async def transcribe(self, audio_path, language=None, progress=None):
        out = await super().transcribe(audio_path, language=language, progress=progress)
        self.last_language = language or self.detected
        return out


MIC = AudioSource(path="/mic.ogg", kind="mic", speaker_label="Me")
SYS = AudioSource(path="/sys.ogg", kind="system")


async def _run(engine, sources):
    return [s async for s in engine.transcribe_sources(sources)]


@pytest.mark.anyio
async def test_language_from_remote_is_off_by_default():
    t = DetectingTranscriber()
    await _run(ComposedEngine(t, None, echo_dedup=False), [MIC, SYS])
    assert t.calls == [("/mic.ogg", None), ("/sys.ogg", None)]  # order and detection unchanged


@pytest.mark.anyio
async def test_language_from_remote_transcribes_remote_first_and_reuses_language():
    t = DetectingTranscriber("sl")
    engine = ComposedEngine(t, None, echo_dedup=False, language_from_remote=True)
    await _run(engine, [MIC, SYS])
    assert t.calls == [("/sys.ogg", None), ("/mic.ogg", "sl")]


@pytest.mark.anyio
async def test_language_from_remote_keeps_other_order_stable():
    t = DetectingTranscriber()
    a = AudioSource(path="/a.ogg", kind="mic", speaker_label="A")
    b = AudioSource(path="/b.ogg", kind="mic", speaker_label="B")
    s1 = AudioSource(path="/s1.ogg", kind="system")
    s2 = AudioSource(path="/s2.ogg", kind="system")
    engine = ComposedEngine(t, None, echo_dedup=False, language_from_remote=True)
    await _run(engine, [a, s1, b, s2])
    assert [p for p, _ in t.calls] == ["/s1.ogg", "/s2.ogg", "/a.ogg", "/b.ogg"]
    assert [lang for _, lang in t.calls] == [None, "sl", "sl", "sl"]


@pytest.mark.anyio
async def test_configured_language_beats_the_detected_one():
    t = DetectingTranscriber("sl")
    engine = ComposedEngine(t, None, language="en", echo_dedup=False, language_from_remote=True)
    await _run(engine, [MIC, SYS])
    assert [lang for _, lang in t.calls] == ["en", "en"]


@pytest.mark.anyio
async def test_transcriber_without_last_language_still_works():
    t = FakeTranscriber()  # no last_language attribute (parakeet, remote, ...)
    engine = ComposedEngine(t, None, echo_dedup=False, language_from_remote=True)
    await _run(engine, [MIC, SYS])
    assert t.calls == [("/sys.ogg", None), ("/mic.ogg", None)]


@pytest.mark.anyio
async def test_progress_weights_stay_paired_with_their_source(monkeypatch):
    monkeypatch.setattr(composed, "source_weights", lambda sources: [0.2, 0.8])  # mic, system
    seen: list[float] = []
    engine = ComposedEngine(
        DetectingTranscriber(), None, echo_dedup=False, language_from_remote=True
    )
    async for _ in engine.transcribe_sources(
        [MIC, SYS], on_progress=lambda stage, frac: seen.append(round(frac, 2))
    ):
        pass
    import asyncio

    await asyncio.sleep(0)
    assert seen == [0.8, 1.0]  # the system source (0.8 of the work) went first


@pytest.mark.anyio
async def test_language_is_logged_once(caplog):
    t = DetectingTranscriber("sl")
    engine = ComposedEngine(t, None, echo_dedup=False, language_from_remote=True)
    with caplog.at_level("INFO", logger="mnemosyne.transcription.composed"):
        await _run(engine, [MIC, SYS, AudioSource(path="/m2.ogg", kind="mic", speaker_label="B")])
    assert sum("Using language sl" in r.message for r in caplog.records) == 1


# ---- sequential WhisperX ----------------------------------------------------------


class _Seg:
    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text


def _fake_modules(monkeypatch):
    """Stand-ins for torch, whisperx and faster_whisper; records what was called."""
    log: dict = {"batched": 0, "sequential": [], "progress": [], "aligned": 0}

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)

    class Batched:
        def transcribe(self, audio, batch_size, language, progress_callback):
            log["batched"] += 1
            return {"segments": [{"start": 0, "end": 1, "text": "hi"}], "language": "en"}

    whisperx = types.ModuleType("whisperx")
    whisperx.load_audio = lambda path: "audio"
    whisperx.load_model = lambda *a, **k: Batched()
    whisperx.load_align_model = lambda language_code, device: ("align", {})

    def align(segments, *a, **k):
        log["aligned"] += 1
        return {"segments": segments}

    whisperx.align = align

    class WhisperModel:
        def __init__(self, size, device, compute_type):
            log["model"] = (size, device, compute_type)

        def transcribe(self, audio, **kwargs):
            log["sequential"].append(kwargs)
            info = types.SimpleNamespace(language="sl", duration=10.0)
            return iter(
                [_Seg(0, 4, " Dober dan."), _Seg(4, 5, "  "), _Seg(5, 10, " Kako ste?")]
            ), info

    faster = types.ModuleType("faster_whisper")
    faster.WhisperModel = WhisperModel
    for name, mod in (("torch", torch), ("whisperx", whisperx), ("faster_whisper", faster)):
        monkeypatch.setitem(sys.modules, name, mod)
    return log


@pytest.mark.anyio
async def test_batched_path_is_the_default(monkeypatch, tmp_path):
    log = _fake_modules(monkeypatch)
    assert Settings(data_dir=tmp_path).whisper_sequential is False
    t = build_transcriber(Settings(data_dir=tmp_path, transcriber="whisperx"))
    assert isinstance(t, WhisperXTranscriber) and t.sequential is False
    out = await t.transcribe("/a.wav")
    assert log["batched"] == 1 and log["sequential"] == [] and "model" not in log
    assert [s.text for s in out] == ["hi"]


@pytest.mark.anyio
async def test_sequential_path_decodes_with_faster_whisper(monkeypatch, tmp_path):
    log = _fake_modules(monkeypatch)
    s = Settings(
        data_dir=tmp_path, transcriber="whisperx", whisper_sequential=True, glossary="Kubernetes"
    )
    t = build_transcriber(s)
    assert t.sequential is True
    progress: list[float] = []
    out = await t.transcribe("/a.wav", language=None, progress=progress.append)

    assert log["batched"] == 0
    assert log["model"][0] == s.whisper_model_size
    (kwargs,) = log["sequential"]
    assert kwargs["vad_filter"] is True
    assert kwargs["condition_on_previous_text"] is False
    assert kwargs["language"] is None  # detected after VAD
    assert kwargs["initial_prompt"] == t.initial_prompt and t.initial_prompt
    assert log["aligned"] == 1  # WhisperX alignment still runs
    assert [s.text for s in out] == ["Dober dan.", "Kako ste?"]
    assert t.last_language == "sl"
    assert progress[-1] == 1.0
    assert 0.34 in [round(p, 2) for p in progress] and 0.85 in progress  # 0.85 * end / duration


@pytest.mark.anyio
async def test_sequential_passes_the_language_through(monkeypatch, tmp_path):
    log = _fake_modules(monkeypatch)
    t = WhisperXTranscriber(sequential=True)
    await t.transcribe("/a.wav", language="sl")
    assert log["sequential"][0]["language"] == "sl"


def test_new_settings_default_off_and_rebuild_the_engine(tmp_path):
    s = Settings(data_dir=tmp_path)
    assert s.whisper_sequential is False and s.language_from_remote is False
    assert "whisper_sequential" in ENGINE_SETTINGS
    assert "language_from_remote" in ENGINE_SETTINGS


def test_changing_new_settings_unloads_engine(client, ctx, fake_engine):
    for key in ("whisper_sequential", "language_from_remote"):
        fake_engine.loaded = True
        assert client.put("/api/settings", json={key: True}).status_code == 200
        assert ctx.models._engine is None
