"""WhisperX transcriber: faster-whisper + wav2vec2 word alignment. No diarization."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ...models.transcript import TranscriptSegment, WordSegment

logger = logging.getLogger(__name__)


class WhisperXTranscriber:
    name = "whisperx"

    def __init__(
        self,
        model_size: str = "medium.en",
        device: str = "cuda",
        compute_type: str = "float16",
        batch_size: int = 8,
        vad_method: str = "silero",
        sequential: bool = False,
        initial_prompt: str | None = None,
    ):
        import torch  # heavy; only when this transcriber is chosen

        self.model_size = model_size
        self.device = device if torch.cuda.is_available() else "cpu"
        self.compute_type = compute_type if self.device == "cuda" else "int8"
        self.batch_size = batch_size
        self.vad_method = vad_method
        self.sequential = sequential
        self.initial_prompt = initial_prompt
        self._model: Any = None
        self._align: dict[str, tuple[Any, Any]] = {}
        # Language Whisper detected (or was given) for the last transcribe() call.
        self.last_language: str | None = None

    def is_loaded(self) -> bool:
        return self._model is not None

    async def load(self) -> None:
        if self._model is not None:
            return
        logger.info(
            "Loading WhisperX model=%s device=%s compute=%s vad=%s",
            self.model_size,
            self.device,
            self.compute_type,
            self.vad_method,
        )

        def _load():
            if self.sequential:
                # faster-whisper on its own: WhisperX's batched pipeline decodes each merged ~30 s
                # VAD window without timestamps and can stop after its first line or loop on a
                # word, with no temperature fallback. WhisperX still aligns the words below.
                import torch  # noqa: F401  loads the bundled CUDA libs (libcublas) for ctranslate2
                from faster_whisper import WhisperModel

                return WhisperModel(
                    self.model_size, device=self.device, compute_type=self.compute_type
                )

            import whisperx

            asr_options = {"initial_prompt": self.initial_prompt} if self.initial_prompt else None
            return whisperx.load_model(
                self.model_size,
                self.device,
                compute_type=self.compute_type,
                vad_method=self.vad_method,
                asr_options=asr_options,
            )

        self._model = await asyncio.to_thread(_load)

    async def unload(self) -> None:
        self._model = None
        self._align.clear()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def transcribe(
        self, audio_path: str, language: str | None = None, progress=None
    ) -> list[TranscriptSegment]:
        if not self.is_loaded():
            await self.load()

        def _run():
            import whisperx

            audio = whisperx.load_audio(audio_path)
            if self.sequential:
                result = self._transcribe_sequential(audio, language, progress)
            else:
                # Transcription is ~85% of the work here, alignment the rest.
                cb = (lambda pct: progress(0.85 * pct / 100.0)) if progress else None
                result = self._model.transcribe(
                    audio,
                    batch_size=self.batch_size,
                    language=language or None,
                    progress_callback=cb,
                )
            lang = result.get("language", language or "en")
            self.last_language = lang
            if lang not in self._align:
                self._align[lang] = whisperx.load_align_model(
                    language_code=lang, device=self.device
                )
            align_model, metadata = self._align[lang]
            aligned = whisperx.align(
                result["segments"],
                align_model,
                metadata,
                audio,
                self.device,
                return_char_alignments=False,
            )
            if progress is not None:
                progress(1.0)
            return aligned["segments"]

        raw = await asyncio.to_thread(_run)
        return [_to_segment(s) for s in raw if s.get("text", "").strip()]

    def _transcribe_sequential(self, audio, language: str | None, progress) -> dict:
        # VAD runs before language detection, so a silent start no longer reads as English.
        # No conditioning on the previous text, so a hallucination loop cannot carry over.
        pieces, info = self._model.transcribe(
            audio,
            language=language or None,
            vad_filter=True,
            condition_on_previous_text=False,
            initial_prompt=self.initial_prompt,
        )
        segments = []
        for piece in pieces:
            if piece.text.strip():
                segments.append({"start": piece.start, "end": piece.end, "text": piece.text})
            if progress is not None and info.duration:
                progress(0.85 * min(piece.end / info.duration, 1.0))
        return {"segments": segments, "language": info.language}


def _to_segment(seg: dict) -> TranscriptSegment:
    words = None
    if seg.get("words"):
        words = [
            WordSegment(
                word=w.get("word", ""),
                start=float(w.get("start", seg.get("start", 0.0))),
                end=float(w.get("end", seg.get("end", 0.0))),
                score=float(w.get("score", 0.0)),
            )
            for w in seg["words"]
            if "word" in w
        ]
    return TranscriptSegment(
        text=seg.get("text", "").strip(),
        speaker="UNKNOWN",
        start=float(seg.get("start", 0.0)),
        end=float(seg.get("end", 0.0)),
        words=words,
    )
