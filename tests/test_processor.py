"""Tests for screen-recorder-analyzer - no GPU/video files required."""
import sys
import os
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_processor_imports_cleanly():
    """Module must import without heavy deps being installed."""
    from screen_recorder_analyzer import processor  # noqa: F401


def test_video_processor_init():
    from screen_recorder_analyzer.processor import VideoProcessor
    p = VideoProcessor(whisper_model_size="tiny", frame_skip=5, max_frames=10)
    assert p.whisper_model_size == "tiny"
    assert p.frame_skip == 5
    assert p.max_frames == 10


def test_video_processor_missing_file():
    from screen_recorder_analyzer.processor import VideoProcessor
    p = VideoProcessor()
    with pytest.raises(Exception):
        p.extract_audio("/nonexistent/video.mp4")


def test_extract_actions_raises_without_api_key(monkeypatch):
    """extract_actions must raise when no API key is set."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from screen_recorder_analyzer.processor import extract_actions
    with pytest.raises(Exception):
        extract_actions({"transcript": "hello", "frame_analysis": []})


def test_action_prompt_structure():
    """The action extraction prompt should include the transcript."""
    from screen_recorder_analyzer.processor import extract_actions
    import unittest.mock as mock

    fake_actions = [{"id": "1", "tools": ["excel"], "action": ["viewing data"]}]

    with mock.patch("screen_recorder_analyzer.llm.ask_llm") as mock_ask:
        mock_ask.return_value = json.dumps(fake_actions)

        result = extract_actions(
            {"transcript": "I opened Excel and sorted column A.", "frame_analysis": []},
        )
        assert isinstance(result, list)
        assert result[0]["tools"] == ["excel"]
        # Verify the prompt included the transcript (may be positional or keyword)
        mock_ask.assert_called_once()
        args, kwargs = mock_ask.call_args
        prompt_text = args[0] if args else kwargs.get("prompt", "")
        assert "Excel" in prompt_text


def test_process_mode_gating(monkeypatch):
    """process(mode=...) must skip the right pipeline stages."""
    from screen_recorder_analyzer.processor import VideoProcessor
    import unittest.mock as mock

    p = VideoProcessor()
    monkeypatch.setattr(p, "get_metadata", lambda _: {})
    monkeypatch.setattr(p, "extract_audio", mock.Mock(return_value="/tmp/a.wav"))
    monkeypatch.setattr(p, "transcribe", mock.Mock(return_value="hello"))
    monkeypatch.setattr(p, "analyze_frames", mock.Mock(return_value=[{"status": "ok"}]))
    # extract_audio is stubbed; the os.remove cleanup path should be a no-op.
    monkeypatch.setattr("screen_recorder_analyzer.processor.os.path.exists", lambda _: False)

    # ocr_only: no audio/transcribe, but frames analyzed.
    r = p.process("video.mp4", mode="ocr_only")
    assert r["mode"] == "ocr_only"
    p.extract_audio.assert_not_called()
    p.analyze_frames.assert_called_once()

    p.analyze_frames.reset_mock()

    # transcription_only: audio transcribed, no OCR.
    r = p.process("video.mp4", mode="transcription_only")
    assert r["transcript"] == "hello"
    assert r["frame_analysis"] == []
    p.analyze_frames.assert_not_called()


def test_api_config_has_mode():
    """The REST ProcessingConfig must expose the new mode + whisper_backend fields."""
    try:
        from screen_recorder_analyzer.api import ProcessingConfig
    except ImportError:
        pytest.skip("fastapi/pydantic not installed")
    cfg = ProcessingConfig()
    assert cfg.mode == "full"
    assert cfg.whisper_backend == "local"


def test_api_app_creates():
    """FastAPI app must be importable."""
    try:
        from screen_recorder_analyzer.api import app
        assert app is not None
    except ImportError:
        pytest.skip("fastapi not installed")
