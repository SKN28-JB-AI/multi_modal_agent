"""씬 내레이션 → 보이스오버 지시문 자동 부착 테스트."""

from app.pipeline.orchestrator import Orchestrator
from app.schemas import Scene

from .conftest import MockBackend, auth_headers, make_sample_pdf, wait_for_job


def test_compose_prompt_embeds_voiceover():
    scene = Scene(index=0, prompt="sunny beach", duration_sec=6,
                  audio_description="gentle waves",
                  narration="시원한 하루를 시작하세요.")
    prompt = Orchestrator._compose_prompt(scene, language="ko",
                                          embed_narration=True)
    # 따옴표 안의 대사 + 언어 명시 + 화자 지정 (3요소)
    assert '"시원한 하루를 시작하세요."' in prompt
    assert "says in Korean" in prompt
    assert "narrator" in prompt
    assert "Audio: gentle waves" in prompt


def test_compose_prompt_skips_voiceover_for_tts_mode():
    scene = Scene(index=0, prompt="sunny beach", narration="대사입니다.")
    prompt = Orchestrator._compose_prompt(scene, language="ko",
                                          embed_narration=False)
    assert "대사입니다" not in prompt   # TTS 사용 시 이중 발화 방지


def test_compose_prompt_without_narration():
    scene = Scene(index=0, prompt="sunny beach")
    prompt = Orchestrator._compose_prompt(scene, "ko", True)
    assert "Voiceover" not in prompt


def test_pdf_job_passes_narration_to_backend(client, tmp_path):
    """E2E: 스토리보드의 씬 내레이션이 비디오 백엔드 프롬프트에 실제로 도달."""
    pdf = tmp_path / "plan.pdf"
    make_sample_pdf(pdf)
    resp = client.post(
        "/v1/videos/pdf",
        files={"file": ("plan.pdf", pdf.read_bytes(), "application/pdf")},
        data={"model": "mock", "options": '{"generation_mode": "scenes"}'},
        headers=auth_headers(),
    )
    body = wait_for_job(client, resp.json()["job_id"])
    assert body["status"] == "completed", body

    prompts = MockBackend.captured_prompts
    assert len(prompts) == 2
    assert 'says in Korean: "시원한 하루를 시작하세요."' in prompts[0]
    assert 'says in Korean: "지금 만나보세요."' in prompts[1]


def test_tts_mode_strips_voiceover_from_prompts(client, tmp_path, monkeypatch):
    """enable_narration=true 면 모델 발화 지시문이 빠져야 한다(이중 발화 방지)."""
    # TTS 호출은 mock 처리
    from app.pipeline import postprocess

    def fake_tts(script, api_key, model, voice, output_path):
        import subprocess
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=600:duration=2",
             str(output_path)], check=True, capture_output=True)

    monkeypatch.setattr(postprocess, "synthesize_narration", fake_tts)

    pdf = tmp_path / "plan.pdf"
    make_sample_pdf(pdf)
    resp = client.post(
        "/v1/videos/pdf",
        files={"file": ("plan.pdf", pdf.read_bytes(), "application/pdf")},
        data={"model": "mock",
              "options": '{"enable_narration": true, "generation_mode": "scenes"}'},
        headers=auth_headers(),
    )
    body = wait_for_job(client, resp.json()["job_id"])
    assert body["status"] == "completed", body
    assert all("Voiceover" not in p for p in MockBackend.captured_prompts)
