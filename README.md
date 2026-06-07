# multi_modal_agent

광고용 멀티모달 동영상 생성 백엔드 서비스 (FastAPI).

텍스트 메시지 또는 **광고 기획서 PDF**를 입력받아, 선택한 비디오 생성 모델
(Sora 2 / Veo 3.1 / LTX-2.3 …)로 동영상을 만들어 반환한다.

## 아키텍처

```
요청(X-App-Key 인증)
 ├─ POST /v1/videos/message  : 프롬프트 1개 → 클립 1개
 └─ POST /v1/videos/pdf      : PDF 기획서 → 광고 영상
        ① 파싱/이해   PyMuPDF(텍스트+페이지 이미지) → GPT 비전 → 브리프
        ② 스토리보드  GPT → 씬 분할(씬별 프롬프트·길이·오디오·카피)
        ③ 씬별 생성   선택한 비디오 백엔드로 클립 생성 (교체 가능 지점)
        ④ 후처리      FFmpeg concat + SRT 자막 + (옵션)자막 번인·내레이션·로고
 → 202 + job_id  →  GET /v1/jobs/{id} 폴링  →  GET /v1/jobs/{id}/video
```

## 비디오 백엔드 (요청 시 `model` 로 선택)

| model | 제공자 | 필요 키 | 비고 |
|---|---|---|---|
| `sora-2`, `sora-2-pro` | OpenAI Videos API | `OPENAI_API_KEY` | ⚠️ 2026-09 API 종료 예정 |
| `veo-3.1`, `veo-3.1-fast` | Google Gemini API | `GEMINI_API_KEY` | 네이티브 오디오 |
| `ltx-2.3`, `ltx-2.3-fast` | fal.ai (Lightricks) | `FAL_API_KEY` | 저비용 |

**새 모델 추가**: `app/backends/` 에 `VideoBackend` 구현 파일 1개 +
`app/backends/__init__.py` 에 `register("이름", 클래스, **파라미터)` 1줄.
같은 클래스를 파라미터만 바꿔 여러 이름으로 등록할 수 있다.
`GET /v1/models` 와 요청 검증에 자동 반영된다.

## 설치/실행

```bash
conda create -n mma python=3.11 -y && conda activate mma
pip install -r requirements.txt

# FFmpeg 필수 (후처리)
winget install Gyan.FFmpeg        # Windows
# apt-get install ffmpeg          # Linux

cp .env.example .env              # 키 입력 (APP_KEYS 는 필수)

python run.py                     # 또는:
# uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000
```

API 문서: http://localhost:8000/docs

## 사용 예

```bash
# 등록된 모델 확인
curl -H "X-App-Key: dev-key-change-me" localhost:8000/v1/models

# 메시지 모드
curl -X POST localhost:8000/v1/videos/message \
  -H "X-App-Key: dev-key-change-me" -H "Content-Type: application/json" \
  -d '{"prompt": "노을 지는 해변, 잔잔한 파도 소리", "model": "ltx-2.3",
       "duration_sec": 8, "aspect_ratio": "16:9"}'

# PDF 기획서 모드 (+선택: 로고 오버레이)
curl -X POST localhost:8000/v1/videos/pdf \
  -H "X-App-Key: dev-key-change-me" \
  -F "file=@광고기획서.pdf" -F "model=veo-3.1" \
  -F 'options={"target_total_duration_sec": 24, "max_scenes": 4,
       "language": "ko", "enable_narration": true}' \
  -F "logo=@logo.png"

# 상태 폴링 → 다운로드
curl -H "X-App-Key: ..." localhost:8000/v1/jobs/{job_id}
curl -H "X-App-Key: ..." -o ad.mp4 localhost:8000/v1/jobs/{job_id}/video
curl -H "X-App-Key: ..." -o ad.srt localhost:8000/v1/jobs/{job_id}/subtitles

# remix — 완료된 잡의 특정 씬을 프롬프트로 부분 수정 (현재 sora-2 / sora-2-pro 지원)
# 새 잡(mode="remix")이 생성되고, 수정된 씬 + 나머지 원본 씬을 재결합한다.
# 원본 잡의 내레이션/로고도 자동 재적용.
curl -X POST localhost:8000/v1/jobs/{job_id}/remix \
  -H "X-App-Key: ..." -H "Content-Type: application/json" \
  -d '{"prompt": "배경을 밤 도시로, 비 내리는 분위기", "scene_index": 1}'
```

### PDF 모드 옵션 (`options` JSON)

| 키 | 기본 | 설명 |
|---|---|---|
| `target_total_duration_sec` | 24 | 목표 총 길이(씬 합계 기준) |
| `max_scenes` | 4 | 최대 씬 수(비용 가드, 1~8) |
| `aspect_ratio` / `resolution` | 16:9 / 1080p | 백엔드 지원 값으로 자동 보정 |
| `language` | ko | 카피·내레이션 언어 |
| `enable_narration` | false | OpenAI TTS 내레이션 합성 |
| `burn_subtitles` | false | SRT 를 영상에 굽기(재인코딩) |

## 설계 결정 (주의점 대응)

- **씬 길이 보정**: 백엔드마다 지원 길이가 달라(Sora 4/8/12s, Veo 4/6/8s,
  LTX 6/8/10s…) 요청 값을 가장 가까운 지원 값으로 자동 보정한다.
- **브랜드 요소**: 로고·화면 텍스트는 생성 모델이 거부/왜곡하므로
  프롬프트에서 배제하고 후처리(오버레이·SRT)로 처리한다(LLM 프롬프트에 명시).
- **부분 실패**: 한 씬이라도 실패하면 잡 전체 failed (부분 광고는 무의미).
  씬별 재시도(`CLIP_RETRIES`)와 동시성 제한(`MAX_CONCURRENT_CLIPS`) 내장.
- **모델 ID 드리프트**: 프리뷰 모델 ID 변경에 대비해 .env 로 덮어쓰기 가능.
- **확장**: 잡 저장소는 단일 프로세스(메모리+JSON) 가정. 다중 워커로 가려면
  `app/jobs/manager.py` 를 Redis/DB + 작업큐로 교체한다.
- **법적 표기**: AI 생성 광고물은 표시 의무(한국 AI기본법 등) 검토 필요 —
  서비스가 자동 처리하지 않으므로 송출 전 별도 확인.

## 테스트

```bash
pip install -r requirements-dev.txt
pytest tests/ -v     # mock 백엔드/LLM 기반 — 외부 API 호출 없음
```
