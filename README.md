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
| `wan-2.2` | Alibaba Model Studio (Wan) | `DASHSCOPE_API_KEY` | 무음, 5초 고정, 480P/1080P |
| `wan-2.5` | Alibaba Model Studio (Wan) | `DASHSCOPE_API_KEY` | 네이티브 오디오(보이스오버), 5/10초 |
| `wan-2.6` | Alibaba Model Studio (Wan) | `DASHSCOPE_API_KEY` | 2~15초(가변), 네이티브 오디오, 720P/1080P |
| `wan-2.7` | Alibaba Model Studio (Wan) | `DASHSCOPE_API_KEY` | 최신, 2~15초(가변), 오디오, 720P/1080P, first/last 제어 |

> Alibaba(DashScope) 모델은 기본 International(싱가포르) 엔드포인트를 쓴다.
> 다른 리전은 `DASHSCOPE_BASE_URL` 로 덮어쓴다(모델·키·엔드포인트는 동일 리전이어야 함).
> t2v/i2v 모델 ID 는 `WAN_T2V_MODEL_DEFAULT` / `WAN_I2V_MODEL_DEFAULT` 로 교체 가능.
> Wan 은 image-to-video(`supports_image_input`)를 지원해 /v2/ads 3단계에서도 쓸 수 있다.

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
# 비디오 생성 전, OpenAI 기본 모델(OPENAI_LLM_MODEL)이 입력 프롬프트를
# 대상 비디오 모델(ltx/veo/sora)에 맞는 프롬프트로 자동 변환한다(선행 단계).
# 끄려면 본문에 "enhance_prompt": false, 또는 서버에서 ENHANCE_MESSAGE_PROMPT=false.
# OPENAI_API_KEY 가 없거나 변환 실패 시 원본 프롬프트로 자동 폴백한다.
curl -X POST localhost:8000/v1/videos/message \
  -H "X-App-Key: dev-key-change-me" -H "Content-Type: application/json" \
  -d '{"prompt": "노을 지는 해변, 잔잔한 파도 소리", "model": "ltx-2.3",
       "duration_sec": 8, "aspect_ratio": "16:9", "enhance_prompt": true}'

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
| `generation_mode` | single | **single**: 샷 타임라인 프롬프트로 1회 생성(백엔드 최대 길이로 보정: Sora 12s/Veo 8s/LTX 10~20s). **scenes**: 씬별 생성 후 결합(긴 광고용) |
| `target_total_duration_sec` | 24 | 목표 총 길이(씬 합계 기준) |
| `max_scenes` | 4 | 최대 씬 수(비용 가드, 1~8) |
| `aspect_ratio` / `resolution` | 16:9 / 1080p | 백엔드 지원 값으로 자동 보정 |
| `language` | ko | 카피·내레이션 언어 |
| `enable_narration` | false | OpenAI TTS 내레이션 합성 |
| `burn_subtitles` | false | SRT 를 영상에 굽기(재인코딩) |
| `logo_name` | 없음 | 서버 `logos/` 폴더의 로고 파일명 |

### 로고 적용 규칙 (PDF 모드)

서버의 `logos/` 폴더에 브랜드 로고(png/jpg/webp)를 넣어두면 광고 영상에
항상 로고가 오버레이된다. 우선순위:

1. 요청 multipart 의 `logo` 업로드 (일회성)
2. `options.logo_name` 으로 지정한 파일 (`GET /v1/logos` 로 목록 조회)
3. `logos/default.png`
4. `logos/` 의 첫 파일(이름순)

폴더가 비어 있고 업로드도 없으면 로고 없이 진행한다.

## 광고 파이프라인 API (/v2/ads) — 기존 /v1 과 독립

프롬프트 → 광고 영상 제작을 4단계로 분리해, 각 단계를 개별 API 로 실행/검수할 수 있다.

```
1) POST /v2/ads/storyboards          프롬프트 → 스토리보드 JSON (잡 생성)
2) POST /v2/ads/{job_id}/images      컷별 첫 장면 이미지 (1 완료 후)
3) POST /v2/ads/{job_id}/videos      이미지를 시작 프레임으로 컷 비디오 + 결합
                                     ★ 2단계(images) 완료 전 호출 시 412
4) POST /v2/ads/{job_id}/proposal    광고 기획서 PDF — 2·3단계와 무관하게 실행 가능

GET /v2/ads/{job_id}                 단계별 상태 + 산출물 URL (폴링)
GET /v2/ads/{job_id}/storyboard      스토리보드 JSON
GET /v2/ads/{job_id}/images/{cut}    컷 이미지(PNG)
GET /v2/ads/{job_id}/videos/{cut}    컷 클립(MP4)
GET /v2/ads/{job_id}/video           최종 결합본(MP4)
GET /v2/ads/{job_id}/proposal        기획서(PDF)
GET /v2/ads/image-models             이미지 모델 목록
```

- 모든 단계는 202 + `job_id` 를 즉시 반환하고 백그라운드로 실행된다.
- 상태 코드: `412` 선행 단계 미완료 / `409` 실행 중·완료(재실행은 `?force=true`) /
  `422` 모델 오류 / `503` API 키 미설정.
- **이미지 모델**(2단계, 요청 `model` 로 선택, 기본 `gpt-image-2`):
  OpenAI `gpt-image-2`·`gpt-image-1` / Google `imagen-4.0` / fal.ai `flux-dev`·`flux-schnell` /
  Alibaba `qwen-image`·`qwen-image-plus`(`DASHSCOPE_API_KEY`).
  생성 후 비디오 해상도로 cover-crop 되어 Sora `input_reference` 픽셀 규격을 만족한다.
- **비디오 모델**(3단계, 기본 `veo-3.1`): 기존 백엔드 3종(Sora/Veo/LTX) + Wan(`wan-2.2`/`wan-2.5`) 모두
  image-to-video 지원(`supports_image_input`). 컷 길이에 맞춰 트리밍 후 FFmpeg 로 결합.
- **기획서 PDF**(4단계): 한글 폰트 자동 탐색(맑은고딕/나눔고딕 → 내장 CID 폴백).
  컷 이미지가 이미 생성돼 있으면 시안으로 함께 삽입된다.

```bash
# 예시
curl -s -X POST localhost:8000/v2/ads/storyboards \
  -H "X-App-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"prompt":"청년 적금 신규 캠페인", "options":{"cut_count":3,"total_duration_sec":16}}'
# → {"job_id":"abc123", ...}
curl -s -X POST localhost:8000/v2/ads/abc123/images   -H "X-App-Key: $KEY" \
  -H "Content-Type: application/json" -d '{"model":"gpt-image-2"}'
curl -s -X POST localhost:8000/v2/ads/abc123/videos   -H "X-App-Key: $KEY" \
  -H "Content-Type: application/json" -d '{"model":"veo-3.1"}'
curl -s -X POST localhost:8000/v2/ads/abc123/proposal -H "X-App-Key: $KEY"
```

## 설계 결정 (주의점 대응)

- **씬 길이 