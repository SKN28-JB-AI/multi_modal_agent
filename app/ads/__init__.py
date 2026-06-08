"""
ads 패키지: 광고 제작 4단계 파이프라인 (기존 /v1 API 와 독립).

  1) POST /v2/ads/storyboards          프롬프트 → 스토리보드 JSON
  2) POST /v2/ads/{job_id}/images      컷별 첫 장면 이미지 (1 완료 후)
  3) POST /v2/ads/{job_id}/videos      이미지 기반 컷 비디오 + 결합 (2 완료 후 ★)
  4) POST /v2/ads/{job_id}/proposal    광고 기획서 PDF (2·3과 무관, 1 완료 후)
"""
