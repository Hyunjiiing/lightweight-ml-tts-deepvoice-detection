# lightweight-ml-tts-deepvoice-detection
정보보호 기말 프로젝트: 경량 ML 기반 딥보이스·TTS 합성 음성 탐지 연구

### 1. 데이터셋 소개 (Dataset)

본 프로젝트는 TTS·보이스 클로닝 기술로 생성된 합성 음성을 탐지하기 위해
ASVspoof 2019 데이터셋을 사용하였다.

#### 1-1. 데이터셋 출처

ASVspoof 2019 데이터셋은 아래 링크에서 제공된다.  
https://datashare.ed.ac.uk/handle/10283/3336
  

#### 1-2. 데이터셋 구성

ASVspoof 2019는 합성 음성 탐지 연구를 위해 구축된 공개 벤치마크로, 다음 두 가지 형태로 구성된다.

1. Logical Access (LA)
TTS/VC 모델로 직접 생성된 순수 합성 음성 데이터

2. Physical Access (PA)
합성 음성을 스피커로 재생한 후 실제 환경에서 마이크로 다시 녹음한 데이터
→ 본 프로젝트는 보이스피싱 환경과 유사한 PA 데이터셋을 중심으로 실험을 진행하였다.
  

#### 1-3. 라이선스

본 데이터셋은 ODC Attribution License (ODC-By 1.0)을 따른다.
사용 시 반드시 아래와 같은 출처 표기가 필요하다.

ASVspoof 2019 Dataset에서 제공된 정보를 포함하고 있으며,
해당 데이터는 ODC Attribution License (ODC-By 1.0)에 따라 제공됩니다.

Wang, X., Yamagishi, J., Todisco, M., Patino, J., Nautsch, A.,
Bonastre, J.-F., Evans, N. (2019).
ASVspoof 2019: A large-scale public database of synthesized,
converted and replayed speech. arXiv:1911.01601.
