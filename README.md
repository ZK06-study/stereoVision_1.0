# 스테레오비전 학습 프로젝트

<div align="center">

![Python](https://img.shields.io/badge/Python-3.7+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.19+-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.3+-11557c?style=for-the-badge&logo=matplotlib&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

**컴퓨터 비전의 스테레오비전 알고리즘과 3D 깊이 추정을 위한 종합 교육 자료**

[시작하기](#시작하기) •
[문서](#문서) •
[예제 코드](#예제-코드) •
[기여하기](#기여하기)

</div>

---

## 📋 목차

- [프로젝트 개요](#-프로젝트-개요)
- [주요 특징](#-주요-특징)
- [프로젝트 구조](#-프로젝트-구조)
- [시작하기](#-시작하기)
- [학습 경로](#-학습-경로)
- [알고리즘 구현](#-알고리즘-구현)
- [성능 평가](#-성능-평가)
- [문서](#-문서)
- [예제 코드](#-예제-코드)
- [기여하기](#-기여하기)
- [라이선스](#-라이선스)
- [참고 자료](#-참고-자료)

---

## 🎯 프로젝트 개요

본 프로젝트는 **스테레오비전 기술의 체계적 학습**을 위해 설계된 종합 교육 플랫폼입니다. 컴퓨터 비전 분야의 핵심 기술인 스테레오비전의 이론적 배경부터 실제 구현까지 단계별로 학습할 수 있는 완전한 교육 자료를 제공합니다.

### 학습 목표

- 스테레오비전의 기본 원리 및 수학적 기초 이해
- 다양한 스테레오 매칭 알고리즘의 원리와 구현 방법 습득
- 3D 깊이 계산 및 포인트 클라우드 생성 기술 학습
- 실무 환경에서 활용 가능한 응용 개발 역량 배양

---

## ✨ 주요 특징

- **📚 체계적인 학습 자료**: 초급부터 고급까지 단계별 학습 과정
- **💻 실습 중심 교육**: Jupyter Notebook을 활용한 인터랙티브 학습
- **🔧 다양한 알고리즘**: StereoBM, StereoSGBM, 사용자 정의 알고리즘 구현
- **📊 성능 분석**: 알고리즘별 처리 속도 및 품질 비교 분석
- **🎨 시각적 학습**: 풍부한 이미지와 다이어그램을 통한 직관적 이해

---

## 📁 프로젝트 구조

```
stereoVision/
├── 📄 README.md                    # 프로젝트 개요 및 사용 가이드
├── 📁 docs/                        # 학습 문서 및 교육 자료
│   ├── 📖 stereo_vision_guide.md   # 스테레오비전 완벽 가이드 (59KB)
│   ├── 📝 concept_summary_sheet.md # 핵심 개념 요약 시트
│   ├── 🎤 presentation_script.md   # 발표용 스크립트
│   ├── 👨‍🏫 tutoring_script.md       # 교육용 스크립트
│   ├── 📋 stereo_vision_quiz.md    # 학습 확인 퀴즈
│   ├── 📝 stereo_vision_worksheet.md # 실습 워크시트
│   ├── 🌐 index.html               # 웹 기반 학습 자료
│   └── 📁 python_examples/         # Python 구현 코드
│       ├── 📄 README.md            # Python 예제 사용 가이드
│       ├── 🐍 basic_stereo.py      # 기본 스테레오비전 구현
│       ├── 🔍 stereo_matching.py   # 스테레오 매칭 알고리즘
│       ├── 📏 depth_calculation.py # 깊이 계산 및 3D 재구성
│       ├── 📓 01_basic_stereo_vision.ipynb        # 기본 개념 튜토리얼
│       ├── 📓 02_stereo_matching_algorithms.ipynb # 알고리즘 심화 학습
│       └── 📓 03_depth_calculation_and_3d.ipynb   # 3D 처리 실습
└── 📁 images/                      # 교육용 이미지 및 다이어그램 (11개 파일)
```

---

## 🚀 시작하기

### 시스템 요구사항

| 구성 요소 | 최소 버전 | 권장 버전 |
|-----------|----------|----------|
| Python    | 3.7      | 3.9+     |
| OpenCV    | 4.0      | 4.5+     |
| NumPy     | 1.19     | 1.21+    |
| Matplotlib| 3.3      | 3.5+     |

### 설치 과정

1. **저장소 복제**
```bash
git clone https://github.com/your-username/stereoVision.git
cd stereoVision
```

2. **가상 환경 설정** (권장)
```bash
python -m venv stereo_env
source stereo_env/bin/activate  # Windows: stereo_env\Scripts\activate
```

3. **필수 패키지 설치**
```bash
# 기본 패키지
pip install opencv-python numpy matplotlib jupyter

# 고급 기능용 패키지 (선택사항)
pip install open3d scikit-image scipy
```

4. **학습 환경 실행**
```bash
jupyter notebook docs/python_examples/
```

---

## 📚 학습 경로

### 🟢 초급 과정 (기본 개념)
**예상 소요 시간: 4-6시간**

1. **이론 학습**: `docs/stereo_vision_guide.md` - 스테레오비전의 수학적 원리
2. **실습**: `01_basic_stereo_vision.ipynb` - 기본 개념 구현
3. **확인**: `docs/stereo_vision_quiz.md` - 이해도 점검

### 🟡 중급 과정 (알고리즘 구현)
**예상 소요 시간: 8-10시간**

1. **알고리즘 심화**: `02_stereo_matching_algorithms.ipynb` - 다양한 매칭 알고리즘
2. **성능 최적화**: `stereo_matching.py` - 사용자 정의 알고리즘 개발
3. **비교 분석**: 알고리즘별 성능 벤치마킹

### 🔴 고급 과정 (3D 재구성)
**예상 소요 시간: 10-12시간**

1. **3D 처리**: `03_depth_calculation_and_3d.ipynb` - 포인트 클라우드 생성
2. **실제 응용**: `depth_calculation.py` - 실시간 거리 측정
3. **프로젝트**: 개인 맞춤형 스테레오비전 시스템 구축

---

## 🔧 알고리즘 구현

### 지원 알고리즘

| 알고리즘 | 유형 | 특징 | 적용 분야 |
|----------|------|------|----------|
| **StereoBM** | 로컬 매칭 | 빠른 처리 속도 | 실시간 응용 |
| **StereoSGBM** | 반전역 매칭 | 높은 정확도 | 정밀 측정 |
| **사용자 정의** | 커스텀 | 학습 목적 | 교육용 |

### 구현 세부사항

- **SAD (Sum of Absolute Differences)**: 기본적인 블록 매칭
- **SSD (Sum of Squared Differences)**: 제곱 차이 기반 매칭
- **Census Transform**: 로버스트한 특징 매칭
- **동적 프로그래밍**: 최적화된 매칭 경로 탐색

---

## 📊 성능 평가

### 벤치마크 결과
*테스트 환경: Intel Core i7, 8GB RAM, 640×480 해상도*

| 알고리즘 | 평균 처리 시간 | 메모리 사용량 | 정확도 지수 | 권장 용도 |
|----------|--------------|-------------|------------|----------|
| StereoBM | ~30ms | 낮음 | 7/10 | 실시간 처리 |
| StereoSGBM | ~150ms | 중간 | 9/10 | 고품질 결과 |
| 사용자 정의 | 가변 | 낮음 | 6/10 | 교육 목적 |

### 품질 평가 지표

- **Disparity Accuracy**: 시차 맵의 정확도
- **Edge Preservation**: 경계 보존 능력
- **Noise Robustness**: 노이즈 내성
- **Computational Efficiency**: 연산 효율성

---

## 📖 문서

### 핵심 문서

- **[완벽 가이드](docs/stereo_vision_guide.md)**: 스테레오비전의 모든 것 (59KB)
- **[개념 요약](docs/concept_summary_sheet.md)**: 핵심 개념 정리
- **[학습 퀴즈](docs/stereo_vision_quiz.md)**: 이해도 확인
- **[실습 워크시트](docs/stereo_vision_worksheet.md)**: 단계별 실습

### 교육 자료

- **[발표 스크립트](docs/presentation_script.md)**: 강의용 자료
- **[교육 스크립트](docs/tutoring_script.md)**: 개별 지도용
- **[웹 자료](docs/index.html)**: 브라우저 기반 학습

---

## 💻 예제 코드

### Jupyter Notebook 시리즈

1. **[기본 스테레오비전](docs/python_examples/01_basic_stereo_vision.ipynb)**
   - 카메라 보정 및 스테레오 설정
   - 기본 시차 맵 생성

2. **[매칭 알고리즘](docs/python_examples/02_stereo_matching_algorithms.ipynb)**
   - 다양한 알고리즘 비교
   - 성능 최적화 기법

3. **[3D 재구성](docs/python_examples/03_depth_calculation_and_3d.ipynb)**
   - 포인트 클라우드 생성
   - 3D 시각화

### Python 모듈

- **[basic_stereo.py](docs/python_examples/basic_stereo.py)**: 기초 구현
- **[stereo_matching.py](docs/python_examples/stereo_matching.py)**: 고급 알고리즘
- **[depth_calculation.py](docs/python_examples/depth_calculation.py)**: 3D 처리

---

## 🤝 기여하기

본 프로젝트는 오픈소스 기여를 환영합니다. 다음 가이드라인을 따라 참여해 주시기 바랍니다:

### 기여 절차

1. **프로젝트 포크**: 개인 계정으로 저장소 복사
2. **기능 브랜치 생성**: `git checkout -b feature/새로운기능`
3. **변경사항 구현**: 코드 작성 및 테스트
4. **커밋**: `git commit -m "기능: 새로운 기능 추가"`
5. **푸시**: `git push origin feature/새로운기능`
6. **풀 리퀘스트 생성**: 상세한 설명과 함께 제출

### 기여 가능 영역

- 새로운 알고리즘 구현
- 문서 개선 및 번역
- 성능 최적화
- 버그 수정
- 테스트 케이스 추가

---

## 📄 라이선스

본 프로젝트는 교육 목적으로 제작되었습니다. 자세한 라이선스 정보는 [LICENSE](LICENSE) 파일을 참조하시기 바랍니다.

---

## 📚 참고 자료

### 주요 참고 문헌

- Scharstein, D., & Szeliski, R. (2002). *A taxonomy and evaluation of dense two-frame stereo correspondence algorithms*
- Hirschmuller, H. (2008). *Stereo processing by semiglobal matching and mutual information*
- Brown, M. Z., Burschka, D., & Hager, G. D. (2003). *Advances in computational stereo*

### 참고 리포지토리

본 프로젝트는 다음 오픈소스 프로젝트를 참고하여 제작되었습니다:
- [SagiK-Repository/StereoVision](https://github.com/SagiK-Repository/StereoVision.git)

### 추가 학습 자료

- [OpenCV 스테레오비전 문서](https://docs.opencv.org/master/dd/d53/tutorial_py_depthmap.html)
- [Middlebury 스테레오 데이터셋](https://vision.middlebury.edu/stereo/)
- [KITTI 비전 벤치마크](http://www.cvlibs.net/datasets/kitti/)

---

<div align="center">

**📧 문의사항이나 제안이 있으시면 [Issues](../../issues)를 통해 연락해 주시기 바랍니다.**


</div>
