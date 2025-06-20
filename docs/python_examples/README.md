# 스테레오비전 Python 실습 예제

이 폴더에는 스테레오비전 학습을 위한 Python 실습 예제들이 포함되어 있습니다.

## 📁 파일 구성

### 1. `basic_stereo.py`
- **목적**: OpenCV를 이용한 기본 스테레오비전 구현
- **주요 기능**:
  - StereoBM과 StereoSGBM 알고리즘 비교
  - 매개변수 조정 및 결과 시각화
  - 샘플 데이터 생성 기능

### 2. `stereo_matching.py`
- **목적**: 다양한 스테레오 매칭 알고리즘 직접 구현
- **주요 기능**:
  - SAD (Sum of Absolute Differences) 구현
  - SSD (Sum of Squared Differences) 구현
  - Census Transform 구현
  - 알고리즘 성능 비교

### 3. `depth_calculation.py`
- **목적**: 시차 맵으로부터 깊이 정보 계산 및 3D 재구성
- **주요 기능**:
  - 시차-깊이 변환
  - 3D 점군 생성
  - 깊이 맵 통계 분석
  - 3D 시각화

## 🚀 실행 방법

### 필요 라이브러리 설치

```bash
pip install opencv-python numpy matplotlib
```

### 실행 순서

1. **기본 스테레오비전 학습**
   ```bash
   python basic_stereo.py
   ```

2. **매칭 알고리즘 비교**
   ```bash
   python stereo_matching.py
   ```

3. **깊이 계산 및 3D 재구성**
   ```bash
   python depth_calculation.py
   ```

## 📚 학습 가이드

### 초급자용
1. `basic_stereo.py`부터 시작하세요
2. OpenCV의 기본 사용법을 익히세요
3. 매개변수 변경이 결과에 미치는 영향을 관찰하세요

### 중급자용
1. `stereo_matching.py`로 알고리즘 내부 동작을 이해하세요
2. 각 알고리즘의 장단점을 비교해보세요
3. 커스텀 매칭 함수를 구현해보세요

### 고급자용
1. `depth_calculation.py`로 3D 재구성을 학습하세요
2. 실제 카메라 데이터로 테스트해보세요
3. 성능 최적화를 시도해보세요

## 🔧 코드 수정 가이드

### 실제 데이터 사용하기

각 파일의 메인 함수에서 경로를 수정하세요:

```python
# basic_stereo.py에서
left_path = 'your_left_image.png'
right_path = 'your_right_image.png'
```

### 매개변수 조정

```python
# 시차 범위 조정
num_disparities = 64  # 16의 배수로 설정

# 윈도우 크기 조정
block_size = 15  # 홀수로 설정

# 카메라 매개변수
focal_length = 700  # 픽셀 단위
baseline = 0.2      # 미터 단위
```

## 📊 예상 결과

### basic_stereo.py 실행 결과
- 좌우 영상과 시차 맵 비교 화면
- StereoBM vs StereoSGBM 결과 비교
- 처리 시간 및 통계 정보

### stereo_matching.py 실행 결과
- SAD, SSD, Census Transform 비교
- 알고리즘별 처리 시간 그래프
- 각 방법의 시차 맵 품질 비교

### depth_calculation.py 실행 결과
- 시차 맵과 깊이 맵 변환
- 3D 점군 시각화
- 깊이 분포 히스토그램

## 🎯 실습 과제

### 과제 1: 매개변수 최적화
- 다양한 `numDisparities`와 `blockSize` 조합을 테스트
- 최적의 조합 찾기

### 과제 2: 알고리즘 성능 분석
- 각 매칭 알고리즘의 장단점 정리
- 처리 시간 vs 품질 트레이드오프 분석

### 과제 3: 실제 데이터 적용
- 본인의 스테레오 영상으로 테스트
- 결과 분석 및 개선 방안 제안

## 🐛 문제 해결

### 자주 발생하는 오류

1. **ModuleNotFoundError: No module named 'cv2'**
   ```bash
   pip install opencv-python
   ```

2. **영상을 불러올 수 없음**
   - 파일 경로 확인
   - 지원되는 이미지 형식인지 확인 (jpg, png, bmp 등)

3. **메모리 부족**
   - 영상 크기를 줄이거나
   - 시차 범위를 축소

4. **처리 시간이 너무 오래 걸림**
   - 윈도우 크기를 줄이거나
   - 시차 범위를 축소
   - 영상 해상도를 낮춤

## 📖 추가 학습 자료

- OpenCV 공식 문서: [Depth Map from Stereo Images](https://docs.opencv.org/master/dd/d53/tutorial_py_depthmap.html)
- 스테레오비전 이론: `../stereo_vision_guide.md`
- 학습지: `../stereo_vision_worksheet.md`
- 퀴즈: `../stereo_vision_quiz.md`

## 💡 팁

1. **디버깅**: 중간 결과를 시각화하여 문제점 파악
2. **최적화**: 병렬 처리나 NumPy 벡터화 활용
3. **검증**: 알려진 거리의 물체로 정확도 검증
4. **실험**: 다양한 조건에서 테스트

---

**즐거운 스테레오비전 학습 되세요!** 🎉