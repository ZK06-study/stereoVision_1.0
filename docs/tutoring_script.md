# 스테레오 비전 기술 교육 자료

## 과정 개요

본 문서는 스테레오 비전의 핵심 원리부터 실제 파이썬 코드 구현, 3D 복원 및 실무 문제 해결까지 총 3차시의 교육 과정으로 구성된 기술 자료입니다. 각 차시는 이론 설명과 구체적인 실습 예제를 포함하여 학습자가 단계적으로 기술을 습득할 수 있도록 설계되었습니다.

---
---

# 1차시: 스테레오 비전 핵심 원리

### **학습 목표**
- 스테레오 비전의 기본 개념과 원리를 이해한다.
- 깊이가 계산되는 핵심 공식을 설명할 수 있다.
- 스테레오 매칭의 개념과 시차 맵(Disparity Map)의 의미를 이해한다.

### **준비 자료**
- `document/stereo_vision_guide.md`
- `document/concept_summary_sheet.md`

---

## 1. 스테레오 비전의 정의 (Definition of Stereo Vision)

스테레오 비전은 2개의 카메라를 사용하여 3차원 깊이 정보를 추출하는 컴퓨터 비전 기술입니다. 이는 인간이 두 눈을 통해 사물과의 거리를 인지하는 **양안시(Binocular Vision)** 원리를 모방한 것입니다.

일반 카메라는 3차원 공간을 2차원 평면 이미지로 투영하므로 깊이 정보가 소실됩니다. 스테레오 비전은 서로 다른 위치에 있는 두 카메라로 촬영한 한 쌍의 이미지에서 이 소실된 깊이 정보를 복원하는 것을 목표로 합니다.

## 2. 시차와 깊이의 관계 (Relationship between Disparity and Depth)

두 카메라가 동일한 객체를 촬영하면, 카메라의 위치 차이로 인해 각 이미지에 객체가 맺히는 위치가 달라집니다. 이 이미지 평면상에서의 픽셀 위치 차이를 **시차(Disparity)** 라고 합니다.

시차와 객체까지의 실제 깊이(거리)는 **반비례** 관계를 가집니다.
- **시차가 크다**: 객체가 카메라에 가깝다.
- **시차가 작다**: 객체가 카메라로부터 멀다.

## 3. 3차원 깊이 계산 원리 (Principle of 3D Depth Calculation)

깊이는 삼각형의 닮음 원리를 통해 수학적으로 계산할 수 있습니다. 핵심 공식은 다음과 같습니다.

> ### **Z = (f × b) / d**

- **Z**: 카메라로부터 객체까지의 **깊이 (Depth)** (단위: meters)
- **f**: 카메라의 **초점 거리 (Focal Length)** (단위: pixels)
- **b**: 두 카메라 사이의 거리 **베이스라인 (Baseline)** (단위: meters)
- **d**: 좌/우 영상에서의 픽셀 차이 **시차 (Disparity)** (단위: pixels)

여기서 `f`와 `b`는 카메라 캘리브레이션을 통해 사전에 측정된 고정값입니다. 따라서 스테레오 매칭을 통해 시차 `d`를 계산하면, 간단한 수식으로 깊이 `Z`를 구할 수 있습니다.

## 4. 스테레오 매칭과 시차 맵 (Stereo Matching and Disparity Map)

**스테레오 매칭**은 왼쪽 이미지의 특정 픽셀(또는 영역)에 대응하는 점을 오른쪽 이미지에서 찾는 과정입니다. 이 과정을 이미지의 모든 픽셀에 대해 수행하여 각 픽셀의 시차(`d`)를 계산합니다.

**시차 맵(Disparity Map)**은 이렇게 계산된 시차 값들을 이미지 형태로 시각화한 것입니다.
- 가까운 객체(시차 큼)는 밝은 값으로 표현됩니다.
- 먼 객체(시차 작음)는 어두운 값으로 표현됩니다.

시차 맵은 2D 이미지처럼 보이지만, 각 픽셀은 단순한 밝기 값이 아닌 '깊이'에 대한 정보를 담고 있는 데이터의 집합체입니다.

## 5. 스테레오 매칭 알고리즘 유형

- **지역 정합 (Local Matching)**: 특정 픽셀 주변의 작은 윈도우(영역) 정보만을 이용하여 비용을 계산합니다. 속도가 빠르지만 정확도는 상대적으로 낮습니다. (예: SAD, SSD, NCC)
- **전역 정합 (Global Matching)**: 이미지 전체의 관계를 고려하여 에너지 함수를 최소화하는 방식으로 시차를 결정합니다. 정확도가 높지만 계산량이 매우 많고 속도가 느립니다. (예: Graph Cuts, SGM)

---
---

# 2차시: OpenCV를 이용한 시차 맵 생성

### **학습 목표**
- OpenCV를 사용하여 스테레오 이미지를 불러오고 시차 맵을 생성할 수 있다.
- `StereoBM`과 `StereoSGBM`의 차이를 이해하고, 각 알고리즘의 기본 파라미터를 조정한다.
- 파라미터 변경이 시차 맵의 품질에 미치는 영향을 확인한다.

### **준비 자료**
- `document/python_examples/`
- `stereo_vision_guide.md` (4장. Python 코드 구현 참고)
- 실습용 스테레오 이미지 (e.g., `tsukuba_l.png`, `tsukuba_r.png`)

---

## 1. 실습 환경 설정

Python 환경에서 `OpenCV`, `Numpy`, `Matplotlib` 라이브러리를 설치합니다.

```bash
pip install opencv-python numpy matplotlib
```

이후 `python_examples` 폴더에 `practice_session_2.py` 파일을 생성하여 실습을 진행합니다.

## 2. 기본 스테레오 매칭: `StereoBM`

`StereoBM` (Stereo Block Matching)은 OpenCV에서 제공하는 가장 기본적인 블록 매칭 알고리즘입니다. 속도가 매우 빠르지만 텍스처가 부족한 영역에서는 성능이 저하될 수 있습니다.

**[코드 예제]**
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 이미지 불러오기 (회색조)
# ※주의: 실제 사용 시에는 보정된(rectified) 이미지를 사용해야 합니다.
try:
    imgL = cv2.imread('tsukuba_l.png', cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread('tsukuba_r.png', cv2.IMREAD_GRAYSCALE)
except Exception:
    # 파일이 없을 경우 다운로드
    import urllib.request
    url_l = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/tsukuba_l.png"
    url_r = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/tsukuba_r.png"
    urllib.request.urlretrieve(url_l, 'tsukuba_l.png')
    urllib.request.urlretrieve(url_r, 'tsukuba_r.png')
    imgL = cv2.imread('tsukuba_l.png', cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread('tsukuba_r.png', cv2.IMREAD_GRAYSCALE)

# 2. StereoBM 객체 생성 및 파라미터 설정
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)

# 3. 시차 맵 계산
disparity_bm = stereo.compute(imgL, imgR)

# 4. 결과 시각화
plt.figure(figsize=(12, 6))
plt.title('Disparity Map (StereoBM)')
plt.imshow(disparity_bm, cmap='jet')
plt.colorbar()
plt.show()
```

- **`numDisparities`**: 최대 시차 탐색 범위. 16의 배수여야 합니다.
- **`blockSize`**: 매칭에 사용할 블록 크기. 홀수여야 하며, 값이 클수록 결과가 뭉개지지만 안정적일 수 있습니다.

## 3. 고급 스테레오 매칭: `StereoSGBM`

`StereoSGBM` (Semi-Global Block Matching)은 블록 매칭에 전역적인 제약 조건을 추가하여 `StereoBM`보다 훨씬 안정적이고 정확한 결과를 생성합니다.

**[코드 예제]**
```python
# 5. StereoSGBM 객체 생성 및 파라미터 설정
window_size = 5
stereo_sgbm = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16 * 5,
    blockSize=window_size,
    P1=8 * 3 * window_size**2,
    P2=32 * 3 * window_size**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

# 6. SGBM 시차 맵 계산 및 정규화
disparity_sgbm = stereo_sgbm.compute(imgL, imgR).astype(np.float32) / 16.0

# 7. SGBM 결과 시각화
plt.figure(figsize=(12, 6))
plt.title('Disparity Map (StereoSGBM)')
plt.imshow(disparity_sgbm, cmap='jet')
plt.colorbar()
plt.show()
```
- **`P1`, `P2`**: 시차의 부드러움(smoothness)을 제어하는 파라미터. `P2 > P1`이어야 합니다.
- **`uniquenessRatio`**: 매칭된 최적 후보의 신뢰도를 결정하는 파라미터. 값이 높을수록 신뢰도가 낮은 매칭 결과가 필터링됩니다.

### **결과 분석**
`StereoSGBM`은 `StereoBM`에 비해 노이즈가 적고 객체의 경계가 명확한, 훨씬 고품질의 시차 맵을 생성하는 것을 확인할 수 있습니다.

---
---

# 3차시: 3D 복원 및 실전 트러블슈팅

### **학습 목표**
- 시차 맵과 카메라 파라미터를 이용해 깊이 맵(Depth Map)을 계산한다.
- 깊이 맵을 3D 점군(Point Cloud)으로 변환하고 시각화하여 3차원 구조를 확인한다.
- 스테레오 비전 시스템에서 발생하는 일반적인 문제와 해결 방안을 이해한다.

### **준비 자료**
- 2차시에서 작성한 `practice_session_2.py` 코드
- `stereo_vision_guide.md` (5, 6장 참고)

---

## 1. 깊이 맵(Depth Map) 계산

2차시에서 생성한 시차 맵(`disparity_sgbm`)과 1차시의 깊이 공식을 이용하여, 각 픽셀의 실제 거리 값(미터 단위)을 담고 있는 깊이 맵을 생성합니다.

**[코드 예제]**
```python
# 8. 깊이 맵 계산
# 실제 카메라 파라미터 (가정)
f = 615.0  # 초점거리 (pixels)
b = 0.075  # 베이스라인 (meters)

# 0 또는 음수 시차 값으로 인한 오류 방지
disparity_sgbm[disparity_sgbm <= 0] = 0.1

# 깊이 맵 계산: Z = (f * b) / d
depth_map = (f * b) / disparity_sgbm
```

## 2. 3D 점군(Point Cloud) 생성 및 시각화

깊이 맵과 카메라 내부 파라미터를 이용하여 2D 이미지 좌표를 3D 공간 좌표로 역투영(reprojection)합니다. OpenCV의 `reprojectImageTo3D` 함수를 사용하면 이 과정을 편리하게 수행할 수 있습니다.

**[코드 예제]**
```python
# 9. 3D 점군 생성
h, w = imgL.shape
cx, cy = w / 2, h / 2
Q = np.float32([[1, 0, 0, -cx],
                [0, 1, 0, -cy],
                [0, 0, 0, f],
                [0, 0, -1/b, 0]])

# 3D 좌표 계산
points_3d = cv2.reprojectImageTo3D(disparity_sgbm, Q)
colors = cv2.cvtColor(imgL, cv2.COLOR_GRAY2BGR)

# 유효한 점들만 필터링 (거리가 너무 멀거나 가까운 점 제외)
mask = (disparity_sgbm > 0) & (depth_map < 20) # 20m 이내
points = points_3d[mask]
colors = colors[mask]

# 10. 3D 시각화
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 시각화 속도를 위해 점군 샘플링
sample_size = 5000
if len(points) > sample_size:
    indices = np.random.choice(len(points), sample_size, replace=False)
    points = points[indices]
    colors = colors[indices]

ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors / 255.0, s=1)
ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
ax.set_title('3D Point Cloud')
ax.view_init(elev=-70, azim=-90)
plt.show()
```
이 코드는 2D 이미지로부터 3차원 공간 구조가 복원되었음을 시각적으로 보여줍니다.

## 3. 주요 문제 및 해결 방안 (Troubleshooting)

| 문제 유형 | 원인 | 해결 방안 |
|---|---|---|
| **높은 노이즈 및 부정확한 깊이** | **카메라 캘리브레이션 오류** | 체커보드를 이용한 정밀한 캘리브레이션을 다시 수행하여 카메라 왜곡을 보정하고, 영상 정렬(Rectification)을 확인합니다. |
| **특정 영역의 매칭 실패 (구멍)** | 텍스처(무늬)가 없는 평면, 빛 반사, 가려짐(Occlusion) 영역 | - `StereoSGBM`의 파라미터(`uniquenessRatio`, `speckleWindowSize`)를 조정합니다.<br>- 후처리 필터(WLS Filter 등)를 적용합니다.<br>- 적외선 패턴 프로젝터를 사용하는 액티브 스테레오 방식을 고려합니다. |
| **느린 처리 속도** | 높은 해상도, 과도한 시차 탐색 범위, 복잡한 알고리즘 | - 이미지 해상도를 낮춥니다. (계산량은 해상도의 제곱에 비례)<br>- 필요한 부분만 처리하는 관심 영역(ROI)을 설정합니다.<br>- `numDisparities`를 최소한으로 설정합니다.<br>- OpenCV의 CUDA 모듈을 사용하여 GPU 가속을 적용합니다. |

## 과정 요약

1.  두 카메라로 **이미지 쌍**을 획득하고 **캘리브레이션**을 통해 보정합니다.
2.  **스테레오 매칭** 알고리즘(`StereoSGBM`)으로 **시차 맵**을 계산합니다.
3.  카메라 파라미터와 깊이 공식을 이용해 **깊이 맵**으로 변환합니다.
4.  2D 좌표와 깊이 맵을 **3D 점군**으로 재구성하여 3차원 정보를 최종적으로 획득합니다. 