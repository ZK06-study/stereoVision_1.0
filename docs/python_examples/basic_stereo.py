#!/usr/bin/env python3
"""
기본 스테레오비전 예제
Basic Stereo Vision Example

이 예제는 OpenCV를 사용하여 기본적인 스테레오비전을 구현합니다.
StereoBM과 StereoSGBM 알고리즘을 비교해볼 수 있습니다.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_stereo_images(left_path, right_path):
    """
    스테레오 영상 쌍을 로드하는 함수
    
    Args:
        left_path (str): 좌측 영상 경로
        right_path (str): 우측 영상 경로
    
    Returns:
        tuple: (좌측 영상, 우측 영상)
    """
    # 그레이스케일로 영상 로드
    imgL = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    
    if imgL is None or imgR is None:
        raise ValueError("영상을 불러올 수 없습니다. 경로를 확인해주세요.")
    
    print(f"좌측 영상 크기: {imgL.shape}")
    print(f"우측 영상 크기: {imgR.shape}")
    
    return imgL, imgR

def create_stereo_bm(num_disparities=64, block_size=15):
    """
    StereoBM 객체를 생성하는 함수
    
    Args:
        num_disparities (int): 시차 범위 (16의 배수)
        block_size (int): 블록 크기 (홀수)
    
    Returns:
        cv2.StereoBM: StereoBM 객체
    """
    stereo = cv2.StereoBM_create(numDisparities=num_disparities, 
                                blockSize=block_size)
    
    # 추가 매개변수 설정
    stereo.setPreFilterCap(31)
    stereo.setMinDisparity(0)
    stereo.setTextureThreshold(10)
    stereo.setUniquenessRatio(10)
    stereo.setSpeckleWindowSize(100)
    stereo.setSpeckleRange(32)
    stereo.setDisp12MaxDiff(1)
    
    return stereo

def create_stereo_sgbm(num_disparities=64, block_size=5):
    """
    StereoSGBM 객체를 생성하는 함수
    
    Args:
        num_disparities (int): 시차 범위 (16의 배수)
        block_size (int): 블록 크기
    
    Returns:
        cv2.StereoSGBM: StereoSGBM 객체
    """
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )
    
    return stereo

def compute_disparity(imgL, imgR, stereo_matcher):
    """
    시차 맵을 계산하는 함수
    
    Args:
        imgL (numpy.ndarray): 좌측 영상
        imgR (numpy.ndarray): 우측 영상
        stereo_matcher: 스테레오 매칭 객체
    
    Returns:
        numpy.ndarray: 시차 맵
    """
    # 시차 계산
    disparity = stereo_matcher.compute(imgL, imgR)
    
    # StereoSGBM의 경우 16으로 나누어 실제 시차값 획득
    if isinstance(stereo_matcher, cv2.StereoSGBM):
        disparity = disparity.astype(np.float32) / 16.0
    
    return disparity

def visualize_results(imgL, imgR, disparity_bm, disparity_sgbm):
    """
    결과를 시각화하는 함수
    
    Args:
        imgL (numpy.ndarray): 좌측 영상
        imgR (numpy.ndarray): 우측 영상
        disparity_bm (numpy.ndarray): StereoBM 시차 맵
        disparity_sgbm (numpy.ndarray): StereoSGBM 시차 맵
    """
    plt.figure(figsize=(15, 10))
    
    # 원본 영상
    plt.subplot(2, 3, 1)
    plt.imshow(imgL, cmap='gray')
    plt.title('Left Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(imgR, cmap='gray')
    plt.title('Right Image')
    plt.axis('off')
    
    # StereoBM 결과
    plt.subplot(2, 3, 4)
    plt.imshow(disparity_bm, cmap='plasma')
    plt.title('StereoBM Disparity')
    plt.colorbar()
    plt.axis('off')
    
    # StereoSGBM 결과
    plt.subplot(2, 3, 5)
    plt.imshow(disparity_sgbm, cmap='plasma')
    plt.title('StereoSGBM Disparity')
    plt.colorbar()
    plt.axis('off')
    
    # 차이 비교
    plt.subplot(2, 3, 6)
    diff = np.abs(disparity_bm.astype(np.float32) - disparity_sgbm)
    plt.imshow(diff, cmap='hot')
    plt.title('Difference (BM - SGBM)')
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    """
    메인 함수
    """
    # 테스트 영상 경로 (실제 경로로 변경 필요)
    left_path = 'left_image.png'
    right_path = 'right_image.png'
    
    try:
        # 영상 로드
        imgL, imgR = load_stereo_images(left_path, right_path)
        
        # 스테레오 매처 생성
        stereo_bm = create_stereo_bm(num_disparities=64, block_size=15)
        stereo_sgbm = create_stereo_sgbm(num_disparities=64, block_size=5)
        
        print("시차 맵 계산 중...")
        
        # 시차 계산
        disparity_bm = compute_disparity(imgL, imgR, stereo_bm)
        disparity_sgbm = compute_disparity(imgL, imgR, stereo_sgbm)
        
        print("시차 맵 계산 완료!")
        
        # 결과 시각화
        visualize_results(imgL, imgR, disparity_bm, disparity_sgbm)
        
        # 통계 정보 출력
        print("\n=== 시차 맵 통계 ===")
        print(f"StereoBM - 최소: {disparity_bm.min():.2f}, 최대: {disparity_bm.max():.2f}")
        print(f"StereoSGBM - 최소: {disparity_sgbm.min():.2f}, 최대: {disparity_sgbm.max():.2f}")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        print("샘플 데이터를 생성하여 테스트해보세요.")
        
        # 샘플 데이터 생성 및 테스트
        create_sample_data()

def create_sample_data():
    """
    테스트용 샘플 데이터 생성
    """
    print("\n샘플 데이터 생성 중...")
    
    # 간단한 패턴 영상 생성
    height, width = 480, 640
    imgL = np.zeros((height, width), dtype=np.uint8)
    imgR = np.zeros((height, width), dtype=np.uint8)
    
    # 체스보드 패턴 생성
    for i in range(0, height, 40):
        for j in range(0, width, 40):
            if (i//40 + j//40) % 2 == 0:
                imgL[i:i+40, j:j+40] = 255
                # 우측 영상은 약간 이동
                if j >= 20:
                    imgR[i:i+40, j-20:j+20] = 255
    
    # 노이즈 추가
    noise = np.random.randint(0, 50, (height, width))
    imgL = np.clip(imgL.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    imgR = np.clip(imgR.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # 스테레오 매처 생성
    stereo_bm = create_stereo_bm(num_disparities=64, block_size=15)
    stereo_sgbm = create_stereo_sgbm(num_disparities=64, block_size=5)
    
    # 시차 계산
    disparity_bm = compute_disparity(imgL, imgR, stereo_bm)
    disparity_sgbm = compute_disparity(imgL, imgR, stereo_sgbm)
    
    # 결과 시각화
    visualize_results(imgL, imgR, disparity_bm, disparity_sgbm)

if __name__ == "__main__":
    main()