#!/usr/bin/env python3
"""
스테레오 매칭 알고리즘 구현 및 비교
Stereo Matching Algorithms Implementation and Comparison

이 예제는 다양한 스테레오 매칭 알고리즘을 직접 구현하고 비교합니다.
- SAD (Sum of Absolute Differences)
- SSD (Sum of Squared Differences)
- Census Transform
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

class StereoMatcher:
    """
    다양한 스테레오 매칭 알고리즘을 구현한 클래스
    """
    
    def __init__(self, max_disparity=64, window_size=5):
        """
        초기화
        
        Args:
            max_disparity (int): 최대 시차 범위
            window_size (int): 윈도우 크기 (홀수)
        """
        if window_size % 2 == 0:
            raise ValueError("윈도우 크기는 홀수여야 합니다.")
        
        self.max_disparity = max_disparity
        self.window_size = window_size
        self.half_window = window_size // 2
    
    def compute_sad(self, left_img, right_img):
        """
        SAD (Sum of Absolute Differences) 알고리즘으로 시차 맵 계산
        
        Args:
            left_img (numpy.ndarray): 좌측 영상
            right_img (numpy.ndarray): 우측 영상
        
        Returns:
            numpy.ndarray: 시차 맵
        """
        h, w = left_img.shape
        disparity = np.zeros((h, w), dtype=np.float32)
        
        print("SAD 계산 중...")
        for y in range(self.half_window, h - self.half_window):
            if y % 50 == 0:
                print(f"진행률: {y/(h-2*self.half_window)*100:.1f}%")
            
            for x in range(self.half_window, w - self.half_window):
                # 좌측 영상의 윈도우
                left_window = left_img[y-self.half_window:y+self.half_window+1,
                                     x-self.half_window:x+self.half_window+1]
                
                min_sad = float('inf')
                best_disparity = 0
                
                # 시차 범위에서 최적의 매칭점 찾기
                for d in range(self.max_disparity):
                    if x - d - self.half_window < 0:
                        break
                    
                    # 우측 영상의 윈도우
                    right_window = right_img[y-self.half_window:y+self.half_window+1,
                                           x-d-self.half_window:x-d+self.half_window+1]
                    
                    # SAD 계산
                    sad = np.sum(np.abs(left_window.astype(np.float32) - 
                                       right_window.astype(np.float32)))
                    
                    if sad < min_sad:
                        min_sad = sad
                        best_disparity = d
                
                disparity[y, x] = best_disparity
        
        return disparity
    
    def compute_ssd(self, left_img, right_img):
        """
        SSD (Sum of Squared Differences) 알고리즘으로 시차 맵 계산
        
        Args:
            left_img (numpy.ndarray): 좌측 영상
            right_img (numpy.ndarray): 우측 영상
        
        Returns:
            numpy.ndarray: 시차 맵
        """
        h, w = left_img.shape
        disparity = np.zeros((h, w), dtype=np.float32)
        
        print("SSD 계산 중...")
        for y in range(self.half_window, h - self.half_window):
            if y % 50 == 0:
                print(f"진행률: {y/(h-2*self.half_window)*100:.1f}%")
            
            for x in range(self.half_window, w - self.half_window):
                left_window = left_img[y-self.half_window:y+self.half_window+1,
                                     x-self.half_window:x+self.half_window+1]
                
                min_ssd = float('inf')
                best_disparity = 0
                
                for d in range(self.max_disparity):
                    if x - d - self.half_window < 0:
                        break
                    
                    right_window = right_img[y-self.half_window:y+self.half_window+1,
                                           x-d-self.half_window:x-d+self.half_window+1]
                    
                    # SSD 계산
                    diff = left_window.astype(np.float32) - right_window.astype(np.float32)
                    ssd = np.sum(diff * diff)
                    
                    if ssd < min_ssd:
                        min_ssd = ssd
                        best_disparity = d
                
                disparity[y, x] = best_disparity
        
        return disparity
    
    def compute_census(self, left_img, right_img):
        """
        Census Transform 알고리즘으로 시차 맵 계산
        
        Args:
            left_img (numpy.ndarray): 좌측 영상
            right_img (numpy.ndarray): 우측 영상
        
        Returns:
            numpy.ndarray: 시차 맵
        """
        # Census Transform 적용
        left_census = self._census_transform(left_img)
        right_census = self._census_transform(right_img)
        
        h, w = left_img.shape
        disparity = np.zeros((h, w), dtype=np.float32)
        
        print("Census Transform 시차 계산 중...")
        for y in range(self.half_window, h - self.half_window):
            if y % 50 == 0:
                print(f"진행률: {y/(h-2*self.half_window)*100:.1f}%")
            
            for x in range(self.half_window, w - self.half_window):
                left_census_window = left_census[y-self.half_window:y+self.half_window+1,
                                               x-self.half_window:x+self.half_window+1]
                
                min_cost = float('inf')
                best_disparity = 0
                
                for d in range(self.max_disparity):
                    if x - d - self.half_window < 0:
                        break
                    
                    right_census_window = right_census[y-self.half_window:y+self.half_window+1,
                                                     x-d-self.half_window:x-d+self.half_window+1]
                    
                    # Hamming Distance 계산 (XOR 후 1의 개수)
                    xor_result = left_census_window ^ right_census_window
                    cost = np.sum([bin(x).count('1') for x in xor_result.flatten()])
                    
                    if cost < min_cost:
                        min_cost = cost
                        best_disparity = d
                
                disparity[y, x] = best_disparity
        
        return disparity
    
    def _census_transform(self, img):
        """
        Census Transform 적용
        
        Args:
            img (numpy.ndarray): 입력 영상
        
        Returns:
            numpy.ndarray: Census transform된 영상
        """
        h, w = img.shape
        census = np.zeros((h, w), dtype=np.uint32)
        
        print("Census Transform 적용 중...")
        for y in range(self.half_window, h - self.half_window):
            for x in range(self.half_window, w - self.half_window):
                center_pixel = img[y, x]
                census_value = 0
                bit_position = 0
                
                # 윈도우 내의 모든 픽셀과 중심 픽셀 비교
                for dy in range(-self.half_window, self.half_window + 1):
                    for dx in range(-self.half_window, self.half_window + 1):
                        if dy == 0 and dx == 0:
                            continue  # 중심 픽셀은 제외
                        
                        neighbor_pixel = img[y + dy, x + dx]
                        if neighbor_pixel < center_pixel:
                            census_value |= (1 << bit_position)
                        bit_position += 1
                
                census[y, x] = census_value
        
        return census

def compare_algorithms(left_img, right_img, max_disparity=32, window_size=5):
    """
    다양한 알고리즘을 비교하는 함수
    
    Args:
        left_img (numpy.ndarray): 좌측 영상
        right_img (numpy.ndarray): 우측 영상
        max_disparity (int): 최대 시차
        window_size (int): 윈도우 크기
    
    Returns:
        dict: 각 알고리즘의 결과와 처리 시간
    """
    matcher = StereoMatcher(max_disparity=max_disparity, window_size=window_size)
    results = {}
    
    # SAD 알고리즘
    start_time = time.time()
    sad_disparity = matcher.compute_sad(left_img, right_img)
    sad_time = time.time() - start_time
    results['SAD'] = {'disparity': sad_disparity, 'time': sad_time}
    
    # SSD 알고리즘
    start_time = time.time()
    ssd_disparity = matcher.compute_ssd(left_img, right_img)
    ssd_time = time.time() - start_time
    results['SSD'] = {'disparity': ssd_disparity, 'time': ssd_time}
    
    # Census Transform (작은 윈도우로 테스트)
    small_matcher = StereoMatcher(max_disparity=max_disparity, window_size=3)
    start_time = time.time()
    census_disparity = small_matcher.compute_census(left_img, right_img)
    census_time = time.time() - start_time
    results['Census'] = {'disparity': census_disparity, 'time': census_time}
    
    return results

def visualize_comparison(left_img, right_img, results):
    """
    알고리즘 비교 결과를 시각화
    
    Args:
        left_img (numpy.ndarray): 좌측 영상
        right_img (numpy.ndarray): 우측 영상
        results (dict): 각 알고리즘의 결과
    """
    plt.figure(figsize=(15, 12))
    
    # 원본 영상
    plt.subplot(2, 3, 1)
    plt.imshow(left_img, cmap='gray')
    plt.title('Left Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(right_img, cmap='gray')
    plt.title('Right Image')
    plt.axis('off')
    
    # 각 알고리즘 결과
    plot_idx = 3
    for algorithm, data in results.items():
        plt.subplot(2, 3, plot_idx)
        plt.imshow(data['disparity'], cmap='plasma')
        plt.title(f'{algorithm}\n(Time: {data["time"]:.2f}s)')
        plt.colorbar()
        plt.axis('off')
        plot_idx += 1
    
    plt.tight_layout()
    plt.show()
    
    # 처리 시간 비교 그래프
    plt.figure(figsize=(10, 6))
    algorithms = list(results.keys())
    times = [results[alg]['time'] for alg in algorithms]
    
    bars = plt.bar(algorithms, times, color=['blue', 'green', 'red'])
    plt.ylabel('Processing Time (seconds)')
    plt.title('Algorithm Processing Time Comparison')
    
    # 막대 위에 시간 값 표시
    for bar, time_val in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{time_val:.2f}s', ha='center', va='bottom')
    
    plt.show()

def create_test_images():
    """
    테스트용 간단한 영상 생성
    
    Returns:
        tuple: (좌측 영상, 우측 영상)
    """
    height, width = 200, 300
    left_img = np.zeros((height, width), dtype=np.uint8)
    right_img = np.zeros((height, width), dtype=np.uint8)
    
    # 간단한 사각형 패턴 생성
    # 좌측 영상
    cv2.rectangle(left_img, (50, 50), (100, 100), 255, -1)
    cv2.rectangle(left_img, (150, 80), (200, 130), 128, -1)
    cv2.circle(left_img, (80, 150), 30, 200, -1)
    
    # 우측 영상 (약간 이동)
    cv2.rectangle(right_img, (40, 50), (90, 100), 255, -1)  # 10픽셀 이동
    cv2.rectangle(right_img, (135, 80), (185, 130), 128, -1)  # 15픽셀 이동
    cv2.circle(right_img, (75, 150), 30, 200, -1)  # 5픽셀 이동
    
    # 노이즈 추가
    noise = np.random.randint(0, 30, (height, width))
    left_img = np.clip(left_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    right_img = np.clip(right_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return left_img, right_img

def main():
    """
    메인 함수
    """
    print("스테레오 매칭 알고리즘 비교 시작")
    
    # 테스트 영상 생성
    left_img, right_img = create_test_images()
    
    print("영상 크기:", left_img.shape)
    print("알고리즘 비교 시작...")
    
    # 작은 매개변수로 빠른 테스트
    results = compare_algorithms(left_img, right_img, 
                               max_disparity=20, window_size=5)
    
    # 결과 출력
    print("\n=== 처리 시간 비교 ===")
    for algorithm, data in results.items():
        print(f"{algorithm}: {data['time']:.3f}초")
    
    # 시각화
    visualize_comparison(left_img, right_img, results)
    
    # 시차 통계
    print("\n=== 시차 맵 통계 ===")
    for algorithm, data in results.items():
        disparity = data['disparity']
        print(f"{algorithm} - 평균: {np.mean(disparity):.2f}, "
              f"최소: {np.min(disparity):.2f}, "
              f"최대: {np.max(disparity):.2f}")

if __name__ == "__main__":
    main()