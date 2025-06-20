#!/usr/bin/env python3
"""
깊이 계산 및 3D 재구성 예제
Depth Calculation and 3D Reconstruction Example

이 예제는 시차 맵으로부터 실제 깊이 정보를 계산하고,
3D 점군을 생성하는 방법을 보여줍니다.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class DepthCalculator:
    """
    깊이 계산 및 3D 재구성을 위한 클래스
    """
    
    def __init__(self, focal_length, baseline, image_width, image_height):
        """
        초기화
        
        Args:
            focal_length (float): 초점거리 (픽셀 단위)
            baseline (float): 베이스라인 (미터 단위)
            image_width (int): 영상 너비
            image_height (int): 영상 높이
        """
        self.focal_length = focal_length
        self.baseline = baseline
        self.image_width = image_width
        self.image_height = image_height
        
        # 카메라 내부 매개변수 행렬
        self.K = np.array([
            [focal_length, 0, image_width/2],
            [0, focal_length, image_height/2],
            [0, 0, 1]
        ])
        
        print(f"카메라 매개변수:")
        print(f"- 초점거리: {focal_length} pixels")
        print(f"- 베이스라인: {baseline} m")
        print(f"- 영상 크기: {image_width} x {image_height}")
    
    def disparity_to_depth(self, disparity_map, min_disparity=1.0):
        """
        시차 맵을 깊이 맵으로 변환
        공식: Z = (f * b) / d
        
        Args:
            disparity_map (numpy.ndarray): 시차 맵
            min_disparity (float): 최소 시차 (0으로 나누기 방지)
        
        Returns:
            numpy.ndarray: 깊이 맵 (미터 단위)
        """
        # 0이나 음수 시차 처리
        valid_disparity = np.where(disparity_map > 0, disparity_map, min_disparity)
        
        # 깊이 계산: Z = (f * b) / d
        depth_map = (self.focal_length * self.baseline) / valid_disparity
        
        # 너무 먼 거리는 제한 (예: 100m)
        depth_map = np.clip(depth_map, 0, 100)
        
        return depth_map
    
    def create_point_cloud(self, left_image, disparity_map, max_depth=50.0):
        """
        깊이 맵으로부터 3D 점군 생성
        
        Args:
            left_image (numpy.ndarray): 좌측 영상 (텍스처용)
            disparity_map (numpy.ndarray): 시차 맵
            max_depth (float): 최대 깊이 제한
        
        Returns:
            tuple: (3D 좌표, 색상 정보)
        """
        # 깊이 맵 계산
        depth_map = self.disparity_to_depth(disparity_map)
        
        # 유효한 깊이 영역만 선택
        valid_mask = (disparity_map > 0) & (depth_map < max_depth)
        
        # 픽셀 좌표 생성
        y_coords, x_coords = np.mgrid[0:self.image_height, 0:self.image_width]
        
        # 유효한 점들만 선택
        valid_x = x_coords[valid_mask]
        valid_y = y_coords[valid_mask]
        valid_depth = depth_map[valid_mask]
        
        # 3D 좌표 계산
        # X = (x - cx) * Z / fx
        # Y = (y - cy) * Z / fy
        # Z = depth
        cx, cy = self.image_width/2, self.image_height/2
        
        points_3d = np.zeros((len(valid_x), 3))
        points_3d[:, 0] = (valid_x - cx) * valid_depth / self.focal_length  # X
        points_3d[:, 1] = (valid_y - cy) * valid_depth / self.focal_length  # Y
        points_3d[:, 2] = valid_depth  # Z
        
        # 색상 정보 추출
        if len(left_image.shape) == 3:
            colors = left_image[valid_mask]
        else:
            # 그레이스케일인 경우 RGB로 변환
            gray_values = left_image[valid_mask]
            colors = np.stack([gray_values, gray_values, gray_values], axis=1)
        
        return points_3d, colors
    
    def analyze_depth_statistics(self, depth_map, disparity_map):
        """
        깊이 맵 통계 분석
        
        Args:
            depth_map (numpy.ndarray): 깊이 맵
            disparity_map (numpy.ndarray): 시차 맵
        
        Returns:
            dict: 통계 정보
        """
        valid_mask = disparity_map > 0
        valid_depths = depth_map[valid_mask]
        valid_disparities = disparity_map[valid_mask]
        
        stats = {
            'depth_mean': np.mean(valid_depths),
            'depth_std': np.std(valid_depths),
            'depth_min': np.min(valid_depths),
            'depth_max': np.max(valid_depths),
            'disparity_mean': np.mean(valid_disparities),
            'disparity_std': np.std(valid_disparities),
            'valid_pixels': np.sum(valid_mask),
            'total_pixels': depth_map.size,
            'valid_ratio': np.sum(valid_mask) / depth_map.size
        }
        
        return stats

def visualize_depth_analysis(left_img, disparity_map, depth_map, stats):
    """
    깊이 분석 결과 시각화
    
    Args:
        left_img (numpy.ndarray): 좌측 영상
        disparity_map (numpy.ndarray): 시차 맵
        depth_map (numpy.ndarray): 깊이 맵
        stats (dict): 통계 정보
    """
    plt.figure(figsize=(15, 10))
    
    # 좌측 영상
    plt.subplot(2, 3, 1)
    plt.imshow(left_img, cmap='gray')
    plt.title('Left Image')
    plt.axis('off')
    
    # 시차 맵
    plt.subplot(2, 3, 2)
    plt.imshow(disparity_map, cmap='plasma')
    plt.title('Disparity Map')
    plt.colorbar(label='Disparity (pixels)')
    plt.axis('off')
    
    # 깊이 맵
    plt.subplot(2, 3, 3)
    plt.imshow(depth_map, cmap='viridis', vmax=20)  # 20m까지만 표시
    plt.title('Depth Map')
    plt.colorbar(label='Depth (meters)')
    plt.axis('off')
    
    # 깊이 히스토그램
    plt.subplot(2, 3, 4)
    valid_depths = depth_map[disparity_map > 0]
    plt.hist(valid_depths[valid_depths < 50], bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Depth (meters)')
    plt.ylabel('Frequency')
    plt.title('Depth Distribution')
    plt.grid(True, alpha=0.3)
    
    # 시차-깊이 관계
    plt.subplot(2, 3, 5)
    valid_disparities = disparity_map[disparity_map > 0]
    valid_depths_plot = depth_map[disparity_map > 0]
    
    # 샘플링 (너무 많은 점은 시각화가 어려움)
    sample_size = min(5000, len(valid_disparities))
    indices = np.random.choice(len(valid_disparities), sample_size, replace=False)
    
    plt.scatter(valid_disparities[indices], valid_depths_plot[indices], 
               alpha=0.5, s=1)
    plt.xlabel('Disparity (pixels)')
    plt.ylabel('Depth (meters)')
    plt.title('Disparity vs Depth Relationship')
    plt.grid(True, alpha=0.3)
    
    # 통계 정보
    plt.subplot(2, 3, 6)
    plt.axis('off')
    stats_text = f"""
    Depth Statistics:
    Mean: {stats['depth_mean']:.2f} m
    Std: {stats['depth_std']:.2f} m
    Min: {stats['depth_min']:.2f} m
    Max: {stats['depth_max']:.2f} m
    
    Disparity Statistics:
    Mean: {stats['disparity_mean']:.2f} px
    Std: {stats['disparity_std']:.2f} px
    
    Coverage:
    Valid pixels: {stats['valid_pixels']:,}
    Total pixels: {stats['total_pixels']:,}
    Valid ratio: {stats['valid_ratio']:.1%}
    """
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()

def visualize_3d_point_cloud(points_3d, colors, max_points=10000):
    """
    3D 점군 시각화
    
    Args:
        points_3d (numpy.ndarray): 3D 좌표 배열
        colors (numpy.ndarray): 색상 배열
        max_points (int): 표시할 최대 점 개수
    """
    # 점이 너무 많으면 샘플링
    if len(points_3d) > max_points:
        indices = np.random.choice(len(points_3d), max_points, replace=False)
        points_3d = points_3d[indices]
        colors = colors[indices]
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 3D 산점도
    if len(colors.shape) == 2 and colors.shape[1] == 3:
        # RGB 색상
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                  c=colors/255.0, s=1, alpha=0.6)
    else:
        # 그레이스케일
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                  c=colors, cmap='gray', s=1, alpha=0.6)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'3D Point Cloud ({len(points_3d):,} points)')
    
    # 시점 조정
    ax.view_init(elev=20, azim=45)
    
    plt.show()

def create_sample_stereo_data():
    """
    테스트용 스테레오 데이터 생성
    
    Returns:
        tuple: (좌측 영상, 시차 맵)
    """
    height, width = 240, 320
    
    # 좌측 영상 생성
    left_img = np.zeros((height, width), dtype=np.uint8)
    
    # 다양한 거리의 객체들 생성
    # 가까운 객체 (큰 시차)
    cv2.rectangle(left_img, (50, 50), (120, 120), 255, -1)
    cv2.circle(left_img, (200, 80), 30, 200, -1)
    
    # 중간 거리 객체
    cv2.rectangle(left_img, (150, 150), (200, 200), 150, -1)
    
    # 먼 거리 객체 (작은 시차)
    cv2.rectangle(left_img, (20, 180), (100, 220), 100, -1)
    
    # 시차 맵 생성 (실제로는 스테레오 매칭으로 계산)
    disparity_map = np.zeros((height, width), dtype=np.float32)
    
    # 가까운 객체들 (높은 시차)
    disparity_map[50:120, 50:120] = 20  # 20 픽셀 시차
    y, x = np.ogrid[:height, :width]
    circle_mask = (x - 200)**2 + (y - 80)**2 <= 30**2
    disparity_map[circle_mask] = 25  # 25 픽셀 시차
    
    # 중간 거리 객체
    disparity_map[150:200, 150:200] = 10  # 10 픽셀 시차
    
    # 먼 거리 객체
    disparity_map[180:220, 20:100] = 5  # 5 픽셀 시차
    
    # 노이즈 추가
    noise = np.random.randint(0, 20, (height, width))
    left_img = np.clip(left_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return left_img, disparity_map

def main():
    """
    메인 함수
    """
    print("깊이 계산 및 3D 재구성 예제 시작")
    
    # 카메라 매개변수 설정
    focal_length = 500  # pixels
    baseline = 0.1      # meters (10cm)
    image_width = 320
    image_height = 240
    
    # 샘플 데이터 생성
    left_img, disparity_map = create_sample_stereo_data()
    
    # 깊이 계산기 생성
    depth_calc = DepthCalculator(focal_length, baseline, image_width, image_height)
    
    # 깊이 맵 계산
    depth_map = depth_calc.disparity_to_depth(disparity_map)
    
    # 통계 분석
    stats = depth_calc.analyze_depth_statistics(depth_map, disparity_map)
    
    print("\n=== 깊이 분석 결과 ===")
    print(f"평균 깊이: {stats['depth_mean']:.2f} m")
    print(f"깊이 범위: {stats['depth_min']:.2f} - {stats['depth_max']:.2f} m")
    print(f"유효 픽셀 비율: {stats['valid_ratio']:.1%}")
    
    # 시각화
    visualize_depth_analysis(left_img, disparity_map, depth_map, stats)
    
    # 3D 점군 생성 및 시각화
    print("\n3D 점군 생성 중...")
    points_3d, colors = depth_calc.create_point_cloud(left_img, disparity_map, max_depth=20.0)
    
    print(f"생성된 3D 점 개수: {len(points_3d):,}")
    
    # 3D 시각화
    visualize_3d_point_cloud(points_3d, colors)
    
    # 거리별 분석
    print("\n=== 거리별 객체 분석 ===")
    for distance_range in [(0, 2), (2, 5), (5, 10), (10, float('inf'))]:
        min_d, max_d = distance_range
        if max_d == float('inf'):
            mask = depth_map >= min_d
            range_str = f"{min_d}m 이상"
        else:
            mask = (depth_map >= min_d) & (depth_map < max_d)
            range_str = f"{min_d}-{max_d}m"
        
        count = np.sum(mask & (disparity_map > 0))
        if count > 0:
            avg_depth = np.mean(depth_map[mask & (disparity_map > 0)])
            print(f"{range_str}: {count:,}개 픽셀, 평균 깊이: {avg_depth:.2f}m")

if __name__ == "__main__":
    main()