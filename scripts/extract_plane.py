import open3d as o3d
import numpy as np
import os
print("현재 경로:", os.getcwd())
# 1. PCD 파일 로드
pcd_path = os.path.join(os.getcwd(), "pcd", "output.pcd")
pcd = o3d.io.read_point_cloud(pcd_path)
points = np.asarray(pcd.points)

# 2. 평면 후보군 판단 함수 (FAST-LIO2의 plane_judge 유사 로직)
def is_planar_patch(pts, threshold=0.001):
    if len(pts) < 5:
        return False, None

    # 중심 정렬
    centroid = np.mean(pts, axis=0)
    centered = pts - centroid

    # SVD로 normal 추정
    cov = np.dot(centered.T, centered)
    _, s, vh = np.linalg.svd(cov)
    normal = vh[2, :]

    # 가장 작은 고유값이 threshold보다 작으면 평면이라고 판단
    if s[2] < threshold:
        return True, normal
    else:
        return False, None

# 3. 포인트 슬라이딩 윈도우로 평면 판단
pl_surf_pts = []
non_plane_pts = []
group_size = 30
for i in range(0, len(points) - group_size, group_size):
    patch = points[i:i+group_size]
    planar, _ = is_planar_patch(patch)
    if planar:
        pl_surf_pts.extend(patch)
    else:
        non_plane_pts.extend(patch)

# 4. 시각화: 평면 빨강, 일반 회색
surf_pcd = o3d.geometry.PointCloud()
surf_pcd.points = o3d.utility.Vector3dVector(np.array(pl_surf_pts))
surf_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # 빨강

if len(non_plane_pts) > 0:
    non_pcd = o3d.geometry.PointCloud()
    non_pcd.points = o3d.utility.Vector3dVector(np.array(non_plane_pts))
    non_pcd.paint_uniform_color([0.0, 1.0, 0.0])
else:
    print("비평면 포인트가 없습니다 (non_plane_pts is empty).")
    non_pcd = None

print(f"전체 포인트 수: {len(points)}")
print(f"평면으로 분류된 포인트 수: {len(pl_surf_pts)}")
print(f"비평면으로 분류된 포인트 수: {len(non_plane_pts)}")


o3d.visualization.draw_geometries([surf_pcd, non_pcd])
