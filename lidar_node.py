#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
import numpy as np


class DBSCAN:
    def __init__(self, eps=0.5, min_samples=8):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        self.labels_ = np.full(X.shape[0], -1)
        self.visited_ = np.zeros(X.shape[0], dtype=bool)
        self.cluster_id = 0
        for i in range(X.shape[0]):
            if not self.visited_[i]:
                self.visited_[i] = True
                neighbors = self._find_neighbors(X, i)
                if len(neighbors) >= self.min_samples:
                    self._expand_cluster(X, i, neighbors, self.cluster_id)
                    self.cluster_id += 1
        return self

    def _find_neighbors(self, X, idx):
        dists = np.linalg.norm(X - X[idx], axis=1)
        return np.where(dists <= self.eps)[0]

    def _expand_cluster(self, X, idx, neighbors, cluster_id):
        self.labels_[idx] = cluster_id
        i = 0
        while i < len(neighbors):
            nb = neighbors[i]
            if not self.visited_[nb]:
                self.visited_[nb] = True
                new_nb = self._find_neighbors(X, nb)
                if len(new_nb) >= self.min_samples:
                    neighbors = np.union1d(neighbors, new_nb)
            if self.labels_[nb] == -1:
                self.labels_[nb] = cluster_id
            i += 1


def _cluster_length(pts):
    if len(pts) < 2:
        return 0.0
    from scipy.spatial.distance import pdist
    d = pdist(pts)
    return float(np.max(d)) if len(d) > 0 else 0.0


class LidarNode(Node):
    SAG_MIN  =   0
    SAG_MAX  =  60
    ORTA_MIN =  60
    ORTA_MAX = 120
    SOL_MIN  = 120
    SOL_MAX  = 180

    def __init__(self):
        super().__init__('lidar_node')

        self.sag_engel    = 0.0
        self.orta_engel   = 0.0
        self.sol_engel    = 0.0
        self.sag_mesafe   = 0.0
        self.orta_mesafe  = 0.0
        self.sol_mesafe   = 0.0
        self.sag_uzunluk  = 0.0
        self.orta_uzunluk = 0.0
        self.sol_uzunluk  = 0.0

        self.dbscan = DBSCAN(eps=0.5, min_samples=8)
        qos = rclpy.qos.qos_profile_sensor_data

        self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile=qos)

        self.pub_sag_engel   = self.create_publisher(Float32, '/sag_engel',    10)
        self.pub_orta_engel  = self.create_publisher(Float32, '/orta_engel',   10)
        self.pub_sol_engel   = self.create_publisher(Float32, '/sol_engel',    10)
        self.pub_sag_mesafe  = self.create_publisher(Float32, '/sag_mesafe',   10)
        self.pub_orta_mesafe = self.create_publisher(Float32, '/orta_mesafe',  10)
        self.pub_sol_mesafe  = self.create_publisher(Float32, '/sol_mesafe',   10)
        self.pub_gui_raw     = self.create_publisher(LaserScan, '/lidar_gui_raw', qos_profile=qos)

        self.get_logger().info(
            f'LiDAR Node başlatıldı — '
            f'Sağ:{self.SAG_MIN}°-{self.SAG_MAX}° | '
            f'Orta:{self.ORTA_MIN}°-{self.ORTA_MAX}° | '
            f'Sol:{self.SOL_MIN}°-{self.SOL_MAX}°'
        )

    def scan_callback(self, msg: LaserScan):
        self.pub_gui_raw.publish(msg)

        ranges = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))
        valid  = np.isfinite(ranges) & (ranges > msg.range_min) & (ranges < msg.range_max)
        valid_ranges = ranges[valid]
        valid_angles = angles[valid]

        if len(valid_ranges) == 0:
            self._reset_all()
            self._publish()
            return

        x = valid_ranges * np.cos(valid_angles)
        y = valid_ranges * np.sin(valid_angles)
        points = np.column_stack((x, y))

        front_mask   = y >= 0
        front_points = points[front_mask]
        front_angles = valid_angles[front_mask]

        if len(front_points) == 0:
            self._reset_all()
            self._publish()
            return

        try:
            self.dbscan.fit(front_points)
            unique_labels = set(self.dbscan.labels_)
            unique_labels.discard(-1)
        except Exception as e:
            self.get_logger().warn(f'DBSCAN hatası: {e}')
            self._reset_all()
            self._publish()
            return

        sag_found  = False
        orta_found = False
        sol_found  = False
        sag_min    = float('inf')
        orta_min   = float('inf')
        sol_min    = float('inf')
        sag_uzun   = 0.0
        orta_uzun  = 0.0
        sol_uzun   = 0.0

        for label in unique_labels:
            mask     = self.dbscan.labels_ == label
            cpts     = front_points[mask]
            cang     = front_angles[mask]
            dists    = np.linalg.norm(cpts, axis=1)
            min_dist = float(np.min(dists))
            uzunluk  = _cluster_length(cpts)
            avg_deg  = np.degrees(float(np.mean(cang))) % 360

            if self.SAG_MIN <= avg_deg < self.SAG_MAX:
                sag_found = True
                if min_dist < sag_min:
                    sag_min  = min_dist
                    sag_uzun = uzunluk
            elif self.ORTA_MIN <= avg_deg < self.ORTA_MAX:
                orta_found = True
                if min_dist < orta_min:
                    orta_min  = min_dist
                    orta_uzun = uzunluk
            elif self.SOL_MIN <= avg_deg <= self.SOL_MAX:
                sol_found = True
                if min_dist < sol_min:
                    sol_min  = min_dist
                    sol_uzun = uzunluk

        self.sag_engel    = 1.0 if sag_found  else 0.0
        self.orta_engel   = 1.0 if orta_found else 0.0
        self.sol_engel    = 1.0 if sol_found  else 0.0
        self.sag_mesafe   = sag_min  if sag_min  != float('inf') else 0.0
        self.orta_mesafe  = orta_min if orta_min != float('inf') else 0.0
        self.sol_mesafe   = sol_min  if sol_min  != float('inf') else 0.0
        self.sag_uzunluk  = sag_uzun
        self.orta_uzunluk = orta_uzun
        self.sol_uzunluk  = sol_uzun

        self._publish()
        self._log()

    def _reset_all(self):
        self.sag_engel = self.orta_engel = self.sol_engel = 0.0
        self.sag_mesafe = self.orta_mesafe = self.sol_mesafe = 0.0
        self.sag_uzunluk = self.orta_uzunluk = self.sol_uzunluk = 0.0

    def _publish(self):
        def f32(v):
            m = Float32()
            m.data = float(v)
            return m
        self.pub_sag_engel.publish(f32(self.sag_engel))
        self.pub_orta_engel.publish(f32(self.orta_engel))
        self.pub_sol_engel.publish(f32(self.sol_engel))
        self.pub_sag_mesafe.publish(f32(self.sag_mesafe))
        self.pub_orta_mesafe.publish(f32(self.orta_mesafe))
        self.pub_sol_mesafe.publish(f32(self.sol_mesafe))

    def _log(self):
        self.get_logger().info(
            f"SAĞ: {'ENGEL' if self.sag_engel else 'TEMİZ':6s} {self.sag_mesafe:.2f}m  |  "
            f"ORTA: {'ENGEL' if self.orta_engel else 'TEMİZ':6s} {self.orta_mesafe:.2f}m  |  "
            f"SOL: {'ENGEL' if self.sol_engel else 'TEMİZ':6s} {self.sol_mesafe:.2f}m"
        )


def main(args=None):
    rclpy.init(args=args)
    node = LidarNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
