#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np

class FollowTheGap(Node):
    def __init__(self):
        super().__init__('follow_the_gap')
        
        # We subscribe to the remapped front scan topic
        self.subscription = self.create_subscription(
            LaserScan,
            '/front/scan',
            self.scan_callback,
            10)
        
        # Publish to cmd_vel to control the robot
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.get_logger().info("Follow The Gap node started!")
        
        # Robot physical parameters
        self.bubble_radius = 0.1  # Safety bubble radius in meters
        self.max_speed = 0.6     # Maximum forward speed
        
    def preprocess_lidar(self, ranges):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """
        proc_ranges = np.array(ranges)
        # Replace infinites and NaNs with maximum valid range (e.g., 5.0m)
        proc_ranges[np.isinf(proc_ranges) | np.isnan(proc_ranges)] = 5.0
        # Clip max range
        proc_ranges = np.clip(proc_ranges, 0, 5.0)
        return proc_ranges

    def find_max_gap(self, free_space_ranges):
        """ Return the start index & end index of the max gap in free_space_ranges """
        # Mask where free_space_ranges > threshold (e.g. non-zero after bubble)
        threshold = 0.1
        mask = free_space_ranges > threshold
        
        # Find consecutive sequences of Trues
        # Trick: pad with False and take diff
        padded = np.insert(np.append(mask, False), 0, False)
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1
        
        if len(starts) == 0:
            return 0, 0
            
        # Find the sequence with the maximum length
        lengths = ends - starts + 1
        max_idx = np.argmax(lengths)
        
        return starts[max_idx], ends[max_idx]
    
    def find_best_point(self, start_i, end_i, ranges):
        """ Start_i & end_i are start and end indicies of max-gap range, respectively
            Return index of best point in ranges
            Naive: Choose the furthest point within the gap and go there
        """
        if start_i == end_i:
            return start_i
            
        gap_ranges = ranges[start_i:end_i+1]
        best_point_idx_in_gap = np.argmax(gap_ranges)
        return start_i + best_point_idx_in_gap

    def scan_callback(self, msg):
        # 1. Preprocess the LiDAR ranges
        # We only consider the field of view in front of the robot (e.g. -90 to +90 degrees)
        # Assuming msg.angle_min is approx -3.14 and msg.angle_max is approx 3.14
        
        ranges = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))
        
        # Filter for front-facing FOV only (e.g., -pi/2 to pi/2)
        front_indices = np.where((angles > -np.pi/2) & (angles < np.pi/2))[0]
        front_ranges = ranges[front_indices]
        front_angles = angles[front_indices]
        
        proc_ranges = self.preprocess_lidar(front_ranges)
        
        # 2. Find closest point to LiDAR
        closest_idx = np.argmin(proc_ranges)
        closest_dist = proc_ranges[closest_idx]
        
        # 3. Eliminate all points inside 'bubble' (set them to zero) 
        # Create a bubble around the closest point
        angle_increment = msg.angle_increment
        # How many indices does the bubble radius cover?
        # arc length = r * theta => theta = arc_length / r
        # We use a simple approximation: theta = bubble_radius / closest_dist
        if closest_dist > 0.05:
            bubble_angle = np.arcsin(min(self.bubble_radius / closest_dist, 1.0))
        else:
            bubble_angle = np.pi/2
            
        num_indices = int(bubble_angle / angle_increment)
        
        # Set values inside bubble to 0
        bubble_start = max(0, closest_idx - num_indices)
        bubble_end = min(len(proc_ranges) - 1, closest_idx + num_indices)
        proc_ranges[bubble_start:bubble_end+1] = 0.0
        
        # 4. Find max length gap 
        start_i, end_i = self.find_max_gap(proc_ranges)
        
        # 5. Find the best point in the gap 
        best_point_idx = self.find_best_point(start_i, end_i, proc_ranges)
        
        # 6. Publish Drive message
        best_angle = front_angles[best_point_idx]
        
        # Simple proportional controller for steering
        twist = Twist()
        # The goal is mostly directly in front, so if the gap is somewhat straight, go fast
        # If the gap requires a sharp turn, slow down
        twist.angular.z = best_angle * 1.5
        twist.linear.x = self.max_speed * max(0.2, (1.0 - abs(best_angle)/(np.pi/2)))
        
        # Safety stop if very close to obstacle
        if closest_dist < 0.2:
            twist.linear.x = -0.5
            twist.angular.z = 0.0
            
        self.publisher.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = FollowTheGap()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
        
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
