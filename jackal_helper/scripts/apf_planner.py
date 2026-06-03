#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import math

def euler_from_quaternion(q):
    t3 = +2.0 * (q.w * q.z + q.x * q.y)
    t4 = +1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(t3, t4)

class APFPlanner(Node):
    def __init__(self):
        super().__init__('apf_planner')
        
        self.declare_parameter('world_idx', 0)
        world_idx = self.get_parameter('world_idx').value
        
        # Determine global goal from world_idx
        if world_idx < 300:  
            INIT_POSITION = [-2.25, 3, 1.57]  
            GOAL_POSITION = [0, 10]  
        else:  
            INIT_POSITION = [11, 0, 3.14]  
            GOAL_POSITION = [-20, 0]  
            
        self.goal_x = INIT_POSITION[0] + GOAL_POSITION[0]
        self.goal_y = INIT_POSITION[1] + GOAL_POSITION[1]

        self.current_x = INIT_POSITION[0]
        self.current_y = INIT_POSITION[1]
        self.current_yaw = INIT_POSITION[2]
        self.yaw_offset = INIT_POSITION[2]  # Gazebo teleports the robot, but EKF odom stays at 0. This is the offset.
        self.odom_received = False
        
        # EMA filter variables to prevent jitter
        self.F_att_x_ema = 0.0
        self.F_att_y_ema = 0.0
        self.F_rep_x_ema = 0.0
        self.F_rep_y_ema = 0.0
        self.ema_alpha = 0.2  # Smoothing factor (lower is smoother)
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/platform/odom/filtered',
            self.odom_callback,
            10)

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/front/scan',
            self.scan_callback,
            10)
        
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/apf_markers', 10)
        
        self.get_logger().info(f"APF Planner started! Goal: ({self.goal_x}, {self.goal_y})")
        
        # APF Parameters
        self.k_att = 5.0
        self.k_rep = 0.05
        self.rep_distance = 1
        self.max_speed = 0.7
        self.k_omega = 2.0
        
    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        # Apply the yaw offset because EKF odometry does not track the initial Gazebo teleportation rotation
        self.current_yaw = euler_from_quaternion(msg.pose.pose.orientation) + self.yaw_offset
        self.odom_received = True

    def scan_callback(self, msg):
        if not self.odom_received:
            return
            
        # 1. Calculate Attractive Force (Global frame)
        dx = self.goal_x - self.current_x
        dy = self.goal_y - self.current_y
        dist_to_goal = math.hypot(dx, dy)
        
        if dist_to_goal < 0.5:
            # We reached the goal
            self.publisher.publish(Twist())
            return
            
        # Normalize attractive force
        F_att_global_x = (dx / dist_to_goal) * self.k_att
        F_att_global_y = (dy / dist_to_goal) * self.k_att
        
        # Transform attractive force to local frame (base_link)
        cos_theta = math.cos(self.current_yaw)
        sin_theta = math.sin(self.current_yaw)
        F_att_local_x = F_att_global_x * cos_theta + F_att_global_y * sin_theta
        F_att_local_y = -F_att_global_x * sin_theta + F_att_global_y * cos_theta

        # 2. Calculate Repulsive Force (Local frame)
        ranges = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))
        
        # Filter for front-facing FOV only (e.g., -pi/2 to pi/2)
        front_indices = np.where((angles > -np.pi/2) & (angles < np.pi/2))[0]
        ranges = ranges[front_indices]
        angles = angles[front_indices]
        
        # Replace infinites and NaNs
        ranges[np.isinf(ranges) | np.isnan(ranges)] = 5.0
        
        F_rep_local_x = 0.0
        F_rep_local_y = 0.0
        
        # Group rays into sectors to prevent too many rays from amplifying force
        sectors = 36
        sector_size = len(ranges) // sectors
        
        for i in range(sectors):
            start = i * sector_size
            end = start + sector_size
            sector_ranges = ranges[start:end]
            
            min_r = np.min(sector_ranges)
            if min_r < self.rep_distance:
                # To prevent division by zero or huge forces if extremely close
                r_eff = max(min_r, 0.1) 
                
                # Standard APF repulsion formula magnitude
                mag = self.k_rep * (1.0 / r_eff - 1.0 / self.rep_distance) * (1.0 / (r_eff**2))
                
                # Get the angle of the closest point in this sector
                alpha = angles[start + np.argmin(sector_ranges)]
                
                # Add to total repulsive force (pushing AWAY from obstacle)
                F_rep_local_x -= mag * math.cos(alpha)
                F_rep_local_y -= mag * math.sin(alpha)
                
        # Apply EMA filter to eliminate jitter
        self.F_att_x_ema = self.ema_alpha * F_att_local_x + (1 - self.ema_alpha) * self.F_att_x_ema
        self.F_att_y_ema = self.ema_alpha * F_att_local_y + (1 - self.ema_alpha) * self.F_att_y_ema
        self.F_rep_x_ema = self.ema_alpha * F_rep_local_x + (1 - self.ema_alpha) * self.F_rep_x_ema
        self.F_rep_y_ema = self.ema_alpha * F_rep_local_y + (1 - self.ema_alpha) * self.F_rep_y_ema
        
        F_att_local_x = self.F_att_x_ema
        F_att_local_y = self.F_att_y_ema
        F_rep_local_x = self.F_rep_x_ema
        F_rep_local_y = self.F_rep_y_ema
                
        # 3. Sum forces
        F_tot_x = F_att_local_x + F_rep_local_x
        F_tot_y = F_att_local_y + F_rep_local_y
        
        # Break symmetry if heading straight into a wall
        if F_rep_local_x < -0.5 and abs(F_rep_local_y) < 0.2:
            F_tot_y += 1.5  # Force a left turn
        
        # 4. Convert force to motion commands
        desired_heading = math.atan2(F_tot_y, F_tot_x)
        
        twist = Twist()
        twist.angular.z = np.clip(self.k_omega * desired_heading, -1.5, 1.5)
        
        # Slow down if we need to turn sharply (and stop if heading is behind us)
        turn_penalty = max(0.0, 1.0 - abs(desired_heading) / (math.pi / 2))
        
        # Drive forward only if the net force is somewhat forward, otherwise turn in place
        if F_tot_x > 0:
            desired_speed = math.hypot(F_tot_x, F_tot_y)
            twist.linear.x = min(float(self.max_speed), desired_speed) * turn_penalty
        else:
            twist.linear.x = 0.0  # Turn in place instead of backing up
            
        # Optional: Print debug info every 20 ticks
        if not hasattr(self, 'tick'):
            self.tick = 0
        self.tick += 1
        if self.tick % 20 == 0:
            self.get_logger().info(f"F_att: ({F_att_local_x:.2f}, {F_att_local_y:.2f}) | F_rep: ({F_rep_local_x:.2f}, {F_rep_local_y:.2f}) | Cmd: v={twist.linear.x:.2f}, w={twist.angular.z:.2f}")
            
        self.publisher.publish(twist)
        self.publish_force_markers(F_att_local_x, F_att_local_y, F_rep_local_x, F_rep_local_y, F_tot_x, F_tot_y, msg.header.stamp)

    def publish_force_markers(self, att_x, att_y, rep_x, rep_y, tot_x, tot_y, stamp):
        marker_array = MarkerArray()
        
        def create_arrow(m_id, x, y, r, g, b):
            marker = Marker()
            marker.header.frame_id = 'base_link'
            marker.header.stamp = stamp
            marker.ns = 'apf_forces'
            marker.id = m_id
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            
            p_start = Point()
            p_start.x = 0.0
            p_start.y = 0.0
            p_start.z = 0.2
            
            p_end = Point()
            # Scale the arrows for visualization (0.5 means 1 N of force = 0.5 meters long)
            scale = 0.5
            p_end.x = x * scale
            p_end.y = y * scale
            p_end.z = 0.2
            
            marker.points = [p_start, p_end]
            marker.scale.x = 0.15  # Shaft diameter
            marker.scale.y = 0.3   # Head diameter
            marker.scale.z = 0.2   # Head length
            
            marker.color.r = float(r)
            marker.color.g = float(g)
            marker.color.b = float(b)
            marker.color.a = 0.8
            return marker

        # Green = Attractive
        marker_array.markers.append(create_arrow(0, att_x, att_y, 0.0, 1.0, 0.0))
        # Red = Repulsive
        marker_array.markers.append(create_arrow(1, rep_x, rep_y, 1.0, 0.0, 0.0))
        # Blue = Total
        marker_array.markers.append(create_arrow(2, tot_x, tot_y, 0.0, 0.5, 1.0))
        
        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = APFPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
