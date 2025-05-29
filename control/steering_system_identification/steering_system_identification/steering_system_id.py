#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from autoware_auto_control_msgs.msg import AckermannControlCommand
from sensor_msgs.msg import SteeringReport
from collections import deque
from datetime import datetime, timedelta

from .mhe import SteeringMHE

class SteeringSystemID(Node):
    def __init__(self):
        super().__init__('steering_system_id')
        
        # Initialize MHE
        self.mhe = SteeringMHE()
        
        # Create subscribers
        self.cmd_sub = self.create_subscription(
            AckermannControlCommand,
            '~/input/control_cmd',
            self.on_control_cmd,
            1)
            
        self.steering_sub = self.create_subscription(
            SteeringReport,
            '~/input/steering_report',
            self.on_steering_report,
            1)
            
        # Create timer for parameter estimation
        self.timer = self.create_timer(0.1, self.on_timer)  # 10 Hz
        
        # Data buffers
        self.commanded_steering = deque(maxlen=1000)  # Store last 1000 commands
        self.measured_steering = deque(maxlen=1000)   # Store last 1000 measurements
        
        self.get_logger().info('Steering system identification node initialized')
        
    def on_control_cmd(self, msg):
        """Callback for control command messages"""
        self.commanded_steering.append({
            'stamp': msg.stamp,
            'steering': msg.lateral.steering_tire_angle
        })
        
    def on_steering_report(self, msg):
        """Callback for steering report messages"""
        self.measured_steering.append({
            'stamp': msg.stamp,
            'steering': msg.steering_tire_angle
        })
        
    def on_timer(self):
        """Timer callback for parameter estimation"""
        if not self.commanded_steering or not self.measured_steering:
            return
            
        # Get latest measurements
        cmd = self.commanded_steering[-1]
        meas = self.measured_steering[-1]
        
        # Calculate time difference
        dt = (meas['stamp'] - cmd['stamp']).to_sec()
        
        # Update MHE
        self.mhe.update(cmd['steering'], meas['steering'], dt)
        
        # Log current estimates
        self.get_logger().info(
            f'Current estimates - Time constant: {self.mhe.get_time_constant():.3f}, '
            f'Delay: {self.mhe.get_delay():.3f}'
        )
        
        # Clean up old data
        now = self.get_clock().now()
        max_age = timedelta(seconds=10.0)
        
        while self.commanded_steering and (now - self.commanded_steering[0]['stamp']) > max_age:
            self.commanded_steering.popleft()
            
        while self.measured_steering and (now - self.measured_steering[0]['stamp']) > max_age:
            self.measured_steering.popleft()

def main(args=None):
    rclpy.init(args=args)
    node = SteeringSystemID()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
