import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, LaserScan
from math import sin, cos, pi
import time

class GoToPose(Node):
    def __init__(self, targets, callback=None):
        super().__init__('navigation_goal_action_client')
        self.action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.callback = callback
        self.targets = targets 
        self.current_target_index = 0  
        self.blue_box_detected = False  
        self.blue_box_position = None  

    def send_next_goal(self):
        if self.blue_box_detected and self.blue_box_position:
            self.get_logger().info(f"Blue box detected! Navigating directly to blue box at {self.blue_box_position}")
            x, y = self.blue_box_position
            yaw = 0.0  
            self.send_goal(x, y, yaw)
        elif self.current_target_index < len(self.targets):
            x, y, yaw = self.targets[self.current_target_index]
            self.send_goal(x, y, yaw)
        else:
            self.get_logger().info("All goals reached!")
            if self.callback:
                self.callback()
            return

    def send_goal(self, x, y, yaw):
        self.get_logger().info(f"Waiting for action server...")
        if not self.action_client.wait_for_server(timeout_sec=10.0):  
            self.get_logger().error("Action server not available!")
            return

        self.get_logger().info(f"Sending goal: x={x}, y={y}, yaw={yaw}")
        
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.orientation.z = sin(yaw / 2)
        goal_msg.pose.pose.orientation.w = cos(yaw / 2)

        self.send_goal_future = self.action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self.send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self.get_result_future = goal_handle.get_result_async()
        self.get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        self.get_logger().info(f'Goal {self.current_target_index + 1} reached.')

        if self.blue_box_detected:
            self.get_logger().info("Blue box was detected, stopping further navigation.")

            return

        self.current_target_index += 1
        self.send_next_goal()
        
        self.get_logger().info("Starting rotation before moving to next goal.")
        self.rotating = True  
        self.rotation_start_time = self.get_clock().now().seconds_nanoseconds()[0]  


    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Distance Remaining: {feedback.distance_remaining}')

    def update_blue_box_position(self, x, y):
        self.get_logger().info(f"Received blue box position: {x}, {y}")
        self.blue_box_detected = True
        self.blue_box_position = (x, y)


class RobotExplorer(Node):
    def __init__(self, nav_node):
        super().__init__('robot_explorer')

        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.nav_node = nav_node 

        self.stage = "exploring"
        self.sensitivity = 10
        self.min_area = 1000  
        self.stop_area = 300000  
        self.blue_detected = False
        self.green_detected = False
        self.red_detected = False
        self.obstacle_distance = float('inf')
        self.blue_box_position = None
        self.grid_map = np.zeros((100, 100))  
        self.reached_blue_box = False
        self.rotating = False

        self.timer = self.create_timer(0.1, self.control_robot)
        
        frame_size=(320,240)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer_camera = cv2.VideoWriter('camera_feed.mp4', fourcc, 10.0,frame_size)
        self.video_writer_green = cv2.VideoWriter('green_mask.mp4', fourcc, 10.0, frame_size)
        self.video_writer_blue = cv2.VideoWriter('blue_mask.mp4', fourcc, 10.0, frame_size)
        self.video_writer_red = cv2.VideoWriter('red_mask.mp4', fourcc, 10.0, frame_size)
        


    def rotate_and_detect_blue(self):
        self.get_logger().info("Starting 360° rotation and color detection.")
        self.rotating = True
        self.rotation_start_time = self.get_clock().now().seconds_nanoseconds()[0]

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        lower_green = np.array([60 - self.sensitivity, 100, 100])
        upper_green = np.array([60 + self.sensitivity, 255, 255])

        lower_blue = np.array([100 - self.sensitivity, 150, 100])
        upper_blue = np.array([140 + self.sensitivity, 255, 255])

        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        mask_red = cv2.bitwise_or(cv2.inRange(hsv_image, lower_red1, upper_red1),
                                  cv2.inRange(hsv_image, lower_red2, upper_red2))
        mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
        mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)

        kernel = np.ones((5, 5), np.uint8)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
        
        green_contours, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        blue_contours, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        red_contours, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        

       
        
        if green_contours:
            c = max(green_contours, key=cv2.contourArea)
            area = cv2.contourArea(c)

            if area > self.min_area:
                self.green_detected = True
                M = cv2.moments(c)
                if M['m00'] != 0:
                    cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
                    (x, y), radius = cv2.minEnclosingCircle(c)
                    center = (int(x), int(y))
                    radius = int(radius)
                    cv2.circle(cv_image, center, radius, (0, 255, 0), 2)   
            else:
                self.green_detected = False
                
        if red_contours:
            c = max(red_contours, key=cv2.contourArea)
            area = cv2.contourArea(c)

            if area > self.min_area:
                self.red_detected = True
                M = cv2.moments(c)
                if M['m00'] != 0:
                    cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
                    (x, y), radius = cv2.minEnclosingCircle(c)
                    center = (int(x), int(y))
                    radius = int(radius)
                    cv2.circle(cv_image, center, radius, (0,0,255), 2)
            else:
                self.red_detected = False

        if blue_contours:
            c = max(blue_contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            self.get_logger().info(f"Detected blue box area: {area}")

            if area > self.min_area:
                self.blue_detected = True
                M = cv2.moments(c)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00']) 
                    cy = int(M['m01'] / M['m00']) 
                    (x, y), radius = cv2.minEnclosingCircle(c)
                    center = (int(x), int(y))
                    radius = int(radius)
                    cv2.circle(cv_image, center, radius, (255, 0, 0), 2)
                    self.blue_box_position = cx  
                    self.blue_box_area = area  
                    self.rotating = False
                    self.stage = "approaching_blue"
                    self.nav_node.update_blue_box_position(x, y) 
            else:
                self.blue_detected = False

        
        # cv2.imshow('Camera Feed', cv_image_resized)
        # cv2.imshow('Green Mask', mask_green_resized)
        # cv2.imshow('Blue Mask', mask_blue_resized)
        # cv2.imshow('Red Mask', mask_red_resized)


        frame_size=(320,240)
        mask_green_resized = cv2.resize(mask_green, frame_size, interpolation=cv2.INTER_AREA)
        mask_blue_resized = cv2.resize(mask_blue,frame_size, interpolation=cv2.INTER_AREA)
        mask_red_resized = cv2.resize(mask_red,frame_size, interpolation=cv2.INTER_AREA)
        cv_image_resized = cv2.resize(cv_image, frame_size, interpolation=cv2.INTER_AREA)
        
        cv2.waitKey(3)
                
        green_bgr = cv2.cvtColor(mask_green_resized, cv2.COLOR_GRAY2BGR)
        blue_bgr = cv2.cvtColor(mask_blue_resized, cv2.COLOR_GRAY2BGR)
        red_bgr = cv2.cvtColor(mask_red_resized, cv2.COLOR_GRAY2BGR)

        self.video_writer_camera.write(cv_image_resized)
        self.video_writer_green.write(green_bgr)
        self.video_writer_blue.write(blue_bgr)
        self.video_writer_red.write(red_bgr)

    def lidar_callback(self, msg):
        self.blue_box_distance = min(msg.ranges) if msg.ranges else float('inf')
        if self.stage == "approaching_blue":
            return
    def control_robot(self):
        twist = Twist()

        if self.rotating:
            twist.angular.z = 0.5  
            elapsed_time = self.get_clock().now().seconds_nanoseconds()[0] - self.rotation_start_time
            if elapsed_time >= (2 * pi / 0.5):  
                self.get_logger().info("360° rotation complete.")
                self.rotating = False  
                if not self.blue_detected:
                    self.nav_node.send_next_goal()  #no blue box detected, go to next goal
            else:
                self.get_logger().info(f"Rotating... {elapsed_time:.2f} sec")

        #
        elif self.stage == "approaching_blue":
            cx = self.blue_box_position
            error = cx - 320  
            twist.angular.z = -0.002 * error  
            self.get_logger().info("Approaching blue")
            
            if self.obstacle_distance < 0.5:  
                self.stage = "avoiding_obstacle" 
                self.get_logger().info("Obstacle detected in front of the blue box, avoiding...")
            else:
                self.get_logger().info(f" blue_box_area: {self.blue_box_area}")
                self.get_logger().info(f" stop_area: {self.stop_area}")
                
                if self.blue_box_area < self.stop_area:
                    twist.linear.x = 0.3
                    twist.angular.z = -0.002 * error
                    self.cmd_vel_pub.publish(twist)
                    self.get_logger().info("Keep moving")
                else:

                    self.get_logger().info("Reached blue box, stopping.")
                    self.reached_blue_box = True 
                    
                    twist.linear.x = 0.0 
                    twist.angular.z = 0.0 

                    
                    self.cleanup()
                    self.get_logger().info("Video saved. Cleanup complete.")
                    
                    self.cmd_vel_pub.publish(twist)
                    self.stage = "stopped"

        elif self.stage == "avoiding_obstacle":
            twist.linear.x = 0.2
            if self.left_distance > self.right_distance:
                twist.angular.z = 0.5  
            else:
                twist.angular.z = -0.5  

            if self.obstacle_distance > 1.0:  
                self.stage = "approaching_blue"  
                self.get_logger().info("Obstacle avoided, re-approaching blue box.")

        elif self.stage == "stopped":
            return

        # self.cmd_vel_pub.publish(twist)

    def cleanup(self):
        self.video_writer_camera.release()
        self.video_writer_green.release()
        self.video_writer_blue.release()
        self.video_writer_red.release()
        self.get_logger().info("Video writers released.")


def main(args=None):
    rclpy.init(args=args)

    targets = [(-0.595,-4.16,-0.00143),(1.79,-4.72,-0.00143),(5.86,-5.41,-0.00143),(5.63,-10.9,-0.00143),(-0.282,-10,-0.00143),(-2.04,-12.1,-0.00143),(-5.42,-12.3,-0.00143),(-6.42,-3.41,-0.00143)]
    nav_node = GoToPose(targets)

    explorer = RobotExplorer(nav_node)

    nav_node.send_next_goal()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(nav_node)
    executor.add_node(explorer)
    executor.spin()
    
    explorer.cleanup()

    nav_node.destroy_node()
    explorer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()