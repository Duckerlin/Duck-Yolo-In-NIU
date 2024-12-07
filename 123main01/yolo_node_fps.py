# File: yolo_realsense_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import time
import csv

class YoloRealSenseNode(Node):
    def __init__(self):
        super().__init__('yolo_realsense_node')

        # Declare ROS parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('model_path', '/home/duckeggs/ros2_ws/models/best.pt'),
                ('output_csv_path', '/home/duckeggs/ros2_ws/yolo_ros2/fps_data.csv'),
                ('frame_width', 640),
                ('frame_height', 480),
                ('fps', 30),
                ('recording_interval', 10),
                ('max_intervals', 6)  # Max intervals for recording data
            ]
        )

        # Retrieve parameters
        self.model_path = self.get_parameter('model_path').value
        self.output_csv_path = self.get_parameter('output_csv_path').value
        self.frame_width = self.get_parameter('frame_width').value
        self.frame_height = self.get_parameter('frame_height').value
        self.fps = self.get_parameter('fps').value
        self.recording_interval = self.get_parameter('recording_interval').value
        self.max_intervals = self.get_parameter('max_intervals').value

        # Initialize model and RealSense
        self.bridge = CvBridge()
        self._initialize_model()
        self._initialize_realsense()

        # Publisher
        self.publisher = self.create_publisher(Image, 'yolo/detections', 10)

        # Initialize counters and statistics
        self.start_time = time.time()
        self.frame_count = 0
        self.total_distance = 0.0
        self.total_angle = 0.0
        self.total_objects = 0
        self.record_count = 0
        self.prev_time = time.time()
        self.previous_fps = None
        self.fps_data = []
        self.frame_time_data = []

    def _initialize_model(self):
        """Initialize the YOLO model."""
        try:
            self.model = YOLO(self.model_path)
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(self.device)
        except Exception as e:
            self.get_logger().error(f"Error initializing YOLO model: {e}")
            raise

    def _initialize_realsense(self):
        """Initialize the RealSense pipeline."""
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.color, self.frame_width, self.frame_height, rs.format.bgr8, self.fps)
            self.config.enable_stream(rs.stream.depth, self.frame_width, self.frame_height, rs.format.z16, self.fps)
            self.pipeline.start(self.config)
            self.align = rs.align(rs.stream.color)
        except Exception as e:
            self.get_logger().error(f"Error initializing RealSense pipeline: {e}")
            raise

    def calculate_angle(self, vector):
        """Calculate the angle between the object vector and the normal vector (0,0,1)."""
        normal_vector = np.array([0, 0, 1], dtype=np.float32)
        vector = np.array(vector, dtype=np.float32)
        dot_product = np.dot(vector, normal_vector)
        cos_theta = dot_product / (np.linalg.norm(vector) * np.linalg.norm(normal_vector))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))

    def process_images(self):
        """Main loop for processing images."""
        try:
            while rclpy.ok():
                frame_start_time = time.time()
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    self.get_logger().warning("No color or depth frame received.")
                    continue

                frame = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                results = self.model(frame)

                if results:
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                            label = result.names[box.cls[0].item()]
                            confidence = box.conf[0].item()
                            object_distance = depth_frame.get_distance((x1 + x2) // 2, (y1 + y2) // 2)

                            # Draw results
                            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                            depth_intri = depth_frame.profile.as_video_stream_profile().intrinsics
                            camera_xyz = rs.rs2_deproject_pixel_to_point(depth_intri, (center_x, center_y), object_distance)
                            angle_deg = self.calculate_angle(camera_xyz)
                            self._update_statistics(object_distance, angle_deg)

                # Publish processed image
                self.publisher.publish(self.bridge.cv2_to_imgmsg(frame, encoding='bgr8'))
                self._update_fps_and_display(frame, frame_start_time)

                # Exit after the max interval
                if self.record_count >= self.max_intervals:
                    break
        except Exception as e:
            self.get_logger().error(f"Error processing images: {e}")
        finally:
            self._save_statistics()

    def _update_statistics(self, object_distance, angle_deg):
        """Update statistical data."""
        self.frame_count += 1
        self.total_distance += object_distance
        self.total_angle += angle_deg
        self.total_objects += 1

    def _update_fps_and_display(self, frame, frame_start_time):
        """Calculate and display FPS."""
        current_time = time.time()
        elapsed_time = current_time - self.prev_time
        if elapsed_time >= self.recording_interval:
            fps = self.frame_count / elapsed_time
            self.fps_data.append(fps)
            self.frame_time_data.append(elapsed_time / self.frame_count * 1000)
            self.record_count += 1
            self.frame_count = 0
            self.prev_time = current_time

    def _save_statistics(self):
        """Save FPS and frame time data to CSV."""
        try:
            with open(self.output_csv_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Interval', 'FPS', 'Frame Time'])
                avg_fps = np.mean(self.fps_data)
                avg_frame_time = np.mean(self.frame_time_data)
                for i, fps in enumerate(self.fps_data):
                    writer.writerow([i + 1, fps, self.frame_time_data[i]])
                writer.writerow(['Average', avg_fps, avg_frame_time])
            self.get_logger().info(f"FPS and frame time data saved to {self.output_csv_path}")
        except Exception as e:
            self.get_logger().error(f"Error saving FPS data to CSV: {e}")

    def destroy_node(self):
        try:
            self.pipeline.stop()
        except Exception as e:
            self.get_logger().error(f"Error stopping RealSense pipeline: {e}")
        finally:
            super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = YoloRealSenseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
