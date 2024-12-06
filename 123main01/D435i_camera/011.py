import torch
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import time

class YoloRealSense:
    def __init__(self, model_path):
        # Load the YOLO model
        self.model = YOLO(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        if self.device == 'cuda':
            self.model.half()

        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Set frame resolution and FPS
        self.frame_width = 640
        self.frame_height = 480
        self.fps = 30
        self.config.enable_stream(rs.stream.color, self.frame_width, self.frame_height, rs.format.bgr8, self.fps)
        self.config.enable_stream(rs.stream.depth, self.frame_width, self.frame_height, rs.format.z16, self.fps)
        
        # Start the RealSense pipeline
        try:
            self.pipeline.start(self.config)
        except Exception as e:
            print(f"Error starting RealSense pipeline: {e}")
            raise

        # Align the color and depth frames
        self.align = rs.align(rs.stream.color)

        # Initialize counters and statistics
        self.start_time = time.time()
        self.frame_count = 0
        self.total_distance = 0.0
        self.total_angle = 0.0
        self.total_objects = 0
        self.recording_interval = 10  # 10 seconds interval
        self.record_count = 0
        self.prev_time = time.time()
        self.previous_fps = None  # Store the previous FPS value to display in subsequent frames

    def calculate_angle(self, vector):
        # 法向量 (0,0,1)
        normal_vector = np.array([0, 0, 1], dtype=np.float32)
        vector = np.array(vector, dtype=np.float32)
        
        # 计算点积
        dot_product = np.dot(vector, normal_vector)
        norm_vector = np.linalg.norm(vector)
        norm_normal = np.linalg.norm(normal_vector)
        
        # 计算夹角（弧度）
        cos_theta = dot_product / (norm_vector * norm_normal)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 确保值在 [-1, 1] 范围内
        angle_rad = np.arccos(cos_theta)
        
        # 转换为度数
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    def process_images(self):
        while True:
            try:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    print("No color or depth frame received.")
                    continue

                frame = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())

                # Use the YOLO model for detection
                results = self.model(frame)

                if results:
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                            label = result.names[box.cls[0].item()]
                            confidence = box.conf[0].item()
                            object_distance = depth_frame.get_distance((x1 + x2) // 2, (y1 + y2) // 2)

                            # Calculate center coordinates
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2

                            # Draw bounding box
                            font_scale = 1
                            thickness = 2
                            color = (255, 255, 255) 

                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                            # Convert to camera coordinates
                            ux, uy = center_x, center_y
                            dis = depth_frame.get_distance(ux, uy)
                            depth_intri = depth_frame.profile.as_video_stream_profile().intrinsics
                            camera_xyz = rs.rs2_deproject_pixel_to_point(depth_intri, (ux, uy), dis)
                            camera_xyz = np.round(np.array(camera_xyz)*100, 2)  # Convert to mm with two decimal places

                            # Draw circle and text for camera coordinates
                            cv2.circle(frame, (ux, uy), 2, color, 2)
                            cv2.putText(frame, f"({camera_xyz[0]:.2f}, {camera_xyz[1]:.2f}, {camera_xyz[2]:.2f})", (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
                            cv2.putText(frame, f"Distance: {object_distance:.3f}m", (x1, y2 + 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
                            

                            # Calculate and display the angle with respect to the normal vector (0,0,1)
                            angle_deg = self.calculate_angle(camera_xyz)
                            cv2.putText(frame, f"Angle: {angle_deg:.2f}", (x1, y2 + 90), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

                            # Display object label
                            cv2.putText(frame, f"{label}({confidence:.2f})", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

                            # Display center coordinates
                            center_text = f"Center: ({center_x}, {center_y})"
                            cv2.putText(frame, center_text, (x1, y2 + 120), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

                            # Update statistics
                            self.frame_count += 1
                            self.total_distance += object_distance
                            self.total_angle += angle_deg
                            self.total_objects += 1

                # Calculate FPS and display for each interval
                current_time = time.time()
                elapsed_time = current_time - self.prev_time
                if elapsed_time >= self.recording_interval:  # Update every 10 seconds
                    # Calculate FPS using the number of frames processed in the interval
                    fps = self.frame_count / elapsed_time
                    self.previous_fps = fps  # Store current FPS for display in the next interval
                    print(f"Interval {self.record_count + 1}: FPS: {fps:.2f}")

                    # Reset for the next interval
                    self.record_count += 1
                    self.frame_count = 0
                    self.prev_time = current_time

                # Calculate total elapsed time and display it with FPS update number
                total_elapsed_time = time.time() - self.start_time
                elapsed_seconds = int(total_elapsed_time)  # Round down to whole seconds
                fps_update_num = self.record_count + 1  # Number of FPS updates

                # Display Time and FPS Update number
                cv2.putText(frame, f"Time: {elapsed_seconds} sec | FPS Update: {fps_update_num}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Display FPS: or the actual FPS based on elapsed time
                if self.previous_fps is None:  # During the first few seconds
                    cv2.putText(frame, "FPS:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                else:
                    cv2.putText(frame, f"FPS: {self.previous_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                # Display the frame
                cv2.imshow('YOLO RealSense', frame)
                if cv2.waitKey(1) & 0xFF == ord('q') or self.record_count >= 6:  # Stop after 6 intervals
                    break

            except Exception as e:
                print(f"Error processing image: {e}")

        cv2.destroyAllWindows()

    def stop(self):
        try:
            self.pipeline.stop()
        except Exception as e:
            print(f"Error stopping RealSense pipeline: {e}")
        finally:
            cv2.destroyAllWindows()

def main():
    model_path = r"D:\123\123main\pt\best.pt"
    
    yolo_realsense = YoloRealSense(model_path)
    yolo_realsense.process_images()
    yolo_realsense.stop()

if __name__ == "__main__":
    main()
