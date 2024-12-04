import cv2
import os

def frames_to_video(frames_dir, output_video, frame_rate):
    
    images = [img for img in os.listdir(frames_dir) if img.endswith((".png", ".jpg", ".jpeg"))]
    images.sort()  
    
    if not images:
        print("No images found in the directory.")
        return

    first_frame_path = os.path.join(frames_dir, images[0])
    frame = cv2.imread(first_frame_path)
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    video_writer = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))
    
    for image in images:
        frame_path = os.path.join(frames_dir, image)
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved as: {output_video}")

frames_dir = "output_frames"        # Directory containing frames
output_video = "output_video16.mp4"   # Output video file
frame_rate = 20                   # Frames per second
frames_to_video(frames_dir, output_video, frame_rate)
