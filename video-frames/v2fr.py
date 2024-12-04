import cv2
import os

# Function to convert video to frames
def video_to_frames(video_path, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Error: Could not open video.")
        return
    
    # Frame counter
    frame_count = 0
    
    while True:
        # Read the next frame
        ret, frame = video.read()
        
        if not ret:
            print("Video processing complete.")
            break
        
        # Save the frame as an image file
        frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
    
    # Release the video object
    video.release()
    print(f"Frames saved to: {output_dir}")

# Usage example
video_path = "C:\\Users\\sakshita\\OneDrive\\Desktop\\Mini Project\\Code\\v2f\\vi.mp4"  # Path to the video file
output_dir = "output_frames"   # Directory to save the frames
video_to_frames(video_path, output_dir)
