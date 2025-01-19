import cv2
import os
import numpy as np
from time import time

def extract_frames_gpu(video_path, output_dir, max_frames=20000):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Check if CUDA is available
    if cv2.cuda.getCudaEnabledDeviceCount() == 0:
        print("No CUDA device found! Falling back to CPU.")
        use_gpu = False
    else:
        print("CUDA device found! Using GPU acceleration.")
        use_gpu = True
        # Create GPU upload and download streams
        stream = cv2.cuda_Stream()

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get video properties
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    # Calculate frame extraction interval
    interval = max(1, total_frames // max_frames)

    frame_count = 0
    saved_count = 0
    start_time = time()

    if use_gpu:
        # Create GPU mat objects for upload
        gpu_frame = cv2.cuda_GpuMat()

    print(f"Total frames in video: {total_frames}")
    print(f"Extracting approximately {max_frames} frames...")

    while True:
        # Read frame
        success, frame = video.read()

        if not success:
            break

        # Save frame at specified intervals
        if frame_count % interval == 0 and saved_count < max_frames:
            if use_gpu:
                # Upload frame to GPU
                gpu_frame.upload(frame, stream)

                # Apply any GPU processing here if needed
                # For example, resize or color conversion
                # gpu_frame = cv2.cuda.resize(gpu_frame, (new_width, new_height))

                # Download processed frame back to CPU
                processed_frame = gpu_frame.download(stream)

                # Save the frame
                frame_filename = os.path.join(output_dir, f'frame_{saved_count:05d}.jpg')
                cv2.imwrite(frame_filename, processed_frame)
            else:
                # CPU processing path
                frame_filename = os.path.join(output_dir, f'frame_{saved_count:05d}.jpg')
                cv2.imwrite(frame_filename, frame)

            saved_count += 1

            # Print progress
            if saved_count % 100 == 0:
                elapsed_time = time() - start_time
                fps = saved_count / elapsed_time
                print(f"Saved {saved_count} frames... ({fps:.2f} frames/second)")

        frame_count += 1

    # Release resources
    video.release()
    if use_gpu:
        gpu_frame.release()
        stream.release()

    total_time = time() - start_time
    print(f"\nExtraction complete! Saved {saved_count} frames to {output_dir}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average speed: {saved_count/total_time:.2f} frames/second")

# Example usage
if __name__ == "__main__":
    video_path = "C:/Users/default.DESKTOP-7FKFEEG/Downloads/farag v elshorbagy 2019 chopped.mp4"  # Replace with your video path
    output_directory = "output frames fvel"  # Replace with your desired output directory

    extract_frames_gpu(video_path, output_directory)
