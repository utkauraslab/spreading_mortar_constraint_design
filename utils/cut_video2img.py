import cv2
import os
import sys






def cut_video_into_frames(video_path, output_folder):
    """
    Cuts a video file into individual frames and saves them as images.

    Args:
        video_path (str): /home/farong/Desktoop/spreading_mortar_constraint_design
        output_folder (str): /home/farong/Desktoop/spreading_mortar_constraint_design/data
    """
    # --- 1. Validate Input and Create Output Directory ---
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return

    if not os.path.exists(output_folder):
        print(f"Output folder '{output_folder}' not found. Creating it...")
        os.makedirs(output_folder)
        print(f"Successfully created directory: '{output_folder}'")

    # --- 2. Open the Video File ---
    video_capture = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not video_capture.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return

    # --- 3. Get Video Properties (for progress tracking) ---
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    print(f"Video Info: {total_frames} total frames at {fps:.2f} FPS.")


    # --- 4. Read and Save Frames ---
    frame_count = 0
    success = True
    while success:
        # Read the next frame
        success, frame = video_capture.read()

        if success:
            # Construct the output filename
            # Using zfill to pad with leading zeros (e.g., 00001, 00002) for proper sorting
            #frame_filename = os.path.join(output_folder, f"frame_{str(frame_count).zfill(5)}.jpg")

            frame_filename = os.path.join(output_folder, f"frame_{str(frame_count).zfill(4)}.png")

            # Save the frame as a JPEG image
            cv2.imwrite(frame_filename, frame)

            frame_count += 1

            # Print progress
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames...")

    # --- 5. Release Resources ---
    video_capture.release()
    print(f"\nDone! Extracted {frame_count} frames and saved them to '{output_folder}'.")


# --- Example Usage ---
if __name__ == '__main__':
    
    # __file__ : /home/farong/Desktoop/spreading_mortar_constraint_design/utils/cut_video2img.py
    # os.path.dirname(__file__): /home/farong/Desktoop/spreading_mortar_constraint_design/utils
    # os.path.dirname(os.path.dirname(__file__)): /home/farong/Desktoop/spreading_mortar_constraint_design
    project_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


    # --- Configuration ---
    # IMPORTANT: Replace this with the actual name of your video file
    video_filename = "spreading_mortar_videos/mortar2.mp4"
    # The video is expected to be in a 'videos' subdirectory
    input_video_path = os.path.join(project_directory, video_filename)

    # Define the directory to save the frames
    output_frames_directory = os.path.join(project_directory, 'data')

    # --- Run the function ---
    # Check if a video file name is provided as a command-line argument
    if len(sys.argv) > 1:
        input_video_path = sys.argv[1]
        print(f"Using video file from command-line argument: {input_video_path}")
    elif not os.path.exists(input_video_path):
         print("---")
         print(f"ERROR: The example video '{video_filename}' was not found at '{input_video_path}'.")
         print("Please do one of the following:")
         print(f"1. Place your video file at that location and rename it to '{video_filename}'.")
         print("2. Update the 'video_filename' variable in this script with your video's name.")
         print("3. Run the script with the path to your video as an argument, e.g.:")
         print(f"   python {os.path.basename(__file__)} /path/to/your/video.mp4")
         print("---")
         sys.exit(1) # Exit the script if the default video isn't found

    cut_video_into_frames(video_path=input_video_path, output_folder=output_frames_directory)


