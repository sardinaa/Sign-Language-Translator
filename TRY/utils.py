import cv2
import numpy as np
from IPython.display import Video, display
import matplotlib.pyplot as plt

def generate_landmarks_comparison_video(input_data, original_video=None, output_video_path="landmarks_comparison_output.webm", frame_size=(640, 640), fps=30, sequential=False):
    """
    Generate a comparison video of multiple landmarks sets or a single input with multiple sequences.
    Optionally include the original video source. If `sequential` is True, each sequence of landmarks
    is displayed one after the other.

    Parameters:
        input_data (list or np.ndarray): Either a list of tuples [(label, landmarks)] or a numpy array
                                         of shape (num_sequences, frames, landmarks, coordinates) or
                                         (frames, landmarks, coordinates).
        original_video (str, optional): Path to the original video file to include as the first region.
        output_video_path (str): Path to save the generated video.
        frame_size (tuple): Size of each region in the video in pixels (width, height).
        fps (int): Frames per second for the video.
        sequential (bool, optional): If True, displays each sequence of landmarks individually.

    Returns:
        None
    """
    # Process input data based on type
    if isinstance(input_data, list):
        landmarks_list = input_data
    elif isinstance(input_data, np.ndarray):
        if input_data.ndim == 4:  # Case: (num_sequences, frames, landmarks, coordinates)
            landmarks_list = [(f"Sequence {i+1}", seq) for i, seq in enumerate(input_data)]
        elif input_data.ndim == 3 and (input_data.shape[2] == 2 or input_data.shape[2] == 3):  # Case: (frames, landmarks, coordinates)
            landmarks_list = [("Sequence 1", input_data)]
        else:
            raise ValueError("Input numpy array must have shape (num_sequences, frames, landmarks, coordinates) or (frames, landmarks, coordinates).")
    else:
        raise ValueError("Input data must be a list of tuples or a numpy array.")

    # Create the video writer for WebM
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'VP90'), fps, frame_size)

    # Load original video frames if provided
    original_video_frames = []
    if original_video:
        cap = cv2.VideoCapture(original_video)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, frame_size)
            original_video_frames.append(frame)
        cap.release()

    if sequential:
        # Sequential mode: Process each sequence one at a time
        for label, landmarks in landmarks_list:
            for frame_idx in range(landmarks.shape[0]):
                frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)

                # Draw landmarks
                for lm in landmarks[frame_idx]:
                    if np.any(lm[:2] != 0):  # Skip zeroed landmarks
                        x = int(lm[0] * frame_size[0])
                        y = int(lm[1] * frame_size[1])
                        cv2.circle(frame, (x, y), radius=3, color=(0, 255, 0), thickness=-1)

                # Add label text in the top-left corner
                text_position = (10, 30)
                cv2.putText(
                    frame, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
                )

                # Write the frame to the video
                video_writer.write(frame)
    else:
        # Side-by-side mode: Combine all regions in one frame
        num_regions = len(landmarks_list) + (1 if original_video else 0)
        total_width = frame_size[0] * num_regions

        # Get the number of frames from the landmarks
        num_frames = landmarks_list[0][1].shape[0]

        for frame_idx in range(num_frames):
            frame = np.zeros((frame_size[1], total_width, 3), dtype=np.uint8)

            # If original video is included, add the corresponding frame
            if original_video and frame_idx < len(original_video_frames):
                frame[:, :frame_size[0]] = original_video_frames[frame_idx]

            # Helper function to draw landmarks on a specific region
            def draw_landmarks(region_offset, landmarks):
                for lm in landmarks:
                    if np.any(lm[:2] != 0):  # Skip zeroed landmarks
                        x = int(lm[0] * frame_size[0]) + region_offset
                        y = int(lm[1] * frame_size[1])
                        cv2.circle(frame, (x, y), radius=3, color=(0, 255, 0), thickness=-1)

            # Draw landmarks for each set and add label
            for i, (label, landmarks) in enumerate(landmarks_list):
                region_offset = (i + (1 if original_video else 0)) * frame_size[0]
                draw_landmarks(region_offset, landmarks[frame_idx])

                # Add label text in the top-left corner of the region
                text_position = (region_offset + 10, 30)
                cv2.putText(
                    frame, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
                )

            # Write the frame to the video
            video_writer.write(frame)

    # Release resources
    video_writer.release()

    # Display the video in the notebook
    display(Video(output_video_path, embed=True))


def plot_landmark_frame(frame, name = None, color = None, size = None):

    # Coordenadas x e y
    x = frame[:, 0]
    y = frame[:, 1]
    plt.figure(figsize=size)
    plt.xlim(0,1)
    plt.ylim(0,1)

    plt.scatter(x, y, s = 2)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().invert_yaxis()  # Invertir eje Y para coordinar con las imágenes
    plt.axis("off")
    if name:
        plt.savefig(name, transparent=True, dpi=500, bbox_inches="tight")
    plt.show()

def count_zero_keypoints_in_frame(landmarks, frame_idx):
    """
    Identifies the positions of landmarks with all zeros in a specific frame and counts them.

    Args:
        landmarks (numpy.ndarray): A 3D array of shape (frames, landmarks, coordinates).
        frame_idx (int): The index of the frame to analyze.

    Returns:
        tuple:
            - int: Number of zero landmarks in the specified frame.
            - list: Indices of the zero landmarks in the frame.
    """
    if 0 <= frame_idx < landmarks.shape[0]:
        # Identify landmarks with all zeros in the specified frame
        zero_mask = np.all(landmarks[frame_idx] == 0, axis=1)
        zero_indices = np.where(zero_mask)[0].tolist()

        # Count the zero landmarks
        zero_count = len(zero_indices)

        return zero_count, zero_indices
    else:
        raise ValueError(f"Invalid frame index: {frame_idx}. It must be between 0 and {landmarks.shape[0] - 1}.")