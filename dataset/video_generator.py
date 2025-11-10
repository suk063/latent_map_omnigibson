import os
import numpy as np
import imageio
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

def create_rgb_video(data_dir, output_path, fps=30, start_frame=0, num_frames=None):
    """
    Creates a video from a sequence of RGB images.
    """
    rgb_dir = os.path.join(data_dir, 'rgb')
    if not os.path.isdir(rgb_dir):
        print(f"RGB directory not found: {rgb_dir}")
        return

    image_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
    
    # Slice the files based on start_frame and num_frames
    image_files = image_files[start_frame:]
    if num_frames is not None:
        image_files = image_files[:num_frames]

    if not image_files:
        print(f"No PNG files found in {rgb_dir}.")
        return

    # Create a video writer
    writer = imageio.get_writer(output_path, fps=fps)

    for image_file in tqdm(image_files, desc="Generating RGB video"):
        file_path = os.path.join(rgb_dir, image_file)
        try:
            image = imageio.imread(file_path)
            writer.append_data(image)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    writer.close()
    print(f"RGB video saved to: {output_path}")

def create_segmentation_video(data_dir, output_path, fps=30, start_frame=0, num_frames=None):
    """
    Creates a video from a sequence of instance segmentation masks.
    """
    seg_dir = os.path.join(data_dir, 'seg_instance_id')
    if not os.path.isdir(seg_dir):
        print(f"Segmentation directory not found: {seg_dir}")
        return

    seg_files = sorted([f for f in os.listdir(seg_dir) if f.endswith('.npy')])

    # Slice the files based on start_frame and num_frames
    seg_files = seg_files[start_frame:]
    if num_frames is not None:
        seg_files = seg_files[:num_frames]

    if not seg_files:
        print(f"No NPY files found in {seg_dir}.")
        return

    # Find all unique instance IDs across all frames
    unique_ids = set([0]) # Assuming background ID is 0
    for seg_file in tqdm(seg_files, desc="Scanning for unique IDs"):
        file_path = os.path.join(seg_dir, seg_file)
        seg_map = np.load(file_path)
        unique_ids.update(np.unique(seg_map))

    # Create a color map for each ID
    sorted_ids = sorted(list(unique_ids))
    
    # Use 'tab20' colormap to generate a variety of colors
    cmap = plt.cm.get_cmap('tab20', 20)
    colors = [cmap(i) for i in range(20)]
    
    color_map = {}
    for i, id_val in enumerate(sorted_ids):
        if id_val == 0:
            color_map[id_val] = (0, 0, 0)  # Black for background
        else:
            # Cycle through the colors
            color_idx = (i - 1) % len(colors)
            color_map[id_val] = (np.array(colors[color_idx][:3]) * 255).astype(np.uint8)

    # Create a video writer
    writer = imageio.get_writer(output_path, fps=fps)

    for seg_file in tqdm(seg_files, desc="Generating segmentation video"):
        file_path = os.path.join(seg_dir, seg_file)
        seg_map = np.load(file_path)
        
        # Create an RGB image from the segmentation map
        h, w = seg_map.shape
        rgb_seg = np.zeros((h, w, 3), dtype=np.uint8)
        
        for id_val, color in color_map.items():
            rgb_seg[seg_map == id_val] = color
            
        writer.append_data(rgb_seg)
        
    writer.close()
    print(f"Segmentation video saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate videos from the processed BEHAVIOR dataset.")
    parser.add_argument(
        "--data_folder",
        type=str,
        default="/home/sunghwan/workspace/omnigibson/DATASETS/behavior",
        help="Path to the root dataset folder."
    )
    parser.add_argument("--task_id", type=int, required=True, help="Task ID.")
    parser.add_argument("--demo_id", type=int, required=True, help="Demo ID.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the output video.")
    parser.add_argument("--start_frame", type=int, default=0, help="Starting frame number.")
    parser.add_argument("--num_frames", type=int, default=None, help="Number of frames to use for the video. If not specified, all frames are used.")
    args = parser.parse_args()

    episode_dir = os.path.join(args.data_folder, "processed_data", f"task-{args.task_id:04d}", f"episode_{args.demo_id:08d}")
    
    if not os.path.isdir(episode_dir):
        print(f"Episode directory not found: {episode_dir}")
        return

    print(f"Starting video generation for {episode_dir}")

    # Generate RGB video
    rgb_video_path = os.path.join(episode_dir, "rgb_video.mp4")
    create_rgb_video(episode_dir, rgb_video_path, args.fps, start_frame=args.start_frame, num_frames=args.num_frames)
    
    # Generate segmentation video
    seg_video_path = os.path.join(episode_dir, "segmentation_video.mp4")
    create_segmentation_video(episode_dir, seg_video_path, args.fps, start_frame=args.start_frame, num_frames=args.num_frames)

if __name__ == "__main__":
    main()
