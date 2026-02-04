"""Save vertex normals for ScanNet scenes as .pth files (automated version)."""
import argparse
import glob
import multiprocessing as mp
import os
import sys
import traceback
from tqdm import tqdm

import numpy as np
import open3d as o3d
import torch

def face_normal(vertex, face):
    """Calculate face normals and areas."""
    v01 = vertex[face[:, 1]] - vertex[face[:, 0]]
    v02 = vertex[face[:, 2]] - vertex[face[:, 0]]
    vec = np.cross(v01, v02)
    length = np.sqrt(np.sum(vec**2, axis=1, keepdims=True)) + 1e-8
    nf = vec / length
    area = length * 0.5
    return nf, area

def vertex_normal(vertex, face):
    """Calculate vertex normals by area-weighted averaging."""
    nf, area = face_normal(vertex, face)
    nf = nf * area
    
    nv = np.zeros_like(vertex)
    for i in range(face.shape[0]):
        nv[face[i]] += nf[i]
    
    length = np.sqrt(np.sum(nv**2, axis=1, keepdims=True)) + 1e-8
    return nv / length

def process_scene(scene_dir, output_dir):
    """Process single scene and save normals."""
    try:
        scene_id = os.path.basename(scene_dir)
        ply_file = os.path.join(scene_dir, f"{scene_id}_vh_clean_2.ply")
        output_path = os.path.join(output_dir, f"{scene_id}_normals.pth")
        
        if not os.path.exists(ply_file):
            print(f"Warning: PLY file not found in {scene_dir}")
            return
            
        if os.path.exists(output_path):
            # print(f"Skipping existing: {output_path}")
            return

        # Read mesh data
        mesh = o3d.io.read_triangle_mesh(ply_file)
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        
        # Calculate normals
        normals = vertex_normal(vertices[:, :3], faces)  # Use XYZ coordinates
        
        # Save as pytorch tensor
        torch.save(torch.from_numpy(normals.astype(np.float32)), output_path)
        # print(f"Saved: {output_path}")
        return True
    
    except Exception as e:
        print(f"\nError processing {scene_dir}: {str(e)}")
        traceback.print_exc()
        return False

def process_dataset(data_dir, output_dir, num_workers=8):
    """Process all scenes in a dataset directory (scans or scans_test)."""
    if not os.path.exists(data_dir):
        print(f"Dataset directory not found: {data_dir}")
        return 0
    
    # Find all scene directories
    scene_dirs = glob.glob(os.path.join(data_dir, "scene*"))
    print(f"Found {len(scene_dirs)} scenes in {data_dir}")
    
    if not scene_dirs:
        return 0
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process in parallel with progress tracking
    success_count = 0
    with mp.Pool(num_workers) as pool:
        results = []
        for scene_dir in scene_dirs:
            results.append(pool.apply_async(process_scene, (scene_dir, output_dir)))
        
        # Create progress bar
        pbar = tqdm(total=len(scene_dirs), desc=f"Processing {os.path.basename(data_dir)}")
        for res in results:
            if res.get():
                success_count += 1
            pbar.update(1)
        pbar.close()
    
    print(f"Processed {success_count}/{len(scene_dirs)} scenes successfully")
    return success_count

def main():
    
    DEFAULT_OUTPUT_DIR = "./scannet_normals"
    
    parser = argparse.ArgumentParser(
        description="Calculate and save vertex normals for ScanNet scenes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                       help="Output directory for normal vectors")
    parser.add_argument("--num_workers", type=int, default=8,
                       help="Number of parallel workers")
    parser.add_argument(
        '--train_scannet_dir', default='scans', help='scannet data directory.')
    parser.add_argument(
        '--test_scannet_dir',
        default='scans_test',
        help='scannet data directory.')
    args = parser.parse_args()
    print(f"Output directory: {args.output_dir}")
    
    # 处理训练/验证集 (scans)
    train_dir = args.train_scannet_dir
    train_output = args.output_dir
    train_count = process_dataset(train_dir, train_output, args.num_workers)
    
    # 处理测试集 (scans_test)
    test_dir = args.test_scannet_dir
    test_output = args.output_dir
    test_count = process_dataset(test_dir, test_output, args.num_workers)
    
    print("\n" + "="*50)
    print(f"Processing summary:")
    print(f"  Train/val scenes processed: {train_count}")
    print(f"  Test scenes processed: {test_count}")
    print(f"Total scenes processed: {train_count + test_count}")
    print("="*50)

if __name__ == "__main__":
    main()