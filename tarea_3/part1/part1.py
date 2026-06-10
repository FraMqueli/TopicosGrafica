from plyfile import PlyData
import pycolmap
import numpy as np
from PIL import Image
from pathlib import Path
import json
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--method", type=str, default="center")
parser.add_argument("--ply", type=str, required=True)
parser.add_argument("--colmap_dir", type=str, required=True)
parser.add_argument("--masks_dir", type=str, required=True)
parser.add_argument("--tau_low", type=float, default=0.3)
parser.add_argument("--tau_high", type=float, default=0.7)
parser.add_argument("--output", type=str, default="output.json")
parser.add_argument("--extent_iter", type=int, default=20)
parser.add_argument("--extent_threshold", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=0)

args = parser.parse_args()

if args.seed != 0:
  np.random.default_rng(seed=args.seed)

def get_gaussian_center(gaussian):
  return np.array([gaussian['x'], gaussian['y'], gaussian['z']])

# Mostly AI generated
def sample_gaussian(gaussian, num_samples):
  # Get center
  mean = get_gaussian_center(gaussian)
  
  # Extract scale (exp is used because 3DGS stores scales in log space)
  scale = np.array([np.exp(gaussian['scale_0']), np.exp(gaussian['scale_1']), np.exp(gaussian['scale_2'])])
  
  # Extract rotation (quaternion)
  q = np.array([gaussian['rot_0'], gaussian['rot_1'], gaussian['rot_2'], gaussian['rot_3']])
  
  # Compute the Rotation Matrix from the Quaternion
  R = np.array([
    [1 - 2*q[2]**2 - 2*q[3]**2, 2*q[1]*q[2] - 2*q[0]*q[3], 2*q[1]*q[3] + 2*q[0]*q[2]],
    [2*q[1]*q[2] + 2*q[0]*q[3], 1 - 2*q[1]**2 - 2*q[3]**2, 2*q[2]*q[3] - 2*q[0]*q[1]],
    [2*q[1]*q[3] - 2*q[0]*q[2], 2*q[2]*q[3] + 2*q[0]*q[1], 1 - 2*q[1]**2 - 2*q[2]**2]
  ])
  
  # Compute the Covariance Matrix
  S = np.diag(scale)
  covariance = R @ S @ S.T @ R.T
  
  # Sample points
  sampled_points = np.random.multivariate_normal(mean, covariance, num_samples)
  
  return sampled_points

def check_if_points_in_mask(points, mask: Image, image: pycolmap.Image, method: str) -> bool:
  included = 0
  camera_points = image.cam_from_world() * points
  image_points = image.camera.img_from_cam(camera_points)

  for i in range(len(points)):
    if camera_points[i][2] <= 0:
      continue
    row, column = image_points[i]
    row = round(row)
    column = round(column)
    if 0 <= row < image.camera.width and 0 <= column < image.camera.height:
      pixel = mask.getpixel((row, column))
      if pixel != 0:
        included += 1
  
  if included/len(points) >= args.extent_threshold or (included == 1 and method == "center") :
    return True



def get_image_path_from_mask(mask_path: str):
  return mask_path.replace("_mask", "", 1).replace(".png", ".JPG", 1)

plydata = PlyData.read(args.ply)
rec = pycolmap.Reconstruction()
rec.read(str(args.colmap_dir))

masks_path = Path(args.masks_dir)
masks = [f.name for f in masks_path.iterdir()]
masks.remove("masks_summary.json")

labels = []
scores = []

mask_image_pairs = []

for mask_name in masks:
  mask = Image.open(str(masks_path / mask_name))
  image_path = get_image_path_from_mask(mask_name)
  image = rec.find_image_with_name(image_path)
  if image is None:
    print(f"Error! Image {image_path} not found")
  else:
    mask_image_pairs.append((mask, image))

print(f"Number of gaussians: {len(plydata["vertex"])}")
progress = 0
p_progress = 0

for gaussian in list(plydata["vertex"]):
  included = 0
  progress += 1
  if(progress/len(plydata["vertex"]) * 100 > p_progress + 1):
    p_progress += 1
    print(f"Avance: {p_progress}%")
  

  points = None
  if args.method == "center":
    points = [get_gaussian_center(gaussian)]

  elif args.method == "extent-aware":
    points = sample_gaussian(gaussian, args.extent_iter)

  else:
    raise ValueError(f"Unknown method: {args.method}")

  for mask, image in mask_image_pairs:
    if check_if_points_in_mask(points, mask, image, args.method):
      included += 1
  
  score = included / len(mask_image_pairs)
  label = 0
  if score < args.tau_low:
    label = 1
  elif score < args.tau_high:
    label = 4
  else:
    label = 2
  
  scores.append(score)
  labels.append(label)

data = {
  "method": args.method,
  "parameters": {"tau_low": args.tau_low, "tau_high": args.tau_high},
  "labels": labels,
  "scores": scores
}

with open(args.output, "w") as file:
  json.dump(data, file)
  

  
  