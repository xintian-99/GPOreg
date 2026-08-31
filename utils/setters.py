import os
import torch
import random
import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

def format_alpha(a):
    """Format a loss weight for use in file names without losing precision.

    Old scheme int(a*100) collapsed everything below 0.01 to '00'
    (0.001 and 0.0 got identical names). '%g' keeps the exact value:
    0 -> '0', 0.001 -> '0.001', 0.02 -> '0.02', 1.0 -> '1'.
    """
    return "%g" % float(a)

def make_run_name(opt):
    """Single source of truth for the experiment file name.

    Used for the results CSV, the loss log and the warped-image folder,
    so the three can never disagree. alpha_g2d is reported as 0 when the
    g2d loss is disabled, so the name always reflects the effective weight.
    """
    eff_ag2d = float(opt.get('alpha_g2d', 0)) if int(opt.get('loss_g2d', 0)) else 0.0
    # L1 runs keep the historical name; L2 runs get an extra token so they never collide
    pen = '_l2' if (int(opt.get('loss_g2d', 0)) and opt.get('g2d_penalty', 'l1') == 'l2') else ''
    return 'ag2d%s%s_anc%s_agc%s_k%d_geo%d_ns%d_gn%d_iter%d_%s' % (
        format_alpha(eff_ag2d), pen,
        format_alpha(opt.get('alpha_ncc', 0)),
        format_alpha(opt.get('alpha_gcc', 0)),
        int(opt.get('K', 0)), int(opt.get('enable_geo_init', 0)),
        int(opt.get('node_shape', 0)), int(opt.get('geo_num', 0)),
        int(opt.get('n_iters', 0)), opt.get('field_split', 'Unknown')
    )

def setGPU(opt):

    os.environ["CUDA_VISIBLE_DEVICES"] = opt['gpu_id']
    if not torch.cuda.is_available():
        print("No GPU found, using CPU ...")
    else:
        print("----->>>> GPU %s is set up ..." % opt['gpu_id'])

def setFoldersLoggers(opt, split_id = None):

    opt['data_path'] = os.path.join(opt['datasets_path'], opt['dataset'])
    opt['log'] = os.path.join(opt['logs_path'], opt['dataset'], opt['model'])

    os.makedirs(opt['log'], exist_ok = True)

    print("----->>>> Log path: %s" % opt['log'])
    print("----->>>> Data set path: %s" % opt['data_path'])

def setSeed(seed=0):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def save_losses(loss_dict, iter_id, sub_id, opt):
    """
    Saves loss values to a structured CSV file and generates a loss plot.

    Parameters:
        loss_dict (dict): Dictionary containing loss values.
        iter_id (int): Current iteration.
        sub_id (str): Subject ID.
        opt (dict): Experiment options containing hyperparameters.
        loss_history (list): List to store loss values for plotting.
    """

    if not isinstance(opt, dict):
        raise TypeError(f"Expected 'opt' to be a dictionary, but got {type(opt)} instead.")

    try:
        # Base filename (no sub_id for CSV) - shared with the results CSV name
        file_base = make_run_name(opt)

        # Define log directory and ensure it exists
        log_dir = os.path.join("logs", "losses")
        os.makedirs(log_dir, exist_ok=True)

        # Full path for CSV and plot file
        csv_fp = os.path.join(log_dir, f"{file_base}.csv")
        plot_fp = os.path.join(log_dir, f"{file_base}{sub_id}.png")  # sub_id for per-image plots

        # Check if CSV exists
        file_exists = os.path.isfile(csv_fp)

        # Save loss values in CSV
        with open(csv_fp, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['iter_id', 'sub_id'] + list(loss_dict.keys()))

            if not file_exists:
                writer.writeheader()  # Write headers only if the file is new

            writer.writerow({'iter_id': iter_id, 'sub_id': sub_id, **loss_dict})


    except Exception as e:
        print(f"Error saving losses: {e}")

def save_warped_rgb(new_coords, sub_id, opt, csv_name):
    """Warp the ORIGINAL RGB fundus photo with the final flow and save it
    without any colormap, alongside the viridis grayscale version.

    new_coords: the (1,1,N,2) normalized sampling grid from the last iteration
    (same one used to produce warped_x)."""
    import torch
    from torch.nn import functional as F

    fp = os.path.join(opt['datasets_path'], 'img', f"{sub_id}_1.jpg")
    rgb = Image.open(fp).convert('RGB')
    t = torch.from_numpy(np.array(rgb)).permute(2, 0, 1).float().unsqueeze(0) / 255.
    t = t.to(new_coords.device)
    t = F.interpolate(t, size=tuple(opt['img_size']), mode='bilinear', align_corners=True)
    with torch.no_grad():
        w = F.grid_sample(t, new_coords.detach(), mode='bilinear', align_corners=True)
    w = w.reshape(3, opt['img_size'][0], opt['img_size'][1])
    arr = (w.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    csv_name_no_ext = os.path.splitext(csv_name)[0]
    save_dir = os.path.join("results", "warped_img", csv_name_no_ext)
    os.makedirs(save_dir, exist_ok=True)
    Image.fromarray(arr).save(os.path.join(save_dir, f"{sub_id}_1_warped_rgb.png"))

def save_warped_images(warped_x, warped_y, sub_id, opt, csv_name):
    """
    Save the warped images as color-enhanced PNG files inside 'warped_images/csv_name/'.

    Parameters:
    - warped_x: Tensor, the warped fixed image (grayscale).
    - warped_y: Tensor, the warped moving image (grayscale).
    - sub_id: The subject ID for file naming.
    - opt: Dictionary of options containing paths and configurations.
    - csv_name: The name of the CSV file (used to create a subfolder).
    """
    # Convert to CPU and detach for saving
    warped_x_np = warped_x.detach().cpu().numpy()
    warped_y_np = warped_y.detach().cpu().numpy()

    # Normalize to [0, 1] range for visualization
    warped_x_np = (warped_x_np - np.min(warped_x_np)) / (np.max(warped_x_np) - np.min(warped_x_np) + 1e-7)
    warped_y_np = (warped_y_np - np.min(warped_y_np)) / (np.max(warped_y_np) - np.min(warped_y_np) + 1e-7)

    # Use a colormap to "recolor" the grayscale image
    colormap = cm.viridis  # Options: 'jet', 'plasma', 'magma', 'coolwarm', etc.

    # Apply the colormap
    warped_x_colored = colormap(warped_x_np.squeeze())  # RGBA output
    warped_y_colored = colormap(warped_y_np.squeeze())

    # Convert to RGB (remove alpha channel)
    warped_x_colored = (warped_x_colored[..., :3] * 255).astype(np.uint8)
    warped_y_colored = (warped_y_colored[..., :3] * 255).astype(np.uint8)

    # Convert numpy arrays to PIL Images
    warped_x_img = Image.fromarray(warped_x_colored)
    warped_y_img = Image.fromarray(warped_y_colored)

    # **Remove .csv extension from csv_name**
    csv_name_no_ext = os.path.splitext(csv_name)[0]  # Removes ".csv"

    # **Define save directory with CSV name as subfolder**
    save_dir = os.path.join("results", "warped_img", csv_name_no_ext)
    os.makedirs(save_dir, exist_ok=True)

    # File paths
    x_filename = os.path.join(save_dir, f"{sub_id}_1_warped.png") 
    # y_filename = os.path.join(save_dir, f"{sub_id}_2.png")

    # Save the images
    warped_x_img.save(x_filename)
    # warped_y_img.save(y_filename)