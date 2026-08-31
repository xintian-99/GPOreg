import os
import time
import torch
import argparse
import pandas as pd
# from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

from utils import setters
from utils.functions import AverageMeter,  GaussianBlur2D
from utils.loss import GccLoss, NccLoss, Grad2d
from models.gaussianSplatComplex import gaussianSplatComplex
from loaders.firereg_loader import firereg_loader
from utils.gpo_utils import make_coords, sparse_sampling_prep


def run(opt):

    if torch.cuda.is_available():
        gpu_id = int(str(opt.get('gpu_id', '0')).split(',')[0])
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')

    fire_loader = firereg_loader(
        root_dir=opt['datasets_path'],
        split=opt['field_split'],
        geo_num=opt['geo_num'],
        enable_geo_init=bool(opt.get('enable_geo_init', 1)),
    )
    only = opt.get("only_sub", "")
    if only:
        if only not in fire_loader.total_list:
            raise ValueError(
                f"--only_sub={only} is invalid for split={opt['field_split']}. "
                f"Valid examples: {opt['field_split']}01 ... "            )
        idx = fire_loader.total_list.index(only)
        data_iter = [fire_loader[idx]] 
    else:
        data_iter = fire_loader  
        
    df_data = []

    eval_dsc = AverageMeter()
    init_dsc = AverageMeter()
    eval_time = AverageMeter()

    criterion_gcc = GccLoss().to(device)
    criterion_ncc = NccLoss().to(device)
    criterion_g2d = Grad2d(penalty=opt['g2d_penalty']).to(device)

    n_total = len(data_iter)
    run_start = time.time()
    print(f"Run: {setters.make_run_name(opt)}  |  {n_total} pairs, {opt['n_iters']} iters, lr={opt['lr']:g}, device={device}", flush=True)

    for pair_idx, data in enumerate(data_iter, 1):
        sub_id = data[-1]
        if opt.get("only_sub", "") and sub_id != opt["only_sub"]:
            continue
        x = data[0].float().to(device).unsqueeze(0) #_1
        y = data[1].float().to(device).unsqueeze(0) #_2

        x_kpts = data[2].float().to(device).squeeze() #control points of fixed image
        y_kpts = data[3].float().to(device).squeeze() #control points of moving image

        if opt['enable_geo_init']:
            x_geos = data[4].long().to(device).squeeze().float()
            y_geos = data[5].long().to(device).squeeze().float()

            x_geos = x_geos[...,[1,0]] / opt['org_size'][0] * opt['img_size'][0]
            y_geos = y_geos[...,[1,0]] / opt['org_size'][0] * opt['img_size'][0]
        else:
            x_geos, y_geos = None, None

        x_kpts = x_kpts[...,[1,0]] 
        y_kpts = y_kpts[...,[1,0]]

        rate = 1024 // opt['img_size'][0] // 2 # 1024//1024//2 = 0
        blur = GaussianBlur2D(1, max(rate, 1)).to(device) #return blurred
        x, y = blur(x), blur(y)
        x_img = F.interpolate(x, size=opt['img_size'], mode='bilinear', align_corners=True)
        y_img = F.interpolate(y, size=opt['img_size'], mode='bilinear', align_corners=True)

        x_msk, y_msk = (x_img > 0), (y_img > 0)

        coords = make_coords(opt['img_size']).to(device).float() #torch.Size([1048576, 2]) 1024*1024 Each row represents a pixel coordinate (y, x).
        coords.requires_grad = True
        if opt['enable_geo_init']:
            model = gaussianSplatComplex(K=opt['K'], x_geos=x_geos, y_geos=y_geos).to(device)
        else:
            model = gaussianSplatComplex(node_shape=opt['node_shape'], K=opt['K']).to(device)
        
        # params = [
        #     {'params': model.node_position, 'name': 'node_position', 'lr': 1},
        #     {'params': model.translation, 'name': 'translation', 'lr': 0.05},
        #     {'params': model.meta_node_radius, 'name': 'meta_node_radius', 'lr': 0.05},
        # ]
        params = [
            {'params': model.node_position, 'name': 'node_position', 'lr': opt['lrs']['node_position']},
            {'params': model.translation, 'name': 'translation', 'lr': opt['lrs']['translation']},
            {'params': model.meta_node_radius, 'name': 'meta_node_radius', 'lr': opt['lrs']['meta_node_radius']},
        ]
        
        optimizer = torch.optim.Adam(params)
        csv_name = setters.make_run_name(opt) + '.csv'
        loss_history = []
        st = time.time() 
        
        for iter_id in range(opt['n_iters']):
            optimizer.zero_grad()

            coords_ = coords.clone()

            pos_flow = model(coords_)
   
            new_coords, ori_coords = sparse_sampling_prep(opt['img_size'], pos_flow, coords_)

            # pull pixel intensities,encourages structural alignment.
            sampled_x = F.grid_sample(x_img, new_coords, mode='bilinear', align_corners=True).squeeze()
            sampled_y = F.grid_sample(y_img, ori_coords, mode='bilinear', align_corners=True).squeeze()
            loss_gcc = criterion_gcc(sampled_x, sampled_y)

            warped_x = sampled_x.view(opt['img_size']).unsqueeze(0).unsqueeze(0)
            warped_y = sampled_y.view(opt['img_size']).unsqueeze(0).unsqueeze(0)
            loss_ncc = criterion_ncc(warped_x, warped_y)
            
            loss = opt['alpha_gcc'] * loss_gcc + opt['alpha_ncc'] * loss_ncc 
            
            if opt['loss_g2d']:
                dpf = pos_flow.view(1, 2, opt['img_size'][0], opt['img_size'][1])  # (1, 2, H, W)
                loss_g2d = criterion_g2d(dpf)
                loss += opt['alpha_g2d'] * loss_g2d
                # print("Loss G2D:", loss_g2d)

            if opt['save_losses']:
                loss_dict = {
                    'total_loss': loss.item(),
                    'gcc_loss': loss_gcc.item(),
                    'ncc_loss': loss_ncc.item()
                }
                if opt['loss_g2d']:
                    loss_dict['g2d_loss'] = loss_g2d.item()

                # Append total loss only when logging
                if iter_id % opt['loss_log_freq'] == 0:
                    loss_history.append(loss_dict['total_loss'])  
                    setters.save_losses(loss_dict, iter_id, sub_id, opt)
          
            if torch.isnan(loss) or torch.isinf(loss):
            #    print(f"NaN detected in loss at iteration {iter_id}. Skipping this step.")
                continue
                
            loss.backward()
            optimizer.step()

            yps = y_kpts / opt['org_size'][0] * opt['img_size'][0]

            kps_flow = model(yps) / opt['img_size'][0] * opt['org_size'][0]
            warped_kps = y_kpts + kps_flow

            init_tre = torch.mean(torch.norm(y_kpts - x_kpts, dim=-1)).item()
            eval_tre = torch.mean(torch.norm(warped_kps - x_kpts, dim=-1)).item()
            
            # Calculate per-point registration errors
            point_errors = torch.norm(warped_kps - x_kpts, dim=-1)

            thresholds = [5, 10, 15, 25, 50]
            auc_scores = []
            
            for threshold in thresholds:        
                threshold_tensor = torch.arange(1, threshold + 1, device=point_errors.device).unsqueeze(1)
                # print(threshold_tensor)
                gs_error = torch.cumsum((point_errors < threshold_tensor).float(), dim=0)
                gs_error = (gs_error[-1] * 100) / len(point_errors)  
                auc_score = torch.sum(gs_error) / (threshold * 100)
                # Convert to Python float and store
                auc_scores.append(auc_score.item())
            if opt['print_freq'] and ((iter_id + 1) % opt['print_freq'] == 0 or iter_id == 0):
                print(f"    [{sub_id}] iter {iter_id + 1:4d}/{opt['n_iters']}  loss {loss.item():.4f}  TRE {eval_tre:.3f}  AUC@5 {auc_scores[0]:.3f}", flush=True)
            # print(f'AUC scores @ {thresholds}px: {[f"{score:.3f}" for score in auc_scores]} | [%d/%d] sub_idx %s, Eval TRE %.4f, Init TRE %.4f, loss %.4f' % (iter_id, opt['n_iters'], sub_id, eval_tre, init_tre, loss.item()), end='\r')
        et = time.time()
        # print("")
        # print("sub_idx %s, Eval TRE %.4f, Init TRE %.4f, Time %.4f, AUC@[5, 10, 15, 25, 50]: %.3f, %.3f, %.3f,%.3f, %.3f" % 
        #       (sub_id, eval_tre, init_tre, et-st, auc_scores[0], auc_scores[1], auc_scores[2], auc_scores[3], auc_scores[4]))
        init_dsc.update(init_tre)
        eval_dsc.update(eval_tre)
        eval_time.update(et-st)
        print(f"[{pair_idx:2d}/{n_total}] {sub_id}  init TRE {init_tre:6.2f} -> TRE {eval_tre:6.3f}   AUC@5/10/25 {auc_scores[0]:.3f}/{auc_scores[1]:.3f}/{auc_scores[3]:.3f}   {et - st:5.0f}s", flush=True)
        
        max_length = 10  
        point_errors_list = point_errors.tolist()
        point_errors_fixed = point_errors_list[:max_length] + [None] * (max_length - len(point_errors_list))
        loop_df_data = [sub_id, eval_tre, init_tre, et-st, auc_scores[0], auc_scores[1], auc_scores[2], auc_scores[3], auc_scores[4]] + point_errors_fixed

        df_data.append(loop_df_data)
        #save warped images
        if opt['save_images']:
            setters.save_warped_images(warped_x, warped_y, sub_id, opt,csv_name)
            setters.save_warped_rgb(new_coords, sub_id, opt, csv_name)
                  

    avg_df_data = ['Avg', eval_dsc.avg, init_dsc.avg, eval_time.avg, 
               np.mean([d[4] for d in df_data]), np.mean([d[5] for d in df_data]), 
               np.mean([d[6] for d in df_data]), np.mean([d[7] for d in df_data]), np.mean([d[8] for d in df_data])] + [None] * max_length

    df_data.append(avg_df_data)

    fp = os.path.join('results', opt['dataset'])
  
    os.makedirs(fp, exist_ok=True)
    csv_fp = os.path.join(fp, csv_name)
    point_error_columns = [f"point_error_{i+1}" for i in range(max_length)]
    columns = ['sub_id', 'eval_tre', 'init_tre', 'eval_time', 'AUC@5', 'AUC@10', 'AUC@15', 'AUC@25', 'AUC@50'] + point_error_columns
    df = pd.DataFrame(df_data, columns=columns)
    print(f"Saving results to {csv_fp}")
    if opt['save_images']:
        print(f"Warped images saved")
    df.to_csv(csv_fp, index=False)

    # ---- final summary ----
    per = df[df['sub_id'] != 'Avg']
    tre = per['eval_tre'].astype(float)
    print('\n' + '=' * 72)
    print(f"SUMMARY  {setters.make_run_name(opt)}")
    print(f"  pairs: {len(per)}   NaN: {int(tre.isna().sum())}   total time: {(time.time() - run_start) / 60:.1f} min")
    print(f"  TRE  : {np.nanmean(tre):.3f}  (init {np.nanmean(per['init_tre'].astype(float)):.3f})")
    print('  AUC  : ' + '  '.join(f"@{t} {np.nanmean(per[f'AUC@{t}'].astype(float)):.3f}" for t in (5, 10, 15, 25, 50)))
    for cat in ('A', 'P', 'S'):
        m = per['sub_id'].str.startswith(cat)
        if m.any():
            print(f"  {cat}-class ({int(m.sum()):2d}): TRE {np.nanmean(tre[m]):.3f}   AUC@25 {np.nanmean(per.loc[m, 'AUC@25'].astype(float)):.3f}")
    worst = per.assign(worst_tre=tre).sort_values('worst_tre', ascending=False).head(3)
    print('  worst: ' + ', '.join(f"{r.sub_id}={r.worst_tre:.1f}" for r in worst.itertuples()))
    print('=' * 72, flush=True)

if __name__ == '__main__':

    opt = {
        'logs_path': './logs',
        'save_freq': 5,
        'n_checkpoints': 2,
        'spacing': np.array([1.0, 1.0]),  # spacing of the images
    }

    parser = argparse.ArgumentParser(description = "retina")
    parser.add_argument("-bs", "--batch_size", type = int, default = 1)
    parser.add_argument("-d", "--dataset", type = str, default = 'firereg')
    parser.add_argument("--gpu_id", type = str, default = '0')
    parser.add_argument("-dp", "--datasets_path", type = str, default = "./data/fire/")
    parser.add_argument(
    "--field_split",
    type=str,
    default="test",
    choices=["A", "P", "S", "train", "val", "test"],
    help="Subset to run on. Use A/P/S for official FIRE subsets, or train/val/test for convenience splits."
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--img_size", type = str, default = '(1024,1024)')
    parser.add_argument("--org_size", type=str, default='(2912,2912)')
    parser.add_argument("--n_iters", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--sample_size", type=int, default=10000)
    parser.add_argument("--enable_geo_init", type=int, default=1)
    parser.add_argument("--geo_num", type=int, default=1000)
    parser.add_argument("--loss_g2d", type=int, default=1, help="Enable Grad2d loss (1 for True, 0 for False)")
    parser.add_argument("--alpha_gcc", type=float, default=1.0, help="Weight for GCC loss")
    parser.add_argument("--alpha_ncc", type=float, default=1.0, help="Weight for NCC loss")
    parser.add_argument("--alpha_g2d", type=float, default=0.035, help="Weight for Grad2d loss")
    parser.add_argument("--g2d_penalty", type=str, default="l1", choices=["l1", "l2"], help="Grad2d penalty: l1 (TV-like, edge-preserving) or l2 (diffusion, smooths discontinuities)")
    parser.add_argument("--node_shape", type=int, default=30, help="Number of nodes per dimension in the Gaussian Splat model")
    parser.add_argument("--K", type=int, default=30, help="Number of Gaussian components")
    parser.add_argument("--save_losses", type=int, default=1, help="Whether to save losses (1 for True, 0 for False)")
    parser.add_argument("--loss_log_freq", type=int, default=1, help="Frequency of logging losses")
    parser.add_argument("--save_images", type=int, default=0, help="Set to 1 to save warped images, 0 to disable")
    parser.add_argument("--print_freq", type=int, default=50, help="Print loss/TRE every N iterations (0 = only per-pair lines)")
    parser.add_argument("--only_sub", type=str, default="", help="Run only one sample id, e.g., P01 or S12, --only_sub must belong to the same --field_split.")


    args, unknowns = parser.parse_known_args()
    opt = {**opt, **vars(args)}
    print(unknowns)
    opt['nkwargs'] = {s.split('=')[0]:s.split('=')[1] for s in unknowns}
    opt['nkwargs']['img_size'] = opt['img_size']
    opt['img_size'] = eval(opt['img_size'])
    opt['org_size'] = eval(opt['org_size'])

    opt['lrs'] = {
        'node_position': 100 * opt['lr'],
        'meta_node_radius': opt['lr'],
        'translation': opt['lr'],
    }

    run(opt)
