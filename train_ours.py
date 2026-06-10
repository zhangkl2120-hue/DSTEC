import warnings
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning
)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
from time import time
import shutil
import argparse
import configparser
import copy
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from ours import make_model
from lib.utils import load_graphdata_channel1, get_adjacency_matrix, compute_val_loss_mstgcn, predict_and_save_results_mstgcn
from tensorboardX import SummaryWriter
from lib.metrics import masked_mape_np,  masked_mae,masked_mse,masked_rmse


parser = argparse.ArgumentParser()

parser.add_argument("--as_path", type=str, default=None)
parser.add_argument("--at_path", type=str, default=None)

parser.add_argument("--config", default='configurations/METR_LA_astgcn.conf', type=str,
                    help="configuration file path")
parser.add_argument("--gpu", type=str, default=None)
parser.add_argument("--ablate_time", action="store_true")
parser.add_argument("--ablate_space", action="store_true")
parser.add_argument("--ablate_guidance", action="store_true")
parser.add_argument("--bank_path", type=str, default=None)

args = parser.parse_args()
if args.ablate_time and args.ablate_space:
    raise ValueError("--ablate_time and --ablate_space cannot be enabled together.")

ABLATION_ACTIVE = args.ablate_time or args.ablate_space
config = configparser.ConfigParser()
config.read(args.config)
data_config = config['Data']
training_config = config['Training']


adj_filename = data_config['adj_filename']
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
if config.has_option('Data', 'id_filename'):
    id_filename = data_config['id_filename']
else:
    id_filename = None

num_of_vertices = int(data_config['num_of_vertices'])
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])
len_input = int(data_config['len_input'])
dataset_name = data_config['dataset_name']

model_name = training_config['model_name']

ctx = training_config['ctx']
os.environ["CUDA_VISIBLE_DEVICES"] = ctx
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("CUDA:", USE_CUDA, DEVICE)

learning_rate = float(training_config['learning_rate'])
weight_decay = float(training_config.get('weight_decay'))
patience = int(training_config.get('patience'))
clip_grad = float(training_config.get('clip_grad', fallback='5.0'))

lr_factor = float(training_config.get('lr_factor', fallback='0.8'))
lr_patience = int(training_config.get('lr_patience', fallback='10'))
lr_threshold = float(training_config.get('lr_threshold', fallback='1e-3'))
lr_cooldown = int(training_config.get('lr_cooldown', fallback='2'))
lr_min = float(training_config.get('lr_min', fallback='1e-5'))

teacher_update_interval = int(training_config.get('teacher_update_interval', fallback='50'))
lambda_guide = float(training_config.get('lambda_guide', fallback='0.1'))
tau_rho = float(training_config.get('tau_rho', fallback='0.5'))

lambda_aux = float(training_config.get('lambda_aux', fallback='0.2'))

rho_delta = float(training_config.get('rho_delta', fallback='0.1'))
rho_M = int(training_config.get('rho_M', fallback='5'))
rho_max_batches = int(training_config.get('rho_max_batches', fallback='30'))

finetune_start = int(training_config.get('finetune_start', fallback='100'))

tw_start = float(training_config.get('time_weight_start', fallback='1.0'))
tw_end   = float(training_config.get('time_weight_end', fallback='2.0'))

exp_tag = training_config.get('exp_tag', fallback='').strip()

epochs = int(training_config['epochs'])
start_epoch = int(training_config['start_epoch'])
batch_size = int(training_config['batch_size'])

num_of_weeks = int(training_config['num_of_weeks'])
num_of_days = int(training_config['num_of_days'])
num_of_hours = int(training_config['num_of_hours'])
time_strides = num_of_hours
nb_time_filter = int(training_config['nb_time_filter'])
in_channels = int(training_config['in_channels'])
nb_block = int(training_config['nb_block'])
loss_function = training_config['loss_function']
metric_method = training_config['metric_method']
missing_value = float(training_config['missing_value'])

folder_dir =  'OURS_'+('%s_h%dd%dw%d_channel%d_%e' % (model_name, num_of_hours, num_of_days, num_of_weeks, in_channels, learning_rate))
if exp_tag:
    folder_dir = folder_dir + "_" + exp_tag
params_path = os.path.join('experiments', dataset_name, folder_dir)

train_loader, train_target_tensor, val_loader, val_target_tensor,test_loader, test_target_tensor, _mean, _std = load_graphdata_channel1(
    graph_signal_matrix_filename,
    num_of_hours, num_of_days, num_of_weeks,
    DEVICE, batch_size,
)

adj_mx, distance_mx = get_adjacency_matrix(adj_filename, num_of_vertices, id_filename)

root_dir = os.path.dirname(graph_signal_matrix_filename)
base_name = os.path.basename(graph_signal_matrix_filename)
dataset_prefix = os.path.splitext(base_name)[0]

if args.as_path is not None and os.path.exists(args.as_path):
    print("Use external As from:", args.as_path)
    As_space = np.load(args.as_path)
else:
    space_adj_path = os.path.join(root_dir, dataset_prefix + "_As_ours.npy")
    if os.path.exists(space_adj_path):
        As_space = np.load(space_adj_path)
    else:
        As_space = adj_mx
def normalize_adj(A, eps=1e-8):
    A = A.copy().astype(np.float32)
    A[A < 0] = 0
    max_val = A.max()
    if max_val < eps:
        return A
    return A / max_val


def compute_selective_guidance_loss(Y_student, Y_teacher):
    return F.smooth_l1_loss(Y_student, Y_teacher.detach())


def select_final_prediction(Yt, Ys, Y):
    if args.ablate_time:
        return Ys
    if args.ablate_space:
        return Yt
    return Y


def weighted_smooth_l1(pred, target, criterion_robust, time_weights, mask=None):
    loss_element = criterion_robust(pred, target)
    if mask is not None:
        return torch.mean(loss_element * mask * time_weights)
    return torch.mean(loss_element * time_weights)


class PredictionSelector(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, *model_args, **model_kwargs):
        outputs = self.base_model(*model_args, **model_kwargs)
        if isinstance(outputs, tuple):
            return select_final_prediction(*outputs[:3])
        return outputs


@torch.no_grad()
def compute_expert_stability(net, loader, num_nodes, delta=0.1, M=5,
                             max_batches=30, DEVICE="cuda"):
    net.eval()
    delta_t_acc = torch.zeros(num_nodes, device=DEVICE)
    delta_s_acc = torch.zeros(num_nodes, device=DEVICE)
    cnt = 0

    for b, (x, _) in enumerate(loader):
        if b >= max_batches:
            break

        out = net(x)
        Ht_clean = out[3]
        Hs_clean = out[4]

        sq_t = torch.zeros_like(Ht_clean)
        sq_s = torch.zeros_like(Hs_clean)

        for _ in range(M):
            x_noisy = x + delta * torch.randn_like(x)
            out_n = net(x_noisy)
            Ht_noisy = out_n[3]
            Hs_noisy = out_n[4]
            sq_t += (Ht_noisy - Ht_clean).pow(2)
            sq_s += (Hs_noisy - Hs_clean).pow(2)

        sq_t = sq_t / M
        sq_s = sq_s / M

        delta_t_acc += sq_t.mean(dim=(0, 2))
        delta_s_acc += sq_s.mean(dim=(0, 2))
        cnt += 1

    delta_t = delta_t_acc / max(cnt, 1)
    delta_s = delta_s_acc / max(cnt, 1)

    d_t = (delta_t - delta_t.min()) / (delta_t.max() - delta_t.min() + 1e-6)
    d_s = (delta_s - delta_s.min()) / (delta_s.max() - delta_s.min() + 1e-6)
    rho_t = torch.exp(-d_t)
    rho_s = torch.exp(-d_s)

    rho_t_scalar = rho_t.mean().item()
    rho_s_scalar = rho_s.mean().item()

    return rho_t, rho_s, rho_t_scalar, rho_s_scalar



net = make_model(
    DEVICE, nb_block, in_channels,
    nb_time_filter, time_strides,
    num_for_predict, len_input, num_of_vertices,
    As_space,
    pattern_bank_path=args.bank_path,
    pattern_top_m=5,
    pattern_tau=0.2
)


def compute_val_loss_ours(net, val_loader, criterion, masked_flag, missing_value, sw, epoch, _mean, _std):
    net.eval()
    tmp = []

    with torch.no_grad():
        for batch_index, batch_data in enumerate(val_loader):
            encoder_inputs, labels = batch_data

            out = net(encoder_inputs)
            outputs = out

            if isinstance(out, tuple):
                outputs = select_final_prediction(*out[:3])

            mean_tensor = torch.as_tensor(_mean, device=DEVICE, dtype=outputs.dtype).squeeze().view(1, 1, 1)
            std_tensor = torch.as_tensor(_std, device=DEVICE, dtype=outputs.dtype).squeeze().view(1, 1, 1).clamp_min(1e-6)
            outputs_raw = outputs * std_tensor + mean_tensor

            if masked_flag:
                loss = criterion(outputs_raw, labels, missing_value)
            else:
                loss = criterion(outputs_raw, labels)

            tmp.append(loss.item())

    validation_loss = sum(tmp) / len(tmp)
    print('validation loss: ', validation_loss)
    if sw:
        sw.add_scalar('validation_loss', validation_loss, epoch)
    return validation_loss


def train_main():
    if (start_epoch == 0) and (not os.path.exists(params_path)):
        os.makedirs(params_path)
    elif (start_epoch == 0) and (os.path.exists(params_path)):
        shutil.rmtree(params_path)
        os.makedirs(params_path)
    elif (start_epoch > 0) and (os.path.exists(params_path)):
        print('train from params directory %s' % (params_path))
    else:
        raise SystemExit('Wrong type of model!')

    masked_flag = 0
    criterion = nn.L1Loss().to(DEVICE)
    criterion_masked = masked_mae
    if loss_function == 'masked_mse':
        criterion_masked = masked_mse
        masked_flag = 1
    elif loss_function == 'masked_mae':
        criterion_masked = masked_mae
        masked_flag = 1
    elif loss_function == 'mae':
        criterion = nn.L1Loss().to(DEVICE)
        masked_flag = 0
    elif loss_function == 'rmse':
        criterion = nn.MSELoss().to(DEVICE)
        masked_flag = 0

    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=lr_factor, patience=lr_patience,
        threshold=lr_threshold, threshold_mode='rel', cooldown=lr_cooldown,
        min_lr=lr_min, verbose=True
    )

    K = max(1, epochs // teacher_update_interval)
    print(f"[DSTEC] Stages: K={K}, teacher_update_interval={teacher_update_interval}, "
          f"lambda_guide={lambda_guide}, lambda_aux={lambda_aux}, tau_rho={tau_rho}")
    if ABLATION_ACTIVE:
        mode = "ablate_time" if args.ablate_time else "ablate_space"
        print(f"[DSTEC] Ablation mode: {mode}; guidance disabled.")

    reference_net = copy.deepcopy(net)
    reference_net.eval()
    for p in reference_net.parameters():
        p.requires_grad_(False)

    sw = SummaryWriter(logdir=params_path, flush_secs=5)

    best_path = os.path.join(params_path, "best.params")
    stage_best_path = os.path.join(params_path, "stage_best.params")

    global_step = 0
    best_epoch = 0
    best_val_loss = np.inf
    start_time = time()
    no_improve = 0

    stage_best_val = float("inf")
    stage_best_epoch = -1
    current_stage = 0

    selected_teacher = None
    rho_t_scalar = 0.0
    rho_s_scalar = 0.0
    guidance_active = False

    if start_epoch > 0:
        params_filename = os.path.join(params_path, 'epoch_%s.params' % start_epoch)
        net.load_state_dict(torch.load(params_filename))
        reference_net.load_state_dict(net.state_dict())
        reference_net.eval()
        for p in reference_net.parameters():
            p.requires_grad_(False)
        print('start epoch:', start_epoch)

    for epoch in range(start_epoch, epochs):
        epoch_main = 0.0
        epoch_aux = 0.0
        epoch_time_aux = 0.0
        epoch_space_aux = 0.0
        epoch_guide = 0.0
        cnt = 0
        epoch_total = 0.0

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_train_start = time()

        mean_tensor = torch.as_tensor(_mean, device=DEVICE).squeeze().view(1, 1, 1)
        std_tensor = torch.as_tensor(_std, device=DEVICE).squeeze().view(1, 1, 1).clamp_min(1e-6)

        if epoch > 0 and epoch % teacher_update_interval == 0:
            current_stage = epoch // teacher_update_interval

            if (not args.ablate_guidance) and (not ABLATION_ACTIVE):
                ref_ckpt = os.path.join(params_path, f"epoch_{best_epoch}.params")
                if os.path.exists(ref_ckpt):
                    print(f"[DSTEC Stage {current_stage}] Load reference from epoch {best_epoch}")
                    reference_net.load_state_dict(torch.load(ref_ckpt, map_location=DEVICE))
                    reference_net.eval()
                    for p in reference_net.parameters():
                        p.requires_grad_(False)

                print(f"[DSTEC Stage {current_stage}] Computing per-expert stability...")
                rho_t, rho_s, rho_t_scalar, rho_s_scalar = compute_expert_stability(
                    reference_net, train_loader, num_of_vertices,
                    delta=rho_delta, M=rho_M, max_batches=rho_max_batches, DEVICE=DEVICE
                )

                if rho_t_scalar >= rho_s_scalar:
                    selected_teacher = 'T'
                    teacher_rho = rho_t_scalar
                else:
                    selected_teacher = 'S'
                    teacher_rho = rho_s_scalar

                guidance_active = (teacher_rho > tau_rho)

                print(f"[DSTEC Stage {current_stage}] ρ_T={rho_t_scalar:.4f}, "
                      f"ρ_S={rho_s_scalar:.4f}, teacher={selected_teacher}, "
                      f"ρ_teacher={teacher_rho:.4f}, τ_ρ={tau_rho}, "
                      f"guidance={'ON' if guidance_active else 'OFF'}")

            else:
                selected_teacher = None
                guidance_active = False

            stage_best_val = float("inf")
            stage_best_epoch = -1

        freeze = (epoch >= finetune_start)
        net.t_graph.set_freeze(freeze)
        net.noise_layer.set_enabled(not freeze)

        net.train()
        for batch_index, batch_data in enumerate(train_loader):
            encoder_inputs, labels = batch_data
            optimizer.zero_grad()

            out = net(encoder_inputs)
            Yt, Ys, Y, Ht, Hs = out
            outputs = select_final_prediction(Yt, Ys, Y)

            labels_norm = (labels - mean_tensor) / std_tensor

            T_out = outputs.shape[-1]
            time_weights = torch.linspace(tw_start, tw_end, steps=T_out, device=DEVICE).view(1, 1, T_out)
            criterion_robust = nn.SmoothL1Loss(reduction='none')
            mask = None
            if masked_flag:
                mask = (labels != 0).float()
                mask /= (mask.mean() + 1e-6)

            main_loss = weighted_smooth_l1(outputs, labels_norm, criterion_robust, time_weights, mask)
            loss_t = torch.tensor(0.0, device=DEVICE)
            loss_s = torch.tensor(0.0, device=DEVICE)
            if not args.ablate_time:
                loss_t = weighted_smooth_l1(Yt, labels_norm, criterion_robust, time_weights, mask)
            if not args.ablate_space:
                loss_s = weighted_smooth_l1(Ys, labels_norm, criterion_robust, time_weights, mask)
            aux_loss = loss_t + loss_s

            guide_loss = torch.tensor(0.0, device=DEVICE)

            if (not args.ablate_guidance) and (not ABLATION_ACTIVE) and guidance_active and selected_teacher is not None:
                if selected_teacher == 'T':
                    guide_loss = compute_selective_guidance_loss(Ys, Yt)
                else:
                    guide_loss = compute_selective_guidance_loss(Yt, Ys)

            loss = main_loss + lambda_aux * aux_loss + lambda_guide * guide_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=clip_grad)
            optimizer.step()

            global_step += 1

            training_loss = loss.item()
            sw.add_scalar('loss/total', training_loss, global_step)
            sw.add_scalar('loss/main', main_loss.item(), global_step)
            sw.add_scalar('loss/time_aux', loss_t.item(), global_step)
            sw.add_scalar('loss/space_aux', loss_s.item(), global_step)
            sw.add_scalar('loss/aux', aux_loss.item(), global_step)
            sw.add_scalar('loss/guide', guide_loss.item(), global_step)

            epoch_total += training_loss
            epoch_main += main_loss.item()
            epoch_aux += aux_loss.item()
            epoch_time_aux += loss_t.item()
            epoch_space_aux += loss_s.item()
            epoch_guide += guide_loss.item()
            cnt += 1

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_train_time = time() - epoch_train_start

        avg_train_total = epoch_total / max(cnt, 1)
        avg_train_main = epoch_main / max(cnt, 1)
        avg_train_aux = epoch_aux / max(cnt, 1)
        avg_train_time_aux = epoch_time_aux / max(cnt, 1)
        avg_train_space_aux = epoch_space_aux / max(cnt, 1)
        avg_train_guide = epoch_guide / max(cnt, 1)

        print(f"[EPOCH {epoch}] total={avg_train_total:.4f} main={avg_train_main:.4f} "
              f"aux={avg_train_aux:.4f} (T={avg_train_time_aux:.4f}, S={avg_train_space_aux:.4f}) "
              f"guide={avg_train_guide:.4f} time={epoch_train_time:.2f}s "
              f"teacher={selected_teacher} lr={optimizer.param_groups[0]['lr']:.2e}")

        if masked_flag:
            val_loss = compute_val_loss_ours(net, val_loader, criterion_masked,
                                              masked_flag, missing_value, sw, epoch, _mean, _std)
        else:
            val_loss = compute_val_loss_ours(net, val_loader, criterion,
                                              masked_flag, missing_value, sw, epoch, _mean, _std)

        scheduler.step(val_loss)

        params_filename = os.path.join(params_path, f'epoch_{epoch}.params')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save(net.state_dict(), params_filename)
            torch.save(net.state_dict(), best_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch}, best epoch={best_epoch}, '
                      f'best val={best_val_loss:.4f}')
                break

        if val_loss < stage_best_val:
            stage_best_val = val_loss
            stage_best_epoch = epoch
            torch.save(net.state_dict(), stage_best_path)

    print(f"HPO_BEST_VAL={best_val_loss:.6f}")
    print('best epoch:', best_epoch)

    predict_main(best_epoch, test_loader, test_target_tensor, metric_method, _mean, _std, 'test')


def predict_main(global_step, data_loader, data_target_tensor,metric_method, _mean, _std, type):

    epoch_path = os.path.join(params_path, f'epoch_{global_step}.params')
    best_path = os.path.join(params_path, "best.params")

    load_path = best_path if os.path.exists(best_path) else epoch_path
    print('load weight from:', load_path)

    net.load_state_dict(torch.load(load_path, map_location=DEVICE))

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    eval_t0 = time()

    eval_net = PredictionSelector(net).to(DEVICE)
    predict_and_save_results_mstgcn(
        eval_net, data_loader, data_target_tensor, global_step,
        metric_method, _mean, _std, params_path, type
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    eval_total = time() - eval_t0
    print(f"[EFFICIENCY] total_evaluation_time: {eval_total:.4f}s")
if __name__ == "__main__":

    train_main()
