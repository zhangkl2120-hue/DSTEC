import os
import numpy as np
from time import time
import torch
import torch.utils.data
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from .metrics import masked_mape_np
from scipy.sparse.linalg import eigs
from .metrics import masked_mape_np,  masked_mae,masked_mse,masked_rmse,masked_mae_test,masked_rmse_test


def re_normalization(x, mean, std):
    mean = np.array(mean)
    std = np.array(std)

    while mean.ndim > x.ndim:
        mean = mean.squeeze(0)
        std = std.squeeze(0)

    return x * std + mean



def max_min_normalization(x, _max, _min):
    x = 1. * (x - _min)/(_max - _min)
    x = x * 2. - 1.
    return x


def re_max_min_normalization(x, _max, _min):
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x

def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)

        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)

        if id_filename:

            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
            return A, distaneA

        else:

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distaneA[i, j] = distance
            return A, distaneA

def scaled_Laplacian(W):
    
    W = W.astype(np.float32)
    assert W.shape[0] == W.shape[1]

    d = np.sum(W, axis=1).astype(np.float32)
    d = np.where(d < 1e-6, 1e-6, d)
    D = np.diag(d)

    L = D - W

    try:
        lambda_max = eigs(L, k=1, which='LR', return_eigenvectors=False)[0].real
    except Exception:
        lambda_max = np.max(np.abs(np.linalg.eigvals(L)))

    if (lambda_max < 1e-6) or np.isnan(lambda_max) or np.isinf(lambda_max):
        lambda_max = 1.0

    L_tilde = (2.0 * L) / float(lambda_max) - np.eye(W.shape[0], dtype=np.float32)
    return L_tilde.astype(np.float32)



def cheb_polynomial(L_tilde, K):

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials




def load_graphdata_channel1(graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks, DEVICE, batch_size, shuffle=True, anomaly_cfg=None,use_rdw=False):

    file = os.path.basename(graph_signal_matrix_filename).split('.')[0]

    dirpath = os.path.dirname(graph_signal_matrix_filename)

    filename = os.path.join(dirpath,
                            file + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks)) +'_astcgn'

    print('load file:', filename)

    file_data = np.load(filename + '.npz')
    train_x = file_data['train_x']
    train_target = file_data['train_target']

    val_x = file_data['val_x']
    val_target = file_data['val_target']

    test_x = file_data['test_x']
    test_target = file_data['test_target']

    if not use_rdw:
        train_x = train_x[:, :, 0:1, :]
        val_x = val_x[:, :, 0:1, :]
        test_x = test_x[:, :, 0:1, :]

    mean = file_data['mean'][:, :, 0:1, :]
    std = file_data['std'][:, :, 0:1, :]
    



    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(DEVICE)
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(DEVICE)

    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor).to(DEVICE)
    val_target_tensor = torch.from_numpy(val_target).type(torch.FloatTensor).to(DEVICE)

    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_target_tensor)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor).to(DEVICE)
    test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).to(DEVICE)

    test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    return train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, mean, std

def compute_val_loss_mstgcn(net, val_loader, criterion, masked_flag, missing_value, sw, epoch):
    net.eval()
    with torch.no_grad():
        tmp = []
        for batch_index, batch_data in enumerate(val_loader):
            encoder_inputs, labels = batch_data

            outputs = net(encoder_inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[2]

            if masked_flag:
                loss = criterion(outputs, labels, missing_value)
            else:
                loss = criterion(outputs, labels)

            tmp.append(loss.item())

        validation_loss = sum(tmp) / len(tmp)
        print('validation loss: ', validation_loss)
        sw.add_scalar('validation_loss', validation_loss, epoch)
    return validation_loss





def predict_and_save_results_mstgcn(net, data_loader, data_target_tensor, global_step,
                                    metric_method, _mean, _std, params_path, type):

    net.train(False)

    with torch.no_grad():
        data_target_tensor = data_target_tensor.cpu().numpy()
        loader_length = len(data_loader)

        prediction = []
        input = []

        infer_total = 0.0
        num_batches = 0
        num_samples = 0

        for batch_index, batch_data in enumerate(data_loader):
            encoder_inputs, labels = batch_data
            input.append(encoder_inputs[:, :, 0:1].cpu().numpy())

            batch_size_cur = encoder_inputs.shape[0]

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time()

            outputs = net(encoder_inputs)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            infer_total += (time() - t0)

            num_batches += 1
            num_samples += batch_size_cur

            if isinstance(outputs, tuple):
                outputs = outputs[2]

            prediction.append(outputs.detach().cpu().numpy())

            if batch_index % 100 == 0:
                print('predicting data set batch %s / %s' % (batch_index + 1, loader_length))

        print(f"[EFFICIENCY] pure_inference_time: {infer_total:.6f}s")
        print(f"[EFFICIENCY] avg_inference_time_per_batch: {infer_total / max(num_batches, 1):.6f}s")
        print(f"[EFFICIENCY] avg_inference_time_per_sample: {infer_total / max(num_samples, 1):.8f}s")
        print(f"[EFFICIENCY] num_batches: {num_batches}, num_samples: {num_samples}")

        input = np.concatenate(input, 0)
        input = re_normalization(input, _mean, _std)

        prediction = np.concatenate(prediction, 0)
        prediction = re_normalization(prediction, _mean, _std)
        prediction = np.clip(prediction, 0.0, None)

        print('input:', input.shape)
        print('prediction:', prediction.shape)
        print("target min/max/mean:", data_target_tensor.min(), data_target_tensor.max(), data_target_tensor.mean())
        print("pred   min/max/mean:", prediction.min(), prediction.max(), prediction.mean())

        print('data_target_tensor:', data_target_tensor.shape)
        output_filename = os.path.join(params_path, 'output_epoch_%s_%s' % (global_step, type))
        np.savez(output_filename, input=input, prediction=prediction, data_target_tensor=data_target_tensor)

        excel_list = []
        prediction_length = prediction.shape[2]

        for i in range(prediction_length):
            assert data_target_tensor.shape[0] == prediction.shape[0]
            print('current epoch: %s, predict %s points' % (global_step, i))
            if metric_method == 'mask':
                mae = masked_mae_test(data_target_tensor[:, :, i], prediction[:, :, i], 0.0)
                rmse = masked_rmse_test(data_target_tensor[:, :, i], prediction[:, :, i], 0.0)
                mape = masked_mape_np(data_target_tensor[:, :, i], prediction[:, :, i], 0)
            else:
                mae = mean_absolute_error(data_target_tensor[:, :, i], prediction[:, :, i])
                rmse = mean_squared_error(data_target_tensor[:, :, i], prediction[:, :, i]) ** 0.5
                mape = masked_mape_np(data_target_tensor[:, :, i], prediction[:, :, i], 0)
            print('MAE: %.2f' % (mae))
            print('RMSE: %.2f' % (rmse))
            print('MAPE: %.2f' % (mape))
            excel_list.extend([mae, rmse, mape])

        if metric_method == 'mask':
            mae = masked_mae_test(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0.0)
            rmse = masked_rmse_test(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0.0)
            mape = masked_mape_np(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)
        else:
            mae = mean_absolute_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1))
            rmse = mean_squared_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1)) ** 0.5
            mape = masked_mape_np(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)

        print('all MAE: %.2f' % (mae))
        print('all RMSE: %.2f' % (rmse))
        print('all MAPE: %.2f' % (mape))
        excel_list.extend([mae, rmse, mape])
        print(excel_list)
