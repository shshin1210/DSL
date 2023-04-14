import torch

def compute_hyp(arg, illum, cam_crf, I, batch_size):
    A = illum * cam_crf
    A = A.reshape(-1, batch_size * arg.num_train_px_per_iter, arg.wvl_num).permute(1,0,2)
    
    list_A = list(A)
    block = torch.block_diag(*list_A)

    x = torch.linalg.lstsq(block, I)
    x = x.solution
    
    return x

def diff_hyp(arg , x, x_gt, batch_size):
    diff = abs(x-x_gt).sum() / (arg.wvl_num * arg.num_train_px_per_iter * batch_size)
    
    return diff

def recon_large(arg, A,b):
    """
        Reconstruction for 
    """
    cam_W, cam_H = arg.cam_W, arg.cam_H
    N = arg.illum_num
    W = arg.wvl_num
    M = cam_H*cam_W
    
    batch_size = 200000
    num_iter = 2000
    
    num_batches = int(torch.ceil(M / batch_size))
    loss_f = torch.nn.L1Loss()
    losses = []
    X_np_all = torch.zeros(M, 1, W, 1)

    # define initial learning rate and decay step
    lr = 0.5
    decay_step = 500

    # training loop over batches
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, M)
        batch_size_ = end_idx - start_idx
        A_batch = torch.from_numpy(A[start_idx:end_idx]).to(arg.device)
        B_batch = torch.from_numpy(b[start_idx:end_idx]).to(arg.device)
        X_est = torch.randn(batch_size_, 1, W, 1, requires_grad=True, device=arg.device)
        optimizer = torch.optim.Adam([X_est], lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=0.5)

        optimizer.zero_grad()
        for i in range(num_iter):
            loss = loss_f(A_batch @ X_est, B_batch)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if i % 100 == 0:
                print(f"Batch {batch_idx + 1}/{num_batches}, Iteration {i}/{num_iter}, Loss: {loss.item()}, LR: {optimizer.param_groups[0]['lr']}")

        X_np_all[start_idx:end_idx] = X_est.detach().cpu()

    X_np_all = X_np_all.numpy()
