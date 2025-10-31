import itertools, time, torch
from copy import deepcopy
from codes.models.paper_global_transformer import SpatiotemporalTransformer


def count_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def measure_throughput(model, batch, runs=60, warmup=10):
    device = next(model.parameters()).device
    model.eval(); torch.cuda.synchronize()
    for _ in range(warmup):
        _ = model(**batch)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(runs):
        _ = model(**batch)
    torch.cuda.synchronize()
    dt = time.time() - t0
    return runs / dt

def one_epoch(model, loader, optimizer, device, loss_fn):
    model.train()
    for xb in loader:
        # EXPECT xb to provide everything your model.forward needs
        # e.g., xb = {'x_seq':..., 'edge_index':..., 'edge_weight':..., 'node_features':...}
        for k,v in xb.items():
            if torch.is_tensor(v): xb[k] = v.to(device)
        y_true = xb.pop('y_true')  # <-- make sure your loader packs targets here
        y_pred = model(**xb)
        loss = loss_fn(y_pred, y_true)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def eval_metric(model, val_loader, device, metric_fn):
    model.eval()
    metrics = []
    for xb in val_loader:
        for k,v in xb.items():
            if torch.is_tensor(v): xb[k] = v.to(device)
        y_true = xb.pop('y_true')
        y_pred = model(**xb)
        metrics.append(metric_fn(y_pred, y_true))
    return float(torch.stack([m if torch.is_tensor(m) else torch.tensor(m) for m in metrics]).mean())

def sweep_spatiotemporal_transformer(
    device,
    train_loader, val_loader,
    base_cfg,  # dict with fixed fields: input_dim, gcn_dim, num_nodes, forecast_dim
    arch_grid, # dict of lists: hidden_dim, nhead_per_hidden, num_layers, attn_dropout, ff_dropout, gat_heads
    opt_grid,  # dict of lists: lr, weight_decay
    metric_fn, # your SMAPE/RMSE/etc. (return scalar tensor)
    loss_fn,   # training loss (e.g., MSELoss)
    epochs=10, # short runs; you only need relative ranking
):
    results = []
    # Prebuild a representative batch for throughput (from first val batch)
    sample_batch = next(iter(val_loader))
    for k,v in sample_batch.items():
        if torch.is_tensor(v): sample_batch[k] = v.to(device)

    for hidden_dim in arch_grid['hidden_dim']:
        # choose heads that divide d_model
        valid_heads = [h for h in arch_grid['nhead_per_hidden'][hidden_dim] if hidden_dim % h == 0]
        for nhead, num_layers, attn_do, ff_do, gat_heads in itertools.product(
            valid_heads, arch_grid['num_layers'], arch_grid['attn_dropout'],
            arch_grid['ff_dropout'], arch_grid['gat_heads']
        ):
            # Build model
            model = SpatiotemporalTransformer(
                input_dim=base_cfg['input_dim'],
                gcn_dim=base_cfg['gcn_dim'],
                hidden_dim=hidden_dim,
                nhead=nhead,
                num_layers=num_layers,
                num_nodes=base_cfg['num_nodes'],
                forecast_dim=base_cfg.get('forecast_dim', 1),
                attn_dropout=attn_do,
                ff_dropout=ff_do,
                gat_heads=gat_heads
            ).to(device)

            for lr, wd in itertools.product(opt_grid['lr'], opt_grid['weight_decay']):
                # Fresh copy each opt setting
                m = deepcopy(model)
                opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)

                # Train a few epochs & time them
                torch.cuda.reset_peak_memory_stats(device)
                t0 = time.time()
                for ep in range(epochs):
                    one_epoch(m, train_loader, opt, device, loss_fn)
                torch.cuda.synchronize()
                epoch_time_total = time.time() - t0
                time_per_epoch = epoch_time_total / epochs
                peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

                # Val metric + throughput
                val_score = eval_metric(m, val_loader, device, metric_fn)
                thr = measure_throughput(m, sample_batch, runs=60, warmup=10)
                params_m = count_params(m) / 1e6

                results.append({
                    'hidden_dim': hidden_dim,
                    'nhead': nhead,
                    'num_layers': num_layers,
                    'attn_dropout': attn_do,
                    'ff_dropout': ff_do,
                    'gat_heads': gat_heads,
                    'lr': lr,
                    'weight_decay': wd,
                    'params_M': round(params_m, 3),
                    'throughput_sps': round(thr, 2),
                    'time_per_epoch_s': round(time_per_epoch, 2),
                    'peak_mem_GB': round(peak_gb, 2),
                    'val_metric': float(val_score),
                })
                # OPTIONAL: early discard if clearly dominated (prune)
                # e.g., if time_per_epoch very large and val_metric not improving

    # Sort best by your main metric (lower is better if RMSE/SMAPE; flip sign if higher is better)
    results.sort(key=lambda r: r['val_metric'])
    return results
