# def debug_print(name, tensor):
#     print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, "
#           f"device={tensor.device}, "
#           f"min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, "
#           f"mean={tensor.mean().item():.4f}, "
#           f"has_nan={tensor.isnan().any().item()}")

# def training_step(model, batch, criterion, optimizer):
#     inputs, labels = batch
#     outputs = model(inputs)
#     loss = criterion(outputs, labels)

#     if loss.item() > 100 or torch.isnan(loss):
#         breakpoint()

#     loss.backward()
#     optimizer.step()

# import logging

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[
#         logging.FileHandler("training.log"),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# logger.info("Starting training: lr=%.4f, batch_size=%d", lr, batch_size)
# logger.warning("Loss spike detected: %.4f at step %d", loss.item(), step)
# logger.error("NaN loss at step %d, stopping", step)

#cpu memory with tracemalloc:
# import tracemalloc

# tracemalloc.start()

# # your code here
# model = build_model()
# data = load_dataset()

# snapshot = tracemalloc.take_snapshot()
# top_stats = snapshot.statistics("lineno")
# for stat in top_stats[:10]:
#     print(stat)

#mapear toda transformação do modelo:
# def check_shapes(model, sample_input):
#     print(f"Input: {sample_input.shape}")
#     hooks = []

#     def make_hook(name):
#         def hook(module, inp, out):
#             in_shape = inp[0].shape if isinstance(inp, tuple) else inp.shape
#             out_shape = out.shape if hasattr(out, "shape") else type(out)
#             print(f"  {name}: {in_shape} -> {out_shape}")
#         return hook

#     for name, module in model.named_modules():
#         hooks.append(module.register_forward_hook(make_hook(name)))

#     with torch.no_grad():
#         model(sample_input)

#     for h in hooks:
#         h.remove()

#detecar NaN
# def detect_nan(model, loss, step):
#     if torch.isnan(loss):
#         print(f"NaN loss at step {step}")
#         for name, param in model.named_parameters():
#             if param.grad is not None:
#                 if torch.isnan(param.grad).any():
#                     print(f"  NaN gradient in {name}")
#                 if torch.isinf(param.grad).any():
#                     print(f"  Inf gradient in {name}")
#         return True
#     return False

#CHECAR DATA LEAKAGE
def check_data_leakage(train_set, test_set, id_column='id'):
    train_ids = set(train_set[id_column])
    test_ids = set(test_set[id_column])
    overlap = train_ids.intersection(test_ids)
    if overlap:
        print(f'Data leakage detected: {len(overlap)} overLapping IDs found: {overlap}')
        return True
    return False