"""PA2-3 evaluation: per-node transition (switching) probability prediction.

Loads a trained checkpoint and reports, on the full dataset:
  - L1 of the model's transition prediction vs the simulator-derived target.
  - L1 of the analytic baseline 2*p*(1-p) (with p = predicted signal prob)
    vs the same target. This baseline is exact under i.i.d. uniform stimulus
    but biased under Markov stimulus, so the gap quantifies what the GNN
    learns beyond the closed form.
  - Per-gate-type breakdown (PI=0, AND=1, NOT=2) of L1 errors.

Deliberately avoids detector_factory / base_detector to keep cv2 out.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import torch
import numpy as np

from config import get_parse_args
from datasets.mlpgate_dataset import MLPGateDataset
from models.model import create_model, load_model

# Map integer gate-type codes to human-readable names for display
GATE_NAME = {0: 'PI', 1: 'AND', 2: 'NOT'}


def evaluate(args):
    # Force batch size 1 and disable multiprocessing workers because the eval
    # loop processes one circuit at a time and the per-gate-type breakdown must
    # keep circuits separate rather than batching them together.
    args.batch_size = 1
    args.num_workers = 0

    # MLPGateDataset loads all circuits from data_dir into memory as a list,
    # so we iterate over it directly without a DataLoader.
    dataset = MLPGateDataset(args.data_dir, args)

    # Build the model architecture and restore weights from the saved checkpoint.
    # The model must have been trained with --Trans_weight > 0 to include a
    # transition readout head; otherwise trans_pred will be None and only the
    # analytic baseline will be reported.
    model = create_model(args)
    device_str = 'cuda' if args.gpus[0] >= 0 else 'cpu'
    model = load_model(model, args.load_model, device=device_str)
    model = model.to(args.device)
    model.eval()

    # Accumulators for per-node absolute errors across all circuits:
    #   abs_err_model    – errors from the GNN transition head
    #   abs_err_analytic – errors from the closed-form baseline 2*p*(1-p)
    #   abs_err_prob     – errors of the raw signal-probability prediction
    #   per_type_err_model – model errors split by gate type (PI / AND / NOT)
    abs_err_model = []
    abs_err_analytic = []
    abs_err_prob = []
    per_type_err_model = {0: [], 1: [], 2: []}

    n_circuits = 0
    with torch.no_grad():
        for g in dataset:
            n_circuits += 1
            g = g.to(args.device)

            # Forward pass; preds is a tuple whose length depends on whether
            # the model was trained with a transition head.
            preds, _max_sim = model(g)

            # Unpack predictions: (hs, hf, prob, trans, is_rc) for new models
            # or (hs, hf, prob, is_rc) for legacy checkpoints without trans head.
            if len(preds) == 5:
                _hs, _hf, prob_pred, trans_pred, _is_rc = preds
            else:
                # Legacy 4-tuple: transition head was not trained
                _hs, _hf, prob_pred, _is_rc = preds
                trans_pred = None

            # Move tensors to CPU numpy arrays; squeeze removes the batch dim.
            prob_pred_np  = prob_pred.detach().cpu().squeeze().numpy()
            prob_target   = g.prob.detach().cpu().squeeze().numpy()
            trans_target  = g.trans_prob.detach().cpu().squeeze().numpy()
            gate          = g.gate.detach().cpu().squeeze().numpy().astype(int)

            # Clip predicted probabilities to [0, 1] before computing errors.
            prob_pred_c = np.clip(prob_pred_np, 0.0, 1.0)

            # L1 error for signal-probability prediction
            e_prob      = np.abs(prob_pred_c - prob_target)

            # Analytic baseline: under i.i.d. uniform stimulus the switching
            # probability equals 2*p*(1-p).  Any model improvement over this
            # baseline indicates the GNN captures temporal/structural correlations.
            e_analytic  = np.abs(2.0 * prob_pred_c * (1.0 - prob_pred_c) - trans_target)

            abs_err_prob.append(e_prob)
            abs_err_analytic.append(e_analytic)

            # Compute model transition errors only when the head is present.
            if trans_pred is not None:
                trans_pred_np = np.clip(trans_pred.detach().cpu().squeeze().numpy(), 0.0, 1.0)
                e_model = np.abs(trans_pred_np - trans_target)
                abs_err_model.append(e_model)

                # Collect errors for each gate type to enable the per-type breakdown.
                for gt in (0, 1, 2):
                    mask = (gate == gt)
                    if mask.any():
                        per_type_err_model[gt].append(e_model[mask])

    # Concatenate per-circuit error arrays into flat arrays over all nodes.
    all_prob     = np.concatenate(abs_err_prob)
    all_analytic = np.concatenate(abs_err_analytic)

    # Print summary header
    print('======================================================================')
    print('  P2-3 SWITCHING-PROBABILITY EVALUATION')
    print('  Checkpoint    : {}'.format(args.load_model))
    print('  Data dir      : {}'.format(args.data_dir))
    print('  Label file    : {}'.format(args.label_file))
    print('  Circuits used : {}'.format(n_circuits))
    print('  Total nodes   : {}'.format(len(all_prob)))
    print('======================================================================')

    # Report signal-probability and analytic-baseline errors
    print('  L1(prob_pred, prob_target)        = {:.6f}'.format(all_prob.mean()))
    print('  L1(2p(1-p)_pred, trans_target)    = {:.6f}   <-- analytic baseline'.format(all_analytic.mean()))

    if abs_err_model:
        # Report model transition errors and how much they improve over the baseline
        all_model = np.concatenate(abs_err_model)
        gap = all_analytic.mean() - all_model.mean()
        print('  L1(trans_pred, trans_target)      = {:.6f}   <-- model'.format(all_model.mean()))
        print('  Gap (analytic - model)            = {:+.6f}   '
              '({})'.format(gap, '+ model beats baseline' if gap > 0 else '- model worse than baseline'))
        print('')

        # Per-gate-type breakdown so we can see which gate types drive the error
        print('  Per-gate-type L1 (model):')
        print('    type  |   N nodes  |   L1 trans')
        print('    ------|------------|-----------')
        for gt in (0, 1, 2):
            if per_type_err_model[gt]:
                arr = np.concatenate(per_type_err_model[gt])
                print('    {:5s} | {:10d} | {:.6f}'.format(GATE_NAME[gt], arr.size, arr.mean()))
            else:
                # Gate type absent in this dataset split
                print('    {:5s} | {:>10s} | {:>9s}'.format(GATE_NAME[gt], '0', 'N/A'))
    else:
        # The checkpoint has no transition head — only baseline numbers are valid.
        print('')
        print('  [WARN] No transition predictions — was the model trained with --Trans_weight > 0?')


def main():
    # Parse command-line arguments defined in config.py
    args = get_parse_args()
    args.local_rank = 0

    # Expose only the requested GPUs to PyTorch via the environment variable
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus_str
    args.device = torch.device('cuda:0' if args.gpus[0] >= 0 else 'cpu')

    # Validate required arguments before doing any expensive work
    if not args.load_model:
        print('[ERROR] --load_model is required.', file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.load_model):
        print('[ERROR] Checkpoint not found: {}'.format(args.load_model), file=sys.stderr)
        sys.exit(1)

    evaluate(args)


if __name__ == '__main__':
    main()
