"""PA3 evaluation: per-node transition (switching) probability prediction.

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


GATE_NAME = {0: 'PI', 1: 'AND', 2: 'NOT'}


def evaluate(args):
    args.batch_size = 1
    args.num_workers = 0

    dataset = MLPGateDataset(args.data_dir, args)

    model = create_model(args)
    device_str = 'cuda' if args.gpus[0] >= 0 else 'cpu'
    model = load_model(model, args.load_model, device=device_str)
    model = model.to(args.device)
    model.eval()

    abs_err_model = []
    abs_err_analytic = []
    abs_err_prob = []
    per_type_err_model = {0: [], 1: [], 2: []}

    n_circuits = 0
    with torch.no_grad():
        for g in dataset:
            n_circuits += 1
            g = g.to(args.device)
            preds, _max_sim = model(g)
            # preds = (hs, hf, prob, trans, is_rc)
            if len(preds) == 5:
                _hs, _hf, prob_pred, trans_pred, _is_rc = preds
            else:
                # legacy 4-tuple: no trans head
                _hs, _hf, prob_pred, _is_rc = preds
                trans_pred = None

            prob_pred_np  = prob_pred.detach().cpu().squeeze().numpy()
            prob_target   = g.prob.detach().cpu().squeeze().numpy()
            trans_target  = g.trans_prob.detach().cpu().squeeze().numpy()
            gate          = g.gate.detach().cpu().squeeze().numpy().astype(int)

            prob_pred_c = np.clip(prob_pred_np, 0.0, 1.0)
            e_prob      = np.abs(prob_pred_c - prob_target)
            e_analytic  = np.abs(2.0 * prob_pred_c * (1.0 - prob_pred_c) - trans_target)
            abs_err_prob.append(e_prob)
            abs_err_analytic.append(e_analytic)

            if trans_pred is not None:
                trans_pred_np = np.clip(trans_pred.detach().cpu().squeeze().numpy(), 0.0, 1.0)
                e_model = np.abs(trans_pred_np - trans_target)
                abs_err_model.append(e_model)
                for gt in (0, 1, 2):
                    mask = (gate == gt)
                    if mask.any():
                        per_type_err_model[gt].append(e_model[mask])

    all_prob     = np.concatenate(abs_err_prob)
    all_analytic = np.concatenate(abs_err_analytic)

    print('======================================================================')
    print('  PA3 SWITCHING-PROBABILITY EVALUATION')
    print('  Checkpoint    : {}'.format(args.load_model))
    print('  Data dir      : {}'.format(args.data_dir))
    print('  Label file    : {}'.format(args.label_file))
    print('  Circuits used : {}'.format(n_circuits))
    print('  Total nodes   : {}'.format(len(all_prob)))
    print('======================================================================')
    print('  L1(prob_pred, prob_target)        = {:.6f}'.format(all_prob.mean()))
    print('  L1(2p(1-p)_pred, trans_target)    = {:.6f}   <-- analytic baseline'.format(all_analytic.mean()))

    if abs_err_model:
        all_model = np.concatenate(abs_err_model)
        gap = all_analytic.mean() - all_model.mean()
        print('  L1(trans_pred, trans_target)      = {:.6f}   <-- model'.format(all_model.mean()))
        print('  Gap (analytic - model)            = {:+.6f}   '
              '({})'.format(gap, '+ model beats baseline' if gap > 0 else '- model worse than baseline'))
        print('')
        print('  Per-gate-type L1 (model):')
        print('    type  |   N nodes  |   L1 trans')
        print('    ------|------------|-----------')
        for gt in (0, 1, 2):
            if per_type_err_model[gt]:
                arr = np.concatenate(per_type_err_model[gt])
                print('    {:5s} | {:10d} | {:.6f}'.format(GATE_NAME[gt], arr.size, arr.mean()))
            else:
                print('    {:5s} | {:>10s} | {:>9s}'.format(GATE_NAME[gt], '0', 'N/A'))
    else:
        print('')
        print('  [WARN] No transition predictions — was the model trained with --Trans_weight > 0?')


def main():
    args = get_parse_args()
    args.local_rank = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus_str
    args.device = torch.device('cuda:0' if args.gpus[0] >= 0 else 'cpu')

    if not args.load_model:
        print('[ERROR] --load_model is required.', file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.load_model):
        print('[ERROR] Checkpoint not found: {}'.format(args.load_model), file=sys.stderr)
        sys.exit(1)

    evaluate(args)


if __name__ == '__main__':
    main()
