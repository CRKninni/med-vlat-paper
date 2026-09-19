#!/usr/bin/env python3
"""MED-VLAT VQA-RAD dual-route merge: FGGF open + closed Y/N specialist + FGGF choice."""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_enhanced import compute_split_accuracy, is_correct_answer

_CLOSED_STARTS = {'is', 'are', 'does', 'do', 'was', 'were', 'can', 'has', 'have', 'did', 'will', 'any'}


def pre_question(question):
    """Question normalization so image+question keys align across branches."""
    q = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        str(question).lower(),
    ).replace(' \t', ' ').replace('is/are', 'is').replace('near/in', 'in')
    q = q.replace('>', 'more than ').replace('-yes/no', '')
    q = q.replace('x ray', 'xray').replace('x-ray', 'xray')
    return q.rstrip(' ')


def is_binary_yesno_question(question):
    q = pre_question(question)
    q = re.sub(r'^[^a-z]+', '', q.strip())
    if re.search(r'\bor\b', q) and not q.startswith(('what', 'where', 'which', 'how', 'name', 'list')):
        return False
    if re.search(r'\bvs\.?\b', q):
        return False
    tokens = q.split()
    if tokens and tokens[0] in _CLOSED_STARTS:
        return True
    if q.startswith('the patient'):
        return True
    return False


def preds_by_qid(path):
    with open(path) as f:
        return {int(r['question_id']): r for r in json.load(f)}


def closed_yn_by_key(path):
    out = {}
    with open(path) as f:
        rows = json.load(f)
    for r in rows:
        image = str(r.get('image_name', r.get('image', ''))).strip().lower()
        key = (image, pre_question(r['question']))
        out[key] = r
    return out


def _pred_from_row(row):
    if row is None:
        return ''
    return str(row.get('answer', row.get('pred', ''))).strip()


def pick_closed(ann, closed_yn_row, choice_row):
    if is_binary_yesno_question(ann['question']):
        pred = _pred_from_row(closed_yn_row)
        if pred:
            return pred, 'closed_yn'
        if choice_row is not None:
            return _pred_from_row(choice_row), 'choice_fallback'
        return '', 'none'
    if choice_row is not None:
        return _pred_from_row(choice_row), 'fggf_choice'
    pred = _pred_from_row(closed_yn_row)
    if pred:
        return pred, 'closed_yn_fallback'
    return '', 'none'


def merge_split(split_json, open_preds, closed_yn_preds, choice_preds):
    open_by_qid = preds_by_qid(open_preds)
    closed_yn_by_q = closed_yn_by_key(closed_yn_preds)
    choice_by_qid = preds_by_qid(choice_preds)
    merged = []
    for ann in json.load(open(split_json)):
        qid = int(ann['question_id'])
        key = (ann['image'].strip().lower(), pre_question(ann['question']))
        atype = str(ann.get('answer_type', '')).lower()
        gt = ann.get('answer', [])
        if not isinstance(gt, list):
            gt = [gt]

        if atype == 'open':
            pred = _pred_from_row(open_by_qid.get(qid))
            source = 'fggf_open'
        else:
            pred, source = pick_closed(ann, closed_yn_by_q.get(key), choice_by_qid.get(qid))

        pred_n = pred.lower().strip()
        gt_n = [str(a).lower().strip() for a in gt]
        merged.append({
            'question_id': qid,
            'answer': pred,
            'correct': is_correct_answer(pred_n, gt_n),
            'answer_type': atype,
            'source': source,
            'ground_truth': gt,
        })
    return merged


def print_results(title, merged):
    overall, open_acc, closed_acc, oc, ot, cc, ct = compute_split_accuracy(merged)
    print('=' * 70)
    print(title)
    print('=' * 70)
    print(f'Overall: {overall:.2f}% ({oc + cc:.0f}/{ot + ct})')
    print(f'OPEN:    {open_acc:.2f}% ({oc:.0f}/{ot})  [FGGF]')
    print(f'CLOSED:  {closed_acc:.2f}% ({cc:.0f}/{ct})  [MED-VLAT Y/N + FGGF choice]')
    sources = {}
    for r in merged:
        if r['answer_type'] == 'closed':
            sources[r['source']] = sources.get(r['source'], 0) + 1
    if sources:
        print(f'CLOSED routing: {sources}')
    print('=' * 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_json', required=True)
    ap.add_argument('--open_preds', required=True)
    ap.add_argument('--closed_yn_preds', default=None, help='Closed yes/no branch predictions')
    ap.add_argument('--mvcm_preds', default=None, help='Deprecated alias for --closed_yn_preds')
    ap.add_argument('--choice_preds', required=True)
    ap.add_argument('--out_json', default=None)
    args = ap.parse_args()
    closed_yn = args.closed_yn_preds or args.mvcm_preds
    if not closed_yn:
        ap.error('Provide --closed_yn_preds (or deprecated --mvcm_preds)')

    merged = merge_split(
        args.test_json, args.open_preds, closed_yn, args.choice_preds,
    )
    print_results('MED-VLAT VQA-RAD DUAL-ROUTE (FGGF open + Y/N + FGGF choice)', merged)
    if args.out_json:
        with open(args.out_json, 'w') as f:
            json.dump(merged, f, indent=2)
        print(f'Merged predictions → {args.out_json}')


if __name__ == '__main__':
    main()
