#!/usr/bin/env python3
"""
计算排除敏感内容后的 EM/F1 指标

用法:
    python scripts/calculate_filtered_metrics.py \
        --results outputs/musique_full_gpt52/result3.json \
        --failed outputs/musique_full_gpt52/failed_extractions/failed_extractions_20260203_100913.json \
        --dataset reproduce/dataset/musique.json
"""

import argparse
import json
import re
from collections import Counter
from typing import List, Set


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_em(gold_list: List[str], predicted: str) -> float:
    """计算 Exact Match"""
    for gold in gold_list:
        if normalize_answer(gold) == normalize_answer(predicted):
            return 1.0
    return 0.0


def compute_f1(gold_list: List[str], predicted: str) -> float:
    """计算 F1 Score"""
    f1_scores = []
    for gold in gold_list:
        gold_tokens = normalize_answer(gold).split()
        predicted_tokens = normalize_answer(predicted).split()
        common = Counter(predicted_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            f1_scores.append(0.0)
            continue

        precision = 1.0 * num_same / len(predicted_tokens) if predicted_tokens else 0
        recall = 1.0 * num_same / len(gold_tokens) if gold_tokens else 0
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * (precision * recall) / (precision + recall))
    return max(f1_scores) if f1_scores else 0.0


def get_affected_qa_indices(failed_extractions_path: str, dataset_path: str) -> Set[int]:
    """根据失败的 extraction 找出受影响的 QA 索引"""

    # 加载失败的 chunk 记录
    with open(failed_extractions_path, 'r') as f:
        failed_data = json.load(f)

    # 提取失败 passage 的标题（第一行通常是标题）
    failed_titles = set()
    for record in failed_data['failed_records']:
        passage = record['passage']
        title = passage.split('\n')[0].strip()
        failed_titles.add(title)

    # 加载 QA 数据集
    with open(dataset_path, 'r') as f:
        samples = json.load(f)

    # 找出受影响的 QA
    affected_indices = set()
    for idx, sample in enumerate(samples):
        paragraphs = sample.get('paragraphs', [])
        supporting_titles = set()
        for p in paragraphs:
            if p.get('is_supporting', False):
                supporting_titles.add(p.get('title', ''))

        # 检查是否有 supporting paragraph 在失败列表中
        if supporting_titles & failed_titles:
            affected_indices.add(idx)

    return affected_indices, failed_titles


def main():
    parser = argparse.ArgumentParser(description='计算排除敏感内容后的 EM/F1 指标')
    parser.add_argument('--results', required=True, help='QA 结果文件路径 (result3.json)')
    parser.add_argument('--failed', required=True, help='失败的 extraction 文件路径')
    parser.add_argument('--dataset', required=True, help='原始数据集路径 (musique.json)')
    parser.add_argument('--output', help='输出结果到 JSON 文件')
    parser.add_argument('--verbose', action='store_true', help='显示详细信息')
    args = parser.parse_args()

    # 加载结果
    with open(args.results, 'r') as f:
        results = json.load(f)

    # 获取受影响的 QA 索引
    affected_indices, failed_titles = get_affected_qa_indices(args.failed, args.dataset)

    print('=' * 70)
    print('EM/F1 计算结果 (排除敏感内容)')
    print('=' * 70)

    if args.verbose:
        print(f'\n失败的文档标题 ({len(failed_titles)} 个):')
        for t in sorted(failed_titles):
            print(f'  - {t}')
        print(f'\n受影响的 QA 索引 ({len(affected_indices)} 个): {sorted(affected_indices)}')

    # 计算原始 EM/F1
    total_em_all = 0
    total_f1_all = 0
    for item in results:
        predicted = item['output']
        gold_list = item['golden_answer']
        total_em_all += compute_em(gold_list, predicted)
        total_f1_all += compute_f1(gold_list, predicted)

    em_all = total_em_all / len(results)
    f1_all = total_f1_all / len(results)

    print(f'\n【原始结果】(分母 = {len(results)})')
    print(f'  ExactMatch: {em_all:.4f} ({em_all*100:.2f}%)')
    print(f'  F1:         {f1_all:.4f} ({f1_all*100:.2f}%)')

    # 计算排除敏感内容后的 EM/F1
    total_em_filtered = 0
    total_f1_filtered = 0
    valid_count = 0

    for i, item in enumerate(results):
        if i in affected_indices:
            continue
        predicted = item['output']
        gold_list = item['golden_answer']
        total_em_filtered += compute_em(gold_list, predicted)
        total_f1_filtered += compute_f1(gold_list, predicted)
        valid_count += 1

    em_filtered = total_em_filtered / valid_count if valid_count > 0 else 0
    f1_filtered = total_f1_filtered / valid_count if valid_count > 0 else 0

    print(f'\n【排除敏感内容后】(分母 = {valid_count}, 排除了 {len(affected_indices)} 个)')
    print(f'  ExactMatch: {em_filtered:.4f} ({em_filtered*100:.2f}%)')
    print(f'  F1:         {f1_filtered:.4f} ({f1_filtered*100:.2f}%)')

    # 被排除问题的表现
    if affected_indices:
        excluded_em = sum(compute_em(results[i]['golden_answer'], results[i]['output']) for i in affected_indices)
        excluded_f1 = sum(compute_f1(results[i]['golden_answer'], results[i]['output']) for i in affected_indices)

        print(f'\n【被排除问题的表现】({len(affected_indices)} 个)')
        print(f'  平均 EM: {excluded_em/len(affected_indices):.4f}')
        print(f'  平均 F1: {excluded_f1/len(affected_indices):.4f}')

    print('\n' + '=' * 70)
    print('总结')
    print('=' * 70)
    print(f'原始:     EM={em_all:.4f}, F1={f1_all:.4f} (n={len(results)})')
    print(f'排除后:   EM={em_filtered:.4f}, F1={f1_filtered:.4f} (n={valid_count})')
    em_diff = (em_filtered - em_all) * 100
    f1_diff = (f1_filtered - f1_all) * 100
    print(f'变化:     EM {em_diff:+.2f}%, F1 {f1_diff:+.2f}%')

    # 保存结果
    if args.output:
        output_data = {
            'original': {
                'n': len(results),
                'ExactMatch': em_all,
                'F1': f1_all
            },
            'filtered': {
                'n': valid_count,
                'excluded_count': len(affected_indices),
                'ExactMatch': em_filtered,
                'F1': f1_filtered
            },
            'excluded_qa_indices': sorted(affected_indices),
            'failed_titles': sorted(failed_titles)
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f'\n结果已保存到: {args.output}')


if __name__ == '__main__':
    main()
