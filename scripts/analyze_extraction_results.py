#!/usr/bin/env python3
"""
分析 OpenIE 提取结果的统计脚本

用法：
    python analyze_extraction_results.py --openie_file <path_to_openie_results.json>
"""

import json
import argparse
from collections import defaultdict


def analyze_extraction_results(openie_file):
    """分析 OpenIE 提取结果"""

    print("=" * 80)
    print("OpenIE 提取结果分析")
    print("=" * 80)
    print()

    with open(openie_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    docs = data.get('docs', [])
    total = len(docs)

    if total == 0:
        print("⚠️  没有找到任何文档")
        return

    failed_count = 0
    retry_counts = []
    failure_reasons = defaultdict(int)
    has_think = 0
    has_memory = 0
    think_lengths = []
    memory_lengths = []

    for doc in docs:
        metadata = doc.get('metadata', {})

        if metadata.get('extraction_failed', False):
            failed_count += 1
            reason = metadata.get('failure_reason', 'unknown')
            failure_reasons[reason] += 1

        retry_count = metadata.get('retry_count', 0)
        retry_counts.append(retry_count)

        if 'think' in doc and doc['think']:
            has_think += 1
            think_lengths.append(len(doc['think']))

        if 'memory' in doc and doc['memory']:
            has_memory += 1
            memory_lengths.append(len(doc['memory']))

    success_count = total - failed_count
    success_rate = success_count / total * 100
    failure_rate = failed_count / total * 100

    avg_retry = sum(retry_counts) / len(retry_counts) if retry_counts else 0
    max_retry = max(retry_counts) if retry_counts else 0

    avg_think_length = sum(think_lengths) / len(think_lengths) if think_lengths else 0
    avg_memory_length = sum(memory_lengths) / len(memory_lengths) if memory_lengths else 0

    print("📊 基本统计")
    print("-" * 80)
    print(f"总文档数:        {total}")
    print(f"成功提取:        {success_count} ({success_rate:.2f}%)")
    print(f"提取失败:        {failed_count} ({failure_rate:.2f}%)")
    print()

    print("🔄 重试统计")
    print("-" * 80)
    print(f"平均重试次数:    {avg_retry:.2f}")
    print(f"最大重试次数:    {max_retry}")

    retry_distribution = defaultdict(int)
    for count in retry_counts:
        retry_distribution[count] += 1

    print("\n重试次数分布:")
    for retry_num in sorted(retry_distribution.keys()):
        count = retry_distribution[retry_num]
        percentage = count / total * 100
        bar = "█" * int(percentage / 2)
        print(f"  {retry_num} 次: {count:4d} ({percentage:5.2f}%) {bar}")
    print()

    if failed_count > 0:
        print("❌ 失败原因分析")
        print("-" * 80)
        for reason, count in failure_reasons.items():
            percentage = count / failed_count * 100
            print(f"  {reason:30s}: {count:4d} ({percentage:5.2f}%)")
        print()

    print("📝 内容统计")
    print("-" * 80)
    print(f"包含 Think:      {has_think} ({has_think/total*100:.2f}%)")
    print(f"包含 Memory:     {has_memory} ({has_memory/total*100:.2f}%)")
    print(f"平均 Think 长度: {avg_think_length:.0f} 字符")
    print(f"平均 Memory 长度:{avg_memory_length:.0f} 字符")
    print()

    total_entities = sum(len(doc.get('extracted_entities', [])) for doc in docs)
    total_triples = sum(len(doc.get('extracted_triples', [])) for doc in docs)

    avg_entities = total_entities / total
    avg_triples = total_triples / total

    print("🔗 图谱统计")
    print("-" * 80)
    print(f"总实体数:        {total_entities}")
    print(f"总三元组数:      {total_triples}")
    print(f"平均实体数:      {avg_entities:.2f} 个/文档")
    print(f"平均三元组数:    {avg_triples:.2f} 个/文档")
    print()

    print("🏥 健康度评估")
    print("-" * 80)

    if failure_rate < 1:
        health = "优秀 ✓"
        color = "🟢"
    elif failure_rate < 5:
        health = "良好 ✓"
        color = "🟡"
    elif failure_rate < 10:
        health = "一般 ⚠️"
        color = "🟠"
    else:
        health = "需要改进 ✗"
        color = "🔴"

    print(f"失败率评级:      {color} {health}")

    if avg_retry < 0.5:
        retry_health = "优秀 ✓"
        retry_color = "🟢"
    elif avg_retry < 1.0:
        retry_health = "良好 ✓"
        retry_color = "🟡"
    elif avg_retry < 1.5:
        retry_health = "一般 ⚠️"
        retry_color = "🟠"
    else:
        retry_health = "需要改进 ✗"
        retry_color = "🔴"

    print(f"重试次数评级:    {retry_color} {retry_health}")
    print()

    print("💡 建议")
    print("-" * 80)

    if failure_rate > 5:
        print("⚠️  失败率较高，建议：")
        print("   1. 检查 prompt 模板是否正确")
        print("   2. 考虑使用更强的 LLM 模型")
        print("   3. 降低 temperature 参数")
        print()

    if avg_retry > 1.0:
        print("⚠️  平均重试次数较高，建议：")
        print("   1. 优化 prompt 模板，使输出更稳定")
        print("   2. 调整 LLM 参数（temperature, top_p）")
        print("   3. 检查输入文本质量")
        print()

    if has_think < total * 0.9:
        print("⚠️  部分文档缺少 Think 内容，建议：")
        print("   1. 检查 prompt 是否明确要求输出 <think> 标签")
        print("   2. 查看失败记录了解原因")
        print()

    if failure_rate < 1 and avg_retry < 0.5:
        print("✓ 提取质量优秀，无需改进")
        print()

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='分析 OpenIE 提取结果')
    parser.add_argument('--openie_file', type=str, required=True,
                       help='OpenIE 结果文件路径 (openie_results_ner_*.json)')

    args = parser.parse_args()

    try:
        analyze_extraction_results(args.openie_file)
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {args.openie_file}")
    except json.JSONDecodeError:
        print(f"❌ 错误: 文件格式不正确，无法解析 JSON")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
