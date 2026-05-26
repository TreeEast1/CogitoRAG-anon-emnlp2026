#!/usr/bin/env python3
"""
检查 Think 文件的完整性和质量

用法：
    python check_think_quality.py --think_dir <path_to_think_storage>
"""

import os
import argparse
import json
from collections import defaultdict


def check_think_quality(think_dir):
    """检查 think 文件的质量"""

    print("=" * 80)
    print("Think 文件质量检查")
    print("=" * 80)
    print()

    if not os.path.exists(think_dir):
        print(f"❌ 错误: 目录不存在 {think_dir}")
        return

    # 获取所有 think 文件
    think_files = [f for f in os.listdir(think_dir) if f.endswith('.txt')]
    total_files = len(think_files)

    if total_files == 0:
        print("⚠️  没有找到任何 think 文件")
        return

    print(f"📊 找到 {total_files} 个 think 文件")
    print()

    # 统计数据
    empty_files = []
    short_files = []  # < 50 字符
    long_files = []   # > 1000 字符
    lengths = []
    word_counts = []

    # 分析每个文件
    print("🔍 分析文件...")
    for filename in think_files:
        filepath = os.path.join(think_dir, filename)

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        length = len(content)
        word_count = len(content.split())

        lengths.append(length)
        word_counts.append(word_count)

        if length == 0:
            empty_files.append(filename)
        elif length < 50:
            short_files.append((filename, length))
        elif length > 1000:
            long_files.append((filename, length))

    # 计算统计数据
    avg_length = sum(lengths) / len(lengths) if lengths else 0
    min_length = min(lengths) if lengths else 0
    max_length = max(lengths) if lengths else 0

    avg_words = sum(word_counts) / len(word_counts) if word_counts else 0

    # 输出统计结果
    print()
    print("=" * 80)
    print("统计结果")
    print("=" * 80)
    print()

    print("📏 长度统计")
    print("-" * 80)
    print(f"平均长度:     {avg_length:.0f} 字符")
    print(f"最小长度:     {min_length} 字符")
    print(f"最大长度:     {max_length} 字符")
    print(f"平均词数:     {avg_words:.0f} 词")
    print()

    print("📊 分布统计")
    print("-" * 80)

    # 长度分布
    length_ranges = {
        '0': 0,
        '1-50': 0,
        '51-100': 0,
        '101-200': 0,
        '201-500': 0,
        '501-1000': 0,
        '1000+': 0
    }

    for length in lengths:
        if length == 0:
            length_ranges['0'] += 1
        elif length <= 50:
            length_ranges['1-50'] += 1
        elif length <= 100:
            length_ranges['51-100'] += 1
        elif length <= 200:
            length_ranges['101-200'] += 1
        elif length <= 500:
            length_ranges['201-500'] += 1
        elif length <= 1000:
            length_ranges['501-1000'] += 1
        else:
            length_ranges['1000+'] += 1

    for range_name, count in length_ranges.items():
        percentage = count / total_files * 100
        bar = "█" * int(percentage / 2)
        print(f"  {range_name:12s}: {count:4d} ({percentage:5.1f}%) {bar}")

    print()

    # 问题文件
    if empty_files or short_files:
        print("⚠️  问题文件")
        print("-" * 80)

        if empty_files:
            print(f"\n空文件 ({len(empty_files)} 个):")
            for filename in empty_files[:10]:  # 只显示前10个
                print(f"  - {filename}")
            if len(empty_files) > 10:
                print(f"  ... 还有 {len(empty_files) - 10} 个")

        if short_files:
            print(f"\n过短文件 (<50字符, {len(short_files)} 个):")
            for filename, length in short_files[:10]:
                print(f"  - {filename} ({length} 字符)")
            if len(short_files) > 10:
                print(f"  ... 还有 {len(short_files) - 10} 个")

        print()

    # 质量评估
    print("🏥 质量评估")
    print("-" * 80)

    empty_rate = len(empty_files) / total_files * 100
    short_rate = len(short_files) / total_files * 100

    if empty_rate == 0 and short_rate < 5:
        quality = "优秀 ✓"
        color = "🟢"
    elif empty_rate < 1 and short_rate < 10:
        quality = "良好 ✓"
        color = "🟡"
    elif empty_rate < 5 and short_rate < 20:
        quality = "一般 ⚠️"
        color = "🟠"
    else:
        quality = "需要改进 ✗"
        color = "🔴"

    print(f"整体质量:     {color} {quality}")
    print(f"空文件率:     {empty_rate:.1f}%")
    print(f"过短文件率:   {short_rate:.1f}%")
    print()

    # 建议
    print("💡 建议")
    print("-" * 80)

    if empty_rate > 0:
        print(f"⚠️  发现 {len(empty_files)} 个空文件，建议：")
        print("   1. 检查对应的 OpenIE 结果，查看是否提取失败")
        print("   2. 查看失败记录，了解失败原因")
        print("   3. 考虑重新处理这些文档")
        print()

    if short_rate > 10:
        print(f"⚠️  过短文件比例较高 ({short_rate:.1f}%)，建议：")
        print("   1. 检查 prompt 是否要求输出足够详细的 think")
        print("   2. 调整 LLM 参数（如 max_tokens）")
        print("   3. 使用更强的 LLM 模型")
        print()

    if empty_rate == 0 and short_rate < 5:
        print("✓ Think 文件质量优秀，无需改进")
        print()

    # 保存报告
    report_file = os.path.join(os.path.dirname(think_dir), 'think_quality_report.json')

    report_data = {
        'total_files': total_files,
        'empty_files': len(empty_files),
        'short_files': len(short_files),
        'long_files': len(long_files),
        'avg_length': avg_length,
        'min_length': min_length,
        'max_length': max_length,
        'avg_words': avg_words,
        'empty_rate': empty_rate,
        'short_rate': short_rate,
        'quality': quality,
        'length_distribution': length_ranges
    }

    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    print(f"✓ 报告已保存到: {report_file}")
    print()

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='检查 think 文件质量')
    parser.add_argument('--think_dir', type=str, required=True,
                       help='Think 文件目录路径')

    args = parser.parse_args()

    try:
        check_think_quality(args.think_dir)
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
