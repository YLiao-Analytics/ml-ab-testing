# ML A/B Testing Framework (机器学习 A/B 测试框架)

![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A comprehensive framework for conducting A/B testing with statistical significance, confidence intervals, and power analysis.  
一个集统计学计算、置信区间评估与统计功效分析于一体的综合性 A/B 测试框架。

---

## 🌟 Features (核心功能)

* **Conversion Tests (转化率检验)** - Binomial metrics (e.g., CTR, Purchase Rate).  
    *适用于二项分布指标，如点击率、转化率。*
* **Continuous Metrics (连续变量检验)** - Mean values (e.g., Revenue, Time on Site).  
    *适用于数值型均值指标，如营收、停留时长。*
* **Sample Size Calculator (样本量计算)** - Experiment planning and duration estimation.  
    *实验前期规划，计算所需最小样本量及实验周期。*
* **Power Analysis (统计功效分析)** - Assessing the probability of detecting real effects.  
    *评估实验检测出真实业务提升的概率，避免“检测效能不足”。*
* **Visualization (可视化)** - Informative reports and statistical charts.  
    *生成直观的统计报告与专业趋势图表。*

---

## 🚀 Quick Start (快速开始)

### Installation (安装)

```bash
# Clone the repository
git clone <repository-url>
cd ml-ab-testing

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

Basic Example (基础示例)
from main import ABTesting, TestType

# Initialize (初始化)
ab_test = ABTesting(alpha=0.05)

# 1. Plan: Calculate Sample Size (规划：计算样本量)
size = ab_test.sample_size_calculator(baseline_rate=0.10, expected_lift=0.05)

# 2. Run: Conversion Test (执行：转化率检验)
result = ab_test.conversion_test(data)

# 3. Report: Generate Insights (报告：生成洞察)
print(ab_test.generate_report(result))
ab_test.plot_results(data, result)

📊 Core Concepts (核心概念)
1. Statistical Significance (统计显著性)
P-value < 0.05: Statistically significant; the effect is likely real.

结果显著，提升很可能不是偶然造成的。

P-value ≥ 0.05: Not significant; cannot rule out random chance.

结果不显著，无法排除随机波动的影响。

2. Statistical Power (统计功效)
Target Power ≥ 0.8: Industry standard to avoid Type II errors (missing a real effect).

工业界标准，确保有 80% 的把握捕捉到真实的业务提升，降低“漏报”风险。

3. Confidence Intervals (置信区间)
If the interval does not contain 0, the result is statistically significant.

如果置信区间不包含 0，说明差异在统计上是可靠的。

📈 Visualizations (可视化展示)
The framework generates comprehensive diagnostic plots (框架提供全方位的诊断图表):

Metric Comparison: Bar charts with error bars. (带误差线的指标对比图。)

Distribution Plots: Probability Density Functions (PDF) for continuous data. (连续变量的概率密度分布图。)

Power vs. Sample Size: Visualizing how sensitivity grows with data. (功效随样本量增长的曲线图。)

⚠️ Important Warnings (注意事项)
[!IMPORTANT]
No Peeking! (不要偷看结果) > Checking results repeatedly before the test ends increases Type I errors (False Positives).

在实验结束前频繁查看结果会显著增加“假阳性”风险。

[!NOTE]
Practical vs. Statistical Significance (实际意义 vs 统计意义) > A tiny lift might be "significant" with huge data, but may not be worth the engineering cost.

在大样本下，极小的提升也可能显著，但需评估其商业价值是否能覆盖开发成本。

📂 Project Structure (项目结构)
Plaintext
ml-ab-testing/
├── main.py                # Core Logic (核心逻辑类)
├── ab_testing_demo.ipynb  # Interactive Demo (交互式演示)
├── requirements.txt       # Dependencies (依赖清单)
├── .gitignore             # Git ignore file (Git 忽略文件)
└── README.md              # Documentation (项目文档)
📄 License (许可证)
This project is licensed under the MIT License.

🎯 Data-Driven Decisions Start Here! (数据驱动决策，从这里开始！)
