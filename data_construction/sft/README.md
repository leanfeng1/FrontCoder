# SFT Data Construction Pipeline - Complete Workflow

FrontCoder SFT 数据构建的完整流程：从 80 个类别到 240K 个高质量代码样本。

## 📊 完整数据流程

```
Step 0: 80 classes + 2,000 subcategories (预定义)
    ↓ [generate_sft1_expand_tasks.py]
    ↓ 为每个subcategory生成10个task（使用Qwen3-Coder-480b）
20,000 tasks
    ↓ [generate_sft2_variants.py]
    ↓ 为每个task生成12个variant（使用Qwen3-Coder-480b）
240,000 variants
    ↓ [generate_sft3_code_minimax.py]
    ↓ 为每个variant生成HTML代码（使用Minimax-M2）
240,000 code samples
    ↓ [minhash_dedup.py]
    ↓ MinHash去重 (Jaccard阈值0.8)
~200,000 deduplicated samples
    ↓ [quality_scorer.py]
    ↓ 25维度质量评分
~167,000 scored samples
    ↓ [filter_sft_data.py]
    ↓ 过滤 (分数≥80%, 长度≤16K tokens)
60,000 final SFT data
```

## 🚀 快速开始

### 完整流程（按顺序执行）

```bash
cd /volume/pt-train/users/wzhang/fj-workspace/code/FrontCoder/data_construction/sft

# Step 1: 生成 20K tasks (每个subcategory生成10个)
python generate_sft1_expand_tasks.py \
    --input_file sft_subcategories_2k.jsonl \
    --output_file sft_tasks_20k.jsonl \
    --workers 50

# Step 2: 生成 240K variants (每个task生成12个)
python generate_sft2_variants.py \
    --input_file sft_tasks_20k.jsonl \
    --output_file sft_variants_240k.jsonl \
    --workers 30

# Step 3: 生成 240K 代码 (使用Minimax-M2)
python generate_sft3_code_minimax.py \
    --input_file sft_variants_240k.jsonl \
    --output_file sft_final_240k.jsonl \
    --workers 100

# Step 4: 去重
python generate_sft4_dedup.py \
    --input sft_final_240k.jsonl \
    --output sft_deduped.jsonl \
    --threshold 0.8

# Step 5: 质量评分
python generate_sft5_scorer.py \
    --input sft_deduped.jsonl \
    --output sft_scored.jsonl

# Step 6: 过滤
python generate_sft6_filter.py \
    --input sft_scored.jsonl \
    --output sft_final_60k.jsonl \
    --min_score 0.8 \
    --max_length 16384
```

---

## 📋 详细步骤说明

### Step 0: 类别定义

**数据文件**:
- `sft_categories_80x2k.json` - 结构化的 80 类 + 2000 子类
- `sft_subcategories_2k.jsonl` - 扁平化的 2000 个子类

**说明**:
- 预定义了 80 个主类别
- 预定义了 2000 个子类别（每个类别平均 25 个子类别）
- 涵盖多个领域：数据可视化、游戏、工具、商业、社交、IoT 等

---

### Step 1: 生成任务 (2K → 20K tasks)

**脚本**: `generate_sft1_expand_tasks.py`

**功能**:
- 为每个子类别生成 10 个不同的具体任务
- 使用 Qwen3-Coder-480b 生成多样化的任务描述

**输入**: `sft_subcategories_2k.jsonl` (2000 个子类别)

**输出**: `sft_tasks_20k.jsonl` (20,000 个任务)

**关键参数**:
- `--workers 50`: 并发数
- `--temperature 0.8`: 生成温度
- `--model Qwen3-Coder-480b`: 使用的模型

**运行**:
```bash
python generate_sft1_expand_tasks.py \
    --input_file sft_subcategories_2k.jsonl \
    --output_file sft_tasks_20k.jsonl \
    --api_key YOUR_API_KEY \
    --base_url YOUR_API_URL \
    --workers 50
```

**测试模式**:
```bash
python generate_sft1_expand_tasks.py --test  # 只处理前3条
```

**数据格式**:
```json
{
  "task_id": 0,
  "subcat_id": 0,
  "class": "Data Science-Data Visualization Dashboards",
  "sub_category": "3D Data Visualization",
  "specific_task": "Create interactive 3D scatter plot with rotation controls"
}
```

---

### Step 2: 生成变体 (20K → 240K variants)

**脚本**: `generate_sft2_variants.py`

**功能**:
- 为每个任务生成 12 个变体
- 使用 12 种预定义的变体类型（颜色方案、布局风格、交互模式等）
- 使用 Qwen3-Coder-480b 生成

**输入**: `sft_tasks_20k.jsonl` (20,000 个任务)

**输出**: `sft_variants_240k.jsonl` (240,000 个变体)

**12 种变体类型**:
1. Color Scheme - 颜色主题变化
2. Layout Style - 布局样式变化
3. Interaction Mode - 交互方式变化
4. Responsive Design - 响应式设计
5. Animation Effects - 动画效果
6. Accessibility Features - 无障碍特性
7. Advanced Features - 高级功能
8. Minimalist Design - 极简设计
9. Data Visualization - 数据可视化
10. Real-time Updates - 实时更新
11. Gamification - 游戏化
12. Internationalization - 国际化

**运行**:
```bash
python generate_sft2_variants.py \
    --input_file sft_tasks_20k.jsonl \
    --output_file sft_variants_240k.jsonl \
    --api_key YOUR_API_KEY \
    --base_url YOUR_API_URL \
    --workers 30 \
    --temperature 0.9
```

**数据格式**:
```json
{
  "variant_id": 0,
  "task_id": 0,
  "variant_type_id": 1,
  "variant_type": "Color Scheme",
  "class": "Data Science-Data Visualization Dashboards",
  "sub_category": "3D Data Visualization",
  "original_task": "Create interactive 3D scatter plot",
  "variant_task": "Create dark-themed interactive 3D scatter plot with #1a1a1a background"
}
```

---

### Step 3: 生成代码 (240K variants → 240K code samples)

**脚本**: `generate_sft3_code_minimax.py`

**功能**:
- 使用 Minimax-M2 为每个变体生成完整的 HTML/CSS/JS 代码
- 高并发处理（100 workers）

**输入**: `sft_variants_240k.jsonl` (240,000 个变体)

**输出**: `sft_final_240k.jsonl` (240,000 个代码样本)

**关键参数**:
- `--workers 100`: 高并发
- `--max_tokens 16384`: 允许生成长代码
- `--temperature 0.7`: 代码生成温度

**运行**:
```bash
python generate_sft3_code_minimax.py \
    --input_file sft_variants_240k.jsonl \
    --output_file sft_final_240k.jsonl \
    --api_key YOUR_MINIMAX_API_KEY \
    --base_url YOUR_MINIMAX_URL \
    --workers 100
```

**数据格式**:
```json
{
  "variant_id": 0,
  "task_id": 0,
  "variant_type": "Color Scheme",
  "class": "Data Science-Data Visualization Dashboards",
  "sub_category": "3D Data Visualization",
  "original_task": "...",
  "variant_task": "...",
  "code": "<!DOCTYPE html>\n<html>...</html>"
}
```

---

### Step 4: MinHash 去重

**脚本**: `generate_sft4_dedup.py`

**功能**: 使用 MinHash LSH 算法去除重复样本

**运行**:
```bash
python generate_sft4_dedup.py \
    --input sft_final_240k.jsonl \
    --output sft_deduped.jsonl \
    --threshold 0.8 \
    --num_perm 128
```

**预期**: 240K → ~200K

---

### Step 5: 质量评分

**脚本**: `generate_sft5_scorer.py`

**功能**: 25 维度质量评分系统

**运行**:
```bash
python generate_sft5_scorer.py \
    --input sft_deduped.jsonl \
    --output sft_scored.jsonl \
    --workers 2000
```

**预期**: ~200K → ~167K (评分后)

---

### Step 6: 过滤

**脚本**: `generate_sft6_filter.py`

**功能**: 基于分数和长度过滤

**运行**:
```bash
python generate_sft6_filter.py \
    --input sft_scored.jsonl \
    --output sft_final_60k.jsonl \
    --min_score 0.8 \
    --max_length 16384 \
    --target_count 60000
```

**预期**: ~167K → 60K

---

## 📁 文件列表

| 文件 | 说明 | 大小（预期） |
|------|------|-------------|
| `generate_sft1_expand_tasks.py` | Step 1: 生成任务 | - |
| `generate_sft2_variants.py` | Step 2: 生成变体 | - |
| `generate_sft3_code_minimax.py` | Step 3: 生成代码 | - |
| `generate_sft4_dedup.py` | Step 4: 去重 | - |
| `generate_sft5_scorer.py` | Step 5: 评分 | - |
| `generate_sft6_filter.py` | Step 6: 过滤 | - |
| `sft_categories_80x2k.json` | 80 类 + 2K 子类 (结构化) | ~500KB |
| `sft_subcategories_2k.jsonl` | 2K 子类 (扁平化) | ~400KB |
| `sft_tasks_20k.jsonl` | 20K 任务 | ~4MB |
| `sft_variants_240k.jsonl` | 240K 变体 | ~50MB |
| `sft_final_240k.jsonl` | 240K 代码样本 | ~10GB |
| `sft_deduped.jsonl` | 去重后 (~200K) | ~8GB |
| `sft_scored.jsonl` | 评分后 (~167K) | ~7GB |
| `sft_final_60k.jsonl` | 最终数据 (60K) | ~3GB |

---

## ⚙️ 配置说明

### API 配置

**Qwen3-Coder-480b** (Step 1 & 2):
- API URL: `https://console.siflow.cn/siflow/auriga/skyinfer/fjing/qwen3-480b-0/v1`
- 用途: 任务生成和变体生成
- 并发: 30-50

**Minimax-M2** (Step 3):
- API URL: 根据部署配置
- 用途: 代码生成
- 并发: 100

### 性能优化

1. **并发控制**: 根据 API 限流调整 `--workers`
2. **断点续传**: 所有脚本支持缓存和断点续传
3. **批量处理**: 使用 `--max_records` 进行分批处理
4. **测试模式**: 使用 `--test` 快速验证流程

---

## 📊 预期时间

基于 API 性能和并发设置：

| 步骤 | 数据量 | 预计时间 |
|------|--------|----------|
| Step 1 | 2K → 20K | 1-2 小时 |
| Step 2 | 20K → 240K | 10-20 小时 |
| Step 3 | 240K → 240K | 20-40 小时 |
| Step 4 | 去重 | 1-2 小时 |
| Step 5 | 评分 | 5-10 小时 |
| Step 6 | 过滤 | 30 分钟 |
| **总计** | - | **40-75 小时** |

---

## 🔧 故障排查

### 常见问题

1. **API 连接失败**
   - 检查 API key 和 base_url
   - 降低并发数 `--workers`

2. **JSON 解析错误**
   - 检查缓存文件是否损坏
   - 删除缓存重新运行

3. **内存不足**
   - 使用 `--max_records` 分批处理
   - 减少并发数

4. **进度丢失**
   - 所有脚本支持断点续传
   - 检查缓存文件 `*_cache.jsonl`

---

## 📝 注意事项

1. **数据质量**: Step 5-6 的评分和过滤参数会影响最终数据质量
2. **成本控制**: API 调用成本较高，建议先用 `--test` 模式验证
3. **存储空间**: 完整流程需要 ~30GB 磁盘空间
4. **备份**: 建议定期备份中间数据

---

## 📖 参考资料

- FrontCoder 论文
- SFT 数据构建方法: 80 类 × 2K 子类 → 20K 任务 × 12 变体 = 240K 样本
- 最终数据量: 60K 高质量样本
