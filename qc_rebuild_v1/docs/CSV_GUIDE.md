# CSV字段及结果口径

| 字段 | 含义 |
|---|---|
| embryo_id | 胚胎标识；本包保留源标识，便于和原图关联 |
| run | 原始RUN编号，不重新编号 |
| image_path | 原始图像位置；历史表包含服务器绝对路径，分享原图时需映射 |
| upstream_split / split | 上游预测任务划分，不等于QC模型自己的划分 |
| qc_split | QC模型开发/验证/独立测试划分 |
| score_delete / score_invalid | 模型类别1的softmax分数；不同checkpoint不能只凭数值直接对比 |
| predicted_action | 对应历史表threshold下的预测，不必然是最终采用的动作 |
| review_action | 人工判断，KEEP/DELETE；空值或待确认值不能当成KEEP |
| sample_reason | 抽样分层，可包含多个类别；分层人数可能重叠 |
| evidence_file / position_in_page | 当时的复核图路径与位置；图片不在小包中 |
| quality_flag / use_scope | 人工标签登记表中的质量或用途备注 |

## 当前可直接重算的标签

`manifests/reviewed_200_corrected.csv`：最后阈值网格参考目录中的200帧更正标签。
它比历史`confirmed_review_labels.csv`更晚；历史审计JSON与prediction_errors.csv可能仍对应更正前的标签。
因此当前重算结果与更正前JSON不一致时，不应反向修改历史JSON。

`manifests/manual_review_registry_923.csv`：此前923帧标签登记；并非923张新的独立测试帧。
它可能与上述200帧或其他历史标签重叠。合并前必须按`embryo_id/run`核对，并明确使用最新人工更正的优先级。

`manifests/audit_0p013_unreviewed_200.csv`：新抽样200帧，人工review_action尚待填写。
这个表记录模型分数和候选动作，不提供新的人工误判结论。

## 四个不同指标

设TN=正常且保留；FP=正常但删除；FN=异常但保留；TP=异常且删除。

- 有效帧误删率：FP/(TN+FP)。
- 异常漏检率：FN/(TP+FN)。
- 保留集错误比例：FN/(TN+FN)。
- 删除集误删比例：FP/(TP+FP)。

分母为0时留空，不报告为0%。新工具对没有合法人工标签或分数的行单独计数。
样本含有临界点富集和不同抽样比例，因此混合样本的指标不能直接推广为全F0误差率。
历史外推文件含有其作者声明的分层假设，不是全F0真实标签统计。

## 新推理入口追加字段

`qc_resolved_image_path`为接收方实际读取路径；`qc_status`为OK/DECODE_ERROR/PREPROCESS_ERROR；
`qc_decode_mode`保留解码类型；`qc_score_delete`为模型分数；`qc_candidate_action`为阈值候选动作；
`qc_threshold`记录本次阈值；`qc_error`记录错误原因。
原始列、顺序和原始image_path均保留，不自动生成最终删除清单或训练mask。
