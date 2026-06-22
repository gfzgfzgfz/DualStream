"""
消融实验模型包装器

支持6种消融配置:
- E1 (base): 独立模式，无粗粒度信息 (不需要缓存)
- E2 (func_feat): 仅使用函数特征增强 (需要缓存)
- E3 (slice_attn): 仅使用切片注意力 (需要缓存)
- E4 (prob_guide): 仅使用概率引导 (需要缓存)
- E5 (full_no_reg): 完整模型但无正则化 (需要缓存)
- E6 (full): 完整模型 (需要缓存)

消融逻辑:
- E1: line_features -> predictor -> line_probs
- E2: line_features + alpha*func_features -> predictor -> line_probs
- E3: line_features * (1 + slice_attention) -> predictor -> line_probs
- E4: line_features -> predictor -> line_probs -> bayesian_update(func_probs) -> adjusted_probs
- E5: E2 + E3 + E4, reg_weight=0
- E6: E2 + E3 + E4, reg_weight>0
"""

import torch
import torch.nn as nn
from transformers import RobertaModel


class LineEncoder(nn.Module):
    """行级编码器"""
    def __init__(self, codebert_path, hidden_dim=768, dropout=0.1):
        super(LineEncoder, self).__init__()
        self.codebert = RobertaModel.from_pretrained(codebert_path)
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        outputs = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        line_features = outputs.last_hidden_state[:, 0, :]
        line_features = self.dropout(line_features)
        return line_features


class AblationModel(nn.Module):
    """消融实验模型

    Args:
        codebert_path: CodeBERT模型路径
        hidden_dim: 隐藏层维度
        alpha: 全局语义增强系数 (E2, E5, E6使用)
        delta: 先验概率调节系数 (E4, E5, E6使用)
        lambda_threshold: 动态阈值调节系数 (E4, E5, E6使用)
        reg_weight: 正则化损失权重 (仅E6使用)
        dropout: Dropout率
        ablation_mode: 消融模式
            - 'base': E1 - 独立模式，无粗粒度信息
            - 'func_feat': E2 - 仅使用函数特征增强
            - 'slice_attn': E3 - 仅使用切片注意力
            - 'prob_guide': E4 - 仅使用概率引导
            - 'full_no_reg': E5 - 完整模型但无正则化
            - 'full': E6 - 完整模型
    """

    VALID_MODES = ['base', 'func_feat', 'slice_attn', 'prob_guide', 'full_no_reg', 'full']

    # 标记哪些模式需要粗粒度缓存
    MODES_NEED_CACHE = ['func_feat', 'slice_attn', 'prob_guide', 'full_no_reg', 'full']

    def __init__(self, codebert_path, hidden_dim=768, alpha=0.3, delta=0.5,
                 lambda_threshold=0.2, reg_weight=0.1, dropout=0.1, ablation_mode='full'):
        super(AblationModel, self).__init__()

        if ablation_mode not in self.VALID_MODES:
            raise ValueError(f"Invalid ablation_mode: {ablation_mode}. Must be one of {self.VALID_MODES}")

        self.ablation_mode = ablation_mode
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.delta = delta
        self.lambda_threshold = lambda_threshold
        # E5模式下reg_weight强制为0
        self.reg_weight = 0.0 if ablation_mode == 'full_no_reg' else reg_weight

        # 行级编码器 (所有模式都需要)
        self.line_encoder = LineEncoder(codebert_path, hidden_dim, dropout)
        self.dropout = nn.Dropout(dropout)

        # 独立预测层 (E1 base模式使用)
        self.independent_predictor = nn.Linear(hidden_dim, 1)

        # 切片注意力层 (E3, E5, E6使用)
        if ablation_mode in ['slice_attn', 'full_no_reg', 'full']:
            self.slice_attention = nn.Linear(hidden_dim, hidden_dim)

        # 交互模式预测层 (E2, E3, E5, E6使用)
        if ablation_mode in ['func_feat', 'slice_attn', 'full_no_reg', 'full']:
            self.interaction_predictor = nn.Linear(hidden_dim, 1)

        # 概率引导预测层 (E4使用，独立于其他组件)
        if ablation_mode == 'prob_guide':
            self.prob_guided_predictor = nn.Linear(hidden_dim, 1)

    def needs_coarse_cache(self):
        """判断当前模式是否需要粗粒度缓存"""
        return self.ablation_mode in self.MODES_NEED_CACHE

    def forward(self, line_input_ids, line_attention_mask, func_features=None,
                slice_features=None, slice_probs=None, func_probs=None, slice_line_mask=None):
        """
        前向传播

        Args:
            line_input_ids: [batch_size, seq_len] 行代码token ids
            line_attention_mask: [batch_size, seq_len] 注意力掩码
            func_features: [batch_size, hidden_dim] 函数特征 (E2, E5, E6需要)
            slice_features: [batch_size, num_slices, hidden_dim] 切片特征 (E3, E5, E6需要)
            slice_probs: [batch_size, num_slices] 切片概率 (E3, E5, E6需要)
            func_probs: [batch_size] 函数概率 (E4, E5, E6需要)
            slice_line_mask: [batch_size, num_slices] 切片-行对齐掩码 (E3, E5, E6需要)

        Returns:
            dict: 包含line_logits, line_probs, reg_loss等
        """
        # 编码行特征
        line_features = self.line_encoder(line_input_ids, line_attention_mask)
        line_features = self.dropout(line_features)

        if self.ablation_mode == 'base':
            return self._forward_base(line_features)
        elif self.ablation_mode == 'func_feat':
            return self._forward_func_feat(line_features, func_features)
        elif self.ablation_mode == 'slice_attn':
            return self._forward_slice_attn(line_features, slice_features, slice_probs, slice_line_mask)
        elif self.ablation_mode == 'prob_guide':
            return self._forward_prob_guide(line_features, func_probs)
        elif self.ablation_mode in ['full_no_reg', 'full']:
            return self._forward_full(line_features, func_features, slice_features,
                                      slice_probs, func_probs, slice_line_mask)

    def _forward_base(self, line_features):
        """E1: 基线模式 - 仅使用行特征，无粗粒度信息"""
        line_logits = self.independent_predictor(line_features).squeeze(-1)
        line_probs = torch.sigmoid(line_logits)

        return {
            'line_logits': line_logits,
            'line_probs': line_probs,
            'line_predictions': (line_probs > 0.5).float(),
            'reg_loss': torch.zeros(line_features.size(0), device=line_features.device)
        }

    def _forward_func_feat(self, line_features, func_features):
        """E2: 仅使用函数特征增强

        公式: enhanced = line_features + alpha * func_features
        """
        # 全局语义增强
        enhanced_features = line_features + self.alpha * func_features
        enhanced_features = self.dropout(enhanced_features)

        line_logits = self.interaction_predictor(enhanced_features).squeeze(-1)
        line_probs = torch.sigmoid(line_logits)

        return {
            'line_logits': line_logits,
            'line_probs': line_probs,
            'line_predictions': (line_probs > 0.5).float(),
            'reg_loss': torch.zeros(line_features.size(0), device=line_features.device)
        }

    def _forward_slice_attn(self, line_features, slice_features, slice_probs, slice_line_mask=None):
        """E3: 仅使用切片注意力

        公式:
        1. relevance = line_features @ W @ slice_features.T
        2. weighted_relevance = relevance * slice_probs * slice_line_mask
        3. attention_weight = sum(weighted_relevance)
        4. enhanced = line_features * (1 + attention_weight)
        """
        batch_size = line_features.size(0)

        if len(slice_features.shape) == 2:
            slice_features = slice_features.unsqueeze(1)
        num_slices = slice_features.size(1)

        # 计算行与各切片的相关性
        line_features_expanded = line_features.unsqueeze(1).expand(-1, num_slices, -1)
        relevance_scores = torch.sum(
            line_features_expanded * self.slice_attention(slice_features),
            dim=-1
        )  # [batch_size, num_slices]

        # 应用切片-行对齐掩码
        if slice_line_mask is not None:
            relevance_scores = relevance_scores * slice_line_mask

        # 应用切片概率权重
        weighted_relevance = relevance_scores * slice_probs
        slice_attention_weights = torch.sum(weighted_relevance, dim=1)  # [batch_size]

        # 注意力增强
        attention_scale = 1.0 + slice_attention_weights.unsqueeze(-1)
        enhanced_features = line_features * attention_scale
        enhanced_features = self.dropout(enhanced_features)

        line_logits = self.interaction_predictor(enhanced_features).squeeze(-1)
        line_probs = torch.sigmoid(line_logits)

        return {
            'line_logits': line_logits,
            'line_probs': line_probs,
            'line_predictions': (line_probs > 0.5).float(),
            'slice_attention_weights': slice_attention_weights,
            'reg_loss': torch.zeros(batch_size, device=line_features.device)
        }

    def _forward_prob_guide(self, line_features, func_probs):
        """E4: 仅使用概率引导

        公式:
        1. prior = (1-delta)*0.5 + delta*func_probs
        2. line_probs = sigmoid(predictor(line_features))
        3. adjusted = bayesian_update(line_probs, prior)
        4. threshold = 0.5 - lambda * confidence * sign(func_probs - 0.5)
        """
        if len(func_probs.shape) > 1:
            func_probs = func_probs.squeeze(-1)

        # 计算先验概率
        prior_probs = (1 - self.delta) * 0.5 + self.delta * func_probs

        # 行级预测
        line_logits = self.prob_guided_predictor(line_features).squeeze(-1)
        line_probs = torch.sigmoid(line_logits)

        # 贝叶斯更新
        log_line_probs = torch.log(line_probs + 1e-10)
        log_prior_probs = torch.log(prior_probs + 1e-10)
        log_complement_line_probs = torch.log(1 - line_probs + 1e-10)
        log_complement_prior_probs = torch.log(1 - prior_probs + 1e-10)

        adjusted_line_probs = torch.exp(
            log_line_probs + log_prior_probs -
            torch.log(
                torch.exp(log_line_probs + log_prior_probs) +
                torch.exp(log_complement_line_probs + log_complement_prior_probs)
            )
        )

        # 动态阈值
        confidence = torch.abs(func_probs - 0.5) * 2
        threshold_shift = torch.sign(func_probs - 0.5) * confidence * self.lambda_threshold
        dynamic_thresholds = 0.5 - threshold_shift

        line_predictions = (adjusted_line_probs > dynamic_thresholds).float()

        return {
            'line_logits': line_logits,
            'line_probs': adjusted_line_probs,  # 返回调整后的概率用于评估
            'adjusted_line_probs': adjusted_line_probs,
            'line_predictions': line_predictions,
            'dynamic_thresholds': dynamic_thresholds,
            'reg_loss': torch.zeros(line_features.size(0), device=line_features.device)
        }

    def _forward_full(self, line_features, func_features, slice_features,
                      slice_probs, func_probs, slice_line_mask=None):
        """E5/E6: 完整模型

        组合: 函数特征增强 + 切片注意力 + 概率引导 + 正则化(仅E6)
        """
        batch_size = line_features.size(0)

        if len(func_probs.shape) > 1:
            func_probs = func_probs.squeeze(-1)

        if len(slice_features.shape) == 2:
            slice_features = slice_features.unsqueeze(1)
        num_slices = slice_features.size(1)

        # 1. 全局语义增强 (来自E2)
        enhanced_features = line_features + self.alpha * func_features

        # 2. 切片注意力 (来自E3)
        line_features_expanded = line_features.unsqueeze(1).expand(-1, num_slices, -1)
        relevance_scores = torch.sum(
            line_features_expanded * self.slice_attention(slice_features),
            dim=-1
        )

        if slice_line_mask is not None:
            relevance_scores = relevance_scores * slice_line_mask

        weighted_relevance = relevance_scores * slice_probs
        slice_attention_weights = torch.sum(weighted_relevance, dim=1)

        attention_scale = 1.0 + slice_attention_weights.unsqueeze(-1)
        enhanced_features = enhanced_features * attention_scale
        enhanced_features = self.dropout(enhanced_features)

        # 3. 交互预测
        line_logits = self.interaction_predictor(enhanced_features).squeeze(-1)
        line_probs = torch.sigmoid(line_logits)

        # 4. 概率引导 (来自E4)
        prior_probs = (1 - self.delta) * 0.5 + self.delta * func_probs

        log_line_probs = torch.log(line_probs + 1e-10)
        log_prior_probs = torch.log(prior_probs + 1e-10)
        log_complement_line_probs = torch.log(1 - line_probs + 1e-10)
        log_complement_prior_probs = torch.log(1 - prior_probs + 1e-10)

        adjusted_line_probs = torch.exp(
            log_line_probs + log_prior_probs -
            torch.log(
                torch.exp(log_line_probs + log_prior_probs) +
                torch.exp(log_complement_line_probs + log_complement_prior_probs)
            )
        )

        # 动态阈值
        confidence = torch.abs(func_probs - 0.5) * 2
        threshold_shift = torch.sign(func_probs - 0.5) * confidence * self.lambda_threshold
        dynamic_thresholds = 0.5 - threshold_shift

        line_predictions = (adjusted_line_probs > dynamic_thresholds).float()

        # 5. 正则化损失 (E5: reg_weight=0, E6: reg_weight>0)
        reg_loss = self.reg_weight * torch.abs(line_probs - func_probs)

        return {
            'line_logits': line_logits,
            'line_probs': line_probs,
            'adjusted_line_probs': adjusted_line_probs,
            'line_predictions': line_predictions,
            'dynamic_thresholds': dynamic_thresholds,
            'slice_attention_weights': slice_attention_weights,
            'reg_loss': reg_loss
        }
