import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel

class LineEncoder(nn.Module):
    """行级编码器，用于编码每行代码及其上下文"""
    def __init__(self, codebert_path, hidden_dim=768, dropout=0.1):
        super(LineEncoder, self).__init__()
        self.codebert = RobertaModel.from_pretrained(codebert_path)
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, attention_mask):
        # 获取CodeBERT的输出
        outputs = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 使用[CLS]标记的表示作为行特征
        line_features = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
        line_features = self.dropout(line_features)
        
        return line_features

class CoarseFineInteraction(nn.Module):
    """粗细粒度交互模块

    支持两种模式:
    1. 批量行模式: line_features [batch_size, num_lines, hidden_dim] - 原始设计
    2. 单行模式: line_features [batch_size, hidden_dim] - 用于缓存加载的交互模式
    """
    def __init__(self, hidden_dim=768, alpha=0.3, dropout=0.1):
        super(CoarseFineInteraction, self).__init__()
        self.hidden_dim = hidden_dim
        self.alpha = alpha

        # 切片注意力传导
        self.slice_attention = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # 行级预测器
        self.line_predictor = nn.Linear(hidden_dim, 1)

    def forward(self, line_features, func_features, slice_features, slice_probs, slice_line_mask=None):
        """
        Args:
            line_features: [batch_size, hidden_dim] 或 [batch_size, num_lines, hidden_dim]
            func_features: [batch_size, hidden_dim]
            slice_features: [batch_size, num_slices, hidden_dim]
            slice_probs: [batch_size, num_slices]
            slice_line_mask: [batch_size, num_slices] 可选，标记哪些切片包含当前行
        """
        # 判断是单行模式还是批量行模式
        if len(line_features.shape) == 2:
            # 单行模式: [batch_size, hidden_dim]
            return self._forward_single_line(line_features, func_features, slice_features, slice_probs, slice_line_mask)
        else:
            # 批量行模式: [batch_size, num_lines, hidden_dim]
            return self._forward_batch_lines(line_features, func_features, slice_features, slice_probs)

    def _forward_single_line(self, line_features, func_features, slice_features, slice_probs, slice_line_mask=None):
        """单行模式的前向传播 - 用于缓存加载的交互模式"""
        batch_size = line_features.size(0)

        # 确保slice_features是三维的
        if len(slice_features.shape) == 2:
            slice_features = slice_features.unsqueeze(1)
        num_slices = slice_features.size(1)

        # 全局语义增强
        enhanced_line_features = line_features + self.alpha * func_features  # [batch_size, hidden_dim]

        # 切片注意力传导
        # 计算行与各切片的相关性
        line_features_expanded = line_features.unsqueeze(1).expand(-1, num_slices, -1)  # [batch_size, num_slices, hidden_dim]

        # 计算相关性得分
        relevance_scores = torch.sum(
            line_features_expanded * self.slice_attention(slice_features),
            dim=-1
        )  # [batch_size, num_slices]

        # 应用切片-行对齐掩码（如果提供）
        if slice_line_mask is not None:
            # 只考虑包含当前行的切片
            relevance_scores = relevance_scores * slice_line_mask

        # 应用切片概率权重
        weighted_relevance = relevance_scores * slice_probs  # [batch_size, num_slices]

        # 计算切片注意力权重（对所有切片求和）
        slice_attention_weights = torch.sum(weighted_relevance, dim=1)  # [batch_size]

        # 注意力增强特征
        attention_scale = 1.0 + slice_attention_weights.unsqueeze(-1)  # [batch_size, 1]
        enhanced_line_features = enhanced_line_features * attention_scale  # [batch_size, hidden_dim]
        enhanced_line_features = self.dropout(enhanced_line_features)

        # 行级预测
        line_logits = self.line_predictor(enhanced_line_features).squeeze(-1)  # [batch_size]
        line_probs = torch.sigmoid(line_logits)  # [batch_size]

        return enhanced_line_features, line_logits, line_probs, slice_attention_weights

    def _forward_batch_lines(self, line_features, func_features, slice_features, slice_probs):
        """批量行模式的前向传播 - 原始设计"""
        batch_size, num_lines, _ = line_features.size()

        # 确保slice_features是三维的
        if len(slice_features.shape) == 2:
            slice_features = slice_features.unsqueeze(1)
        _, num_slices, _ = slice_features.size()

        # 全局语义增强
        func_features_expanded = func_features.unsqueeze(1).expand(-1, num_lines, -1)
        enhanced_line_features = line_features + self.alpha * func_features_expanded

        # 切片注意力传导
        line_features_expanded = line_features.unsqueeze(2).expand(-1, -1, num_slices, -1)
        slice_features_expanded = slice_features.unsqueeze(1).expand(-1, num_lines, -1, -1)

        # 计算相关性得分
        relevance_scores = torch.sum(
            line_features_expanded * self.slice_attention(slice_features_expanded),
            dim=-1
        )  # [batch_size, num_lines, num_slices]

        # 应用切片概率权重
        slice_probs_expanded = slice_probs.unsqueeze(1).expand(-1, num_lines, -1)
        weighted_relevance = relevance_scores * slice_probs_expanded

        # 计算切片注意力权重
        slice_attention_weights = torch.sum(weighted_relevance, dim=2)  # [batch_size, num_lines]

        # 注意力增强特征
        attention_scale = 1.0 + slice_attention_weights.unsqueeze(-1)
        enhanced_line_features = enhanced_line_features * attention_scale
        enhanced_line_features = self.dropout(enhanced_line_features)

        # 行级预测
        line_logits = self.line_predictor(enhanced_line_features).squeeze(-1)  # [batch_size, num_lines]
        line_probs = torch.sigmoid(line_logits)  # [batch_size, num_lines]

        return enhanced_line_features, line_logits, line_probs, slice_attention_weights

class ProbabilityGuidedMechanism(nn.Module):
    """概率引导机制

    支持两种模式:
    1. 批量行模式: line_features [batch_size, num_lines, hidden_dim]
    2. 单行模式: line_features [batch_size, hidden_dim]
    """
    def __init__(self, hidden_dim=768, delta=0.5, lambda_threshold=0.2, dropout=0.1):
        super(ProbabilityGuidedMechanism, self).__init__()
        self.hidden_dim = hidden_dim
        self.delta = delta
        self.lambda_threshold = lambda_threshold
        self.dropout = nn.Dropout(dropout)

        # 行级预测器
        self.line_predictor = nn.Linear(hidden_dim, 1)

    def forward(self, line_features, func_probs):
        """
        Args:
            line_features: [batch_size, hidden_dim] 或 [batch_size, num_lines, hidden_dim]
            func_probs: [batch_size] 或 [batch_size, 1]
        """
        # 确保 func_probs 是一维的
        if len(func_probs.shape) > 1:
            func_probs = func_probs.squeeze(-1)  # [batch_size]

        # 判断是单行模式还是批量行模式
        if len(line_features.shape) == 2:
            return self._forward_single_line(line_features, func_probs)
        else:
            return self._forward_batch_lines(line_features, func_probs)

    def _forward_single_line(self, line_features, func_probs):
        """单行模式的前向传播"""
        # 应用dropout
        line_features = self.dropout(line_features)

        # 计算先验概率
        prior_probs = (1 - self.delta) * 0.5 + self.delta * func_probs  # [batch_size]

        # 初始行级预测
        line_logits = self.line_predictor(line_features).squeeze(-1)  # [batch_size]
        line_probs = torch.sigmoid(line_logits)  # [batch_size]

        # 先验调整后的概率（贝叶斯更新）
        log_line_probs = torch.log(line_probs + 1e-10)
        log_prior_probs = torch.log(prior_probs + 1e-10)
        log_complement_line_probs = torch.log(1 - line_probs + 1e-10)
        log_complement_prior_probs = torch.log(1 - prior_probs + 1e-10)

        # 计算调整后的概率
        adjusted_line_probs = torch.exp(
            log_line_probs + log_prior_probs -
            torch.log(
                torch.exp(log_line_probs + log_prior_probs) +
                torch.exp(log_complement_line_probs + log_complement_prior_probs)
            )
        )

        # 计算动态阈值
        base_threshold = 0.5
        confidence = torch.abs(func_probs - 0.5) * 2  # [batch_size]
        threshold_shift = torch.sign(func_probs - 0.5) * confidence * self.lambda_threshold
        dynamic_thresholds = base_threshold - threshold_shift  # [batch_size]

        # 应用动态阈值
        line_predictions = (adjusted_line_probs > dynamic_thresholds).float()  # [batch_size]

        return adjusted_line_probs, line_predictions, dynamic_thresholds

    def _forward_batch_lines(self, line_features, func_probs):
        """批量行模式的前向传播 - 原始设计"""
        batch_size, num_lines, _ = line_features.size()

        # 应用dropout
        line_features = self.dropout(line_features)

        # 计算先验概率
        prior_probs = (1 - self.delta) * 0.5 + self.delta * func_probs

        # 初始行级预测
        line_logits = self.line_predictor(line_features).squeeze(-1)  # [batch_size, num_lines]
        line_probs = torch.sigmoid(line_logits)  # [batch_size, num_lines]

        # 先验调整后的概率（贝叶斯更新）
        log_line_probs = torch.log(line_probs + 1e-10)
        log_prior_probs = torch.log(prior_probs + 1e-10)
        log_complement_line_probs = torch.log(1 - line_probs + 1e-10)
        log_complement_prior_probs = torch.log(1 - prior_probs + 1e-10)

        # 计算调整后的概率
        adjusted_line_probs = torch.exp(
            log_line_probs + log_prior_probs.unsqueeze(-1) -
            torch.log(
                torch.exp(log_line_probs + log_prior_probs.unsqueeze(-1)) +
                torch.exp(log_complement_line_probs + log_complement_prior_probs.unsqueeze(-1))
            )
        )

        # 计算动态阈值
        base_threshold = 0.5
        confidence = torch.abs(func_probs - 0.5) * 2  # [batch_size]
        threshold_shift = torch.sign(func_probs - 0.5) * confidence * self.lambda_threshold
        dynamic_thresholds = base_threshold - threshold_shift.unsqueeze(-1)  # [batch_size, 1]

        # 应用动态阈值
        line_predictions = (adjusted_line_probs > dynamic_thresholds).float()  # [batch_size, num_lines]

        return adjusted_line_probs, line_predictions, dynamic_thresholds

class FineGrainedModel(nn.Module):
    """细粒度漏洞检测模型

    支持两种模式:
    1. 独立模式 (use_interaction=False): 仅使用行级特征进行预测
    2. 交互模式 (use_interaction=True): 结合粗粒度特征进行预测
    """
    def __init__(self, codebert_path, hidden_dim=768, alpha=0.3, delta=0.5, lambda_threshold=0.2, reg_weight=0.1, use_interaction=True, dropout=0.1):
        super(FineGrainedModel, self).__init__()
        self.use_interaction = use_interaction
        self.hidden_dim = hidden_dim
        self.line_encoder = LineEncoder(codebert_path, hidden_dim, dropout)

        # 交互相关组件（仅在交互模式下使用）
        if use_interaction:
            self.coarse_fine_interaction = CoarseFineInteraction(hidden_dim, alpha, dropout)
            self.probability_guided = ProbabilityGuidedMechanism(hidden_dim, delta, lambda_threshold, dropout)
            self.reg_weight = reg_weight

        # 独立模式下的预测层
        self.independent_predictor = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, line_input_ids, line_attention_mask, func_features=None, slice_features=None,
                slice_probs=None, func_probs=None, slice_line_mask=None):
        """
        Args:
            line_input_ids: [batch_size, seq_len]
            line_attention_mask: [batch_size, seq_len]
            func_features: [batch_size, hidden_dim] 函数特征 (交互模式)
            slice_features: [batch_size, num_slices, hidden_dim] 切片特征 (交互模式)
            slice_probs: [batch_size, num_slices] 切片概率 (交互模式)
            func_probs: [batch_size] 函数概率 (交互模式)
            slice_line_mask: [batch_size, num_slices] 切片-行对齐掩码 (交互模式)
        """
        # 编码每行代码
        line_features = self.line_encoder(line_input_ids, line_attention_mask)  # [batch_size, hidden_dim]
        line_features = self.dropout(line_features)

        if self.use_interaction and all(x is not None for x in [func_features, slice_features, slice_probs, func_probs]):
            # 交互模式
            # 粗细粒度交互
            enhanced_line_features, line_logits, line_probs, slice_attention_weights = self.coarse_fine_interaction(
                line_features, func_features, slice_features, slice_probs, slice_line_mask
            )

            # 概率引导机制
            adjusted_line_probs, line_predictions, dynamic_thresholds = self.probability_guided(
                enhanced_line_features, func_probs
            )

            # 计算约束正则化损失
            reg_loss = self.reg_weight * torch.abs(line_probs - func_probs)

            return {
                'line_logits': line_logits,  # 用于损失计算
                'line_probs': line_probs,
                'adjusted_line_probs': adjusted_line_probs,
                'line_predictions': line_predictions,
                'dynamic_thresholds': dynamic_thresholds,
                'slice_attention_weights': slice_attention_weights,
                'reg_loss': reg_loss
            }
        else:
            # 独立模式
            line_logits = self.independent_predictor(line_features).squeeze(-1)  # [batch_size]
            line_probs = torch.sigmoid(line_logits)  # [batch_size]
            line_predictions = (line_probs > 0.5).float()  # [batch_size]

            return {
                'line_logits': line_logits,  # 用于训练时的损失计算
                'line_probs': line_probs,    # 用于评估和推理
                'line_predictions': line_predictions
            } 