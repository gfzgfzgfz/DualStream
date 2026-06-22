import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel

class FunctionEncoder(nn.Module):
    """函数编码器，用于编码函数代码"""
    def __init__(self, func_codebert_path, hidden_dim=768):
        super(FunctionEncoder, self).__init__()
        self.codebert = RobertaModel.from_pretrained(func_codebert_path)
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, input_ids, attention_mask):
        # 获取CodeBERT的输出
        outputs = self.codebert(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
        
        # 计算注意力权重
        attention_scores = self.attention(hidden_states).squeeze(-1)  # [batch_size, seq_len]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_len]
        
        # 获取[CLS]标记的表示
        cls_representation = hidden_states[:, 0, :]  # [batch_size, hidden_dim]
        
        # 加权求和得到函数表示
        weighted_sum = torch.sum(hidden_states * attention_weights.unsqueeze(-1), dim=1)  # [batch_size, hidden_dim]
        
        # 最终函数表示
        function_representation = cls_representation + weighted_sum  # [batch_size, hidden_dim]
        
        return function_representation, attention_weights

class SliceEncoder(nn.Module):
    """切片编码器，用于编码代码切片"""
    def __init__(self, slice_codebert_path, hidden_dim=768):
        super(SliceEncoder, self).__init__()
        self.codebert = RobertaModel.from_pretrained(slice_codebert_path)
        
    def forward(self, input_ids, attention_mask):
        # 获取CodeBERT的输出，不使用token_type_ids
        outputs = self.codebert(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        
        # 使用[CLS]标记的表示作为切片表示
        slice_representation = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
        
        return slice_representation

class SliceFilter(nn.Module):
    """切片过滤模块，用于选择最相关的切片"""
    def __init__(self, hidden_dim=768, k=5, lambda_mmr=0.7):
        super(SliceFilter, self).__init__()
        # 使用更复杂的相关性评分网络
        self.relevance_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.k = k
        self.lambda_mmr = lambda_mmr
        
    def forward(self, slice_features):
        # 计算相关性得分
        relevance_scores = torch.sigmoid(self.relevance_scorer(slice_features)).squeeze(-1)  # [batch_size, num_slices]
        
        # 选择得分最高的k个切片
        top_k_scores, top_k_indices = torch.topk(relevance_scores, min(self.k, relevance_scores.size(1)), dim=1)
        
        # 获取选中的切片特征
        selected_slices = torch.gather(slice_features, 1, top_k_indices.unsqueeze(-1).expand(-1, -1, slice_features.size(-1)))
        
        return selected_slices, top_k_scores, top_k_indices

class HierarchicalAttention(nn.Module):
    """双向层次注意力机制"""
    def __init__(self, hidden_dim=768):
        super(HierarchicalAttention, self).__init__()
        # 函数到切片方向的对齐权重投影 W_q
        self.func_to_slice = nn.Linear(hidden_dim, hidden_dim)
        # 切片到函数方向的对齐权重投影 W_s
        self.slice_to_func = nn.Linear(hidden_dim, hidden_dim)
        # 切片到函数方向的值投影：W_v 作用于切片，U_v 作用于函数
        self.value_slice = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_func = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, func_features, slice_features):
        batch_size, num_slices, hidden_dim = slice_features.size()

        # 函数到切片的注意力
        func_expanded = func_features.unsqueeze(1).expand(-1, num_slices, -1)  # [batch_size, num_slices, hidden_dim]
        func_to_slice_scores = torch.sum(func_expanded * self.func_to_slice(slice_features), dim=-1)  # [batch_size, num_slices]
        func_to_slice_weights = F.softmax(func_to_slice_scores, dim=1)  # [batch_size, num_slices]

        # 加权切片表示
        enhanced_slice = torch.sum(slice_features * func_to_slice_weights.unsqueeze(-1), dim=1)  # [batch_size, hidden_dim]

        # 切片到函数的注意力：eta_i = softmax(h_{s_i}^T W_s h_f)
        slice_to_func_scores = torch.sum(slice_features * self.slice_to_func(func_expanded), dim=-1)  # [batch_size, num_slices]
        slice_to_func_weights = F.softmax(slice_to_func_scores, dim=1)  # [batch_size, num_slices]

        # 函数感知的局部组合表示：c_{s->f} = sum_i eta_i (W_v h_{s_i} + U_v h_f)
        # 由于 sum_i eta_i = 1，可等价写为 sum_i eta_i W_v h_{s_i} + U_v h_f
        weighted_slice_value = torch.sum(
            self.value_slice(slice_features) * slice_to_func_weights.unsqueeze(-1), dim=1
        )  # [batch_size, hidden_dim]
        enhanced_func = weighted_slice_value + self.value_func(func_features)  # [batch_size, hidden_dim]

        return enhanced_func, enhanced_slice, func_to_slice_weights, slice_to_func_weights

class FeatureFusion(nn.Module):
    """特征融合门控机制"""
    def __init__(self, hidden_dim=768):
        super(FeatureFusion, self).__init__()
        self.slice_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        self.func_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(self, func_enhanced, slice_enhanced, slice_features):
        # 计算切片平均表示
        slice_avg = torch.mean(slice_features, dim=1)  # [batch_size, hidden_dim]
        
        # 切片信息门
        slice_gate_input = torch.cat([func_enhanced, slice_enhanced], dim=1)  # [batch_size, hidden_dim*2]
        slice_gate = self.slice_gate(slice_gate_input)  # [batch_size, hidden_dim]
        
        # 函数信息门
        func_gate_input = torch.cat([slice_avg, func_enhanced], dim=1)  # [batch_size, hidden_dim*2]
        func_gate = self.func_gate(func_gate_input)  # [batch_size, hidden_dim]
        
        # 最终融合特征
        final_features = func_gate * func_enhanced + slice_gate * slice_enhanced  # [batch_size, hidden_dim]
        
        return final_features

class PredictionLayer(nn.Module):
    """增强的预测层，用于函数级和切片级预测"""
    def __init__(self, hidden_dim=768):
        super(PredictionLayer, self).__init__()
        # 增强函数级预测器
        self.func_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 增强切片级预测器
        self.slice_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # 增加输入维度，包含全局+局部信息
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 添加注意力机制，从函数特征捕获上下文
        self.context_attn = nn.Linear(hidden_dim, 1)
        
    def forward(self, final_features, slice_features):
        batch_size, num_slices, hidden_dim = slice_features.size()
        
        # 函数级预测
        func_logits = self.func_predictor(final_features)  # [batch_size, 1]
        func_probs = torch.sigmoid(func_logits)  # [batch_size, 1]
        
        # 计算函数上下文对切片的重要性
        context_weights = torch.sigmoid(self.context_attn(final_features))  # [batch_size, 1]
        
        # 对每个切片特征增加函数级上下文
        func_context = final_features.unsqueeze(1).expand(-1, num_slices, -1)  # [batch_size, num_slices, hidden_dim]
        
        # 构建增强切片特征，包含全局函数上下文和局部切片信息
        enhanced_slice_features = torch.cat([
            slice_features,  # 局部切片特征
            func_context     # 全局函数上下文
        ], dim=2)  # [batch_size, num_slices, hidden_dim*2]
        
        # 切片级预测
        enhanced_slice_features_flat = enhanced_slice_features.view(-1, hidden_dim * 2)
        slice_logits = self.slice_predictor(enhanced_slice_features_flat)  # [batch_size*num_slices, 1]
        slice_probs = torch.sigmoid(slice_logits)  # [batch_size*num_slices, 1]
        slice_probs = slice_probs.view(batch_size, num_slices)  # [batch_size, num_slices]
        
        func_influence = func_probs.squeeze(-1).unsqueeze(1).expand(-1, num_slices)  # [batch_size, num_slices]
        adjusted_slice_probs = slice_probs * (0.3 + 0.7 * func_influence)  # 如果函数不可能有漏洞，切片概率会降低
        
        return func_probs, adjusted_slice_probs

class CoarseGrainedModel(nn.Module):
    """粗粒度漏洞检测模型"""
    def __init__(self, func_codebert_path, slice_codebert_path, hidden_dim=768, k=5, lambda_mmr=0.7, ablation_mode='full'):
        super(CoarseGrainedModel, self).__init__()
        self.ablation_mode = ablation_mode  # 'func_only', 'slice_only', 'no_fusion', 'full'
        self.hidden_dim = hidden_dim
        self.func_encoder = FunctionEncoder(func_codebert_path, hidden_dim)
        self.slice_encoder = SliceEncoder(slice_codebert_path, hidden_dim)
        self.slice_filter = SliceFilter(hidden_dim, k, lambda_mmr)
        self.hierarchical_attention = HierarchicalAttention(hidden_dim)
        self.feature_fusion = FeatureFusion(hidden_dim)
        self.prediction_layer = PredictionLayer(hidden_dim)
        self.k = k
        
    def forward(self, func_input_ids, func_attention_mask, slice_input_ids, slice_attention_mask, slice_indices=None):
        try:
            # 根据消融模式选择不同的前向传播路径
            if self.ablation_mode == 'func_only':
                return self._forward_func_only(func_input_ids, func_attention_mask)
            elif self.ablation_mode == 'slice_only':
                return self._forward_slice_only(slice_input_ids, slice_attention_mask, slice_indices)
            elif self.ablation_mode == 'no_fusion':
                return self._forward_no_fusion(func_input_ids, func_attention_mask, slice_input_ids, slice_attention_mask, slice_indices)
            else:  # 'full' - 完整模型（默认）
                return self._forward_full(func_input_ids, func_attention_mask, slice_input_ids, slice_attention_mask, slice_indices)
        except Exception as e:
            import traceback
            print(f"模型前向传播异常: {e}")
            print(traceback.format_exc())
            raise

    def _forward_full(self, func_input_ids, func_attention_mask, slice_input_ids, slice_attention_mask, slice_indices=None):
        """完整模型的前向传播（原始逻辑）"""
        # 获取批次大小和切片信息
        batch_size, num_slices, seq_len = slice_input_ids.size()

        # 编码函数
        func_features, func_attention = self.func_encoder(func_input_ids, func_attention_mask)

        # 处理切片输入 - 将三维张量重塑为二维进行处理 [batch_size * num_slices, seq_len]
        slice_input_ids_reshaped = slice_input_ids.view(batch_size * num_slices, seq_len)
        slice_attention_mask_reshaped = slice_attention_mask.view(batch_size * num_slices, seq_len)

        # 编码所有切片
        slice_features_flat = self.slice_encoder(slice_input_ids_reshaped, slice_attention_mask_reshaped)

        # 将结果重塑回 [batch_size, num_slices, hidden_dim]
        slice_features = slice_features_flat.view(batch_size, num_slices, -1)

        # 过滤切片
        selected_slices, slice_scores, selected_indices = self.slice_filter(slice_features)

        # 层次注意力
        func_enhanced, slice_enhanced, func_to_slice_weights, slice_to_func_weights = self.hierarchical_attention(
            func_features, selected_slices)

        # 特征融合
        final_features = self.feature_fusion(func_enhanced, slice_enhanced, selected_slices)

        # 预测
        func_probs, slice_probs = self.prediction_layer(final_features, selected_slices)

        # 处理索引映射 - 向量化版本
        original_indices = selected_indices
        if slice_indices is not None and selected_indices.size(1) > 0:
            try:
                # 确保索引在有效范围内
                max_index = slice_indices.size(1) - 1
                if max_index >= 0:
                    safe_indices = torch.clamp(selected_indices, 0, max_index)
                    # 使用gather操作一次性获取所有索引
                    original_indices = torch.gather(slice_indices, 1, safe_indices)
            except Exception as e:
                # 如果映射失败，直接使用选中的索引
                original_indices = selected_indices

        return {
            'func_probs': func_probs,
            'slice_probs': slice_probs,
            'func_attention': func_attention,
            'func_to_slice_weights': func_to_slice_weights,
            'slice_to_func_weights': slice_to_func_weights,
            'slice_scores': slice_scores,
            'selected_indices': original_indices,
            'func_features': func_features,
            'selected_slice_features': selected_slices
        }

    def _forward_func_only(self, func_input_ids, func_attention_mask):
        """只使用函数特征的前向传播"""
        batch_size = func_input_ids.size(0)

        # 编码函数
        func_features, func_attention = self.func_encoder(func_input_ids, func_attention_mask)

        # 直接用函数特征预测
        func_probs = torch.sigmoid(self.prediction_layer.func_predictor(func_features))

        # 创建虚拟的切片预测（全零）
        slice_probs = torch.zeros(batch_size, self.k, device=func_features.device)

        return {
            'func_probs': func_probs,
            'slice_probs': slice_probs,
            'func_attention': func_attention,
            'func_to_slice_weights': None,
            'slice_to_func_weights': None,
            'slice_scores': None,
            'selected_indices': torch.zeros(batch_size, self.k, dtype=torch.long, device=func_features.device),
            'func_features': func_features,
            'selected_slice_features': None
        }

    def _forward_slice_only(self, slice_input_ids, slice_attention_mask, slice_indices=None):
        """只使用切片特征的前向传播"""
        batch_size, num_slices, seq_len = slice_input_ids.size()

        # 编码切片
        slice_input_ids_reshaped = slice_input_ids.view(batch_size * num_slices, seq_len)
        slice_attention_mask_reshaped = slice_attention_mask.view(batch_size * num_slices, seq_len)
        slice_features_flat = self.slice_encoder(slice_input_ids_reshaped, slice_attention_mask_reshaped)
        slice_features = slice_features_flat.view(batch_size, num_slices, -1)

        # 过滤切片
        selected_slices, slice_scores, selected_indices = self.slice_filter(slice_features)

        # 对切片特征做平均池化得到全局表示
        slice_avg = torch.mean(selected_slices, dim=1)  # [batch_size, hidden_dim]

        # 用聚合后的切片特征预测函数级漏洞
        func_probs = torch.sigmoid(self.prediction_layer.func_predictor(slice_avg))

        # 切片级预测
        slice_probs_list = []
        for i in range(selected_slices.size(1)):
            slice_feat = selected_slices[:, i, :]
            enhanced_feat = torch.cat([slice_feat, slice_avg], dim=1)
            slice_prob = torch.sigmoid(self.prediction_layer.slice_predictor(enhanced_feat))
            slice_probs_list.append(slice_prob)
        slice_probs = torch.cat(slice_probs_list, dim=1)

        # 处理索引映射
        original_indices = selected_indices
        if slice_indices is not None and selected_indices.size(1) > 0:
            try:
                max_index = slice_indices.size(1) - 1
                if max_index >= 0:
                    safe_indices = torch.clamp(selected_indices, 0, max_index)
                    original_indices = torch.gather(slice_indices, 1, safe_indices)
            except:
                pass

        return {
            'func_probs': func_probs,
            'slice_probs': slice_probs,
            'func_attention': None,
            'func_to_slice_weights': None,
            'slice_to_func_weights': None,
            'slice_scores': slice_scores,
            'selected_indices': original_indices,
            'func_features': None,
            'selected_slice_features': selected_slices
        }

    def _forward_no_fusion(self, func_input_ids, func_attention_mask, slice_input_ids, slice_attention_mask, slice_indices=None):
        """函数+切片特征，但不使用融合机制（简单拼接）"""
        batch_size, num_slices, seq_len = slice_input_ids.size()

        # 编码函数
        func_features, func_attention = self.func_encoder(func_input_ids, func_attention_mask)

        # 编码切片
        slice_input_ids_reshaped = slice_input_ids.view(batch_size * num_slices, seq_len)
        slice_attention_mask_reshaped = slice_attention_mask.view(batch_size * num_slices, seq_len)
        slice_features_flat = self.slice_encoder(slice_input_ids_reshaped, slice_attention_mask_reshaped)
        slice_features = slice_features_flat.view(batch_size, num_slices, -1)

        # 过滤切片
        selected_slices, slice_scores, selected_indices = self.slice_filter(slice_features)

        # 简单拼接：函数特征 + 切片平均特征
        slice_avg = torch.mean(selected_slices, dim=1)
        combined_features = func_features + slice_avg  # 简单相加而不是门控融合

        # 预测
        func_probs, slice_probs = self.prediction_layer(combined_features, selected_slices)

        # 处理索引映射
        original_indices = selected_indices
        if slice_indices is not None and selected_indices.size(1) > 0:
            try:
                max_index = slice_indices.size(1) - 1
                if max_index >= 0:
                    safe_indices = torch.clamp(selected_indices, 0, max_index)
                    original_indices = torch.gather(slice_indices, 1, safe_indices)
            except:
                pass

        return {
            'func_probs': func_probs,
            'slice_probs': slice_probs,
            'func_attention': func_attention,
            'func_to_slice_weights': None,
            'slice_to_func_weights': None,
            'slice_scores': slice_scores,
            'selected_indices': original_indices,
            'func_features': func_features,
            'selected_slice_features': selected_slices
        } 