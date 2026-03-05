
import torch
from math import inf
from math import nan
NoneType = type(None)
import torch
from torch import device
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree

from torch.nn import *
class DecoderModelMerged(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model_attn_mask_reformat_past_key_subgraph_Shape = torch.load(r'decoder_model_merged_torch/model_attn_mask_reformat_past_key_subgraph_Shape.pt', weights_only=False) # OnnxShape()
        self.model_attn_mask_reformat_input_ids_subgraph_Shape_2 = torch.load(r'decoder_model_merged_torch/model_attn_mask_reformat_input_ids_subgraph_Shape_2.pt', weights_only=False) # OnnxShape()
        self.model_attn_mask_reformat_attn_mask_subgraph_Shape_1 = torch.load(r'decoder_model_merged_torch/model_attn_mask_reformat_attn_mask_subgraph_Shape_1.pt', weights_only=False) # OnnxShape()
        self.initializers = torch.load(r'decoder_model_merged_torch/initializers.pt', weights_only=False) # Module()
        self.model_layers_0_input_layernorm_LayerNorm = torch.load(r'decoder_model_merged_torch/model_layers_0_input_layernorm_LayerNorm.pt', weights_only=False) # SimplifiedLayerNormOp()
        self.model_attn_mask_reformat_attn_mask_subgraph_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_attn_mask_reformat_attn_mask_subgraph_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_attn_mask_reformat_past_key_subgraph_Gather = torch.load(r'decoder_model_merged_torch/model_attn_mask_reformat_past_key_subgraph_Gather.pt', weights_only=False) # OnnxGather()
        self.model_attn_mask_reformat_input_ids_subgraph_Gather_2 = torch.load(r'decoder_model_merged_torch/model_attn_mask_reformat_input_ids_subgraph_Gather_2.pt', weights_only=False) # OnnxGather()
        self.model_attn_mask_reformat_input_ids_subgraph_Gather_1 = torch.load(r'decoder_model_merged_torch/model_attn_mask_reformat_input_ids_subgraph_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_attn_mask_reformat_attn_mask_subgraph_Gather_1 = torch.load(r'decoder_model_merged_torch/model_attn_mask_reformat_attn_mask_subgraph_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_attn_mask_reformat_attn_mask_subgraph_Gather_2 = torch.load(r'decoder_model_merged_torch/model_attn_mask_reformat_attn_mask_subgraph_Gather_2.pt', weights_only=False) # OnnxGather()
        self.model_layers_0_attn_q_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_0_attn_q_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_0_attn_k_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_0_attn_k_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_0_attn_v_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_0_attn_v_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_attn_mask_reformat_input_ids_subgraph_Add_1 = torch.load(r'decoder_model_merged_torch/model_attn_mask_reformat_input_ids_subgraph_Add_1.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_attn_mask_reformat_input_ids_subgraph_Unsqueeze_4 = torch.load(r'decoder_model_merged_torch/model_attn_mask_reformat_input_ids_subgraph_Unsqueeze_4.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_attn_mask_reformat_input_ids_subgraph_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_attn_mask_reformat_input_ids_subgraph_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_attn_mask_reformat_attn_mask_subgraph_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_attn_mask_reformat_attn_mask_subgraph_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_attn_mask_reformat_attn_mask_subgraph_Unsqueeze_2 = torch.load(r'decoder_model_merged_torch/model_attn_mask_reformat_attn_mask_subgraph_Unsqueeze_2.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.com_microsoft__model_layers_0_attn_q_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_0_attn_q_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.com_microsoft__model_layers_0_attn_k_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_0_attn_k_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.model_layers_0_attn_v_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_0_attn_v_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_attn_mask_reformat_input_ids_subgraph_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_attn_mask_reformat_input_ids_subgraph_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_attn_mask_reformat_input_ids_subgraph_Concat_2 = torch.load(r'decoder_model_merged_torch/model_attn_mask_reformat_input_ids_subgraph_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_attn_mask_reformat_attn_mask_subgraph_Concat = torch.load(r'decoder_model_merged_torch/model_attn_mask_reformat_attn_mask_subgraph_Concat.pt', weights_only=False) # OnnxConcat()
        self.model_layers_0_attn_k_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_0_attn_k_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_0_attn_v_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_0_attn_v_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_attn_mask_reformat_input_ids_subgraph_ConstantOfShape_2 = torch.load(r'decoder_model_merged_torch/model_attn_mask_reformat_input_ids_subgraph_ConstantOfShape_2.pt', weights_only=False) # OnnxConstantOfShape()
        self.model_attn_mask_reformat_input_ids_subgraph_Concat_1 = torch.load(r'decoder_model_merged_torch/model_attn_mask_reformat_input_ids_subgraph_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_0_attn_k_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_0_attn_k_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_0_attn_v_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_0_attn_v_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_attn_mask_reformat_attn_mask_subgraph_Equal = torch.load(r'decoder_model_merged_torch/model_attn_mask_reformat_attn_mask_subgraph_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_attn_mask_reformat_input_ids_subgraph_Shape_4 = torch.load(r'decoder_model_merged_torch/model_attn_mask_reformat_input_ids_subgraph_Shape_4.pt', weights_only=False) # OnnxShape()
        self.model_layers_0_attn_k_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_0_attn_k_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_0_attn_v_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_0_attn_v_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_0_attn_v_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_0_attn_v_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_attn_mask_reformat_input_ids_subgraph_Equal = torch.load(r'decoder_model_merged_torch/model_attn_mask_reformat_input_ids_subgraph_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_attn_mask_reformat_attn_mask_subgraph_Where_1 = torch.load(r'decoder_model_merged_torch/model_attn_mask_reformat_attn_mask_subgraph_Where_1.pt', weights_only=False) # OnnxWhere()
        self.model_attn_mask_reformat_input_ids_subgraph_Slice_1 = torch.load(r'decoder_model_merged_torch/model_attn_mask_reformat_input_ids_subgraph_Slice_1.pt', weights_only=False) # OnnxSlice()
        self.model_layers_0_attn_k_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_0_attn_k_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_0_attn_k_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_0_attn_k_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_0_attn_v_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_0_attn_v_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_0_attn_v_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_0_attn_v_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_attn_mask_reformat_input_ids_subgraph_Where_1 = torch.load(r'decoder_model_merged_torch/model_attn_mask_reformat_input_ids_subgraph_Where_1.pt', weights_only=False) # OnnxWhere()
        self.model_attn_mask_reformat_attn_mask_subgraph_Expand = torch.load(r'decoder_model_merged_torch/model_attn_mask_reformat_attn_mask_subgraph_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_attn_mask_reformat_input_ids_subgraph_Squeeze_1 = torch.load(r'decoder_model_merged_torch/model_attn_mask_reformat_input_ids_subgraph_Squeeze_1.pt', weights_only=False) # OnnxSqueezeDynamicAxes()
        self.model_layers_0_attn_k_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_0_attn_k_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_0_attn_k_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_0_attn_k_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_0_attn_v_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_0_attn_v_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_0_attn_v_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_0_attn_v_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_attn_mask_reformat_attn_mask_subgraph_Cast_1 = torch.load(r'decoder_model_merged_torch/model_attn_mask_reformat_attn_mask_subgraph_Cast_1.pt', weights_only=False) # OnnxCast()
        self.model_attn_mask_reformat_input_ids_subgraph_output_0 = torch.load(r'decoder_model_merged_torch/model_attn_mask_reformat_input_ids_subgraph_output_0.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_attn_mask_reformat_input_ids_subgraph_Range = torch.load(r'decoder_model_merged_torch/model_attn_mask_reformat_input_ids_subgraph_Range.pt', weights_only=False) # OnnxRange()
        self.model_layers_0_attn_k_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_0_attn_k_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_0_attn_k_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_0_attn_k_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_0_attn_v_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_0_attn_v_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_0_attn_v_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_0_attn_v_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_attn_mask_reformat_attn_mask_subgraph_Sub = torch.load(r'decoder_model_merged_torch/model_attn_mask_reformat_attn_mask_subgraph_Sub.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_attn_mask_reformat_input_ids_subgraph_Concat_3 = torch.load(r'decoder_model_merged_torch/model_attn_mask_reformat_input_ids_subgraph_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_attn_mask_reformat_input_ids_subgraph_Add_2 = torch.load(r'decoder_model_merged_torch/model_attn_mask_reformat_input_ids_subgraph_Add_2.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_0_attn_k_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_0_attn_k_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_0_attn_k_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_0_attn_k_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_attn_mask_reformat_attn_mask_subgraph_Cast_2 = torch.load(r'decoder_model_merged_torch/model_attn_mask_reformat_attn_mask_subgraph_Cast_2.pt', weights_only=False) # OnnxCast()
        self.model_layers_0_attn_v_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_0_attn_v_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_attn_mask_reformat_input_ids_subgraph_Reshape = torch.load(r'decoder_model_merged_torch/model_attn_mask_reformat_input_ids_subgraph_Reshape.pt', weights_only=False) # OnnxReshape()
        self.model_attn_mask_reformat_attn_mask_subgraph_Where_2 = torch.load(r'decoder_model_merged_torch/model_attn_mask_reformat_attn_mask_subgraph_Where_2.pt', weights_only=False) # OnnxWhere()
        self.model_layers_0_attn_k_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_0_attn_k_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_0_attn_v_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_0_attn_v_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_attn_mask_reformat_input_ids_subgraph_Less = torch.load(r'decoder_model_merged_torch/model_attn_mask_reformat_input_ids_subgraph_Less.pt', weights_only=False) # OnnxCompare()
        self.model_layers_0_attn_k_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_0_attn_k_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_0_attn_v_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_0_attn_v_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_attn_mask_reformat_input_ids_subgraph_Where_2 = torch.load(r'decoder_model_merged_torch/model_attn_mask_reformat_input_ids_subgraph_Where_2.pt', weights_only=False) # OnnxWhere()
        self.model_layers_0_attn_k_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_0_attn_k_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_0_attn_v_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_0_attn_v_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_attn_mask_reformat_input_ids_subgraph_Unsqueeze_8 = torch.load(r'decoder_model_merged_torch/model_attn_mask_reformat_input_ids_subgraph_Unsqueeze_8.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_0_attn_k_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_0_attn_k_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_0_attn_v_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_0_attn_v_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_attn_mask_reformat_input_ids_subgraph_Expand = torch.load(r'decoder_model_merged_torch/model_attn_mask_reformat_input_ids_subgraph_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_0_attn_k_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_0_attn_k_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_0_attn_v_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_0_attn_v_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.model_attn_mask_reformat_Add = torch.load(r'decoder_model_merged_torch/model_attn_mask_reformat_Add.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_0_attn_k_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_0_attn_k_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.model_attn_mask_reformat_Tile = torch.load(r'decoder_model_merged_torch/model_attn_mask_reformat_Tile.pt', weights_only=False) # OnnxTile()
        self.com_microsoft__model_layers_0_attn_MultiHeadAttention = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_0_attn_MultiHeadAttention.pt', weights_only=False) # MultiHeadAttentionOp()
        self.model_layers_0_attn_o_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_0_attn_o_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_0_post_attention_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_0_post_attention_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_0_mlp_gate_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_0_mlp_gate_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_0_mlp_up_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_0_mlp_up_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_0_mlp_act_fn_Sigmoid = torch.load(r'decoder_model_merged_torch/model_layers_0_mlp_act_fn_Sigmoid.pt', weights_only=False) # Sigmoid()
        self.model_layers_0_mlp_act_fn_Mul = torch.load(r'decoder_model_merged_torch/model_layers_0_mlp_act_fn_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_0_mlp_Mul = torch.load(r'decoder_model_merged_torch/model_layers_0_mlp_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_0_mlp_down_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_0_mlp_down_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_1_input_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_1_input_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_1_attn_q_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_1_attn_q_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_1_attn_k_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_1_attn_k_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_1_attn_v_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_1_attn_v_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_1_attn_q_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_1_attn_q_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.com_microsoft__model_layers_1_attn_k_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_1_attn_k_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.model_layers_1_attn_v_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_1_attn_v_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_1_attn_k_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_1_attn_k_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_1_attn_v_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_1_attn_v_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_1_attn_k_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_1_attn_k_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_1_attn_v_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_1_attn_v_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_1_attn_k_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_1_attn_k_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_1_attn_v_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_1_attn_v_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_1_attn_v_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_1_attn_v_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_1_attn_k_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_1_attn_k_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_1_attn_k_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_1_attn_k_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_1_attn_v_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_1_attn_v_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_1_attn_v_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_1_attn_v_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_1_attn_k_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_1_attn_k_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_1_attn_k_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_1_attn_k_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_1_attn_v_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_1_attn_v_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_1_attn_v_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_1_attn_v_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_1_attn_k_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_1_attn_k_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_1_attn_k_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_1_attn_k_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_1_attn_v_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_1_attn_v_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_1_attn_v_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_1_attn_v_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_1_attn_k_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_1_attn_k_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_1_attn_k_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_1_attn_k_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_1_attn_v_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_1_attn_v_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_1_attn_k_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_1_attn_k_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_1_attn_v_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_1_attn_v_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_1_attn_k_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_1_attn_k_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_1_attn_v_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_1_attn_v_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_1_attn_k_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_1_attn_k_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_1_attn_v_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_1_attn_v_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_1_attn_k_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_1_attn_k_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_1_attn_v_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_1_attn_v_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_1_attn_k_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_1_attn_k_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_1_attn_v_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_1_attn_v_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.model_layers_1_attn_k_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_1_attn_k_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.com_microsoft__model_layers_1_attn_MultiHeadAttention = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_1_attn_MultiHeadAttention.pt', weights_only=False) # MultiHeadAttentionOp()
        self.model_layers_1_attn_o_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_1_attn_o_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_1_post_attention_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_1_post_attention_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_1_mlp_gate_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_1_mlp_gate_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_1_mlp_up_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_1_mlp_up_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_1_mlp_act_fn_Sigmoid = torch.load(r'decoder_model_merged_torch/model_layers_1_mlp_act_fn_Sigmoid.pt', weights_only=False) # Sigmoid()
        self.model_layers_1_mlp_act_fn_Mul = torch.load(r'decoder_model_merged_torch/model_layers_1_mlp_act_fn_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_1_mlp_Mul = torch.load(r'decoder_model_merged_torch/model_layers_1_mlp_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_1_mlp_down_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_1_mlp_down_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_2_input_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_2_input_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_2_attn_q_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_2_attn_q_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_2_attn_k_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_2_attn_k_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_2_attn_v_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_2_attn_v_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_2_attn_q_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_2_attn_q_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.com_microsoft__model_layers_2_attn_k_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_2_attn_k_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.model_layers_2_attn_v_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_2_attn_v_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_2_attn_k_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_2_attn_k_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_2_attn_v_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_2_attn_v_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_2_attn_k_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_2_attn_k_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_2_attn_v_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_2_attn_v_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_2_attn_k_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_2_attn_k_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_2_attn_v_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_2_attn_v_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_2_attn_v_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_2_attn_v_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_2_attn_k_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_2_attn_k_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_2_attn_k_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_2_attn_k_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_2_attn_v_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_2_attn_v_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_2_attn_v_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_2_attn_v_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_2_attn_k_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_2_attn_k_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_2_attn_k_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_2_attn_k_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_2_attn_v_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_2_attn_v_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_2_attn_v_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_2_attn_v_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_2_attn_k_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_2_attn_k_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_2_attn_k_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_2_attn_k_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_2_attn_v_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_2_attn_v_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_2_attn_v_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_2_attn_v_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_2_attn_k_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_2_attn_k_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_2_attn_k_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_2_attn_k_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_2_attn_v_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_2_attn_v_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_2_attn_k_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_2_attn_k_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_2_attn_v_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_2_attn_v_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_2_attn_k_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_2_attn_k_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_2_attn_v_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_2_attn_v_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_2_attn_k_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_2_attn_k_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_2_attn_v_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_2_attn_v_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_2_attn_k_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_2_attn_k_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_2_attn_v_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_2_attn_v_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_2_attn_k_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_2_attn_k_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_2_attn_v_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_2_attn_v_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.model_layers_2_attn_k_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_2_attn_k_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.com_microsoft__model_layers_2_attn_MultiHeadAttention = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_2_attn_MultiHeadAttention.pt', weights_only=False) # MultiHeadAttentionOp()
        self.model_layers_2_attn_o_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_2_attn_o_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_2_post_attention_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_2_post_attention_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_2_mlp_gate_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_2_mlp_gate_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_2_mlp_up_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_2_mlp_up_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_2_mlp_act_fn_Sigmoid = torch.load(r'decoder_model_merged_torch/model_layers_2_mlp_act_fn_Sigmoid.pt', weights_only=False) # Sigmoid()
        self.model_layers_2_mlp_act_fn_Mul = torch.load(r'decoder_model_merged_torch/model_layers_2_mlp_act_fn_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_2_mlp_Mul = torch.load(r'decoder_model_merged_torch/model_layers_2_mlp_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_2_mlp_down_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_2_mlp_down_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_3_input_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_3_input_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_3_attn_q_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_3_attn_q_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_3_attn_k_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_3_attn_k_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_3_attn_v_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_3_attn_v_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_3_attn_q_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_3_attn_q_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.com_microsoft__model_layers_3_attn_k_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_3_attn_k_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.model_layers_3_attn_v_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_3_attn_v_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_3_attn_k_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_3_attn_k_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_3_attn_v_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_3_attn_v_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_3_attn_k_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_3_attn_k_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_3_attn_v_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_3_attn_v_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_3_attn_k_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_3_attn_k_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_3_attn_v_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_3_attn_v_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_3_attn_v_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_3_attn_v_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_3_attn_k_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_3_attn_k_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_3_attn_k_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_3_attn_k_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_3_attn_v_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_3_attn_v_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_3_attn_v_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_3_attn_v_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_3_attn_k_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_3_attn_k_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_3_attn_k_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_3_attn_k_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_3_attn_v_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_3_attn_v_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_3_attn_v_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_3_attn_v_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_3_attn_k_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_3_attn_k_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_3_attn_k_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_3_attn_k_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_3_attn_v_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_3_attn_v_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_3_attn_v_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_3_attn_v_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_3_attn_k_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_3_attn_k_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_3_attn_k_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_3_attn_k_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_3_attn_v_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_3_attn_v_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_3_attn_k_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_3_attn_k_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_3_attn_v_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_3_attn_v_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_3_attn_k_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_3_attn_k_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_3_attn_v_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_3_attn_v_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_3_attn_k_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_3_attn_k_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_3_attn_v_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_3_attn_v_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_3_attn_k_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_3_attn_k_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_3_attn_v_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_3_attn_v_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_3_attn_k_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_3_attn_k_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_3_attn_v_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_3_attn_v_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.model_layers_3_attn_k_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_3_attn_k_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.com_microsoft__model_layers_3_attn_MultiHeadAttention = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_3_attn_MultiHeadAttention.pt', weights_only=False) # MultiHeadAttentionOp()
        self.model_layers_3_attn_o_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_3_attn_o_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_3_post_attention_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_3_post_attention_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_3_mlp_gate_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_3_mlp_gate_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_3_mlp_up_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_3_mlp_up_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_3_mlp_act_fn_Sigmoid = torch.load(r'decoder_model_merged_torch/model_layers_3_mlp_act_fn_Sigmoid.pt', weights_only=False) # Sigmoid()
        self.model_layers_3_mlp_act_fn_Mul = torch.load(r'decoder_model_merged_torch/model_layers_3_mlp_act_fn_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_3_mlp_Mul = torch.load(r'decoder_model_merged_torch/model_layers_3_mlp_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_3_mlp_down_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_3_mlp_down_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_4_input_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_4_input_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_4_attn_q_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_4_attn_q_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_4_attn_k_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_4_attn_k_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_4_attn_v_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_4_attn_v_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_4_attn_q_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_4_attn_q_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.com_microsoft__model_layers_4_attn_k_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_4_attn_k_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.model_layers_4_attn_v_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_4_attn_v_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_4_attn_k_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_4_attn_k_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_4_attn_v_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_4_attn_v_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_4_attn_k_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_4_attn_k_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_4_attn_v_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_4_attn_v_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_4_attn_k_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_4_attn_k_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_4_attn_v_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_4_attn_v_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_4_attn_v_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_4_attn_v_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_4_attn_k_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_4_attn_k_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_4_attn_k_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_4_attn_k_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_4_attn_v_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_4_attn_v_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_4_attn_v_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_4_attn_v_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_4_attn_k_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_4_attn_k_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_4_attn_k_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_4_attn_k_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_4_attn_v_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_4_attn_v_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_4_attn_v_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_4_attn_v_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_4_attn_k_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_4_attn_k_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_4_attn_k_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_4_attn_k_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_4_attn_v_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_4_attn_v_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_4_attn_v_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_4_attn_v_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_4_attn_k_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_4_attn_k_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_4_attn_k_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_4_attn_k_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_4_attn_v_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_4_attn_v_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_4_attn_k_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_4_attn_k_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_4_attn_v_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_4_attn_v_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_4_attn_k_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_4_attn_k_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_4_attn_v_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_4_attn_v_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_4_attn_k_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_4_attn_k_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_4_attn_v_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_4_attn_v_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_4_attn_k_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_4_attn_k_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_4_attn_v_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_4_attn_v_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_4_attn_k_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_4_attn_k_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_4_attn_v_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_4_attn_v_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.model_layers_4_attn_k_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_4_attn_k_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.com_microsoft__model_layers_4_attn_MultiHeadAttention = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_4_attn_MultiHeadAttention.pt', weights_only=False) # MultiHeadAttentionOp()
        self.model_layers_4_attn_o_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_4_attn_o_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_4_post_attention_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_4_post_attention_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_4_mlp_gate_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_4_mlp_gate_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_4_mlp_up_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_4_mlp_up_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_4_mlp_act_fn_Sigmoid = torch.load(r'decoder_model_merged_torch/model_layers_4_mlp_act_fn_Sigmoid.pt', weights_only=False) # Sigmoid()
        self.model_layers_4_mlp_act_fn_Mul = torch.load(r'decoder_model_merged_torch/model_layers_4_mlp_act_fn_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_4_mlp_Mul = torch.load(r'decoder_model_merged_torch/model_layers_4_mlp_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_4_mlp_down_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_4_mlp_down_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_5_input_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_5_input_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_5_attn_q_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_5_attn_q_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_5_attn_k_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_5_attn_k_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_5_attn_v_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_5_attn_v_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_5_attn_q_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_5_attn_q_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.com_microsoft__model_layers_5_attn_k_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_5_attn_k_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.model_layers_5_attn_v_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_5_attn_v_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_5_attn_k_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_5_attn_k_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_5_attn_v_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_5_attn_v_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_5_attn_k_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_5_attn_k_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_5_attn_v_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_5_attn_v_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_5_attn_k_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_5_attn_k_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_5_attn_v_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_5_attn_v_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_5_attn_v_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_5_attn_v_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_5_attn_k_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_5_attn_k_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_5_attn_k_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_5_attn_k_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_5_attn_v_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_5_attn_v_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_5_attn_v_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_5_attn_v_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_5_attn_k_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_5_attn_k_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_5_attn_k_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_5_attn_k_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_5_attn_v_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_5_attn_v_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_5_attn_v_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_5_attn_v_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_5_attn_k_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_5_attn_k_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_5_attn_k_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_5_attn_k_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_5_attn_v_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_5_attn_v_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_5_attn_v_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_5_attn_v_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_5_attn_k_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_5_attn_k_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_5_attn_k_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_5_attn_k_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_5_attn_v_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_5_attn_v_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_5_attn_k_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_5_attn_k_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_5_attn_v_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_5_attn_v_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_5_attn_k_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_5_attn_k_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_5_attn_v_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_5_attn_v_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_5_attn_k_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_5_attn_k_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_5_attn_v_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_5_attn_v_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_5_attn_k_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_5_attn_k_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_5_attn_v_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_5_attn_v_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_5_attn_k_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_5_attn_k_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_5_attn_v_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_5_attn_v_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.model_layers_5_attn_k_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_5_attn_k_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.com_microsoft__model_layers_5_attn_MultiHeadAttention = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_5_attn_MultiHeadAttention.pt', weights_only=False) # MultiHeadAttentionOp()
        self.model_layers_5_attn_o_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_5_attn_o_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_5_post_attention_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_5_post_attention_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_5_mlp_gate_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_5_mlp_gate_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_5_mlp_up_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_5_mlp_up_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_5_mlp_act_fn_Sigmoid = torch.load(r'decoder_model_merged_torch/model_layers_5_mlp_act_fn_Sigmoid.pt', weights_only=False) # Sigmoid()
        self.model_layers_5_mlp_act_fn_Mul = torch.load(r'decoder_model_merged_torch/model_layers_5_mlp_act_fn_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_5_mlp_Mul = torch.load(r'decoder_model_merged_torch/model_layers_5_mlp_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_5_mlp_down_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_5_mlp_down_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_6_input_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_6_input_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_6_attn_q_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_6_attn_q_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_6_attn_k_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_6_attn_k_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_6_attn_v_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_6_attn_v_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_6_attn_q_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_6_attn_q_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.com_microsoft__model_layers_6_attn_k_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_6_attn_k_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.model_layers_6_attn_v_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_6_attn_v_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_6_attn_k_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_6_attn_k_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_6_attn_v_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_6_attn_v_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_6_attn_k_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_6_attn_k_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_6_attn_v_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_6_attn_v_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_6_attn_k_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_6_attn_k_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_6_attn_v_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_6_attn_v_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_6_attn_v_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_6_attn_v_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_6_attn_k_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_6_attn_k_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_6_attn_k_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_6_attn_k_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_6_attn_v_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_6_attn_v_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_6_attn_v_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_6_attn_v_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_6_attn_k_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_6_attn_k_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_6_attn_k_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_6_attn_k_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_6_attn_v_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_6_attn_v_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_6_attn_v_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_6_attn_v_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_6_attn_k_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_6_attn_k_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_6_attn_k_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_6_attn_k_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_6_attn_v_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_6_attn_v_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_6_attn_v_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_6_attn_v_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_6_attn_k_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_6_attn_k_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_6_attn_k_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_6_attn_k_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_6_attn_v_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_6_attn_v_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_6_attn_k_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_6_attn_k_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_6_attn_v_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_6_attn_v_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_6_attn_k_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_6_attn_k_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_6_attn_v_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_6_attn_v_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_6_attn_k_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_6_attn_k_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_6_attn_v_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_6_attn_v_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_6_attn_k_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_6_attn_k_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_6_attn_v_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_6_attn_v_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_6_attn_k_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_6_attn_k_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_6_attn_v_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_6_attn_v_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.model_layers_6_attn_k_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_6_attn_k_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.com_microsoft__model_layers_6_attn_MultiHeadAttention = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_6_attn_MultiHeadAttention.pt', weights_only=False) # MultiHeadAttentionOp()
        self.model_layers_6_attn_o_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_6_attn_o_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_6_post_attention_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_6_post_attention_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_6_mlp_gate_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_6_mlp_gate_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_6_mlp_up_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_6_mlp_up_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_6_mlp_act_fn_Sigmoid = torch.load(r'decoder_model_merged_torch/model_layers_6_mlp_act_fn_Sigmoid.pt', weights_only=False) # Sigmoid()
        self.model_layers_6_mlp_act_fn_Mul = torch.load(r'decoder_model_merged_torch/model_layers_6_mlp_act_fn_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_6_mlp_Mul = torch.load(r'decoder_model_merged_torch/model_layers_6_mlp_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_6_mlp_down_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_6_mlp_down_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_7_input_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_7_input_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_7_attn_q_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_7_attn_q_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_7_attn_k_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_7_attn_k_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_7_attn_v_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_7_attn_v_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_7_attn_q_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_7_attn_q_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.com_microsoft__model_layers_7_attn_k_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_7_attn_k_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.model_layers_7_attn_v_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_7_attn_v_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_7_attn_k_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_7_attn_k_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_7_attn_v_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_7_attn_v_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_7_attn_k_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_7_attn_k_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_7_attn_v_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_7_attn_v_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_7_attn_k_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_7_attn_k_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_7_attn_v_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_7_attn_v_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_7_attn_v_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_7_attn_v_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_7_attn_k_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_7_attn_k_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_7_attn_k_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_7_attn_k_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_7_attn_v_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_7_attn_v_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_7_attn_v_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_7_attn_v_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_7_attn_k_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_7_attn_k_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_7_attn_k_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_7_attn_k_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_7_attn_v_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_7_attn_v_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_7_attn_v_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_7_attn_v_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_7_attn_k_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_7_attn_k_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_7_attn_k_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_7_attn_k_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_7_attn_v_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_7_attn_v_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_7_attn_v_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_7_attn_v_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_7_attn_k_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_7_attn_k_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_7_attn_k_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_7_attn_k_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_7_attn_v_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_7_attn_v_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_7_attn_k_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_7_attn_k_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_7_attn_v_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_7_attn_v_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_7_attn_k_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_7_attn_k_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_7_attn_v_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_7_attn_v_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_7_attn_k_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_7_attn_k_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_7_attn_v_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_7_attn_v_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_7_attn_k_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_7_attn_k_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_7_attn_v_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_7_attn_v_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_7_attn_k_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_7_attn_k_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_7_attn_v_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_7_attn_v_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.model_layers_7_attn_k_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_7_attn_k_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.com_microsoft__model_layers_7_attn_MultiHeadAttention = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_7_attn_MultiHeadAttention.pt', weights_only=False) # MultiHeadAttentionOp()
        self.model_layers_7_attn_o_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_7_attn_o_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_7_post_attention_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_7_post_attention_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_7_mlp_gate_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_7_mlp_gate_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_7_mlp_up_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_7_mlp_up_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_7_mlp_act_fn_Sigmoid = torch.load(r'decoder_model_merged_torch/model_layers_7_mlp_act_fn_Sigmoid.pt', weights_only=False) # Sigmoid()
        self.model_layers_7_mlp_act_fn_Mul = torch.load(r'decoder_model_merged_torch/model_layers_7_mlp_act_fn_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_7_mlp_Mul = torch.load(r'decoder_model_merged_torch/model_layers_7_mlp_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_7_mlp_down_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_7_mlp_down_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_8_input_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_8_input_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_8_attn_q_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_8_attn_q_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_8_attn_k_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_8_attn_k_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_8_attn_v_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_8_attn_v_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_8_attn_q_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_8_attn_q_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.com_microsoft__model_layers_8_attn_k_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_8_attn_k_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.model_layers_8_attn_v_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_8_attn_v_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_8_attn_k_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_8_attn_k_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_8_attn_v_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_8_attn_v_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_8_attn_k_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_8_attn_k_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_8_attn_v_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_8_attn_v_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_8_attn_k_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_8_attn_k_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_8_attn_v_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_8_attn_v_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_8_attn_v_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_8_attn_v_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_8_attn_k_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_8_attn_k_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_8_attn_k_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_8_attn_k_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_8_attn_v_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_8_attn_v_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_8_attn_v_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_8_attn_v_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_8_attn_k_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_8_attn_k_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_8_attn_k_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_8_attn_k_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_8_attn_v_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_8_attn_v_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_8_attn_v_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_8_attn_v_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_8_attn_k_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_8_attn_k_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_8_attn_k_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_8_attn_k_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_8_attn_v_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_8_attn_v_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_8_attn_v_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_8_attn_v_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_8_attn_k_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_8_attn_k_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_8_attn_k_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_8_attn_k_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_8_attn_v_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_8_attn_v_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_8_attn_k_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_8_attn_k_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_8_attn_v_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_8_attn_v_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_8_attn_k_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_8_attn_k_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_8_attn_v_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_8_attn_v_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_8_attn_k_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_8_attn_k_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_8_attn_v_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_8_attn_v_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_8_attn_k_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_8_attn_k_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_8_attn_v_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_8_attn_v_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_8_attn_k_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_8_attn_k_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_8_attn_v_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_8_attn_v_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.model_layers_8_attn_k_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_8_attn_k_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.com_microsoft__model_layers_8_attn_MultiHeadAttention = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_8_attn_MultiHeadAttention.pt', weights_only=False) # MultiHeadAttentionOp()
        self.model_layers_8_attn_o_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_8_attn_o_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_8_post_attention_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_8_post_attention_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_8_mlp_gate_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_8_mlp_gate_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_8_mlp_up_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_8_mlp_up_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_8_mlp_act_fn_Sigmoid = torch.load(r'decoder_model_merged_torch/model_layers_8_mlp_act_fn_Sigmoid.pt', weights_only=False) # Sigmoid()
        self.model_layers_8_mlp_act_fn_Mul = torch.load(r'decoder_model_merged_torch/model_layers_8_mlp_act_fn_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_8_mlp_Mul = torch.load(r'decoder_model_merged_torch/model_layers_8_mlp_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_8_mlp_down_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_8_mlp_down_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_9_input_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_9_input_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_9_attn_q_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_9_attn_q_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_9_attn_k_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_9_attn_k_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_9_attn_v_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_9_attn_v_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_9_attn_q_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_9_attn_q_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.com_microsoft__model_layers_9_attn_k_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_9_attn_k_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.model_layers_9_attn_v_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_9_attn_v_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_9_attn_k_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_9_attn_k_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_9_attn_v_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_9_attn_v_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_9_attn_k_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_9_attn_k_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_9_attn_v_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_9_attn_v_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_9_attn_k_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_9_attn_k_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_9_attn_v_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_9_attn_v_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_9_attn_v_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_9_attn_v_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_9_attn_k_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_9_attn_k_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_9_attn_k_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_9_attn_k_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_9_attn_v_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_9_attn_v_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_9_attn_v_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_9_attn_v_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_9_attn_k_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_9_attn_k_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_9_attn_k_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_9_attn_k_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_9_attn_v_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_9_attn_v_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_9_attn_v_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_9_attn_v_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_9_attn_k_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_9_attn_k_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_9_attn_k_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_9_attn_k_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_9_attn_v_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_9_attn_v_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_9_attn_v_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_9_attn_v_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_9_attn_k_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_9_attn_k_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_9_attn_k_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_9_attn_k_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_9_attn_v_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_9_attn_v_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_9_attn_k_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_9_attn_k_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_9_attn_v_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_9_attn_v_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_9_attn_k_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_9_attn_k_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_9_attn_v_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_9_attn_v_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_9_attn_k_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_9_attn_k_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_9_attn_v_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_9_attn_v_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_9_attn_k_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_9_attn_k_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_9_attn_v_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_9_attn_v_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_9_attn_k_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_9_attn_k_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_9_attn_v_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_9_attn_v_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.model_layers_9_attn_k_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_9_attn_k_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.com_microsoft__model_layers_9_attn_MultiHeadAttention = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_9_attn_MultiHeadAttention.pt', weights_only=False) # MultiHeadAttentionOp()
        self.model_layers_9_attn_o_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_9_attn_o_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_9_post_attention_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_9_post_attention_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_9_mlp_gate_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_9_mlp_gate_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_9_mlp_up_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_9_mlp_up_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_9_mlp_act_fn_Sigmoid = torch.load(r'decoder_model_merged_torch/model_layers_9_mlp_act_fn_Sigmoid.pt', weights_only=False) # Sigmoid()
        self.model_layers_9_mlp_act_fn_Mul = torch.load(r'decoder_model_merged_torch/model_layers_9_mlp_act_fn_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_9_mlp_Mul = torch.load(r'decoder_model_merged_torch/model_layers_9_mlp_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_9_mlp_down_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_9_mlp_down_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_10_input_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_10_input_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_10_attn_q_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_10_attn_q_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_10_attn_k_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_10_attn_k_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_10_attn_v_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_10_attn_v_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_10_attn_q_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_10_attn_q_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.com_microsoft__model_layers_10_attn_k_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_10_attn_k_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.model_layers_10_attn_v_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_10_attn_v_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_10_attn_k_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_10_attn_k_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_10_attn_v_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_10_attn_v_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_10_attn_k_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_10_attn_k_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_10_attn_v_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_10_attn_v_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_10_attn_k_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_10_attn_k_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_10_attn_v_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_10_attn_v_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_10_attn_v_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_10_attn_v_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_10_attn_k_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_10_attn_k_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_10_attn_k_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_10_attn_k_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_10_attn_v_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_10_attn_v_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_10_attn_v_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_10_attn_v_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_10_attn_k_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_10_attn_k_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_10_attn_k_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_10_attn_k_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_10_attn_v_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_10_attn_v_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_10_attn_v_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_10_attn_v_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_10_attn_k_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_10_attn_k_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_10_attn_k_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_10_attn_k_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_10_attn_v_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_10_attn_v_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_10_attn_v_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_10_attn_v_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_10_attn_k_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_10_attn_k_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_10_attn_k_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_10_attn_k_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_10_attn_v_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_10_attn_v_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_10_attn_k_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_10_attn_k_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_10_attn_v_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_10_attn_v_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_10_attn_k_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_10_attn_k_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_10_attn_v_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_10_attn_v_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_10_attn_k_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_10_attn_k_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_10_attn_v_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_10_attn_v_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_10_attn_k_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_10_attn_k_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_10_attn_v_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_10_attn_v_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_10_attn_k_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_10_attn_k_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_10_attn_v_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_10_attn_v_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.model_layers_10_attn_k_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_10_attn_k_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.com_microsoft__model_layers_10_attn_MultiHeadAttention = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_10_attn_MultiHeadAttention.pt', weights_only=False) # MultiHeadAttentionOp()
        self.model_layers_10_attn_o_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_10_attn_o_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_10_post_attention_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_10_post_attention_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_10_mlp_gate_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_10_mlp_gate_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_10_mlp_up_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_10_mlp_up_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_10_mlp_act_fn_Sigmoid = torch.load(r'decoder_model_merged_torch/model_layers_10_mlp_act_fn_Sigmoid.pt', weights_only=False) # Sigmoid()
        self.model_layers_10_mlp_act_fn_Mul = torch.load(r'decoder_model_merged_torch/model_layers_10_mlp_act_fn_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_10_mlp_Mul = torch.load(r'decoder_model_merged_torch/model_layers_10_mlp_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_10_mlp_down_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_10_mlp_down_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_11_input_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_11_input_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_11_attn_q_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_11_attn_q_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_11_attn_k_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_11_attn_k_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_11_attn_v_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_11_attn_v_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_11_attn_q_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_11_attn_q_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.com_microsoft__model_layers_11_attn_k_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_11_attn_k_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.model_layers_11_attn_v_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_11_attn_v_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_11_attn_k_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_11_attn_k_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_11_attn_v_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_11_attn_v_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_11_attn_k_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_11_attn_k_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_11_attn_v_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_11_attn_v_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_11_attn_k_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_11_attn_k_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_11_attn_v_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_11_attn_v_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_11_attn_v_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_11_attn_v_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_11_attn_k_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_11_attn_k_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_11_attn_k_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_11_attn_k_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_11_attn_v_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_11_attn_v_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_11_attn_v_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_11_attn_v_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_11_attn_k_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_11_attn_k_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_11_attn_k_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_11_attn_k_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_11_attn_v_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_11_attn_v_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_11_attn_v_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_11_attn_v_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_11_attn_k_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_11_attn_k_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_11_attn_k_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_11_attn_k_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_11_attn_v_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_11_attn_v_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_11_attn_v_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_11_attn_v_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_11_attn_k_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_11_attn_k_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_11_attn_k_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_11_attn_k_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_11_attn_v_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_11_attn_v_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_11_attn_k_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_11_attn_k_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_11_attn_v_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_11_attn_v_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_11_attn_k_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_11_attn_k_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_11_attn_v_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_11_attn_v_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_11_attn_k_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_11_attn_k_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_11_attn_v_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_11_attn_v_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_11_attn_k_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_11_attn_k_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_11_attn_v_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_11_attn_v_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_11_attn_k_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_11_attn_k_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_11_attn_v_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_11_attn_v_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.model_layers_11_attn_k_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_11_attn_k_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.com_microsoft__model_layers_11_attn_MultiHeadAttention = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_11_attn_MultiHeadAttention.pt', weights_only=False) # MultiHeadAttentionOp()
        self.model_layers_11_attn_o_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_11_attn_o_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_11_post_attention_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_11_post_attention_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_11_mlp_gate_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_11_mlp_gate_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_11_mlp_up_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_11_mlp_up_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_11_mlp_act_fn_Sigmoid = torch.load(r'decoder_model_merged_torch/model_layers_11_mlp_act_fn_Sigmoid.pt', weights_only=False) # Sigmoid()
        self.model_layers_11_mlp_act_fn_Mul = torch.load(r'decoder_model_merged_torch/model_layers_11_mlp_act_fn_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_11_mlp_Mul = torch.load(r'decoder_model_merged_torch/model_layers_11_mlp_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_11_mlp_down_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_11_mlp_down_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_12_input_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_12_input_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_12_attn_q_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_12_attn_q_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_12_attn_k_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_12_attn_k_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_12_attn_v_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_12_attn_v_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_12_attn_q_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_12_attn_q_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.com_microsoft__model_layers_12_attn_k_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_12_attn_k_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.model_layers_12_attn_v_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_12_attn_v_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_12_attn_k_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_12_attn_k_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_12_attn_v_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_12_attn_v_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_12_attn_k_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_12_attn_k_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_12_attn_v_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_12_attn_v_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_12_attn_k_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_12_attn_k_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_12_attn_v_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_12_attn_v_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_12_attn_v_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_12_attn_v_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_12_attn_k_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_12_attn_k_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_12_attn_k_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_12_attn_k_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_12_attn_v_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_12_attn_v_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_12_attn_v_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_12_attn_v_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_12_attn_k_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_12_attn_k_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_12_attn_k_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_12_attn_k_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_12_attn_v_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_12_attn_v_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_12_attn_v_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_12_attn_v_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_12_attn_k_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_12_attn_k_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_12_attn_k_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_12_attn_k_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_12_attn_v_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_12_attn_v_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_12_attn_v_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_12_attn_v_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_12_attn_k_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_12_attn_k_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_12_attn_k_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_12_attn_k_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_12_attn_v_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_12_attn_v_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_12_attn_k_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_12_attn_k_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_12_attn_v_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_12_attn_v_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_12_attn_k_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_12_attn_k_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_12_attn_v_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_12_attn_v_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_12_attn_k_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_12_attn_k_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_12_attn_v_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_12_attn_v_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_12_attn_k_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_12_attn_k_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_12_attn_v_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_12_attn_v_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_12_attn_k_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_12_attn_k_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_12_attn_v_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_12_attn_v_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.model_layers_12_attn_k_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_12_attn_k_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.com_microsoft__model_layers_12_attn_MultiHeadAttention = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_12_attn_MultiHeadAttention.pt', weights_only=False) # MultiHeadAttentionOp()
        self.model_layers_12_attn_o_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_12_attn_o_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_12_post_attention_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_12_post_attention_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_12_mlp_gate_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_12_mlp_gate_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_12_mlp_up_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_12_mlp_up_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_12_mlp_act_fn_Sigmoid = torch.load(r'decoder_model_merged_torch/model_layers_12_mlp_act_fn_Sigmoid.pt', weights_only=False) # Sigmoid()
        self.model_layers_12_mlp_act_fn_Mul = torch.load(r'decoder_model_merged_torch/model_layers_12_mlp_act_fn_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_12_mlp_Mul = torch.load(r'decoder_model_merged_torch/model_layers_12_mlp_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_12_mlp_down_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_12_mlp_down_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_13_input_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_13_input_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_13_attn_q_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_13_attn_q_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_13_attn_k_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_13_attn_k_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_13_attn_v_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_13_attn_v_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_13_attn_q_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_13_attn_q_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.com_microsoft__model_layers_13_attn_k_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_13_attn_k_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.model_layers_13_attn_v_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_13_attn_v_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_13_attn_k_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_13_attn_k_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_13_attn_v_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_13_attn_v_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_13_attn_k_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_13_attn_k_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_13_attn_v_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_13_attn_v_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_13_attn_k_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_13_attn_k_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_13_attn_v_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_13_attn_v_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_13_attn_v_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_13_attn_v_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_13_attn_k_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_13_attn_k_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_13_attn_k_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_13_attn_k_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_13_attn_v_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_13_attn_v_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_13_attn_v_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_13_attn_v_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_13_attn_k_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_13_attn_k_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_13_attn_k_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_13_attn_k_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_13_attn_v_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_13_attn_v_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_13_attn_v_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_13_attn_v_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_13_attn_k_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_13_attn_k_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_13_attn_k_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_13_attn_k_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_13_attn_v_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_13_attn_v_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_13_attn_v_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_13_attn_v_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_13_attn_k_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_13_attn_k_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_13_attn_k_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_13_attn_k_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_13_attn_v_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_13_attn_v_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_13_attn_k_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_13_attn_k_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_13_attn_v_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_13_attn_v_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_13_attn_k_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_13_attn_k_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_13_attn_v_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_13_attn_v_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_13_attn_k_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_13_attn_k_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_13_attn_v_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_13_attn_v_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_13_attn_k_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_13_attn_k_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_13_attn_v_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_13_attn_v_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_13_attn_k_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_13_attn_k_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_13_attn_v_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_13_attn_v_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.model_layers_13_attn_k_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_13_attn_k_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.com_microsoft__model_layers_13_attn_MultiHeadAttention = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_13_attn_MultiHeadAttention.pt', weights_only=False) # MultiHeadAttentionOp()
        self.model_layers_13_attn_o_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_13_attn_o_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_13_post_attention_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_13_post_attention_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_13_mlp_gate_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_13_mlp_gate_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_13_mlp_up_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_13_mlp_up_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_13_mlp_act_fn_Sigmoid = torch.load(r'decoder_model_merged_torch/model_layers_13_mlp_act_fn_Sigmoid.pt', weights_only=False) # Sigmoid()
        self.model_layers_13_mlp_act_fn_Mul = torch.load(r'decoder_model_merged_torch/model_layers_13_mlp_act_fn_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_13_mlp_Mul = torch.load(r'decoder_model_merged_torch/model_layers_13_mlp_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_13_mlp_down_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_13_mlp_down_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_14_input_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_14_input_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_14_attn_q_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_14_attn_q_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_14_attn_k_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_14_attn_k_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_14_attn_v_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_14_attn_v_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_14_attn_q_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_14_attn_q_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.com_microsoft__model_layers_14_attn_k_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_14_attn_k_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.model_layers_14_attn_v_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_14_attn_v_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_14_attn_k_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_14_attn_k_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_14_attn_v_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_14_attn_v_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_14_attn_k_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_14_attn_k_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_14_attn_v_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_14_attn_v_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_14_attn_k_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_14_attn_k_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_14_attn_v_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_14_attn_v_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_14_attn_v_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_14_attn_v_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_14_attn_k_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_14_attn_k_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_14_attn_k_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_14_attn_k_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_14_attn_v_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_14_attn_v_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_14_attn_v_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_14_attn_v_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_14_attn_k_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_14_attn_k_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_14_attn_k_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_14_attn_k_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_14_attn_v_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_14_attn_v_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_14_attn_v_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_14_attn_v_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_14_attn_k_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_14_attn_k_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_14_attn_k_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_14_attn_k_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_14_attn_v_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_14_attn_v_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_14_attn_v_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_14_attn_v_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_14_attn_k_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_14_attn_k_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_14_attn_k_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_14_attn_k_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_14_attn_v_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_14_attn_v_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_14_attn_k_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_14_attn_k_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_14_attn_v_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_14_attn_v_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_14_attn_k_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_14_attn_k_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_14_attn_v_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_14_attn_v_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_14_attn_k_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_14_attn_k_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_14_attn_v_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_14_attn_v_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_14_attn_k_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_14_attn_k_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_14_attn_v_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_14_attn_v_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_14_attn_k_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_14_attn_k_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_14_attn_v_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_14_attn_v_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.model_layers_14_attn_k_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_14_attn_k_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.com_microsoft__model_layers_14_attn_MultiHeadAttention = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_14_attn_MultiHeadAttention.pt', weights_only=False) # MultiHeadAttentionOp()
        self.model_layers_14_attn_o_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_14_attn_o_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_14_post_attention_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_14_post_attention_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_14_mlp_gate_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_14_mlp_gate_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_14_mlp_up_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_14_mlp_up_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_14_mlp_act_fn_Sigmoid = torch.load(r'decoder_model_merged_torch/model_layers_14_mlp_act_fn_Sigmoid.pt', weights_only=False) # Sigmoid()
        self.model_layers_14_mlp_act_fn_Mul = torch.load(r'decoder_model_merged_torch/model_layers_14_mlp_act_fn_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_14_mlp_Mul = torch.load(r'decoder_model_merged_torch/model_layers_14_mlp_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_14_mlp_down_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_14_mlp_down_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_15_input_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_15_input_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_15_attn_q_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_15_attn_q_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_15_attn_k_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_15_attn_k_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_15_attn_v_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_15_attn_v_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_15_attn_q_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_15_attn_q_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.com_microsoft__model_layers_15_attn_k_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_15_attn_k_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.model_layers_15_attn_v_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_15_attn_v_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_15_attn_k_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_15_attn_k_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_15_attn_v_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_15_attn_v_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_15_attn_k_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_15_attn_k_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_15_attn_v_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_15_attn_v_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_15_attn_k_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_15_attn_k_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_15_attn_v_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_15_attn_v_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_15_attn_v_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_15_attn_v_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_15_attn_k_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_15_attn_k_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_15_attn_k_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_15_attn_k_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_15_attn_v_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_15_attn_v_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_15_attn_v_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_15_attn_v_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_15_attn_k_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_15_attn_k_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_15_attn_k_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_15_attn_k_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_15_attn_v_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_15_attn_v_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_15_attn_v_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_15_attn_v_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_15_attn_k_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_15_attn_k_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_15_attn_k_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_15_attn_k_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_15_attn_v_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_15_attn_v_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_15_attn_v_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_15_attn_v_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_15_attn_k_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_15_attn_k_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_15_attn_k_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_15_attn_k_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_15_attn_v_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_15_attn_v_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_15_attn_k_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_15_attn_k_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_15_attn_v_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_15_attn_v_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_15_attn_k_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_15_attn_k_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_15_attn_v_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_15_attn_v_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_15_attn_k_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_15_attn_k_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_15_attn_v_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_15_attn_v_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_15_attn_k_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_15_attn_k_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_15_attn_v_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_15_attn_v_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_15_attn_k_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_15_attn_k_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_15_attn_v_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_15_attn_v_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.model_layers_15_attn_k_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_15_attn_k_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.com_microsoft__model_layers_15_attn_MultiHeadAttention = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_15_attn_MultiHeadAttention.pt', weights_only=False) # MultiHeadAttentionOp()
        self.model_layers_15_attn_o_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_15_attn_o_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_15_post_attention_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_15_post_attention_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_15_mlp_gate_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_15_mlp_gate_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_15_mlp_up_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_15_mlp_up_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_15_mlp_act_fn_Sigmoid = torch.load(r'decoder_model_merged_torch/model_layers_15_mlp_act_fn_Sigmoid.pt', weights_only=False) # Sigmoid()
        self.model_layers_15_mlp_act_fn_Mul = torch.load(r'decoder_model_merged_torch/model_layers_15_mlp_act_fn_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_15_mlp_Mul = torch.load(r'decoder_model_merged_torch/model_layers_15_mlp_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_15_mlp_down_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_15_mlp_down_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_16_input_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_16_input_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_16_attn_q_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_16_attn_q_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_16_attn_k_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_16_attn_k_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_16_attn_v_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_16_attn_v_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_16_attn_q_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_16_attn_q_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.com_microsoft__model_layers_16_attn_k_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_16_attn_k_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.model_layers_16_attn_v_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_16_attn_v_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_16_attn_k_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_16_attn_k_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_16_attn_v_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_16_attn_v_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_16_attn_k_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_16_attn_k_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_16_attn_v_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_16_attn_v_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_16_attn_k_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_16_attn_k_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_16_attn_v_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_16_attn_v_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_16_attn_v_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_16_attn_v_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_16_attn_k_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_16_attn_k_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_16_attn_k_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_16_attn_k_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_16_attn_v_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_16_attn_v_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_16_attn_v_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_16_attn_v_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_16_attn_k_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_16_attn_k_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_16_attn_k_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_16_attn_k_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_16_attn_v_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_16_attn_v_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_16_attn_v_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_16_attn_v_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_16_attn_k_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_16_attn_k_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_16_attn_k_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_16_attn_k_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_16_attn_v_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_16_attn_v_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_16_attn_v_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_16_attn_v_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_16_attn_k_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_16_attn_k_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_16_attn_k_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_16_attn_k_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_16_attn_v_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_16_attn_v_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_16_attn_k_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_16_attn_k_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_16_attn_v_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_16_attn_v_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_16_attn_k_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_16_attn_k_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_16_attn_v_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_16_attn_v_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_16_attn_k_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_16_attn_k_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_16_attn_v_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_16_attn_v_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_16_attn_k_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_16_attn_k_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_16_attn_v_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_16_attn_v_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_16_attn_k_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_16_attn_k_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_16_attn_v_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_16_attn_v_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.model_layers_16_attn_k_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_16_attn_k_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.com_microsoft__model_layers_16_attn_MultiHeadAttention = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_16_attn_MultiHeadAttention.pt', weights_only=False) # MultiHeadAttentionOp()
        self.model_layers_16_attn_o_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_16_attn_o_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_16_post_attention_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_16_post_attention_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_16_mlp_gate_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_16_mlp_gate_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_16_mlp_up_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_16_mlp_up_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_16_mlp_act_fn_Sigmoid = torch.load(r'decoder_model_merged_torch/model_layers_16_mlp_act_fn_Sigmoid.pt', weights_only=False) # Sigmoid()
        self.model_layers_16_mlp_act_fn_Mul = torch.load(r'decoder_model_merged_torch/model_layers_16_mlp_act_fn_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_16_mlp_Mul = torch.load(r'decoder_model_merged_torch/model_layers_16_mlp_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_16_mlp_down_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_16_mlp_down_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_17_input_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_17_input_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_17_attn_q_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_17_attn_q_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_17_attn_k_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_17_attn_k_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_17_attn_v_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_17_attn_v_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_17_attn_q_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_17_attn_q_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.com_microsoft__model_layers_17_attn_k_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_17_attn_k_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.model_layers_17_attn_v_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_17_attn_v_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_17_attn_k_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_17_attn_k_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_17_attn_v_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_17_attn_v_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_17_attn_k_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_17_attn_k_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_17_attn_v_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_17_attn_v_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_17_attn_k_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_17_attn_k_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_17_attn_v_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_17_attn_v_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_17_attn_v_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_17_attn_v_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_17_attn_k_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_17_attn_k_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_17_attn_k_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_17_attn_k_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_17_attn_v_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_17_attn_v_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_17_attn_v_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_17_attn_v_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_17_attn_k_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_17_attn_k_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_17_attn_k_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_17_attn_k_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_17_attn_v_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_17_attn_v_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_17_attn_v_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_17_attn_v_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_17_attn_k_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_17_attn_k_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_17_attn_k_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_17_attn_k_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_17_attn_v_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_17_attn_v_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_17_attn_v_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_17_attn_v_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_17_attn_k_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_17_attn_k_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_17_attn_k_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_17_attn_k_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_17_attn_v_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_17_attn_v_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_17_attn_k_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_17_attn_k_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_17_attn_v_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_17_attn_v_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_17_attn_k_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_17_attn_k_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_17_attn_v_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_17_attn_v_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_17_attn_k_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_17_attn_k_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_17_attn_v_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_17_attn_v_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_17_attn_k_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_17_attn_k_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_17_attn_v_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_17_attn_v_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_17_attn_k_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_17_attn_k_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_17_attn_v_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_17_attn_v_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.model_layers_17_attn_k_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_17_attn_k_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.com_microsoft__model_layers_17_attn_MultiHeadAttention = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_17_attn_MultiHeadAttention.pt', weights_only=False) # MultiHeadAttentionOp()
        self.model_layers_17_attn_o_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_17_attn_o_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_17_post_attention_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_17_post_attention_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_17_mlp_gate_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_17_mlp_gate_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_17_mlp_up_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_17_mlp_up_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_17_mlp_act_fn_Sigmoid = torch.load(r'decoder_model_merged_torch/model_layers_17_mlp_act_fn_Sigmoid.pt', weights_only=False) # Sigmoid()
        self.model_layers_17_mlp_act_fn_Mul = torch.load(r'decoder_model_merged_torch/model_layers_17_mlp_act_fn_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_17_mlp_Mul = torch.load(r'decoder_model_merged_torch/model_layers_17_mlp_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_17_mlp_down_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_17_mlp_down_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_18_input_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_18_input_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_18_attn_q_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_18_attn_q_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_18_attn_k_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_18_attn_k_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_18_attn_v_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_18_attn_v_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_18_attn_q_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_18_attn_q_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.com_microsoft__model_layers_18_attn_k_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_18_attn_k_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.model_layers_18_attn_v_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_18_attn_v_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_18_attn_k_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_18_attn_k_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_18_attn_v_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_18_attn_v_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_18_attn_k_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_18_attn_k_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_18_attn_v_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_18_attn_v_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_18_attn_k_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_18_attn_k_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_18_attn_v_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_18_attn_v_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_18_attn_v_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_18_attn_v_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_18_attn_k_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_18_attn_k_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_18_attn_k_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_18_attn_k_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_18_attn_v_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_18_attn_v_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_18_attn_v_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_18_attn_v_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_18_attn_k_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_18_attn_k_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_18_attn_k_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_18_attn_k_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_18_attn_v_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_18_attn_v_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_18_attn_v_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_18_attn_v_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_18_attn_k_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_18_attn_k_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_18_attn_k_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_18_attn_k_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_18_attn_v_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_18_attn_v_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_18_attn_v_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_18_attn_v_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_18_attn_k_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_18_attn_k_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_18_attn_k_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_18_attn_k_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_18_attn_v_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_18_attn_v_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_18_attn_k_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_18_attn_k_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_18_attn_v_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_18_attn_v_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_18_attn_k_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_18_attn_k_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_18_attn_v_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_18_attn_v_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_18_attn_k_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_18_attn_k_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_18_attn_v_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_18_attn_v_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_18_attn_k_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_18_attn_k_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_18_attn_v_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_18_attn_v_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_18_attn_k_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_18_attn_k_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_18_attn_v_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_18_attn_v_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.model_layers_18_attn_k_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_18_attn_k_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.com_microsoft__model_layers_18_attn_MultiHeadAttention = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_18_attn_MultiHeadAttention.pt', weights_only=False) # MultiHeadAttentionOp()
        self.model_layers_18_attn_o_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_18_attn_o_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_18_post_attention_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_18_post_attention_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_18_mlp_gate_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_18_mlp_gate_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_18_mlp_up_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_18_mlp_up_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_18_mlp_act_fn_Sigmoid = torch.load(r'decoder_model_merged_torch/model_layers_18_mlp_act_fn_Sigmoid.pt', weights_only=False) # Sigmoid()
        self.model_layers_18_mlp_act_fn_Mul = torch.load(r'decoder_model_merged_torch/model_layers_18_mlp_act_fn_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_18_mlp_Mul = torch.load(r'decoder_model_merged_torch/model_layers_18_mlp_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_18_mlp_down_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_18_mlp_down_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_19_input_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_19_input_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_19_attn_q_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_19_attn_q_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_19_attn_k_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_19_attn_k_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_19_attn_v_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_19_attn_v_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_19_attn_q_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_19_attn_q_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.com_microsoft__model_layers_19_attn_k_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_19_attn_k_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.model_layers_19_attn_v_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_19_attn_v_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_19_attn_k_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_19_attn_k_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_19_attn_v_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_19_attn_v_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_19_attn_k_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_19_attn_k_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_19_attn_v_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_19_attn_v_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_19_attn_k_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_19_attn_k_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_19_attn_v_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_19_attn_v_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_19_attn_v_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_19_attn_v_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_19_attn_k_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_19_attn_k_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_19_attn_k_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_19_attn_k_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_19_attn_v_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_19_attn_v_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_19_attn_v_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_19_attn_v_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_19_attn_k_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_19_attn_k_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_19_attn_k_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_19_attn_k_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_19_attn_v_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_19_attn_v_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_19_attn_v_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_19_attn_v_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_19_attn_k_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_19_attn_k_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_19_attn_k_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_19_attn_k_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_19_attn_v_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_19_attn_v_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_19_attn_v_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_19_attn_v_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_19_attn_k_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_19_attn_k_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_19_attn_k_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_19_attn_k_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_19_attn_v_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_19_attn_v_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_19_attn_k_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_19_attn_k_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_19_attn_v_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_19_attn_v_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_19_attn_k_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_19_attn_k_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_19_attn_v_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_19_attn_v_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_19_attn_k_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_19_attn_k_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_19_attn_v_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_19_attn_v_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_19_attn_k_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_19_attn_k_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_19_attn_v_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_19_attn_v_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_19_attn_k_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_19_attn_k_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_19_attn_v_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_19_attn_v_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.model_layers_19_attn_k_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_19_attn_k_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.com_microsoft__model_layers_19_attn_MultiHeadAttention = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_19_attn_MultiHeadAttention.pt', weights_only=False) # MultiHeadAttentionOp()
        self.model_layers_19_attn_o_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_19_attn_o_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_19_post_attention_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_19_post_attention_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_19_mlp_gate_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_19_mlp_gate_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_19_mlp_up_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_19_mlp_up_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_19_mlp_act_fn_Sigmoid = torch.load(r'decoder_model_merged_torch/model_layers_19_mlp_act_fn_Sigmoid.pt', weights_only=False) # Sigmoid()
        self.model_layers_19_mlp_act_fn_Mul = torch.load(r'decoder_model_merged_torch/model_layers_19_mlp_act_fn_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_19_mlp_Mul = torch.load(r'decoder_model_merged_torch/model_layers_19_mlp_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_19_mlp_down_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_19_mlp_down_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_20_input_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_20_input_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_20_attn_q_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_20_attn_q_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_20_attn_k_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_20_attn_k_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_20_attn_v_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_20_attn_v_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_20_attn_q_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_20_attn_q_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.com_microsoft__model_layers_20_attn_k_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_20_attn_k_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.model_layers_20_attn_v_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_20_attn_v_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_20_attn_k_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_20_attn_k_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_20_attn_v_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_20_attn_v_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_20_attn_k_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_20_attn_k_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_20_attn_v_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_20_attn_v_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_20_attn_k_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_20_attn_k_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_20_attn_v_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_20_attn_v_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_20_attn_v_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_20_attn_v_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_20_attn_k_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_20_attn_k_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_20_attn_k_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_20_attn_k_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_20_attn_v_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_20_attn_v_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_20_attn_v_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_20_attn_v_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_20_attn_k_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_20_attn_k_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_20_attn_k_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_20_attn_k_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_20_attn_v_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_20_attn_v_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_20_attn_v_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_20_attn_v_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_20_attn_k_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_20_attn_k_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_20_attn_k_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_20_attn_k_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_20_attn_v_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_20_attn_v_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_20_attn_v_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_20_attn_v_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_20_attn_k_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_20_attn_k_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_20_attn_k_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_20_attn_k_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_20_attn_v_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_20_attn_v_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_20_attn_k_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_20_attn_k_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_20_attn_v_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_20_attn_v_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_20_attn_k_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_20_attn_k_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_20_attn_v_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_20_attn_v_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_20_attn_k_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_20_attn_k_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_20_attn_v_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_20_attn_v_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_20_attn_k_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_20_attn_k_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_20_attn_v_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_20_attn_v_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_20_attn_k_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_20_attn_k_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_20_attn_v_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_20_attn_v_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.model_layers_20_attn_k_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_20_attn_k_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.com_microsoft__model_layers_20_attn_MultiHeadAttention = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_20_attn_MultiHeadAttention.pt', weights_only=False) # MultiHeadAttentionOp()
        self.model_layers_20_attn_o_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_20_attn_o_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_20_post_attention_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_20_post_attention_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_20_mlp_gate_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_20_mlp_gate_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_20_mlp_up_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_20_mlp_up_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_20_mlp_act_fn_Sigmoid = torch.load(r'decoder_model_merged_torch/model_layers_20_mlp_act_fn_Sigmoid.pt', weights_only=False) # Sigmoid()
        self.model_layers_20_mlp_act_fn_Mul = torch.load(r'decoder_model_merged_torch/model_layers_20_mlp_act_fn_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_20_mlp_Mul = torch.load(r'decoder_model_merged_torch/model_layers_20_mlp_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_20_mlp_down_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_20_mlp_down_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_21_input_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_21_input_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_21_attn_q_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_21_attn_q_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_21_attn_k_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_21_attn_k_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_21_attn_v_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_21_attn_v_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_21_attn_q_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_21_attn_q_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.com_microsoft__model_layers_21_attn_k_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_21_attn_k_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.model_layers_21_attn_v_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_21_attn_v_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_21_attn_k_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_21_attn_k_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_21_attn_v_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_21_attn_v_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_21_attn_k_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_21_attn_k_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_21_attn_v_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_21_attn_v_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_21_attn_k_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_21_attn_k_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_21_attn_v_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_21_attn_v_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_21_attn_v_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_21_attn_v_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_21_attn_k_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_21_attn_k_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_21_attn_k_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_21_attn_k_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_21_attn_v_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_21_attn_v_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_21_attn_v_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_21_attn_v_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_21_attn_k_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_21_attn_k_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_21_attn_k_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_21_attn_k_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_21_attn_v_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_21_attn_v_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_21_attn_v_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_21_attn_v_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_21_attn_k_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_21_attn_k_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_21_attn_k_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_21_attn_k_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_21_attn_v_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_21_attn_v_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_21_attn_v_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_21_attn_v_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_21_attn_k_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_21_attn_k_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_21_attn_k_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_21_attn_k_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_21_attn_v_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_21_attn_v_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_21_attn_k_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_21_attn_k_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_21_attn_v_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_21_attn_v_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_21_attn_k_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_21_attn_k_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_21_attn_v_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_21_attn_v_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_21_attn_k_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_21_attn_k_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_21_attn_v_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_21_attn_v_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_21_attn_k_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_21_attn_k_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_21_attn_v_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_21_attn_v_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_21_attn_k_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_21_attn_k_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_21_attn_v_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_21_attn_v_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.model_layers_21_attn_k_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_21_attn_k_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.com_microsoft__model_layers_21_attn_MultiHeadAttention = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_21_attn_MultiHeadAttention.pt', weights_only=False) # MultiHeadAttentionOp()
        self.model_layers_21_attn_o_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_21_attn_o_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_21_post_attention_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_21_post_attention_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_21_mlp_gate_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_21_mlp_gate_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_21_mlp_up_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_21_mlp_up_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_21_mlp_act_fn_Sigmoid = torch.load(r'decoder_model_merged_torch/model_layers_21_mlp_act_fn_Sigmoid.pt', weights_only=False) # Sigmoid()
        self.model_layers_21_mlp_act_fn_Mul = torch.load(r'decoder_model_merged_torch/model_layers_21_mlp_act_fn_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_21_mlp_Mul = torch.load(r'decoder_model_merged_torch/model_layers_21_mlp_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_21_mlp_down_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_21_mlp_down_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_22_input_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_22_input_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_22_attn_q_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_22_attn_q_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_22_attn_k_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_22_attn_k_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_22_attn_v_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_22_attn_v_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_22_attn_q_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_22_attn_q_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.com_microsoft__model_layers_22_attn_k_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_22_attn_k_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.model_layers_22_attn_v_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_22_attn_v_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_22_attn_k_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_22_attn_k_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_22_attn_v_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_22_attn_v_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_22_attn_k_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_22_attn_k_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_22_attn_v_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_22_attn_v_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_22_attn_k_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_22_attn_k_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_22_attn_v_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_22_attn_v_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_22_attn_v_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_22_attn_v_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_22_attn_k_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_22_attn_k_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_22_attn_k_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_22_attn_k_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_22_attn_v_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_22_attn_v_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_22_attn_v_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_22_attn_v_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_22_attn_k_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_22_attn_k_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_22_attn_k_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_22_attn_k_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_22_attn_v_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_22_attn_v_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_22_attn_v_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_22_attn_v_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_22_attn_k_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_22_attn_k_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_22_attn_k_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_22_attn_k_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_22_attn_v_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_22_attn_v_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_22_attn_v_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_22_attn_v_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_22_attn_k_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_22_attn_k_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_22_attn_k_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_22_attn_k_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_22_attn_v_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_22_attn_v_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_22_attn_k_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_22_attn_k_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_22_attn_v_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_22_attn_v_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_22_attn_k_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_22_attn_k_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_22_attn_v_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_22_attn_v_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_22_attn_k_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_22_attn_k_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_22_attn_v_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_22_attn_v_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_22_attn_k_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_22_attn_k_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_22_attn_v_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_22_attn_v_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_22_attn_k_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_22_attn_k_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_22_attn_v_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_22_attn_v_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.model_layers_22_attn_k_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_22_attn_k_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.com_microsoft__model_layers_22_attn_MultiHeadAttention = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_22_attn_MultiHeadAttention.pt', weights_only=False) # MultiHeadAttentionOp()
        self.model_layers_22_attn_o_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_22_attn_o_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_22_post_attention_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_22_post_attention_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_22_mlp_gate_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_22_mlp_gate_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_22_mlp_up_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_22_mlp_up_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_22_mlp_act_fn_Sigmoid = torch.load(r'decoder_model_merged_torch/model_layers_22_mlp_act_fn_Sigmoid.pt', weights_only=False) # Sigmoid()
        self.model_layers_22_mlp_act_fn_Mul = torch.load(r'decoder_model_merged_torch/model_layers_22_mlp_act_fn_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_22_mlp_Mul = torch.load(r'decoder_model_merged_torch/model_layers_22_mlp_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_22_mlp_down_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_22_mlp_down_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_23_input_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_23_input_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_23_attn_q_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_23_attn_q_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_23_attn_k_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_23_attn_k_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_23_attn_v_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_23_attn_v_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_23_attn_q_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_23_attn_q_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.com_microsoft__model_layers_23_attn_k_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_23_attn_k_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.model_layers_23_attn_v_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_23_attn_v_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_23_attn_k_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_23_attn_k_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_23_attn_v_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_23_attn_v_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_23_attn_k_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_23_attn_k_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_23_attn_v_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_23_attn_v_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_23_attn_k_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_23_attn_k_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_23_attn_v_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_23_attn_v_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_23_attn_v_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_23_attn_v_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_23_attn_k_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_23_attn_k_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_23_attn_k_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_23_attn_k_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_23_attn_v_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_23_attn_v_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_23_attn_v_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_23_attn_v_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_23_attn_k_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_23_attn_k_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_23_attn_k_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_23_attn_k_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_23_attn_v_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_23_attn_v_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_23_attn_v_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_23_attn_v_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_23_attn_k_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_23_attn_k_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_23_attn_k_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_23_attn_k_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_23_attn_v_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_23_attn_v_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_23_attn_v_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_23_attn_v_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_23_attn_k_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_23_attn_k_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_23_attn_k_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_23_attn_k_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_23_attn_v_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_23_attn_v_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_23_attn_k_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_23_attn_k_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_23_attn_v_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_23_attn_v_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_23_attn_k_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_23_attn_k_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_23_attn_v_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_23_attn_v_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_23_attn_k_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_23_attn_k_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_23_attn_v_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_23_attn_v_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_23_attn_k_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_23_attn_k_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_23_attn_v_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_23_attn_v_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_23_attn_k_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_23_attn_k_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_23_attn_v_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_23_attn_v_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.model_layers_23_attn_k_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_23_attn_k_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.com_microsoft__model_layers_23_attn_MultiHeadAttention = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_23_attn_MultiHeadAttention.pt', weights_only=False) # MultiHeadAttentionOp()
        self.model_layers_23_attn_o_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_23_attn_o_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_23_post_attention_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_23_post_attention_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_23_mlp_gate_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_23_mlp_gate_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_23_mlp_up_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_23_mlp_up_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_23_mlp_act_fn_Sigmoid = torch.load(r'decoder_model_merged_torch/model_layers_23_mlp_act_fn_Sigmoid.pt', weights_only=False) # Sigmoid()
        self.model_layers_23_mlp_act_fn_Mul = torch.load(r'decoder_model_merged_torch/model_layers_23_mlp_act_fn_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_23_mlp_Mul = torch.load(r'decoder_model_merged_torch/model_layers_23_mlp_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_23_mlp_down_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_23_mlp_down_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_24_input_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_24_input_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_24_attn_q_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_24_attn_q_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_24_attn_k_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_24_attn_k_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_24_attn_v_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_24_attn_v_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_24_attn_q_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_24_attn_q_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.com_microsoft__model_layers_24_attn_k_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_24_attn_k_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.model_layers_24_attn_v_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_24_attn_v_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_24_attn_k_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_24_attn_k_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_24_attn_v_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_24_attn_v_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_24_attn_k_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_24_attn_k_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_24_attn_v_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_24_attn_v_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_24_attn_k_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_24_attn_k_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_24_attn_v_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_24_attn_v_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_24_attn_v_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_24_attn_v_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_24_attn_k_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_24_attn_k_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_24_attn_k_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_24_attn_k_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_24_attn_v_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_24_attn_v_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_24_attn_v_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_24_attn_v_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_24_attn_k_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_24_attn_k_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_24_attn_k_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_24_attn_k_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_24_attn_v_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_24_attn_v_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_24_attn_v_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_24_attn_v_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_24_attn_k_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_24_attn_k_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_24_attn_k_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_24_attn_k_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_24_attn_v_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_24_attn_v_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_24_attn_v_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_24_attn_v_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_24_attn_k_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_24_attn_k_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_24_attn_k_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_24_attn_k_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_24_attn_v_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_24_attn_v_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_24_attn_k_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_24_attn_k_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_24_attn_v_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_24_attn_v_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_24_attn_k_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_24_attn_k_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_24_attn_v_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_24_attn_v_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_24_attn_k_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_24_attn_k_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_24_attn_v_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_24_attn_v_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_24_attn_k_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_24_attn_k_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_24_attn_v_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_24_attn_v_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_24_attn_k_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_24_attn_k_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_24_attn_v_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_24_attn_v_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.model_layers_24_attn_k_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_24_attn_k_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.com_microsoft__model_layers_24_attn_MultiHeadAttention = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_24_attn_MultiHeadAttention.pt', weights_only=False) # MultiHeadAttentionOp()
        self.model_layers_24_attn_o_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_24_attn_o_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_24_post_attention_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_24_post_attention_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_24_mlp_gate_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_24_mlp_gate_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_24_mlp_up_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_24_mlp_up_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_24_mlp_act_fn_Sigmoid = torch.load(r'decoder_model_merged_torch/model_layers_24_mlp_act_fn_Sigmoid.pt', weights_only=False) # Sigmoid()
        self.model_layers_24_mlp_act_fn_Mul = torch.load(r'decoder_model_merged_torch/model_layers_24_mlp_act_fn_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_24_mlp_Mul = torch.load(r'decoder_model_merged_torch/model_layers_24_mlp_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_24_mlp_down_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_24_mlp_down_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_25_input_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_25_input_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_25_attn_q_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_25_attn_q_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_25_attn_k_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_25_attn_k_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_25_attn_v_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_25_attn_v_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_25_attn_q_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_25_attn_q_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.com_microsoft__model_layers_25_attn_k_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_25_attn_k_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.model_layers_25_attn_v_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_25_attn_v_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_25_attn_k_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_25_attn_k_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_25_attn_v_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_25_attn_v_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_25_attn_k_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_25_attn_k_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_25_attn_v_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_25_attn_v_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_25_attn_k_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_25_attn_k_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_25_attn_v_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_25_attn_v_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_25_attn_v_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_25_attn_v_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_25_attn_k_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_25_attn_k_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_25_attn_k_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_25_attn_k_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_25_attn_v_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_25_attn_v_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_25_attn_v_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_25_attn_v_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_25_attn_k_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_25_attn_k_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_25_attn_k_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_25_attn_k_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_25_attn_v_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_25_attn_v_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_25_attn_v_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_25_attn_v_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_25_attn_k_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_25_attn_k_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_25_attn_k_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_25_attn_k_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_25_attn_v_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_25_attn_v_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_25_attn_v_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_25_attn_v_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_25_attn_k_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_25_attn_k_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_25_attn_k_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_25_attn_k_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_25_attn_v_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_25_attn_v_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_25_attn_k_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_25_attn_k_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_25_attn_v_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_25_attn_v_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_25_attn_k_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_25_attn_k_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_25_attn_v_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_25_attn_v_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_25_attn_k_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_25_attn_k_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_25_attn_v_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_25_attn_v_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_25_attn_k_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_25_attn_k_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_25_attn_v_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_25_attn_v_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_25_attn_k_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_25_attn_k_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_25_attn_v_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_25_attn_v_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.model_layers_25_attn_k_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_25_attn_k_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.com_microsoft__model_layers_25_attn_MultiHeadAttention = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_25_attn_MultiHeadAttention.pt', weights_only=False) # MultiHeadAttentionOp()
        self.model_layers_25_attn_o_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_25_attn_o_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_25_post_attention_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_25_post_attention_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_25_mlp_gate_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_25_mlp_gate_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_25_mlp_up_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_25_mlp_up_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_25_mlp_act_fn_Sigmoid = torch.load(r'decoder_model_merged_torch/model_layers_25_mlp_act_fn_Sigmoid.pt', weights_only=False) # Sigmoid()
        self.model_layers_25_mlp_act_fn_Mul = torch.load(r'decoder_model_merged_torch/model_layers_25_mlp_act_fn_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_25_mlp_Mul = torch.load(r'decoder_model_merged_torch/model_layers_25_mlp_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_25_mlp_down_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_25_mlp_down_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_26_input_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_26_input_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_26_attn_q_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_26_attn_q_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_26_attn_k_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_26_attn_k_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_26_attn_v_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_26_attn_v_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_26_attn_q_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_26_attn_q_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.com_microsoft__model_layers_26_attn_k_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_26_attn_k_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.model_layers_26_attn_v_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_26_attn_v_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_26_attn_k_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_26_attn_k_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_26_attn_v_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_26_attn_v_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_26_attn_k_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_26_attn_k_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_26_attn_v_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_26_attn_v_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_26_attn_k_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_26_attn_k_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_26_attn_v_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_26_attn_v_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_26_attn_v_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_26_attn_v_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_26_attn_k_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_26_attn_k_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_26_attn_k_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_26_attn_k_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_26_attn_v_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_26_attn_v_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_26_attn_v_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_26_attn_v_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_26_attn_k_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_26_attn_k_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_26_attn_k_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_26_attn_k_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_26_attn_v_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_26_attn_v_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_26_attn_v_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_26_attn_v_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_26_attn_k_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_26_attn_k_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_26_attn_k_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_26_attn_k_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_26_attn_v_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_26_attn_v_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_26_attn_v_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_26_attn_v_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_26_attn_k_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_26_attn_k_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_26_attn_k_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_26_attn_k_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_26_attn_v_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_26_attn_v_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_26_attn_k_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_26_attn_k_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_26_attn_v_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_26_attn_v_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_26_attn_k_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_26_attn_k_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_26_attn_v_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_26_attn_v_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_26_attn_k_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_26_attn_k_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_26_attn_v_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_26_attn_v_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_26_attn_k_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_26_attn_k_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_26_attn_v_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_26_attn_v_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_26_attn_k_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_26_attn_k_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_26_attn_v_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_26_attn_v_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.model_layers_26_attn_k_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_26_attn_k_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.com_microsoft__model_layers_26_attn_MultiHeadAttention = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_26_attn_MultiHeadAttention.pt', weights_only=False) # MultiHeadAttentionOp()
        self.model_layers_26_attn_o_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_26_attn_o_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_26_post_attention_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_26_post_attention_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_26_mlp_gate_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_26_mlp_gate_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_26_mlp_up_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_26_mlp_up_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_26_mlp_act_fn_Sigmoid = torch.load(r'decoder_model_merged_torch/model_layers_26_mlp_act_fn_Sigmoid.pt', weights_only=False) # Sigmoid()
        self.model_layers_26_mlp_act_fn_Mul = torch.load(r'decoder_model_merged_torch/model_layers_26_mlp_act_fn_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_26_mlp_Mul = torch.load(r'decoder_model_merged_torch/model_layers_26_mlp_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_26_mlp_down_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_26_mlp_down_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_27_input_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_27_input_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_27_attn_q_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_27_attn_q_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_27_attn_k_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_27_attn_k_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_27_attn_v_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_27_attn_v_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_27_attn_q_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_27_attn_q_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.com_microsoft__model_layers_27_attn_k_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_27_attn_k_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.model_layers_27_attn_v_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_27_attn_v_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_27_attn_k_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_27_attn_k_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_27_attn_v_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_27_attn_v_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_27_attn_k_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_27_attn_k_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_27_attn_v_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_27_attn_v_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_27_attn_k_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_27_attn_k_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_27_attn_v_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_27_attn_v_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_27_attn_v_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_27_attn_v_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_27_attn_k_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_27_attn_k_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_27_attn_k_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_27_attn_k_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_27_attn_v_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_27_attn_v_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_27_attn_v_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_27_attn_v_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_27_attn_k_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_27_attn_k_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_27_attn_k_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_27_attn_k_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_27_attn_v_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_27_attn_v_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_27_attn_v_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_27_attn_v_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_27_attn_k_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_27_attn_k_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_27_attn_k_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_27_attn_k_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_27_attn_v_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_27_attn_v_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_27_attn_v_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_27_attn_v_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_27_attn_k_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_27_attn_k_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_27_attn_k_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_27_attn_k_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_27_attn_v_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_27_attn_v_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_27_attn_k_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_27_attn_k_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_27_attn_v_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_27_attn_v_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_27_attn_k_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_27_attn_k_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_27_attn_v_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_27_attn_v_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_27_attn_k_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_27_attn_k_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_27_attn_v_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_27_attn_v_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_27_attn_k_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_27_attn_k_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_27_attn_v_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_27_attn_v_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_27_attn_k_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_27_attn_k_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_27_attn_v_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_27_attn_v_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.model_layers_27_attn_k_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_27_attn_k_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.com_microsoft__model_layers_27_attn_MultiHeadAttention = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_27_attn_MultiHeadAttention.pt', weights_only=False) # MultiHeadAttentionOp()
        self.model_layers_27_attn_o_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_27_attn_o_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_27_post_attention_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_27_post_attention_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_27_mlp_gate_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_27_mlp_gate_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_27_mlp_up_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_27_mlp_up_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_27_mlp_act_fn_Sigmoid = torch.load(r'decoder_model_merged_torch/model_layers_27_mlp_act_fn_Sigmoid.pt', weights_only=False) # Sigmoid()
        self.model_layers_27_mlp_act_fn_Mul = torch.load(r'decoder_model_merged_torch/model_layers_27_mlp_act_fn_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_27_mlp_Mul = torch.load(r'decoder_model_merged_torch/model_layers_27_mlp_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_27_mlp_down_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_27_mlp_down_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_28_input_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_28_input_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_28_attn_q_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_28_attn_q_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_28_attn_k_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_28_attn_k_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_28_attn_v_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_28_attn_v_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_28_attn_q_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_28_attn_q_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.com_microsoft__model_layers_28_attn_k_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_28_attn_k_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.model_layers_28_attn_v_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_28_attn_v_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_28_attn_k_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_28_attn_k_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_28_attn_v_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_28_attn_v_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_28_attn_k_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_28_attn_k_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_28_attn_v_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_28_attn_v_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_28_attn_k_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_28_attn_k_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_28_attn_v_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_28_attn_v_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_28_attn_v_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_28_attn_v_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_28_attn_k_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_28_attn_k_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_28_attn_k_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_28_attn_k_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_28_attn_v_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_28_attn_v_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_28_attn_v_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_28_attn_v_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_28_attn_k_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_28_attn_k_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_28_attn_k_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_28_attn_k_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_28_attn_v_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_28_attn_v_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_28_attn_v_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_28_attn_v_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_28_attn_k_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_28_attn_k_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_28_attn_k_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_28_attn_k_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_28_attn_v_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_28_attn_v_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_28_attn_v_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_28_attn_v_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_28_attn_k_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_28_attn_k_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_28_attn_k_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_28_attn_k_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_28_attn_v_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_28_attn_v_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_28_attn_k_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_28_attn_k_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_28_attn_v_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_28_attn_v_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_28_attn_k_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_28_attn_k_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_28_attn_v_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_28_attn_v_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_28_attn_k_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_28_attn_k_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_28_attn_v_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_28_attn_v_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_28_attn_k_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_28_attn_k_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_28_attn_v_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_28_attn_v_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_28_attn_k_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_28_attn_k_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_28_attn_v_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_28_attn_v_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.model_layers_28_attn_k_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_28_attn_k_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.com_microsoft__model_layers_28_attn_MultiHeadAttention = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_28_attn_MultiHeadAttention.pt', weights_only=False) # MultiHeadAttentionOp()
        self.model_layers_28_attn_o_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_28_attn_o_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_28_post_attention_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_28_post_attention_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_28_mlp_gate_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_28_mlp_gate_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_28_mlp_up_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_28_mlp_up_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_28_mlp_act_fn_Sigmoid = torch.load(r'decoder_model_merged_torch/model_layers_28_mlp_act_fn_Sigmoid.pt', weights_only=False) # Sigmoid()
        self.model_layers_28_mlp_act_fn_Mul = torch.load(r'decoder_model_merged_torch/model_layers_28_mlp_act_fn_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_28_mlp_Mul = torch.load(r'decoder_model_merged_torch/model_layers_28_mlp_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_28_mlp_down_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_28_mlp_down_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_29_input_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_29_input_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_29_attn_q_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_29_attn_q_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_29_attn_k_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_29_attn_k_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_29_attn_v_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_29_attn_v_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_29_attn_q_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_29_attn_q_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.com_microsoft__model_layers_29_attn_k_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_29_attn_k_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.model_layers_29_attn_v_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_29_attn_v_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_29_attn_k_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_29_attn_k_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_29_attn_v_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_29_attn_v_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_29_attn_k_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_29_attn_k_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_29_attn_v_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_29_attn_v_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_29_attn_k_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_29_attn_k_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_29_attn_v_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_29_attn_v_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_29_attn_v_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_29_attn_v_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_29_attn_k_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_29_attn_k_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_29_attn_k_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_29_attn_k_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_29_attn_v_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_29_attn_v_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_29_attn_v_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_29_attn_v_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_29_attn_k_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_29_attn_k_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_29_attn_k_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_29_attn_k_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_29_attn_v_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_29_attn_v_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_29_attn_v_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_29_attn_v_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_29_attn_k_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_29_attn_k_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_29_attn_k_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_29_attn_k_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_29_attn_v_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_29_attn_v_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_29_attn_v_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_29_attn_v_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_29_attn_k_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_29_attn_k_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_29_attn_k_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_29_attn_k_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_29_attn_v_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_29_attn_v_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_29_attn_k_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_29_attn_k_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_29_attn_v_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_29_attn_v_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_29_attn_k_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_29_attn_k_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_29_attn_v_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_29_attn_v_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_29_attn_k_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_29_attn_k_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_29_attn_v_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_29_attn_v_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_29_attn_k_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_29_attn_k_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_29_attn_v_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_29_attn_v_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_29_attn_k_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_29_attn_k_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_29_attn_v_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_29_attn_v_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.model_layers_29_attn_k_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_29_attn_k_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.com_microsoft__model_layers_29_attn_MultiHeadAttention = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_29_attn_MultiHeadAttention.pt', weights_only=False) # MultiHeadAttentionOp()
        self.model_layers_29_attn_o_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_29_attn_o_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_29_post_attention_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_29_post_attention_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_29_mlp_gate_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_29_mlp_gate_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_29_mlp_up_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_29_mlp_up_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_29_mlp_act_fn_Sigmoid = torch.load(r'decoder_model_merged_torch/model_layers_29_mlp_act_fn_Sigmoid.pt', weights_only=False) # Sigmoid()
        self.model_layers_29_mlp_act_fn_Mul = torch.load(r'decoder_model_merged_torch/model_layers_29_mlp_act_fn_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_29_mlp_Mul = torch.load(r'decoder_model_merged_torch/model_layers_29_mlp_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_29_mlp_down_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_29_mlp_down_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_30_input_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_30_input_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_30_attn_q_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_30_attn_q_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_30_attn_k_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_30_attn_k_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_30_attn_v_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_30_attn_v_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_30_attn_q_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_30_attn_q_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.com_microsoft__model_layers_30_attn_k_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_30_attn_k_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.model_layers_30_attn_v_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_30_attn_v_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_30_attn_k_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_30_attn_k_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_30_attn_v_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_30_attn_v_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_30_attn_k_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_30_attn_k_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_30_attn_v_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_30_attn_v_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_30_attn_k_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_30_attn_k_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_30_attn_v_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_30_attn_v_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_30_attn_v_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_30_attn_v_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_30_attn_k_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_30_attn_k_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_30_attn_k_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_30_attn_k_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_30_attn_v_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_30_attn_v_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_30_attn_v_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_30_attn_v_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_30_attn_k_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_30_attn_k_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_30_attn_k_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_30_attn_k_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_30_attn_v_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_30_attn_v_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_30_attn_v_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_30_attn_v_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_30_attn_k_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_30_attn_k_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_30_attn_k_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_30_attn_k_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_30_attn_v_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_30_attn_v_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_30_attn_v_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_30_attn_v_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_30_attn_k_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_30_attn_k_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_30_attn_k_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_30_attn_k_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_30_attn_v_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_30_attn_v_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_30_attn_k_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_30_attn_k_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_30_attn_v_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_30_attn_v_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_30_attn_k_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_30_attn_k_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_30_attn_v_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_30_attn_v_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_30_attn_k_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_30_attn_k_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_30_attn_v_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_30_attn_v_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_30_attn_k_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_30_attn_k_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_30_attn_v_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_30_attn_v_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_30_attn_k_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_30_attn_k_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_30_attn_v_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_30_attn_v_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.model_layers_30_attn_k_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_30_attn_k_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.com_microsoft__model_layers_30_attn_MultiHeadAttention = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_30_attn_MultiHeadAttention.pt', weights_only=False) # MultiHeadAttentionOp()
        self.model_layers_30_attn_o_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_30_attn_o_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_30_post_attention_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_30_post_attention_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_30_mlp_gate_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_30_mlp_gate_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_30_mlp_up_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_30_mlp_up_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_30_mlp_act_fn_Sigmoid = torch.load(r'decoder_model_merged_torch/model_layers_30_mlp_act_fn_Sigmoid.pt', weights_only=False) # Sigmoid()
        self.model_layers_30_mlp_act_fn_Mul = torch.load(r'decoder_model_merged_torch/model_layers_30_mlp_act_fn_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_30_mlp_Mul = torch.load(r'decoder_model_merged_torch/model_layers_30_mlp_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_30_mlp_down_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_30_mlp_down_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_31_input_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_31_input_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_31_attn_q_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_31_attn_q_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_31_attn_k_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_31_attn_k_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_31_attn_v_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_31_attn_v_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_31_attn_q_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_31_attn_q_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.com_microsoft__model_layers_31_attn_k_rotary_RotaryEmbedding = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_31_attn_k_rotary_RotaryEmbedding.pt', weights_only=False) # RotaryEmbeddingOp()
        self.model_layers_31_attn_v_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_31_attn_v_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_31_attn_k_proj_repeat_kv_Reshape_1 = torch.load(r'decoder_model_merged_torch/model_layers_31_attn_k_proj_repeat_kv_Reshape_1.pt', weights_only=False) # OnnxReshape()
        self.model_layers_31_attn_v_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_31_attn_v_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_31_attn_k_proj_repeat_kv_Transpose_1 = torch.load(r'decoder_model_merged_torch/model_layers_31_attn_k_proj_repeat_kv_Transpose_1.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_31_attn_v_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_31_attn_v_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_31_attn_k_proj_repeat_kv_Concat_1 = torch.load(r'decoder_model_merged_torch/model_layers_31_attn_k_proj_repeat_kv_Concat_1.pt', weights_only=False) # OnnxConcat()
        self.model_layers_31_attn_v_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_31_attn_v_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_31_attn_v_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_31_attn_v_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_31_attn_k_proj_repeat_kv_Shape_1 = torch.load(r'decoder_model_merged_torch/model_layers_31_attn_k_proj_repeat_kv_Shape_1.pt', weights_only=False) # OnnxShape()
        self.model_layers_31_attn_k_proj_repeat_kv_Unsqueeze_5 = torch.load(r'decoder_model_merged_torch/model_layers_31_attn_k_proj_repeat_kv_Unsqueeze_5.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_31_attn_v_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_31_attn_v_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_31_attn_v_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_31_attn_v_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_31_attn_k_proj_repeat_kv_Gather_1 = torch.load(r'decoder_model_merged_torch/model_layers_31_attn_k_proj_repeat_kv_Gather_1.pt', weights_only=False) # OnnxGather()
        self.model_layers_31_attn_k_proj_repeat_kv_Gather_3 = torch.load(r'decoder_model_merged_torch/model_layers_31_attn_k_proj_repeat_kv_Gather_3.pt', weights_only=False) # OnnxGather()
        self.model_layers_31_attn_v_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_31_attn_v_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_31_attn_v_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_31_attn_v_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_31_attn_k_proj_repeat_kv_Unsqueeze_1 = torch.load(r'decoder_model_merged_torch/model_layers_31_attn_k_proj_repeat_kv_Unsqueeze_1.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_31_attn_k_proj_repeat_kv_Unsqueeze_3 = torch.load(r'decoder_model_merged_torch/model_layers_31_attn_k_proj_repeat_kv_Unsqueeze_3.pt', weights_only=False) # OnnxUnsqueezeStaticAxes()
        self.model_layers_31_attn_v_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_31_attn_v_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_31_attn_v_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_31_attn_v_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_31_attn_k_proj_repeat_kv_Concat_2 = torch.load(r'decoder_model_merged_torch/model_layers_31_attn_k_proj_repeat_kv_Concat_2.pt', weights_only=False) # OnnxConcat()
        self.model_layers_31_attn_k_proj_repeat_kv_Concat_3 = torch.load(r'decoder_model_merged_torch/model_layers_31_attn_k_proj_repeat_kv_Concat_3.pt', weights_only=False) # OnnxConcat()
        self.model_layers_31_attn_v_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_31_attn_v_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_31_attn_k_proj_repeat_kv_Equal = torch.load(r'decoder_model_merged_torch/model_layers_31_attn_k_proj_repeat_kv_Equal.pt', weights_only=False) # OnnxCompare()
        self.model_layers_31_attn_v_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_31_attn_v_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_31_attn_k_proj_repeat_kv_Where = torch.load(r'decoder_model_merged_torch/model_layers_31_attn_k_proj_repeat_kv_Where.pt', weights_only=False) # OnnxWhere()
        self.model_layers_31_attn_v_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_31_attn_v_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_31_attn_k_proj_repeat_kv_Expand = torch.load(r'decoder_model_merged_torch/model_layers_31_attn_k_proj_repeat_kv_Expand.pt', weights_only=False) # OnnxExpand()
        self.model_layers_31_attn_v_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_31_attn_v_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_31_attn_k_proj_repeat_kv_Reshape_3 = torch.load(r'decoder_model_merged_torch/model_layers_31_attn_k_proj_repeat_kv_Reshape_3.pt', weights_only=False) # OnnxReshape()
        self.model_layers_31_attn_v_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_31_attn_v_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_31_attn_k_proj_repeat_kv_Transpose_2 = torch.load(r'decoder_model_merged_torch/model_layers_31_attn_k_proj_repeat_kv_Transpose_2.pt', weights_only=False) # OnnxTranspose()
        self.model_layers_31_attn_v_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_31_attn_v_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.model_layers_31_attn_k_proj_repeat_kv_Reshape_4 = torch.load(r'decoder_model_merged_torch/model_layers_31_attn_k_proj_repeat_kv_Reshape_4.pt', weights_only=False) # OnnxReshape()
        self.com_microsoft__model_layers_31_attn_MultiHeadAttention = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_31_attn_MultiHeadAttention.pt', weights_only=False) # MultiHeadAttentionOp()
        self.model_layers_31_attn_o_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_31_attn_o_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_31_post_attention_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_31_post_attention_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.model_layers_31_mlp_gate_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_31_mlp_gate_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_31_mlp_up_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_31_mlp_up_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.model_layers_31_mlp_act_fn_Sigmoid = torch.load(r'decoder_model_merged_torch/model_layers_31_mlp_act_fn_Sigmoid.pt', weights_only=False) # Sigmoid()
        self.model_layers_31_mlp_act_fn_Mul = torch.load(r'decoder_model_merged_torch/model_layers_31_mlp_act_fn_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_31_mlp_Mul = torch.load(r'decoder_model_merged_torch/model_layers_31_mlp_Mul.pt', weights_only=False) # OnnxBinaryMathOperation()
        self.model_layers_31_mlp_down_proj_MatMul = torch.load(r'decoder_model_merged_torch/model_layers_31_mlp_down_proj_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.com_microsoft__model_layers_32_final_norm_layernorm_SkipLayerNorm = torch.load(r'decoder_model_merged_torch/com_microsoft__model_layers_32_final_norm_layernorm_SkipLayerNorm.pt', weights_only=False) # SkipSimplifiedLayerNormOp()
        self.lm_head_MatMul = torch.load(r'decoder_model_merged_torch/lm_head_MatMul.pt', weights_only=False) # OnnxMatMul()
        self.load_state_dict(torch.load(r'decoder_model_merged_torch/state_dict.pt'))

    
    
    def forward(self, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21, input_22, input_23, input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67):
        model_attn_mask_reformat_past_key_subgraph_shape = self.model_attn_mask_reformat_past_key_subgraph_Shape(input_4)
        model_attn_mask_reformat_input_ids_subgraph_shape_2 = self.model_attn_mask_reformat_input_ids_subgraph_Shape_2(input_1)
        model_attn_mask_reformat_attn_mask_subgraph_shape_1 = self.model_attn_mask_reformat_attn_mask_subgraph_Shape_1(input_2)
        initializers_onnx_initializer_0 = self.initializers.onnx_initializer_0
        model_layers_0_input_layernorm_layer_norm = self.model_layers_0_input_layernorm_LayerNorm(input_1, initializers_onnx_initializer_0);  initializers_onnx_initializer_0 = None
        model_attn_mask_reformat_attn_mask_subgraph_unsqueeze_3 = self.model_attn_mask_reformat_attn_mask_subgraph_Unsqueeze_3(input_2);  input_2 = None
        initializers_onnx_initializer_1 = self.initializers.onnx_initializer_1
        model_attn_mask_reformat_past_key_subgraph_gather = self.model_attn_mask_reformat_past_key_subgraph_Gather(model_attn_mask_reformat_past_key_subgraph_shape, initializers_onnx_initializer_1);  model_attn_mask_reformat_past_key_subgraph_shape = initializers_onnx_initializer_1 = None
        initializers_onnx_initializer_2 = self.initializers.onnx_initializer_2
        model_attn_mask_reformat_input_ids_subgraph_gather_2 = self.model_attn_mask_reformat_input_ids_subgraph_Gather_2(model_attn_mask_reformat_input_ids_subgraph_shape_2, initializers_onnx_initializer_2);  initializers_onnx_initializer_2 = None
        initializers_onnx_initializer_3 = self.initializers.onnx_initializer_3
        model_attn_mask_reformat_input_ids_subgraph_gather_1 = self.model_attn_mask_reformat_input_ids_subgraph_Gather_1(model_attn_mask_reformat_input_ids_subgraph_shape_2, initializers_onnx_initializer_3);  model_attn_mask_reformat_input_ids_subgraph_shape_2 = initializers_onnx_initializer_3 = None
        initializers_onnx_initializer_4 = self.initializers.onnx_initializer_4
        model_attn_mask_reformat_attn_mask_subgraph_gather_1 = self.model_attn_mask_reformat_attn_mask_subgraph_Gather_1(model_attn_mask_reformat_attn_mask_subgraph_shape_1, initializers_onnx_initializer_4);  initializers_onnx_initializer_4 = None
        initializers_onnx_initializer_5 = self.initializers.onnx_initializer_5
        model_attn_mask_reformat_attn_mask_subgraph_gather_2 = self.model_attn_mask_reformat_attn_mask_subgraph_Gather_2(model_attn_mask_reformat_attn_mask_subgraph_shape_1, initializers_onnx_initializer_5);  model_attn_mask_reformat_attn_mask_subgraph_shape_1 = initializers_onnx_initializer_5 = None
        initializers_onnx_initializer_6 = self.initializers.onnx_initializer_6
        model_layers_0_attn_q_proj_mat_mul = self.model_layers_0_attn_q_proj_MatMul(model_layers_0_input_layernorm_layer_norm, initializers_onnx_initializer_6);  initializers_onnx_initializer_6 = None
        initializers_onnx_initializer_7 = self.initializers.onnx_initializer_7
        model_layers_0_attn_k_proj_mat_mul = self.model_layers_0_attn_k_proj_MatMul(model_layers_0_input_layernorm_layer_norm, initializers_onnx_initializer_7);  initializers_onnx_initializer_7 = None
        initializers_onnx_initializer_8 = self.initializers.onnx_initializer_8
        model_layers_0_attn_v_proj_mat_mul = self.model_layers_0_attn_v_proj_MatMul(model_layers_0_input_layernorm_layer_norm, initializers_onnx_initializer_8);  model_layers_0_input_layernorm_layer_norm = initializers_onnx_initializer_8 = None
        model_attn_mask_reformat_input_ids_subgraph_add_1 = self.model_attn_mask_reformat_input_ids_subgraph_Add_1(model_attn_mask_reformat_input_ids_subgraph_gather_2, model_attn_mask_reformat_past_key_subgraph_gather);  model_attn_mask_reformat_past_key_subgraph_gather = None
        model_attn_mask_reformat_input_ids_subgraph_unsqueeze_4 = self.model_attn_mask_reformat_input_ids_subgraph_Unsqueeze_4(model_attn_mask_reformat_input_ids_subgraph_gather_2);  model_attn_mask_reformat_input_ids_subgraph_gather_2 = None
        model_attn_mask_reformat_input_ids_subgraph_unsqueeze_1 = self.model_attn_mask_reformat_input_ids_subgraph_Unsqueeze_1(model_attn_mask_reformat_input_ids_subgraph_gather_1);  model_attn_mask_reformat_input_ids_subgraph_gather_1 = None
        model_attn_mask_reformat_attn_mask_subgraph_unsqueeze_1 = self.model_attn_mask_reformat_attn_mask_subgraph_Unsqueeze_1(model_attn_mask_reformat_attn_mask_subgraph_gather_1);  model_attn_mask_reformat_attn_mask_subgraph_gather_1 = None
        model_attn_mask_reformat_attn_mask_subgraph_unsqueeze_2 = self.model_attn_mask_reformat_attn_mask_subgraph_Unsqueeze_2(model_attn_mask_reformat_attn_mask_subgraph_gather_2);  model_attn_mask_reformat_attn_mask_subgraph_gather_2 = None
        initializers_onnx_initializer_9 = self.initializers.onnx_initializer_9
        initializers_onnx_initializer_10 = self.initializers.onnx_initializer_10
        com_microsoft__model_layers_0_attn_q_rotary_rotary_embedding = self.com_microsoft__model_layers_0_attn_q_rotary_RotaryEmbedding(model_layers_0_attn_q_proj_mat_mul, input_3, initializers_onnx_initializer_9, initializers_onnx_initializer_10);  model_layers_0_attn_q_proj_mat_mul = initializers_onnx_initializer_9 = initializers_onnx_initializer_10 = None
        initializers_onnx_initializer_11 = self.initializers.onnx_initializer_11
        initializers_onnx_initializer_12 = self.initializers.onnx_initializer_12
        com_microsoft__model_layers_0_attn_k_rotary_rotary_embedding = self.com_microsoft__model_layers_0_attn_k_rotary_RotaryEmbedding(model_layers_0_attn_k_proj_mat_mul, input_3, initializers_onnx_initializer_11, initializers_onnx_initializer_12);  model_layers_0_attn_k_proj_mat_mul = initializers_onnx_initializer_11 = initializers_onnx_initializer_12 = None
        initializers_onnx_initializer_13 = self.initializers.onnx_initializer_13
        model_layers_0_attn_v_proj_repeat_kv_reshape_1 = self.model_layers_0_attn_v_proj_repeat_kv_Reshape_1(model_layers_0_attn_v_proj_mat_mul, initializers_onnx_initializer_13);  model_layers_0_attn_v_proj_mat_mul = initializers_onnx_initializer_13 = None
        model_attn_mask_reformat_input_ids_subgraph_unsqueeze_3 = self.model_attn_mask_reformat_input_ids_subgraph_Unsqueeze_3(model_attn_mask_reformat_input_ids_subgraph_add_1);  model_attn_mask_reformat_input_ids_subgraph_add_1 = None
        model_attn_mask_reformat_input_ids_subgraph_concat_2 = self.model_attn_mask_reformat_input_ids_subgraph_Concat_2(model_attn_mask_reformat_input_ids_subgraph_unsqueeze_4, model_attn_mask_reformat_input_ids_subgraph_unsqueeze_4)
        initializers_onnx_initializer_14 = self.initializers.onnx_initializer_14
        model_attn_mask_reformat_attn_mask_subgraph_concat = self.model_attn_mask_reformat_attn_mask_subgraph_Concat(model_attn_mask_reformat_attn_mask_subgraph_unsqueeze_1, initializers_onnx_initializer_14, model_attn_mask_reformat_input_ids_subgraph_unsqueeze_4, model_attn_mask_reformat_attn_mask_subgraph_unsqueeze_2);  model_attn_mask_reformat_attn_mask_subgraph_unsqueeze_1 = initializers_onnx_initializer_14 = model_attn_mask_reformat_attn_mask_subgraph_unsqueeze_2 = None
        initializers_onnx_initializer_15 = self.initializers.onnx_initializer_15
        model_layers_0_attn_k_proj_repeat_kv_reshape_1 = self.model_layers_0_attn_k_proj_repeat_kv_Reshape_1(com_microsoft__model_layers_0_attn_k_rotary_rotary_embedding, initializers_onnx_initializer_15);  com_microsoft__model_layers_0_attn_k_rotary_rotary_embedding = initializers_onnx_initializer_15 = None
        model_layers_0_attn_v_proj_repeat_kv_transpose_1 = self.model_layers_0_attn_v_proj_repeat_kv_Transpose_1(model_layers_0_attn_v_proj_repeat_kv_reshape_1);  model_layers_0_attn_v_proj_repeat_kv_reshape_1 = None
        model_attn_mask_reformat_input_ids_subgraph_constant_of_shape_2 = self.model_attn_mask_reformat_input_ids_subgraph_ConstantOfShape_2(model_attn_mask_reformat_input_ids_subgraph_concat_2);  model_attn_mask_reformat_input_ids_subgraph_concat_2 = None
        initializers_onnx_initializer_16 = self.initializers.onnx_initializer_16
        model_attn_mask_reformat_input_ids_subgraph_concat_1 = self.model_attn_mask_reformat_input_ids_subgraph_Concat_1(model_attn_mask_reformat_input_ids_subgraph_unsqueeze_1, initializers_onnx_initializer_16, model_attn_mask_reformat_input_ids_subgraph_unsqueeze_4, model_attn_mask_reformat_input_ids_subgraph_unsqueeze_3);  model_attn_mask_reformat_input_ids_subgraph_unsqueeze_1 = initializers_onnx_initializer_16 = model_attn_mask_reformat_input_ids_subgraph_unsqueeze_4 = model_attn_mask_reformat_input_ids_subgraph_unsqueeze_3 = None
        model_layers_0_attn_k_proj_repeat_kv_transpose_1 = self.model_layers_0_attn_k_proj_repeat_kv_Transpose_1(model_layers_0_attn_k_proj_repeat_kv_reshape_1);  model_layers_0_attn_k_proj_repeat_kv_reshape_1 = None
        model_layers_0_attn_v_proj_repeat_kv_concat_1 = self.model_layers_0_attn_v_proj_repeat_kv_Concat_1(input_5, model_layers_0_attn_v_proj_repeat_kv_transpose_1);  input_5 = model_layers_0_attn_v_proj_repeat_kv_transpose_1 = None
        initializers_onnx_initializer_17 = self.initializers.onnx_initializer_17
        model_attn_mask_reformat_attn_mask_subgraph_equal = self.model_attn_mask_reformat_attn_mask_subgraph_Equal(model_attn_mask_reformat_attn_mask_subgraph_concat, initializers_onnx_initializer_17);  initializers_onnx_initializer_17 = None
        model_attn_mask_reformat_input_ids_subgraph_shape_4 = self.model_attn_mask_reformat_input_ids_subgraph_Shape_4(model_attn_mask_reformat_input_ids_subgraph_constant_of_shape_2)
        model_layers_0_attn_k_proj_repeat_kv_concat_1 = self.model_layers_0_attn_k_proj_repeat_kv_Concat_1(input_4, model_layers_0_attn_k_proj_repeat_kv_transpose_1);  input_4 = model_layers_0_attn_k_proj_repeat_kv_transpose_1 = None
        model_layers_0_attn_v_proj_repeat_kv_shape_1 = self.model_layers_0_attn_v_proj_repeat_kv_Shape_1(model_layers_0_attn_v_proj_repeat_kv_concat_1)
        model_layers_0_attn_v_proj_repeat_kv_unsqueeze_5 = self.model_layers_0_attn_v_proj_repeat_kv_Unsqueeze_5(model_layers_0_attn_v_proj_repeat_kv_concat_1)
        initializers_onnx_initializer_18 = self.initializers.onnx_initializer_18
        model_attn_mask_reformat_input_ids_subgraph_equal = self.model_attn_mask_reformat_input_ids_subgraph_Equal(model_attn_mask_reformat_input_ids_subgraph_concat_1, initializers_onnx_initializer_18);  initializers_onnx_initializer_18 = None
        initializers_onnx_initializer_19 = self.initializers.onnx_initializer_19
        model_attn_mask_reformat_attn_mask_subgraph_where_1 = self.model_attn_mask_reformat_attn_mask_subgraph_Where_1(model_attn_mask_reformat_attn_mask_subgraph_equal, initializers_onnx_initializer_19, model_attn_mask_reformat_attn_mask_subgraph_concat);  model_attn_mask_reformat_attn_mask_subgraph_equal = initializers_onnx_initializer_19 = model_attn_mask_reformat_attn_mask_subgraph_concat = None
        initializers_onnx_initializer_20 = self.initializers.onnx_initializer_20
        initializers_onnx_initializer_21 = self.initializers.onnx_initializer_21
        initializers_onnx_initializer_22 = self.initializers.onnx_initializer_22
        model_attn_mask_reformat_input_ids_subgraph_slice_1 = self.model_attn_mask_reformat_input_ids_subgraph_Slice_1(model_attn_mask_reformat_input_ids_subgraph_shape_4, initializers_onnx_initializer_20, initializers_onnx_initializer_21, initializers_onnx_initializer_22);  model_attn_mask_reformat_input_ids_subgraph_shape_4 = initializers_onnx_initializer_20 = initializers_onnx_initializer_21 = initializers_onnx_initializer_22 = None
        model_layers_0_attn_k_proj_repeat_kv_shape_1 = self.model_layers_0_attn_k_proj_repeat_kv_Shape_1(model_layers_0_attn_k_proj_repeat_kv_concat_1)
        model_layers_0_attn_k_proj_repeat_kv_unsqueeze_5 = self.model_layers_0_attn_k_proj_repeat_kv_Unsqueeze_5(model_layers_0_attn_k_proj_repeat_kv_concat_1)
        initializers_onnx_initializer_23 = self.initializers.onnx_initializer_23
        model_layers_0_attn_v_proj_repeat_kv_gather_1 = self.model_layers_0_attn_v_proj_repeat_kv_Gather_1(model_layers_0_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_23);  initializers_onnx_initializer_23 = None
        initializers_onnx_initializer_24 = self.initializers.onnx_initializer_24
        model_layers_0_attn_v_proj_repeat_kv_gather_3 = self.model_layers_0_attn_v_proj_repeat_kv_Gather_3(model_layers_0_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_24);  model_layers_0_attn_v_proj_repeat_kv_shape_1 = initializers_onnx_initializer_24 = None
        initializers_onnx_initializer_25 = self.initializers.onnx_initializer_25
        model_attn_mask_reformat_input_ids_subgraph_where_1 = self.model_attn_mask_reformat_input_ids_subgraph_Where_1(model_attn_mask_reformat_input_ids_subgraph_equal, initializers_onnx_initializer_25, model_attn_mask_reformat_input_ids_subgraph_concat_1);  model_attn_mask_reformat_input_ids_subgraph_equal = initializers_onnx_initializer_25 = model_attn_mask_reformat_input_ids_subgraph_concat_1 = None
        model_attn_mask_reformat_attn_mask_subgraph_expand = self.model_attn_mask_reformat_attn_mask_subgraph_Expand(model_attn_mask_reformat_attn_mask_subgraph_unsqueeze_3, model_attn_mask_reformat_attn_mask_subgraph_where_1);  model_attn_mask_reformat_attn_mask_subgraph_unsqueeze_3 = model_attn_mask_reformat_attn_mask_subgraph_where_1 = None
        initializers_onnx_initializer_26 = self.initializers.onnx_initializer_26
        model_attn_mask_reformat_input_ids_subgraph_squeeze_1 = self.model_attn_mask_reformat_input_ids_subgraph_Squeeze_1(model_attn_mask_reformat_input_ids_subgraph_slice_1, initializers_onnx_initializer_26);  model_attn_mask_reformat_input_ids_subgraph_slice_1 = initializers_onnx_initializer_26 = None
        initializers_onnx_initializer_27 = self.initializers.onnx_initializer_27
        model_layers_0_attn_k_proj_repeat_kv_gather_1 = self.model_layers_0_attn_k_proj_repeat_kv_Gather_1(model_layers_0_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_27);  initializers_onnx_initializer_27 = None
        initializers_onnx_initializer_28 = self.initializers.onnx_initializer_28
        model_layers_0_attn_k_proj_repeat_kv_gather_3 = self.model_layers_0_attn_k_proj_repeat_kv_Gather_3(model_layers_0_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_28);  model_layers_0_attn_k_proj_repeat_kv_shape_1 = initializers_onnx_initializer_28 = None
        model_layers_0_attn_v_proj_repeat_kv_unsqueeze_1 = self.model_layers_0_attn_v_proj_repeat_kv_Unsqueeze_1(model_layers_0_attn_v_proj_repeat_kv_gather_1);  model_layers_0_attn_v_proj_repeat_kv_gather_1 = None
        model_layers_0_attn_v_proj_repeat_kv_unsqueeze_3 = self.model_layers_0_attn_v_proj_repeat_kv_Unsqueeze_3(model_layers_0_attn_v_proj_repeat_kv_gather_3);  model_layers_0_attn_v_proj_repeat_kv_gather_3 = None
        model_attn_mask_reformat_attn_mask_subgraph_cast_1 = self.model_attn_mask_reformat_attn_mask_subgraph_Cast_1(model_attn_mask_reformat_attn_mask_subgraph_expand);  model_attn_mask_reformat_attn_mask_subgraph_expand = None
        model_attn_mask_reformat_input_ids_subgraph_output_0 = self.model_attn_mask_reformat_input_ids_subgraph_output_0(model_attn_mask_reformat_input_ids_subgraph_squeeze_1)
        initializers_onnx_initializer_29 = self.initializers.onnx_initializer_29
        initializers_onnx_initializer_30 = self.initializers.onnx_initializer_30
        model_attn_mask_reformat_input_ids_subgraph_range = self.model_attn_mask_reformat_input_ids_subgraph_Range(initializers_onnx_initializer_29, model_attn_mask_reformat_input_ids_subgraph_squeeze_1, initializers_onnx_initializer_30);  initializers_onnx_initializer_29 = model_attn_mask_reformat_input_ids_subgraph_squeeze_1 = initializers_onnx_initializer_30 = None
        model_layers_0_attn_k_proj_repeat_kv_unsqueeze_1 = self.model_layers_0_attn_k_proj_repeat_kv_Unsqueeze_1(model_layers_0_attn_k_proj_repeat_kv_gather_1);  model_layers_0_attn_k_proj_repeat_kv_gather_1 = None
        model_layers_0_attn_k_proj_repeat_kv_unsqueeze_3 = self.model_layers_0_attn_k_proj_repeat_kv_Unsqueeze_3(model_layers_0_attn_k_proj_repeat_kv_gather_3);  model_layers_0_attn_k_proj_repeat_kv_gather_3 = None
        initializers_onnx_initializer_31 = self.initializers.onnx_initializer_31
        initializers_onnx_initializer_32 = self.initializers.onnx_initializer_32
        initializers_onnx_initializer_33 = self.initializers.onnx_initializer_33
        model_layers_0_attn_v_proj_repeat_kv_concat_2 = self.model_layers_0_attn_v_proj_repeat_kv_Concat_2(model_layers_0_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_31, initializers_onnx_initializer_32, model_layers_0_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_33);  initializers_onnx_initializer_31 = initializers_onnx_initializer_32 = initializers_onnx_initializer_33 = None
        initializers_onnx_initializer_34 = self.initializers.onnx_initializer_34
        initializers_onnx_initializer_35 = self.initializers.onnx_initializer_35
        model_layers_0_attn_v_proj_repeat_kv_concat_3 = self.model_layers_0_attn_v_proj_repeat_kv_Concat_3(model_layers_0_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_34, model_layers_0_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_35);  model_layers_0_attn_v_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_34 = model_layers_0_attn_v_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_35 = None
        initializers_onnx_initializer_36 = self.initializers.onnx_initializer_36
        model_attn_mask_reformat_attn_mask_subgraph_sub = self.model_attn_mask_reformat_attn_mask_subgraph_Sub(initializers_onnx_initializer_36, model_attn_mask_reformat_attn_mask_subgraph_cast_1);  initializers_onnx_initializer_36 = model_attn_mask_reformat_attn_mask_subgraph_cast_1 = None
        initializers_onnx_initializer_37 = self.initializers.onnx_initializer_37
        model_attn_mask_reformat_input_ids_subgraph_concat_3 = self.model_attn_mask_reformat_input_ids_subgraph_Concat_3(model_attn_mask_reformat_input_ids_subgraph_output_0, initializers_onnx_initializer_37);  model_attn_mask_reformat_input_ids_subgraph_output_0 = initializers_onnx_initializer_37 = None
        initializers_onnx_initializer_38 = self.initializers.onnx_initializer_38
        model_attn_mask_reformat_input_ids_subgraph_add_2 = self.model_attn_mask_reformat_input_ids_subgraph_Add_2(model_attn_mask_reformat_input_ids_subgraph_range, initializers_onnx_initializer_38);  initializers_onnx_initializer_38 = None
        initializers_onnx_initializer_39 = self.initializers.onnx_initializer_39
        initializers_onnx_initializer_40 = self.initializers.onnx_initializer_40
        initializers_onnx_initializer_41 = self.initializers.onnx_initializer_41
        model_layers_0_attn_k_proj_repeat_kv_concat_2 = self.model_layers_0_attn_k_proj_repeat_kv_Concat_2(model_layers_0_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_39, initializers_onnx_initializer_40, model_layers_0_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_41);  initializers_onnx_initializer_39 = initializers_onnx_initializer_40 = initializers_onnx_initializer_41 = None
        initializers_onnx_initializer_42 = self.initializers.onnx_initializer_42
        initializers_onnx_initializer_43 = self.initializers.onnx_initializer_43
        model_layers_0_attn_k_proj_repeat_kv_concat_3 = self.model_layers_0_attn_k_proj_repeat_kv_Concat_3(model_layers_0_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_42, model_layers_0_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_43);  model_layers_0_attn_k_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_42 = model_layers_0_attn_k_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_43 = None
        model_attn_mask_reformat_attn_mask_subgraph_cast_2 = self.model_attn_mask_reformat_attn_mask_subgraph_Cast_2(model_attn_mask_reformat_attn_mask_subgraph_sub)
        initializers_onnx_initializer_44 = self.initializers.onnx_initializer_44
        model_layers_0_attn_v_proj_repeat_kv_equal = self.model_layers_0_attn_v_proj_repeat_kv_Equal(model_layers_0_attn_v_proj_repeat_kv_concat_2, initializers_onnx_initializer_44);  initializers_onnx_initializer_44 = None
        model_attn_mask_reformat_input_ids_subgraph_reshape = self.model_attn_mask_reformat_input_ids_subgraph_Reshape(model_attn_mask_reformat_input_ids_subgraph_add_2, model_attn_mask_reformat_input_ids_subgraph_concat_3);  model_attn_mask_reformat_input_ids_subgraph_add_2 = model_attn_mask_reformat_input_ids_subgraph_concat_3 = None
        initializers_onnx_initializer_45 = self.initializers.onnx_initializer_45
        model_attn_mask_reformat_attn_mask_subgraph_where_2 = self.model_attn_mask_reformat_attn_mask_subgraph_Where_2(model_attn_mask_reformat_attn_mask_subgraph_cast_2, initializers_onnx_initializer_45, model_attn_mask_reformat_attn_mask_subgraph_sub);  model_attn_mask_reformat_attn_mask_subgraph_cast_2 = initializers_onnx_initializer_45 = model_attn_mask_reformat_attn_mask_subgraph_sub = None
        initializers_onnx_initializer_46 = self.initializers.onnx_initializer_46
        model_layers_0_attn_k_proj_repeat_kv_equal = self.model_layers_0_attn_k_proj_repeat_kv_Equal(model_layers_0_attn_k_proj_repeat_kv_concat_2, initializers_onnx_initializer_46);  initializers_onnx_initializer_46 = None
        initializers_onnx_initializer_47 = self.initializers.onnx_initializer_47
        model_layers_0_attn_v_proj_repeat_kv_where = self.model_layers_0_attn_v_proj_repeat_kv_Where(model_layers_0_attn_v_proj_repeat_kv_equal, initializers_onnx_initializer_47, model_layers_0_attn_v_proj_repeat_kv_concat_2);  model_layers_0_attn_v_proj_repeat_kv_equal = initializers_onnx_initializer_47 = model_layers_0_attn_v_proj_repeat_kv_concat_2 = None
        model_attn_mask_reformat_input_ids_subgraph_less = self.model_attn_mask_reformat_input_ids_subgraph_Less(model_attn_mask_reformat_input_ids_subgraph_range, model_attn_mask_reformat_input_ids_subgraph_reshape);  model_attn_mask_reformat_input_ids_subgraph_range = model_attn_mask_reformat_input_ids_subgraph_reshape = None
        initializers_onnx_initializer_48 = self.initializers.onnx_initializer_48
        model_layers_0_attn_k_proj_repeat_kv_where = self.model_layers_0_attn_k_proj_repeat_kv_Where(model_layers_0_attn_k_proj_repeat_kv_equal, initializers_onnx_initializer_48, model_layers_0_attn_k_proj_repeat_kv_concat_2);  model_layers_0_attn_k_proj_repeat_kv_equal = initializers_onnx_initializer_48 = model_layers_0_attn_k_proj_repeat_kv_concat_2 = None
        model_layers_0_attn_v_proj_repeat_kv_expand = self.model_layers_0_attn_v_proj_repeat_kv_Expand(model_layers_0_attn_v_proj_repeat_kv_unsqueeze_5, model_layers_0_attn_v_proj_repeat_kv_where);  model_layers_0_attn_v_proj_repeat_kv_unsqueeze_5 = model_layers_0_attn_v_proj_repeat_kv_where = None
        initializers_onnx_initializer_49 = self.initializers.onnx_initializer_49
        model_attn_mask_reformat_input_ids_subgraph_where_2 = self.model_attn_mask_reformat_input_ids_subgraph_Where_2(model_attn_mask_reformat_input_ids_subgraph_less, initializers_onnx_initializer_49, model_attn_mask_reformat_input_ids_subgraph_constant_of_shape_2);  model_attn_mask_reformat_input_ids_subgraph_less = initializers_onnx_initializer_49 = model_attn_mask_reformat_input_ids_subgraph_constant_of_shape_2 = None
        model_layers_0_attn_k_proj_repeat_kv_expand = self.model_layers_0_attn_k_proj_repeat_kv_Expand(model_layers_0_attn_k_proj_repeat_kv_unsqueeze_5, model_layers_0_attn_k_proj_repeat_kv_where);  model_layers_0_attn_k_proj_repeat_kv_unsqueeze_5 = model_layers_0_attn_k_proj_repeat_kv_where = None
        model_layers_0_attn_v_proj_repeat_kv_reshape_3 = self.model_layers_0_attn_v_proj_repeat_kv_Reshape_3(model_layers_0_attn_v_proj_repeat_kv_expand, model_layers_0_attn_v_proj_repeat_kv_concat_3);  model_layers_0_attn_v_proj_repeat_kv_expand = model_layers_0_attn_v_proj_repeat_kv_concat_3 = None
        model_attn_mask_reformat_input_ids_subgraph_unsqueeze_8 = self.model_attn_mask_reformat_input_ids_subgraph_Unsqueeze_8(model_attn_mask_reformat_input_ids_subgraph_where_2);  model_attn_mask_reformat_input_ids_subgraph_where_2 = None
        model_layers_0_attn_k_proj_repeat_kv_reshape_3 = self.model_layers_0_attn_k_proj_repeat_kv_Reshape_3(model_layers_0_attn_k_proj_repeat_kv_expand, model_layers_0_attn_k_proj_repeat_kv_concat_3);  model_layers_0_attn_k_proj_repeat_kv_expand = model_layers_0_attn_k_proj_repeat_kv_concat_3 = None
        model_layers_0_attn_v_proj_repeat_kv_transpose_2 = self.model_layers_0_attn_v_proj_repeat_kv_Transpose_2(model_layers_0_attn_v_proj_repeat_kv_reshape_3);  model_layers_0_attn_v_proj_repeat_kv_reshape_3 = None
        model_attn_mask_reformat_input_ids_subgraph_expand = self.model_attn_mask_reformat_input_ids_subgraph_Expand(model_attn_mask_reformat_input_ids_subgraph_unsqueeze_8, model_attn_mask_reformat_input_ids_subgraph_where_1);  model_attn_mask_reformat_input_ids_subgraph_unsqueeze_8 = model_attn_mask_reformat_input_ids_subgraph_where_1 = None
        model_layers_0_attn_k_proj_repeat_kv_transpose_2 = self.model_layers_0_attn_k_proj_repeat_kv_Transpose_2(model_layers_0_attn_k_proj_repeat_kv_reshape_3);  model_layers_0_attn_k_proj_repeat_kv_reshape_3 = None
        initializers_onnx_initializer_50 = self.initializers.onnx_initializer_50
        model_layers_0_attn_v_proj_repeat_kv_reshape_4 = self.model_layers_0_attn_v_proj_repeat_kv_Reshape_4(model_layers_0_attn_v_proj_repeat_kv_transpose_2, initializers_onnx_initializer_50);  model_layers_0_attn_v_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_50 = None
        model_attn_mask_reformat_add = self.model_attn_mask_reformat_Add(model_attn_mask_reformat_attn_mask_subgraph_where_2, model_attn_mask_reformat_input_ids_subgraph_expand);  model_attn_mask_reformat_attn_mask_subgraph_where_2 = model_attn_mask_reformat_input_ids_subgraph_expand = None
        initializers_onnx_initializer_51 = self.initializers.onnx_initializer_51
        model_layers_0_attn_k_proj_repeat_kv_reshape_4 = self.model_layers_0_attn_k_proj_repeat_kv_Reshape_4(model_layers_0_attn_k_proj_repeat_kv_transpose_2, initializers_onnx_initializer_51);  model_layers_0_attn_k_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_51 = None
        initializers_onnx_initializer_52 = self.initializers.onnx_initializer_52
        model_attn_mask_reformat_tile = self.model_attn_mask_reformat_Tile(model_attn_mask_reformat_add, initializers_onnx_initializer_52);  model_attn_mask_reformat_add = initializers_onnx_initializer_52 = None
        com_microsoft__model_layers_0_attn_multi_head_attention = self.com_microsoft__model_layers_0_attn_MultiHeadAttention(com_microsoft__model_layers_0_attn_q_rotary_rotary_embedding, model_layers_0_attn_k_proj_repeat_kv_reshape_4, model_layers_0_attn_v_proj_repeat_kv_reshape_4, attention_mask = model_attn_mask_reformat_tile);  com_microsoft__model_layers_0_attn_q_rotary_rotary_embedding = model_layers_0_attn_k_proj_repeat_kv_reshape_4 = model_layers_0_attn_v_proj_repeat_kv_reshape_4 = None
        getitem = com_microsoft__model_layers_0_attn_multi_head_attention[0];  com_microsoft__model_layers_0_attn_multi_head_attention = None
        initializers_onnx_initializer_53 = self.initializers.onnx_initializer_53
        model_layers_0_attn_o_proj_mat_mul = self.model_layers_0_attn_o_proj_MatMul(getitem, initializers_onnx_initializer_53);  getitem = initializers_onnx_initializer_53 = None
        initializers_onnx_initializer_54 = self.initializers.onnx_initializer_54
        com_microsoft__model_layers_0_post_attention_layernorm_skip_layer_norm = self.com_microsoft__model_layers_0_post_attention_layernorm_SkipLayerNorm(input_1, model_layers_0_attn_o_proj_mat_mul, initializers_onnx_initializer_54);  input_1 = model_layers_0_attn_o_proj_mat_mul = initializers_onnx_initializer_54 = None
        getitem_1 = com_microsoft__model_layers_0_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_55 = self.initializers.onnx_initializer_55
        model_layers_0_mlp_gate_proj_mat_mul = self.model_layers_0_mlp_gate_proj_MatMul(getitem_1, initializers_onnx_initializer_55);  getitem_1 = initializers_onnx_initializer_55 = None
        getitem_2 = com_microsoft__model_layers_0_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_56 = self.initializers.onnx_initializer_56
        model_layers_0_mlp_up_proj_mat_mul = self.model_layers_0_mlp_up_proj_MatMul(getitem_2, initializers_onnx_initializer_56);  getitem_2 = initializers_onnx_initializer_56 = None
        model_layers_0_mlp_act_fn_sigmoid = self.model_layers_0_mlp_act_fn_Sigmoid(model_layers_0_mlp_gate_proj_mat_mul)
        model_layers_0_mlp_act_fn_mul = self.model_layers_0_mlp_act_fn_Mul(model_layers_0_mlp_gate_proj_mat_mul, model_layers_0_mlp_act_fn_sigmoid);  model_layers_0_mlp_gate_proj_mat_mul = model_layers_0_mlp_act_fn_sigmoid = None
        model_layers_0_mlp_mul = self.model_layers_0_mlp_Mul(model_layers_0_mlp_act_fn_mul, model_layers_0_mlp_up_proj_mat_mul);  model_layers_0_mlp_act_fn_mul = model_layers_0_mlp_up_proj_mat_mul = None
        initializers_onnx_initializer_57 = self.initializers.onnx_initializer_57
        model_layers_0_mlp_down_proj_mat_mul = self.model_layers_0_mlp_down_proj_MatMul(model_layers_0_mlp_mul, initializers_onnx_initializer_57);  model_layers_0_mlp_mul = initializers_onnx_initializer_57 = None
        getitem_3 = com_microsoft__model_layers_0_post_attention_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_0_post_attention_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_58 = self.initializers.onnx_initializer_58
        com_microsoft__model_layers_1_input_layernorm_skip_layer_norm = self.com_microsoft__model_layers_1_input_layernorm_SkipLayerNorm(getitem_3, model_layers_0_mlp_down_proj_mat_mul, initializers_onnx_initializer_58);  getitem_3 = model_layers_0_mlp_down_proj_mat_mul = initializers_onnx_initializer_58 = None
        getitem_4 = com_microsoft__model_layers_1_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_59 = self.initializers.onnx_initializer_59
        model_layers_1_attn_q_proj_mat_mul = self.model_layers_1_attn_q_proj_MatMul(getitem_4, initializers_onnx_initializer_59);  getitem_4 = initializers_onnx_initializer_59 = None
        getitem_5 = com_microsoft__model_layers_1_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_60 = self.initializers.onnx_initializer_60
        model_layers_1_attn_k_proj_mat_mul = self.model_layers_1_attn_k_proj_MatMul(getitem_5, initializers_onnx_initializer_60);  getitem_5 = initializers_onnx_initializer_60 = None
        getitem_6 = com_microsoft__model_layers_1_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_61 = self.initializers.onnx_initializer_61
        model_layers_1_attn_v_proj_mat_mul = self.model_layers_1_attn_v_proj_MatMul(getitem_6, initializers_onnx_initializer_61);  getitem_6 = initializers_onnx_initializer_61 = None
        initializers_onnx_initializer_62 = self.initializers.onnx_initializer_62
        initializers_onnx_initializer_63 = self.initializers.onnx_initializer_63
        com_microsoft__model_layers_1_attn_q_rotary_rotary_embedding = self.com_microsoft__model_layers_1_attn_q_rotary_RotaryEmbedding(model_layers_1_attn_q_proj_mat_mul, input_3, initializers_onnx_initializer_62, initializers_onnx_initializer_63);  model_layers_1_attn_q_proj_mat_mul = initializers_onnx_initializer_62 = initializers_onnx_initializer_63 = None
        initializers_onnx_initializer_64 = self.initializers.onnx_initializer_64
        initializers_onnx_initializer_65 = self.initializers.onnx_initializer_65
        com_microsoft__model_layers_1_attn_k_rotary_rotary_embedding = self.com_microsoft__model_layers_1_attn_k_rotary_RotaryEmbedding(model_layers_1_attn_k_proj_mat_mul, input_3, initializers_onnx_initializer_64, initializers_onnx_initializer_65);  model_layers_1_attn_k_proj_mat_mul = initializers_onnx_initializer_64 = initializers_onnx_initializer_65 = None
        initializers_onnx_initializer_66 = self.initializers.onnx_initializer_66
        model_layers_1_attn_v_proj_repeat_kv_reshape_1 = self.model_layers_1_attn_v_proj_repeat_kv_Reshape_1(model_layers_1_attn_v_proj_mat_mul, initializers_onnx_initializer_66);  model_layers_1_attn_v_proj_mat_mul = initializers_onnx_initializer_66 = None
        initializers_onnx_initializer_67 = self.initializers.onnx_initializer_67
        model_layers_1_attn_k_proj_repeat_kv_reshape_1 = self.model_layers_1_attn_k_proj_repeat_kv_Reshape_1(com_microsoft__model_layers_1_attn_k_rotary_rotary_embedding, initializers_onnx_initializer_67);  com_microsoft__model_layers_1_attn_k_rotary_rotary_embedding = initializers_onnx_initializer_67 = None
        model_layers_1_attn_v_proj_repeat_kv_transpose_1 = self.model_layers_1_attn_v_proj_repeat_kv_Transpose_1(model_layers_1_attn_v_proj_repeat_kv_reshape_1);  model_layers_1_attn_v_proj_repeat_kv_reshape_1 = None
        model_layers_1_attn_k_proj_repeat_kv_transpose_1 = self.model_layers_1_attn_k_proj_repeat_kv_Transpose_1(model_layers_1_attn_k_proj_repeat_kv_reshape_1);  model_layers_1_attn_k_proj_repeat_kv_reshape_1 = None
        model_layers_1_attn_v_proj_repeat_kv_concat_1 = self.model_layers_1_attn_v_proj_repeat_kv_Concat_1(input_7, model_layers_1_attn_v_proj_repeat_kv_transpose_1);  input_7 = model_layers_1_attn_v_proj_repeat_kv_transpose_1 = None
        model_layers_1_attn_k_proj_repeat_kv_concat_1 = self.model_layers_1_attn_k_proj_repeat_kv_Concat_1(input_6, model_layers_1_attn_k_proj_repeat_kv_transpose_1);  input_6 = model_layers_1_attn_k_proj_repeat_kv_transpose_1 = None
        model_layers_1_attn_v_proj_repeat_kv_shape_1 = self.model_layers_1_attn_v_proj_repeat_kv_Shape_1(model_layers_1_attn_v_proj_repeat_kv_concat_1)
        model_layers_1_attn_v_proj_repeat_kv_unsqueeze_5 = self.model_layers_1_attn_v_proj_repeat_kv_Unsqueeze_5(model_layers_1_attn_v_proj_repeat_kv_concat_1)
        model_layers_1_attn_k_proj_repeat_kv_shape_1 = self.model_layers_1_attn_k_proj_repeat_kv_Shape_1(model_layers_1_attn_k_proj_repeat_kv_concat_1)
        model_layers_1_attn_k_proj_repeat_kv_unsqueeze_5 = self.model_layers_1_attn_k_proj_repeat_kv_Unsqueeze_5(model_layers_1_attn_k_proj_repeat_kv_concat_1)
        initializers_onnx_initializer_68 = self.initializers.onnx_initializer_68
        model_layers_1_attn_v_proj_repeat_kv_gather_1 = self.model_layers_1_attn_v_proj_repeat_kv_Gather_1(model_layers_1_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_68);  initializers_onnx_initializer_68 = None
        initializers_onnx_initializer_69 = self.initializers.onnx_initializer_69
        model_layers_1_attn_v_proj_repeat_kv_gather_3 = self.model_layers_1_attn_v_proj_repeat_kv_Gather_3(model_layers_1_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_69);  model_layers_1_attn_v_proj_repeat_kv_shape_1 = initializers_onnx_initializer_69 = None
        initializers_onnx_initializer_70 = self.initializers.onnx_initializer_70
        model_layers_1_attn_k_proj_repeat_kv_gather_1 = self.model_layers_1_attn_k_proj_repeat_kv_Gather_1(model_layers_1_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_70);  initializers_onnx_initializer_70 = None
        initializers_onnx_initializer_71 = self.initializers.onnx_initializer_71
        model_layers_1_attn_k_proj_repeat_kv_gather_3 = self.model_layers_1_attn_k_proj_repeat_kv_Gather_3(model_layers_1_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_71);  model_layers_1_attn_k_proj_repeat_kv_shape_1 = initializers_onnx_initializer_71 = None
        model_layers_1_attn_v_proj_repeat_kv_unsqueeze_1 = self.model_layers_1_attn_v_proj_repeat_kv_Unsqueeze_1(model_layers_1_attn_v_proj_repeat_kv_gather_1);  model_layers_1_attn_v_proj_repeat_kv_gather_1 = None
        model_layers_1_attn_v_proj_repeat_kv_unsqueeze_3 = self.model_layers_1_attn_v_proj_repeat_kv_Unsqueeze_3(model_layers_1_attn_v_proj_repeat_kv_gather_3);  model_layers_1_attn_v_proj_repeat_kv_gather_3 = None
        model_layers_1_attn_k_proj_repeat_kv_unsqueeze_1 = self.model_layers_1_attn_k_proj_repeat_kv_Unsqueeze_1(model_layers_1_attn_k_proj_repeat_kv_gather_1);  model_layers_1_attn_k_proj_repeat_kv_gather_1 = None
        model_layers_1_attn_k_proj_repeat_kv_unsqueeze_3 = self.model_layers_1_attn_k_proj_repeat_kv_Unsqueeze_3(model_layers_1_attn_k_proj_repeat_kv_gather_3);  model_layers_1_attn_k_proj_repeat_kv_gather_3 = None
        initializers_onnx_initializer_72 = self.initializers.onnx_initializer_72
        initializers_onnx_initializer_73 = self.initializers.onnx_initializer_73
        initializers_onnx_initializer_74 = self.initializers.onnx_initializer_74
        model_layers_1_attn_v_proj_repeat_kv_concat_2 = self.model_layers_1_attn_v_proj_repeat_kv_Concat_2(model_layers_1_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_72, initializers_onnx_initializer_73, model_layers_1_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_74);  initializers_onnx_initializer_72 = initializers_onnx_initializer_73 = initializers_onnx_initializer_74 = None
        initializers_onnx_initializer_75 = self.initializers.onnx_initializer_75
        initializers_onnx_initializer_76 = self.initializers.onnx_initializer_76
        model_layers_1_attn_v_proj_repeat_kv_concat_3 = self.model_layers_1_attn_v_proj_repeat_kv_Concat_3(model_layers_1_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_75, model_layers_1_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_76);  model_layers_1_attn_v_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_75 = model_layers_1_attn_v_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_76 = None
        initializers_onnx_initializer_77 = self.initializers.onnx_initializer_77
        initializers_onnx_initializer_78 = self.initializers.onnx_initializer_78
        initializers_onnx_initializer_79 = self.initializers.onnx_initializer_79
        model_layers_1_attn_k_proj_repeat_kv_concat_2 = self.model_layers_1_attn_k_proj_repeat_kv_Concat_2(model_layers_1_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_77, initializers_onnx_initializer_78, model_layers_1_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_79);  initializers_onnx_initializer_77 = initializers_onnx_initializer_78 = initializers_onnx_initializer_79 = None
        initializers_onnx_initializer_80 = self.initializers.onnx_initializer_80
        initializers_onnx_initializer_81 = self.initializers.onnx_initializer_81
        model_layers_1_attn_k_proj_repeat_kv_concat_3 = self.model_layers_1_attn_k_proj_repeat_kv_Concat_3(model_layers_1_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_80, model_layers_1_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_81);  model_layers_1_attn_k_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_80 = model_layers_1_attn_k_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_81 = None
        initializers_onnx_initializer_82 = self.initializers.onnx_initializer_82
        model_layers_1_attn_v_proj_repeat_kv_equal = self.model_layers_1_attn_v_proj_repeat_kv_Equal(model_layers_1_attn_v_proj_repeat_kv_concat_2, initializers_onnx_initializer_82);  initializers_onnx_initializer_82 = None
        initializers_onnx_initializer_83 = self.initializers.onnx_initializer_83
        model_layers_1_attn_k_proj_repeat_kv_equal = self.model_layers_1_attn_k_proj_repeat_kv_Equal(model_layers_1_attn_k_proj_repeat_kv_concat_2, initializers_onnx_initializer_83);  initializers_onnx_initializer_83 = None
        initializers_onnx_initializer_84 = self.initializers.onnx_initializer_84
        model_layers_1_attn_v_proj_repeat_kv_where = self.model_layers_1_attn_v_proj_repeat_kv_Where(model_layers_1_attn_v_proj_repeat_kv_equal, initializers_onnx_initializer_84, model_layers_1_attn_v_proj_repeat_kv_concat_2);  model_layers_1_attn_v_proj_repeat_kv_equal = initializers_onnx_initializer_84 = model_layers_1_attn_v_proj_repeat_kv_concat_2 = None
        initializers_onnx_initializer_85 = self.initializers.onnx_initializer_85
        model_layers_1_attn_k_proj_repeat_kv_where = self.model_layers_1_attn_k_proj_repeat_kv_Where(model_layers_1_attn_k_proj_repeat_kv_equal, initializers_onnx_initializer_85, model_layers_1_attn_k_proj_repeat_kv_concat_2);  model_layers_1_attn_k_proj_repeat_kv_equal = initializers_onnx_initializer_85 = model_layers_1_attn_k_proj_repeat_kv_concat_2 = None
        model_layers_1_attn_v_proj_repeat_kv_expand = self.model_layers_1_attn_v_proj_repeat_kv_Expand(model_layers_1_attn_v_proj_repeat_kv_unsqueeze_5, model_layers_1_attn_v_proj_repeat_kv_where);  model_layers_1_attn_v_proj_repeat_kv_unsqueeze_5 = model_layers_1_attn_v_proj_repeat_kv_where = None
        model_layers_1_attn_k_proj_repeat_kv_expand = self.model_layers_1_attn_k_proj_repeat_kv_Expand(model_layers_1_attn_k_proj_repeat_kv_unsqueeze_5, model_layers_1_attn_k_proj_repeat_kv_where);  model_layers_1_attn_k_proj_repeat_kv_unsqueeze_5 = model_layers_1_attn_k_proj_repeat_kv_where = None
        model_layers_1_attn_v_proj_repeat_kv_reshape_3 = self.model_layers_1_attn_v_proj_repeat_kv_Reshape_3(model_layers_1_attn_v_proj_repeat_kv_expand, model_layers_1_attn_v_proj_repeat_kv_concat_3);  model_layers_1_attn_v_proj_repeat_kv_expand = model_layers_1_attn_v_proj_repeat_kv_concat_3 = None
        model_layers_1_attn_k_proj_repeat_kv_reshape_3 = self.model_layers_1_attn_k_proj_repeat_kv_Reshape_3(model_layers_1_attn_k_proj_repeat_kv_expand, model_layers_1_attn_k_proj_repeat_kv_concat_3);  model_layers_1_attn_k_proj_repeat_kv_expand = model_layers_1_attn_k_proj_repeat_kv_concat_3 = None
        model_layers_1_attn_v_proj_repeat_kv_transpose_2 = self.model_layers_1_attn_v_proj_repeat_kv_Transpose_2(model_layers_1_attn_v_proj_repeat_kv_reshape_3);  model_layers_1_attn_v_proj_repeat_kv_reshape_3 = None
        model_layers_1_attn_k_proj_repeat_kv_transpose_2 = self.model_layers_1_attn_k_proj_repeat_kv_Transpose_2(model_layers_1_attn_k_proj_repeat_kv_reshape_3);  model_layers_1_attn_k_proj_repeat_kv_reshape_3 = None
        initializers_onnx_initializer_86 = self.initializers.onnx_initializer_86
        model_layers_1_attn_v_proj_repeat_kv_reshape_4 = self.model_layers_1_attn_v_proj_repeat_kv_Reshape_4(model_layers_1_attn_v_proj_repeat_kv_transpose_2, initializers_onnx_initializer_86);  model_layers_1_attn_v_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_86 = None
        initializers_onnx_initializer_87 = self.initializers.onnx_initializer_87
        model_layers_1_attn_k_proj_repeat_kv_reshape_4 = self.model_layers_1_attn_k_proj_repeat_kv_Reshape_4(model_layers_1_attn_k_proj_repeat_kv_transpose_2, initializers_onnx_initializer_87);  model_layers_1_attn_k_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_87 = None
        com_microsoft__model_layers_1_attn_multi_head_attention = self.com_microsoft__model_layers_1_attn_MultiHeadAttention(com_microsoft__model_layers_1_attn_q_rotary_rotary_embedding, model_layers_1_attn_k_proj_repeat_kv_reshape_4, model_layers_1_attn_v_proj_repeat_kv_reshape_4, attention_mask = model_attn_mask_reformat_tile);  com_microsoft__model_layers_1_attn_q_rotary_rotary_embedding = model_layers_1_attn_k_proj_repeat_kv_reshape_4 = model_layers_1_attn_v_proj_repeat_kv_reshape_4 = None
        getitem_7 = com_microsoft__model_layers_1_attn_multi_head_attention[0];  com_microsoft__model_layers_1_attn_multi_head_attention = None
        initializers_onnx_initializer_88 = self.initializers.onnx_initializer_88
        model_layers_1_attn_o_proj_mat_mul = self.model_layers_1_attn_o_proj_MatMul(getitem_7, initializers_onnx_initializer_88);  getitem_7 = initializers_onnx_initializer_88 = None
        getitem_8 = com_microsoft__model_layers_1_input_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_1_input_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_89 = self.initializers.onnx_initializer_89
        com_microsoft__model_layers_1_post_attention_layernorm_skip_layer_norm = self.com_microsoft__model_layers_1_post_attention_layernorm_SkipLayerNorm(getitem_8, model_layers_1_attn_o_proj_mat_mul, initializers_onnx_initializer_89);  getitem_8 = model_layers_1_attn_o_proj_mat_mul = initializers_onnx_initializer_89 = None
        getitem_9 = com_microsoft__model_layers_1_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_90 = self.initializers.onnx_initializer_90
        model_layers_1_mlp_gate_proj_mat_mul = self.model_layers_1_mlp_gate_proj_MatMul(getitem_9, initializers_onnx_initializer_90);  getitem_9 = initializers_onnx_initializer_90 = None
        getitem_10 = com_microsoft__model_layers_1_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_91 = self.initializers.onnx_initializer_91
        model_layers_1_mlp_up_proj_mat_mul = self.model_layers_1_mlp_up_proj_MatMul(getitem_10, initializers_onnx_initializer_91);  getitem_10 = initializers_onnx_initializer_91 = None
        model_layers_1_mlp_act_fn_sigmoid = self.model_layers_1_mlp_act_fn_Sigmoid(model_layers_1_mlp_gate_proj_mat_mul)
        model_layers_1_mlp_act_fn_mul = self.model_layers_1_mlp_act_fn_Mul(model_layers_1_mlp_gate_proj_mat_mul, model_layers_1_mlp_act_fn_sigmoid);  model_layers_1_mlp_gate_proj_mat_mul = model_layers_1_mlp_act_fn_sigmoid = None
        model_layers_1_mlp_mul = self.model_layers_1_mlp_Mul(model_layers_1_mlp_act_fn_mul, model_layers_1_mlp_up_proj_mat_mul);  model_layers_1_mlp_act_fn_mul = model_layers_1_mlp_up_proj_mat_mul = None
        initializers_onnx_initializer_92 = self.initializers.onnx_initializer_92
        model_layers_1_mlp_down_proj_mat_mul = self.model_layers_1_mlp_down_proj_MatMul(model_layers_1_mlp_mul, initializers_onnx_initializer_92);  model_layers_1_mlp_mul = initializers_onnx_initializer_92 = None
        getitem_11 = com_microsoft__model_layers_1_post_attention_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_1_post_attention_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_93 = self.initializers.onnx_initializer_93
        com_microsoft__model_layers_2_input_layernorm_skip_layer_norm = self.com_microsoft__model_layers_2_input_layernorm_SkipLayerNorm(getitem_11, model_layers_1_mlp_down_proj_mat_mul, initializers_onnx_initializer_93);  getitem_11 = model_layers_1_mlp_down_proj_mat_mul = initializers_onnx_initializer_93 = None
        getitem_12 = com_microsoft__model_layers_2_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_94 = self.initializers.onnx_initializer_94
        model_layers_2_attn_q_proj_mat_mul = self.model_layers_2_attn_q_proj_MatMul(getitem_12, initializers_onnx_initializer_94);  getitem_12 = initializers_onnx_initializer_94 = None
        getitem_13 = com_microsoft__model_layers_2_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_95 = self.initializers.onnx_initializer_95
        model_layers_2_attn_k_proj_mat_mul = self.model_layers_2_attn_k_proj_MatMul(getitem_13, initializers_onnx_initializer_95);  getitem_13 = initializers_onnx_initializer_95 = None
        getitem_14 = com_microsoft__model_layers_2_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_96 = self.initializers.onnx_initializer_96
        model_layers_2_attn_v_proj_mat_mul = self.model_layers_2_attn_v_proj_MatMul(getitem_14, initializers_onnx_initializer_96);  getitem_14 = initializers_onnx_initializer_96 = None
        initializers_onnx_initializer_97 = self.initializers.onnx_initializer_97
        initializers_onnx_initializer_98 = self.initializers.onnx_initializer_98
        com_microsoft__model_layers_2_attn_q_rotary_rotary_embedding = self.com_microsoft__model_layers_2_attn_q_rotary_RotaryEmbedding(model_layers_2_attn_q_proj_mat_mul, input_3, initializers_onnx_initializer_97, initializers_onnx_initializer_98);  model_layers_2_attn_q_proj_mat_mul = initializers_onnx_initializer_97 = initializers_onnx_initializer_98 = None
        initializers_onnx_initializer_99 = self.initializers.onnx_initializer_99
        initializers_onnx_initializer_100 = self.initializers.onnx_initializer_100
        com_microsoft__model_layers_2_attn_k_rotary_rotary_embedding = self.com_microsoft__model_layers_2_attn_k_rotary_RotaryEmbedding(model_layers_2_attn_k_proj_mat_mul, input_3, initializers_onnx_initializer_99, initializers_onnx_initializer_100);  model_layers_2_attn_k_proj_mat_mul = initializers_onnx_initializer_99 = initializers_onnx_initializer_100 = None
        initializers_onnx_initializer_101 = self.initializers.onnx_initializer_101
        model_layers_2_attn_v_proj_repeat_kv_reshape_1 = self.model_layers_2_attn_v_proj_repeat_kv_Reshape_1(model_layers_2_attn_v_proj_mat_mul, initializers_onnx_initializer_101);  model_layers_2_attn_v_proj_mat_mul = initializers_onnx_initializer_101 = None
        initializers_onnx_initializer_102 = self.initializers.onnx_initializer_102
        model_layers_2_attn_k_proj_repeat_kv_reshape_1 = self.model_layers_2_attn_k_proj_repeat_kv_Reshape_1(com_microsoft__model_layers_2_attn_k_rotary_rotary_embedding, initializers_onnx_initializer_102);  com_microsoft__model_layers_2_attn_k_rotary_rotary_embedding = initializers_onnx_initializer_102 = None
        model_layers_2_attn_v_proj_repeat_kv_transpose_1 = self.model_layers_2_attn_v_proj_repeat_kv_Transpose_1(model_layers_2_attn_v_proj_repeat_kv_reshape_1);  model_layers_2_attn_v_proj_repeat_kv_reshape_1 = None
        model_layers_2_attn_k_proj_repeat_kv_transpose_1 = self.model_layers_2_attn_k_proj_repeat_kv_Transpose_1(model_layers_2_attn_k_proj_repeat_kv_reshape_1);  model_layers_2_attn_k_proj_repeat_kv_reshape_1 = None
        model_layers_2_attn_v_proj_repeat_kv_concat_1 = self.model_layers_2_attn_v_proj_repeat_kv_Concat_1(input_9, model_layers_2_attn_v_proj_repeat_kv_transpose_1);  input_9 = model_layers_2_attn_v_proj_repeat_kv_transpose_1 = None
        model_layers_2_attn_k_proj_repeat_kv_concat_1 = self.model_layers_2_attn_k_proj_repeat_kv_Concat_1(input_8, model_layers_2_attn_k_proj_repeat_kv_transpose_1);  input_8 = model_layers_2_attn_k_proj_repeat_kv_transpose_1 = None
        model_layers_2_attn_v_proj_repeat_kv_shape_1 = self.model_layers_2_attn_v_proj_repeat_kv_Shape_1(model_layers_2_attn_v_proj_repeat_kv_concat_1)
        model_layers_2_attn_v_proj_repeat_kv_unsqueeze_5 = self.model_layers_2_attn_v_proj_repeat_kv_Unsqueeze_5(model_layers_2_attn_v_proj_repeat_kv_concat_1)
        model_layers_2_attn_k_proj_repeat_kv_shape_1 = self.model_layers_2_attn_k_proj_repeat_kv_Shape_1(model_layers_2_attn_k_proj_repeat_kv_concat_1)
        model_layers_2_attn_k_proj_repeat_kv_unsqueeze_5 = self.model_layers_2_attn_k_proj_repeat_kv_Unsqueeze_5(model_layers_2_attn_k_proj_repeat_kv_concat_1)
        initializers_onnx_initializer_103 = self.initializers.onnx_initializer_103
        model_layers_2_attn_v_proj_repeat_kv_gather_1 = self.model_layers_2_attn_v_proj_repeat_kv_Gather_1(model_layers_2_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_103);  initializers_onnx_initializer_103 = None
        initializers_onnx_initializer_104 = self.initializers.onnx_initializer_104
        model_layers_2_attn_v_proj_repeat_kv_gather_3 = self.model_layers_2_attn_v_proj_repeat_kv_Gather_3(model_layers_2_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_104);  model_layers_2_attn_v_proj_repeat_kv_shape_1 = initializers_onnx_initializer_104 = None
        initializers_onnx_initializer_105 = self.initializers.onnx_initializer_105
        model_layers_2_attn_k_proj_repeat_kv_gather_1 = self.model_layers_2_attn_k_proj_repeat_kv_Gather_1(model_layers_2_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_105);  initializers_onnx_initializer_105 = None
        initializers_onnx_initializer_106 = self.initializers.onnx_initializer_106
        model_layers_2_attn_k_proj_repeat_kv_gather_3 = self.model_layers_2_attn_k_proj_repeat_kv_Gather_3(model_layers_2_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_106);  model_layers_2_attn_k_proj_repeat_kv_shape_1 = initializers_onnx_initializer_106 = None
        model_layers_2_attn_v_proj_repeat_kv_unsqueeze_1 = self.model_layers_2_attn_v_proj_repeat_kv_Unsqueeze_1(model_layers_2_attn_v_proj_repeat_kv_gather_1);  model_layers_2_attn_v_proj_repeat_kv_gather_1 = None
        model_layers_2_attn_v_proj_repeat_kv_unsqueeze_3 = self.model_layers_2_attn_v_proj_repeat_kv_Unsqueeze_3(model_layers_2_attn_v_proj_repeat_kv_gather_3);  model_layers_2_attn_v_proj_repeat_kv_gather_3 = None
        model_layers_2_attn_k_proj_repeat_kv_unsqueeze_1 = self.model_layers_2_attn_k_proj_repeat_kv_Unsqueeze_1(model_layers_2_attn_k_proj_repeat_kv_gather_1);  model_layers_2_attn_k_proj_repeat_kv_gather_1 = None
        model_layers_2_attn_k_proj_repeat_kv_unsqueeze_3 = self.model_layers_2_attn_k_proj_repeat_kv_Unsqueeze_3(model_layers_2_attn_k_proj_repeat_kv_gather_3);  model_layers_2_attn_k_proj_repeat_kv_gather_3 = None
        initializers_onnx_initializer_107 = self.initializers.onnx_initializer_107
        initializers_onnx_initializer_108 = self.initializers.onnx_initializer_108
        initializers_onnx_initializer_109 = self.initializers.onnx_initializer_109
        model_layers_2_attn_v_proj_repeat_kv_concat_2 = self.model_layers_2_attn_v_proj_repeat_kv_Concat_2(model_layers_2_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_107, initializers_onnx_initializer_108, model_layers_2_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_109);  initializers_onnx_initializer_107 = initializers_onnx_initializer_108 = initializers_onnx_initializer_109 = None
        initializers_onnx_initializer_110 = self.initializers.onnx_initializer_110
        initializers_onnx_initializer_111 = self.initializers.onnx_initializer_111
        model_layers_2_attn_v_proj_repeat_kv_concat_3 = self.model_layers_2_attn_v_proj_repeat_kv_Concat_3(model_layers_2_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_110, model_layers_2_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_111);  model_layers_2_attn_v_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_110 = model_layers_2_attn_v_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_111 = None
        initializers_onnx_initializer_112 = self.initializers.onnx_initializer_112
        initializers_onnx_initializer_113 = self.initializers.onnx_initializer_113
        initializers_onnx_initializer_114 = self.initializers.onnx_initializer_114
        model_layers_2_attn_k_proj_repeat_kv_concat_2 = self.model_layers_2_attn_k_proj_repeat_kv_Concat_2(model_layers_2_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_112, initializers_onnx_initializer_113, model_layers_2_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_114);  initializers_onnx_initializer_112 = initializers_onnx_initializer_113 = initializers_onnx_initializer_114 = None
        initializers_onnx_initializer_115 = self.initializers.onnx_initializer_115
        initializers_onnx_initializer_116 = self.initializers.onnx_initializer_116
        model_layers_2_attn_k_proj_repeat_kv_concat_3 = self.model_layers_2_attn_k_proj_repeat_kv_Concat_3(model_layers_2_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_115, model_layers_2_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_116);  model_layers_2_attn_k_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_115 = model_layers_2_attn_k_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_116 = None
        initializers_onnx_initializer_117 = self.initializers.onnx_initializer_117
        model_layers_2_attn_v_proj_repeat_kv_equal = self.model_layers_2_attn_v_proj_repeat_kv_Equal(model_layers_2_attn_v_proj_repeat_kv_concat_2, initializers_onnx_initializer_117);  initializers_onnx_initializer_117 = None
        initializers_onnx_initializer_118 = self.initializers.onnx_initializer_118
        model_layers_2_attn_k_proj_repeat_kv_equal = self.model_layers_2_attn_k_proj_repeat_kv_Equal(model_layers_2_attn_k_proj_repeat_kv_concat_2, initializers_onnx_initializer_118);  initializers_onnx_initializer_118 = None
        initializers_onnx_initializer_119 = self.initializers.onnx_initializer_119
        model_layers_2_attn_v_proj_repeat_kv_where = self.model_layers_2_attn_v_proj_repeat_kv_Where(model_layers_2_attn_v_proj_repeat_kv_equal, initializers_onnx_initializer_119, model_layers_2_attn_v_proj_repeat_kv_concat_2);  model_layers_2_attn_v_proj_repeat_kv_equal = initializers_onnx_initializer_119 = model_layers_2_attn_v_proj_repeat_kv_concat_2 = None
        initializers_onnx_initializer_120 = self.initializers.onnx_initializer_120
        model_layers_2_attn_k_proj_repeat_kv_where = self.model_layers_2_attn_k_proj_repeat_kv_Where(model_layers_2_attn_k_proj_repeat_kv_equal, initializers_onnx_initializer_120, model_layers_2_attn_k_proj_repeat_kv_concat_2);  model_layers_2_attn_k_proj_repeat_kv_equal = initializers_onnx_initializer_120 = model_layers_2_attn_k_proj_repeat_kv_concat_2 = None
        model_layers_2_attn_v_proj_repeat_kv_expand = self.model_layers_2_attn_v_proj_repeat_kv_Expand(model_layers_2_attn_v_proj_repeat_kv_unsqueeze_5, model_layers_2_attn_v_proj_repeat_kv_where);  model_layers_2_attn_v_proj_repeat_kv_unsqueeze_5 = model_layers_2_attn_v_proj_repeat_kv_where = None
        model_layers_2_attn_k_proj_repeat_kv_expand = self.model_layers_2_attn_k_proj_repeat_kv_Expand(model_layers_2_attn_k_proj_repeat_kv_unsqueeze_5, model_layers_2_attn_k_proj_repeat_kv_where);  model_layers_2_attn_k_proj_repeat_kv_unsqueeze_5 = model_layers_2_attn_k_proj_repeat_kv_where = None
        model_layers_2_attn_v_proj_repeat_kv_reshape_3 = self.model_layers_2_attn_v_proj_repeat_kv_Reshape_3(model_layers_2_attn_v_proj_repeat_kv_expand, model_layers_2_attn_v_proj_repeat_kv_concat_3);  model_layers_2_attn_v_proj_repeat_kv_expand = model_layers_2_attn_v_proj_repeat_kv_concat_3 = None
        model_layers_2_attn_k_proj_repeat_kv_reshape_3 = self.model_layers_2_attn_k_proj_repeat_kv_Reshape_3(model_layers_2_attn_k_proj_repeat_kv_expand, model_layers_2_attn_k_proj_repeat_kv_concat_3);  model_layers_2_attn_k_proj_repeat_kv_expand = model_layers_2_attn_k_proj_repeat_kv_concat_3 = None
        model_layers_2_attn_v_proj_repeat_kv_transpose_2 = self.model_layers_2_attn_v_proj_repeat_kv_Transpose_2(model_layers_2_attn_v_proj_repeat_kv_reshape_3);  model_layers_2_attn_v_proj_repeat_kv_reshape_3 = None
        model_layers_2_attn_k_proj_repeat_kv_transpose_2 = self.model_layers_2_attn_k_proj_repeat_kv_Transpose_2(model_layers_2_attn_k_proj_repeat_kv_reshape_3);  model_layers_2_attn_k_proj_repeat_kv_reshape_3 = None
        initializers_onnx_initializer_121 = self.initializers.onnx_initializer_121
        model_layers_2_attn_v_proj_repeat_kv_reshape_4 = self.model_layers_2_attn_v_proj_repeat_kv_Reshape_4(model_layers_2_attn_v_proj_repeat_kv_transpose_2, initializers_onnx_initializer_121);  model_layers_2_attn_v_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_121 = None
        initializers_onnx_initializer_122 = self.initializers.onnx_initializer_122
        model_layers_2_attn_k_proj_repeat_kv_reshape_4 = self.model_layers_2_attn_k_proj_repeat_kv_Reshape_4(model_layers_2_attn_k_proj_repeat_kv_transpose_2, initializers_onnx_initializer_122);  model_layers_2_attn_k_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_122 = None
        com_microsoft__model_layers_2_attn_multi_head_attention = self.com_microsoft__model_layers_2_attn_MultiHeadAttention(com_microsoft__model_layers_2_attn_q_rotary_rotary_embedding, model_layers_2_attn_k_proj_repeat_kv_reshape_4, model_layers_2_attn_v_proj_repeat_kv_reshape_4, attention_mask = model_attn_mask_reformat_tile);  com_microsoft__model_layers_2_attn_q_rotary_rotary_embedding = model_layers_2_attn_k_proj_repeat_kv_reshape_4 = model_layers_2_attn_v_proj_repeat_kv_reshape_4 = None
        getitem_15 = com_microsoft__model_layers_2_attn_multi_head_attention[0];  com_microsoft__model_layers_2_attn_multi_head_attention = None
        initializers_onnx_initializer_123 = self.initializers.onnx_initializer_123
        model_layers_2_attn_o_proj_mat_mul = self.model_layers_2_attn_o_proj_MatMul(getitem_15, initializers_onnx_initializer_123);  getitem_15 = initializers_onnx_initializer_123 = None
        getitem_16 = com_microsoft__model_layers_2_input_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_2_input_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_124 = self.initializers.onnx_initializer_124
        com_microsoft__model_layers_2_post_attention_layernorm_skip_layer_norm = self.com_microsoft__model_layers_2_post_attention_layernorm_SkipLayerNorm(getitem_16, model_layers_2_attn_o_proj_mat_mul, initializers_onnx_initializer_124);  getitem_16 = model_layers_2_attn_o_proj_mat_mul = initializers_onnx_initializer_124 = None
        getitem_17 = com_microsoft__model_layers_2_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_125 = self.initializers.onnx_initializer_125
        model_layers_2_mlp_gate_proj_mat_mul = self.model_layers_2_mlp_gate_proj_MatMul(getitem_17, initializers_onnx_initializer_125);  getitem_17 = initializers_onnx_initializer_125 = None
        getitem_18 = com_microsoft__model_layers_2_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_126 = self.initializers.onnx_initializer_126
        model_layers_2_mlp_up_proj_mat_mul = self.model_layers_2_mlp_up_proj_MatMul(getitem_18, initializers_onnx_initializer_126);  getitem_18 = initializers_onnx_initializer_126 = None
        model_layers_2_mlp_act_fn_sigmoid = self.model_layers_2_mlp_act_fn_Sigmoid(model_layers_2_mlp_gate_proj_mat_mul)
        model_layers_2_mlp_act_fn_mul = self.model_layers_2_mlp_act_fn_Mul(model_layers_2_mlp_gate_proj_mat_mul, model_layers_2_mlp_act_fn_sigmoid);  model_layers_2_mlp_gate_proj_mat_mul = model_layers_2_mlp_act_fn_sigmoid = None
        model_layers_2_mlp_mul = self.model_layers_2_mlp_Mul(model_layers_2_mlp_act_fn_mul, model_layers_2_mlp_up_proj_mat_mul);  model_layers_2_mlp_act_fn_mul = model_layers_2_mlp_up_proj_mat_mul = None
        initializers_onnx_initializer_127 = self.initializers.onnx_initializer_127
        model_layers_2_mlp_down_proj_mat_mul = self.model_layers_2_mlp_down_proj_MatMul(model_layers_2_mlp_mul, initializers_onnx_initializer_127);  model_layers_2_mlp_mul = initializers_onnx_initializer_127 = None
        getitem_19 = com_microsoft__model_layers_2_post_attention_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_2_post_attention_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_128 = self.initializers.onnx_initializer_128
        com_microsoft__model_layers_3_input_layernorm_skip_layer_norm = self.com_microsoft__model_layers_3_input_layernorm_SkipLayerNorm(getitem_19, model_layers_2_mlp_down_proj_mat_mul, initializers_onnx_initializer_128);  getitem_19 = model_layers_2_mlp_down_proj_mat_mul = initializers_onnx_initializer_128 = None
        getitem_20 = com_microsoft__model_layers_3_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_129 = self.initializers.onnx_initializer_129
        model_layers_3_attn_q_proj_mat_mul = self.model_layers_3_attn_q_proj_MatMul(getitem_20, initializers_onnx_initializer_129);  getitem_20 = initializers_onnx_initializer_129 = None
        getitem_21 = com_microsoft__model_layers_3_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_130 = self.initializers.onnx_initializer_130
        model_layers_3_attn_k_proj_mat_mul = self.model_layers_3_attn_k_proj_MatMul(getitem_21, initializers_onnx_initializer_130);  getitem_21 = initializers_onnx_initializer_130 = None
        getitem_22 = com_microsoft__model_layers_3_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_131 = self.initializers.onnx_initializer_131
        model_layers_3_attn_v_proj_mat_mul = self.model_layers_3_attn_v_proj_MatMul(getitem_22, initializers_onnx_initializer_131);  getitem_22 = initializers_onnx_initializer_131 = None
        initializers_onnx_initializer_132 = self.initializers.onnx_initializer_132
        initializers_onnx_initializer_133 = self.initializers.onnx_initializer_133
        com_microsoft__model_layers_3_attn_q_rotary_rotary_embedding = self.com_microsoft__model_layers_3_attn_q_rotary_RotaryEmbedding(model_layers_3_attn_q_proj_mat_mul, input_3, initializers_onnx_initializer_132, initializers_onnx_initializer_133);  model_layers_3_attn_q_proj_mat_mul = initializers_onnx_initializer_132 = initializers_onnx_initializer_133 = None
        initializers_onnx_initializer_134 = self.initializers.onnx_initializer_134
        initializers_onnx_initializer_135 = self.initializers.onnx_initializer_135
        com_microsoft__model_layers_3_attn_k_rotary_rotary_embedding = self.com_microsoft__model_layers_3_attn_k_rotary_RotaryEmbedding(model_layers_3_attn_k_proj_mat_mul, input_3, initializers_onnx_initializer_134, initializers_onnx_initializer_135);  model_layers_3_attn_k_proj_mat_mul = initializers_onnx_initializer_134 = initializers_onnx_initializer_135 = None
        initializers_onnx_initializer_136 = self.initializers.onnx_initializer_136
        model_layers_3_attn_v_proj_repeat_kv_reshape_1 = self.model_layers_3_attn_v_proj_repeat_kv_Reshape_1(model_layers_3_attn_v_proj_mat_mul, initializers_onnx_initializer_136);  model_layers_3_attn_v_proj_mat_mul = initializers_onnx_initializer_136 = None
        initializers_onnx_initializer_137 = self.initializers.onnx_initializer_137
        model_layers_3_attn_k_proj_repeat_kv_reshape_1 = self.model_layers_3_attn_k_proj_repeat_kv_Reshape_1(com_microsoft__model_layers_3_attn_k_rotary_rotary_embedding, initializers_onnx_initializer_137);  com_microsoft__model_layers_3_attn_k_rotary_rotary_embedding = initializers_onnx_initializer_137 = None
        model_layers_3_attn_v_proj_repeat_kv_transpose_1 = self.model_layers_3_attn_v_proj_repeat_kv_Transpose_1(model_layers_3_attn_v_proj_repeat_kv_reshape_1);  model_layers_3_attn_v_proj_repeat_kv_reshape_1 = None
        model_layers_3_attn_k_proj_repeat_kv_transpose_1 = self.model_layers_3_attn_k_proj_repeat_kv_Transpose_1(model_layers_3_attn_k_proj_repeat_kv_reshape_1);  model_layers_3_attn_k_proj_repeat_kv_reshape_1 = None
        model_layers_3_attn_v_proj_repeat_kv_concat_1 = self.model_layers_3_attn_v_proj_repeat_kv_Concat_1(input_11, model_layers_3_attn_v_proj_repeat_kv_transpose_1);  input_11 = model_layers_3_attn_v_proj_repeat_kv_transpose_1 = None
        model_layers_3_attn_k_proj_repeat_kv_concat_1 = self.model_layers_3_attn_k_proj_repeat_kv_Concat_1(input_10, model_layers_3_attn_k_proj_repeat_kv_transpose_1);  input_10 = model_layers_3_attn_k_proj_repeat_kv_transpose_1 = None
        model_layers_3_attn_v_proj_repeat_kv_shape_1 = self.model_layers_3_attn_v_proj_repeat_kv_Shape_1(model_layers_3_attn_v_proj_repeat_kv_concat_1)
        model_layers_3_attn_v_proj_repeat_kv_unsqueeze_5 = self.model_layers_3_attn_v_proj_repeat_kv_Unsqueeze_5(model_layers_3_attn_v_proj_repeat_kv_concat_1)
        model_layers_3_attn_k_proj_repeat_kv_shape_1 = self.model_layers_3_attn_k_proj_repeat_kv_Shape_1(model_layers_3_attn_k_proj_repeat_kv_concat_1)
        model_layers_3_attn_k_proj_repeat_kv_unsqueeze_5 = self.model_layers_3_attn_k_proj_repeat_kv_Unsqueeze_5(model_layers_3_attn_k_proj_repeat_kv_concat_1)
        initializers_onnx_initializer_138 = self.initializers.onnx_initializer_138
        model_layers_3_attn_v_proj_repeat_kv_gather_1 = self.model_layers_3_attn_v_proj_repeat_kv_Gather_1(model_layers_3_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_138);  initializers_onnx_initializer_138 = None
        initializers_onnx_initializer_139 = self.initializers.onnx_initializer_139
        model_layers_3_attn_v_proj_repeat_kv_gather_3 = self.model_layers_3_attn_v_proj_repeat_kv_Gather_3(model_layers_3_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_139);  model_layers_3_attn_v_proj_repeat_kv_shape_1 = initializers_onnx_initializer_139 = None
        initializers_onnx_initializer_140 = self.initializers.onnx_initializer_140
        model_layers_3_attn_k_proj_repeat_kv_gather_1 = self.model_layers_3_attn_k_proj_repeat_kv_Gather_1(model_layers_3_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_140);  initializers_onnx_initializer_140 = None
        initializers_onnx_initializer_141 = self.initializers.onnx_initializer_141
        model_layers_3_attn_k_proj_repeat_kv_gather_3 = self.model_layers_3_attn_k_proj_repeat_kv_Gather_3(model_layers_3_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_141);  model_layers_3_attn_k_proj_repeat_kv_shape_1 = initializers_onnx_initializer_141 = None
        model_layers_3_attn_v_proj_repeat_kv_unsqueeze_1 = self.model_layers_3_attn_v_proj_repeat_kv_Unsqueeze_1(model_layers_3_attn_v_proj_repeat_kv_gather_1);  model_layers_3_attn_v_proj_repeat_kv_gather_1 = None
        model_layers_3_attn_v_proj_repeat_kv_unsqueeze_3 = self.model_layers_3_attn_v_proj_repeat_kv_Unsqueeze_3(model_layers_3_attn_v_proj_repeat_kv_gather_3);  model_layers_3_attn_v_proj_repeat_kv_gather_3 = None
        model_layers_3_attn_k_proj_repeat_kv_unsqueeze_1 = self.model_layers_3_attn_k_proj_repeat_kv_Unsqueeze_1(model_layers_3_attn_k_proj_repeat_kv_gather_1);  model_layers_3_attn_k_proj_repeat_kv_gather_1 = None
        model_layers_3_attn_k_proj_repeat_kv_unsqueeze_3 = self.model_layers_3_attn_k_proj_repeat_kv_Unsqueeze_3(model_layers_3_attn_k_proj_repeat_kv_gather_3);  model_layers_3_attn_k_proj_repeat_kv_gather_3 = None
        initializers_onnx_initializer_142 = self.initializers.onnx_initializer_142
        initializers_onnx_initializer_143 = self.initializers.onnx_initializer_143
        initializers_onnx_initializer_144 = self.initializers.onnx_initializer_144
        model_layers_3_attn_v_proj_repeat_kv_concat_2 = self.model_layers_3_attn_v_proj_repeat_kv_Concat_2(model_layers_3_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_142, initializers_onnx_initializer_143, model_layers_3_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_144);  initializers_onnx_initializer_142 = initializers_onnx_initializer_143 = initializers_onnx_initializer_144 = None
        initializers_onnx_initializer_145 = self.initializers.onnx_initializer_145
        initializers_onnx_initializer_146 = self.initializers.onnx_initializer_146
        model_layers_3_attn_v_proj_repeat_kv_concat_3 = self.model_layers_3_attn_v_proj_repeat_kv_Concat_3(model_layers_3_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_145, model_layers_3_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_146);  model_layers_3_attn_v_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_145 = model_layers_3_attn_v_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_146 = None
        initializers_onnx_initializer_147 = self.initializers.onnx_initializer_147
        initializers_onnx_initializer_148 = self.initializers.onnx_initializer_148
        initializers_onnx_initializer_149 = self.initializers.onnx_initializer_149
        model_layers_3_attn_k_proj_repeat_kv_concat_2 = self.model_layers_3_attn_k_proj_repeat_kv_Concat_2(model_layers_3_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_147, initializers_onnx_initializer_148, model_layers_3_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_149);  initializers_onnx_initializer_147 = initializers_onnx_initializer_148 = initializers_onnx_initializer_149 = None
        initializers_onnx_initializer_150 = self.initializers.onnx_initializer_150
        initializers_onnx_initializer_151 = self.initializers.onnx_initializer_151
        model_layers_3_attn_k_proj_repeat_kv_concat_3 = self.model_layers_3_attn_k_proj_repeat_kv_Concat_3(model_layers_3_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_150, model_layers_3_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_151);  model_layers_3_attn_k_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_150 = model_layers_3_attn_k_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_151 = None
        initializers_onnx_initializer_152 = self.initializers.onnx_initializer_152
        model_layers_3_attn_v_proj_repeat_kv_equal = self.model_layers_3_attn_v_proj_repeat_kv_Equal(model_layers_3_attn_v_proj_repeat_kv_concat_2, initializers_onnx_initializer_152);  initializers_onnx_initializer_152 = None
        initializers_onnx_initializer_153 = self.initializers.onnx_initializer_153
        model_layers_3_attn_k_proj_repeat_kv_equal = self.model_layers_3_attn_k_proj_repeat_kv_Equal(model_layers_3_attn_k_proj_repeat_kv_concat_2, initializers_onnx_initializer_153);  initializers_onnx_initializer_153 = None
        initializers_onnx_initializer_154 = self.initializers.onnx_initializer_154
        model_layers_3_attn_v_proj_repeat_kv_where = self.model_layers_3_attn_v_proj_repeat_kv_Where(model_layers_3_attn_v_proj_repeat_kv_equal, initializers_onnx_initializer_154, model_layers_3_attn_v_proj_repeat_kv_concat_2);  model_layers_3_attn_v_proj_repeat_kv_equal = initializers_onnx_initializer_154 = model_layers_3_attn_v_proj_repeat_kv_concat_2 = None
        initializers_onnx_initializer_155 = self.initializers.onnx_initializer_155
        model_layers_3_attn_k_proj_repeat_kv_where = self.model_layers_3_attn_k_proj_repeat_kv_Where(model_layers_3_attn_k_proj_repeat_kv_equal, initializers_onnx_initializer_155, model_layers_3_attn_k_proj_repeat_kv_concat_2);  model_layers_3_attn_k_proj_repeat_kv_equal = initializers_onnx_initializer_155 = model_layers_3_attn_k_proj_repeat_kv_concat_2 = None
        model_layers_3_attn_v_proj_repeat_kv_expand = self.model_layers_3_attn_v_proj_repeat_kv_Expand(model_layers_3_attn_v_proj_repeat_kv_unsqueeze_5, model_layers_3_attn_v_proj_repeat_kv_where);  model_layers_3_attn_v_proj_repeat_kv_unsqueeze_5 = model_layers_3_attn_v_proj_repeat_kv_where = None
        model_layers_3_attn_k_proj_repeat_kv_expand = self.model_layers_3_attn_k_proj_repeat_kv_Expand(model_layers_3_attn_k_proj_repeat_kv_unsqueeze_5, model_layers_3_attn_k_proj_repeat_kv_where);  model_layers_3_attn_k_proj_repeat_kv_unsqueeze_5 = model_layers_3_attn_k_proj_repeat_kv_where = None
        model_layers_3_attn_v_proj_repeat_kv_reshape_3 = self.model_layers_3_attn_v_proj_repeat_kv_Reshape_3(model_layers_3_attn_v_proj_repeat_kv_expand, model_layers_3_attn_v_proj_repeat_kv_concat_3);  model_layers_3_attn_v_proj_repeat_kv_expand = model_layers_3_attn_v_proj_repeat_kv_concat_3 = None
        model_layers_3_attn_k_proj_repeat_kv_reshape_3 = self.model_layers_3_attn_k_proj_repeat_kv_Reshape_3(model_layers_3_attn_k_proj_repeat_kv_expand, model_layers_3_attn_k_proj_repeat_kv_concat_3);  model_layers_3_attn_k_proj_repeat_kv_expand = model_layers_3_attn_k_proj_repeat_kv_concat_3 = None
        model_layers_3_attn_v_proj_repeat_kv_transpose_2 = self.model_layers_3_attn_v_proj_repeat_kv_Transpose_2(model_layers_3_attn_v_proj_repeat_kv_reshape_3);  model_layers_3_attn_v_proj_repeat_kv_reshape_3 = None
        model_layers_3_attn_k_proj_repeat_kv_transpose_2 = self.model_layers_3_attn_k_proj_repeat_kv_Transpose_2(model_layers_3_attn_k_proj_repeat_kv_reshape_3);  model_layers_3_attn_k_proj_repeat_kv_reshape_3 = None
        initializers_onnx_initializer_156 = self.initializers.onnx_initializer_156
        model_layers_3_attn_v_proj_repeat_kv_reshape_4 = self.model_layers_3_attn_v_proj_repeat_kv_Reshape_4(model_layers_3_attn_v_proj_repeat_kv_transpose_2, initializers_onnx_initializer_156);  model_layers_3_attn_v_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_156 = None
        initializers_onnx_initializer_157 = self.initializers.onnx_initializer_157
        model_layers_3_attn_k_proj_repeat_kv_reshape_4 = self.model_layers_3_attn_k_proj_repeat_kv_Reshape_4(model_layers_3_attn_k_proj_repeat_kv_transpose_2, initializers_onnx_initializer_157);  model_layers_3_attn_k_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_157 = None
        com_microsoft__model_layers_3_attn_multi_head_attention = self.com_microsoft__model_layers_3_attn_MultiHeadAttention(com_microsoft__model_layers_3_attn_q_rotary_rotary_embedding, model_layers_3_attn_k_proj_repeat_kv_reshape_4, model_layers_3_attn_v_proj_repeat_kv_reshape_4, attention_mask = model_attn_mask_reformat_tile);  com_microsoft__model_layers_3_attn_q_rotary_rotary_embedding = model_layers_3_attn_k_proj_repeat_kv_reshape_4 = model_layers_3_attn_v_proj_repeat_kv_reshape_4 = None
        getitem_23 = com_microsoft__model_layers_3_attn_multi_head_attention[0];  com_microsoft__model_layers_3_attn_multi_head_attention = None
        initializers_onnx_initializer_158 = self.initializers.onnx_initializer_158
        model_layers_3_attn_o_proj_mat_mul = self.model_layers_3_attn_o_proj_MatMul(getitem_23, initializers_onnx_initializer_158);  getitem_23 = initializers_onnx_initializer_158 = None
        getitem_24 = com_microsoft__model_layers_3_input_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_3_input_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_159 = self.initializers.onnx_initializer_159
        com_microsoft__model_layers_3_post_attention_layernorm_skip_layer_norm = self.com_microsoft__model_layers_3_post_attention_layernorm_SkipLayerNorm(getitem_24, model_layers_3_attn_o_proj_mat_mul, initializers_onnx_initializer_159);  getitem_24 = model_layers_3_attn_o_proj_mat_mul = initializers_onnx_initializer_159 = None
        getitem_25 = com_microsoft__model_layers_3_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_160 = self.initializers.onnx_initializer_160
        model_layers_3_mlp_gate_proj_mat_mul = self.model_layers_3_mlp_gate_proj_MatMul(getitem_25, initializers_onnx_initializer_160);  getitem_25 = initializers_onnx_initializer_160 = None
        getitem_26 = com_microsoft__model_layers_3_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_161 = self.initializers.onnx_initializer_161
        model_layers_3_mlp_up_proj_mat_mul = self.model_layers_3_mlp_up_proj_MatMul(getitem_26, initializers_onnx_initializer_161);  getitem_26 = initializers_onnx_initializer_161 = None
        model_layers_3_mlp_act_fn_sigmoid = self.model_layers_3_mlp_act_fn_Sigmoid(model_layers_3_mlp_gate_proj_mat_mul)
        model_layers_3_mlp_act_fn_mul = self.model_layers_3_mlp_act_fn_Mul(model_layers_3_mlp_gate_proj_mat_mul, model_layers_3_mlp_act_fn_sigmoid);  model_layers_3_mlp_gate_proj_mat_mul = model_layers_3_mlp_act_fn_sigmoid = None
        model_layers_3_mlp_mul = self.model_layers_3_mlp_Mul(model_layers_3_mlp_act_fn_mul, model_layers_3_mlp_up_proj_mat_mul);  model_layers_3_mlp_act_fn_mul = model_layers_3_mlp_up_proj_mat_mul = None
        initializers_onnx_initializer_162 = self.initializers.onnx_initializer_162
        model_layers_3_mlp_down_proj_mat_mul = self.model_layers_3_mlp_down_proj_MatMul(model_layers_3_mlp_mul, initializers_onnx_initializer_162);  model_layers_3_mlp_mul = initializers_onnx_initializer_162 = None
        getitem_27 = com_microsoft__model_layers_3_post_attention_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_3_post_attention_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_163 = self.initializers.onnx_initializer_163
        com_microsoft__model_layers_4_input_layernorm_skip_layer_norm = self.com_microsoft__model_layers_4_input_layernorm_SkipLayerNorm(getitem_27, model_layers_3_mlp_down_proj_mat_mul, initializers_onnx_initializer_163);  getitem_27 = model_layers_3_mlp_down_proj_mat_mul = initializers_onnx_initializer_163 = None
        getitem_28 = com_microsoft__model_layers_4_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_164 = self.initializers.onnx_initializer_164
        model_layers_4_attn_q_proj_mat_mul = self.model_layers_4_attn_q_proj_MatMul(getitem_28, initializers_onnx_initializer_164);  getitem_28 = initializers_onnx_initializer_164 = None
        getitem_29 = com_microsoft__model_layers_4_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_165 = self.initializers.onnx_initializer_165
        model_layers_4_attn_k_proj_mat_mul = self.model_layers_4_attn_k_proj_MatMul(getitem_29, initializers_onnx_initializer_165);  getitem_29 = initializers_onnx_initializer_165 = None
        getitem_30 = com_microsoft__model_layers_4_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_166 = self.initializers.onnx_initializer_166
        model_layers_4_attn_v_proj_mat_mul = self.model_layers_4_attn_v_proj_MatMul(getitem_30, initializers_onnx_initializer_166);  getitem_30 = initializers_onnx_initializer_166 = None
        initializers_onnx_initializer_167 = self.initializers.onnx_initializer_167
        initializers_onnx_initializer_168 = self.initializers.onnx_initializer_168
        com_microsoft__model_layers_4_attn_q_rotary_rotary_embedding = self.com_microsoft__model_layers_4_attn_q_rotary_RotaryEmbedding(model_layers_4_attn_q_proj_mat_mul, input_3, initializers_onnx_initializer_167, initializers_onnx_initializer_168);  model_layers_4_attn_q_proj_mat_mul = initializers_onnx_initializer_167 = initializers_onnx_initializer_168 = None
        initializers_onnx_initializer_169 = self.initializers.onnx_initializer_169
        initializers_onnx_initializer_170 = self.initializers.onnx_initializer_170
        com_microsoft__model_layers_4_attn_k_rotary_rotary_embedding = self.com_microsoft__model_layers_4_attn_k_rotary_RotaryEmbedding(model_layers_4_attn_k_proj_mat_mul, input_3, initializers_onnx_initializer_169, initializers_onnx_initializer_170);  model_layers_4_attn_k_proj_mat_mul = initializers_onnx_initializer_169 = initializers_onnx_initializer_170 = None
        initializers_onnx_initializer_171 = self.initializers.onnx_initializer_171
        model_layers_4_attn_v_proj_repeat_kv_reshape_1 = self.model_layers_4_attn_v_proj_repeat_kv_Reshape_1(model_layers_4_attn_v_proj_mat_mul, initializers_onnx_initializer_171);  model_layers_4_attn_v_proj_mat_mul = initializers_onnx_initializer_171 = None
        initializers_onnx_initializer_172 = self.initializers.onnx_initializer_172
        model_layers_4_attn_k_proj_repeat_kv_reshape_1 = self.model_layers_4_attn_k_proj_repeat_kv_Reshape_1(com_microsoft__model_layers_4_attn_k_rotary_rotary_embedding, initializers_onnx_initializer_172);  com_microsoft__model_layers_4_attn_k_rotary_rotary_embedding = initializers_onnx_initializer_172 = None
        model_layers_4_attn_v_proj_repeat_kv_transpose_1 = self.model_layers_4_attn_v_proj_repeat_kv_Transpose_1(model_layers_4_attn_v_proj_repeat_kv_reshape_1);  model_layers_4_attn_v_proj_repeat_kv_reshape_1 = None
        model_layers_4_attn_k_proj_repeat_kv_transpose_1 = self.model_layers_4_attn_k_proj_repeat_kv_Transpose_1(model_layers_4_attn_k_proj_repeat_kv_reshape_1);  model_layers_4_attn_k_proj_repeat_kv_reshape_1 = None
        model_layers_4_attn_v_proj_repeat_kv_concat_1 = self.model_layers_4_attn_v_proj_repeat_kv_Concat_1(input_13, model_layers_4_attn_v_proj_repeat_kv_transpose_1);  input_13 = model_layers_4_attn_v_proj_repeat_kv_transpose_1 = None
        model_layers_4_attn_k_proj_repeat_kv_concat_1 = self.model_layers_4_attn_k_proj_repeat_kv_Concat_1(input_12, model_layers_4_attn_k_proj_repeat_kv_transpose_1);  input_12 = model_layers_4_attn_k_proj_repeat_kv_transpose_1 = None
        model_layers_4_attn_v_proj_repeat_kv_shape_1 = self.model_layers_4_attn_v_proj_repeat_kv_Shape_1(model_layers_4_attn_v_proj_repeat_kv_concat_1)
        model_layers_4_attn_v_proj_repeat_kv_unsqueeze_5 = self.model_layers_4_attn_v_proj_repeat_kv_Unsqueeze_5(model_layers_4_attn_v_proj_repeat_kv_concat_1)
        model_layers_4_attn_k_proj_repeat_kv_shape_1 = self.model_layers_4_attn_k_proj_repeat_kv_Shape_1(model_layers_4_attn_k_proj_repeat_kv_concat_1)
        model_layers_4_attn_k_proj_repeat_kv_unsqueeze_5 = self.model_layers_4_attn_k_proj_repeat_kv_Unsqueeze_5(model_layers_4_attn_k_proj_repeat_kv_concat_1)
        initializers_onnx_initializer_173 = self.initializers.onnx_initializer_173
        model_layers_4_attn_v_proj_repeat_kv_gather_1 = self.model_layers_4_attn_v_proj_repeat_kv_Gather_1(model_layers_4_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_173);  initializers_onnx_initializer_173 = None
        initializers_onnx_initializer_174 = self.initializers.onnx_initializer_174
        model_layers_4_attn_v_proj_repeat_kv_gather_3 = self.model_layers_4_attn_v_proj_repeat_kv_Gather_3(model_layers_4_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_174);  model_layers_4_attn_v_proj_repeat_kv_shape_1 = initializers_onnx_initializer_174 = None
        initializers_onnx_initializer_175 = self.initializers.onnx_initializer_175
        model_layers_4_attn_k_proj_repeat_kv_gather_1 = self.model_layers_4_attn_k_proj_repeat_kv_Gather_1(model_layers_4_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_175);  initializers_onnx_initializer_175 = None
        initializers_onnx_initializer_176 = self.initializers.onnx_initializer_176
        model_layers_4_attn_k_proj_repeat_kv_gather_3 = self.model_layers_4_attn_k_proj_repeat_kv_Gather_3(model_layers_4_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_176);  model_layers_4_attn_k_proj_repeat_kv_shape_1 = initializers_onnx_initializer_176 = None
        model_layers_4_attn_v_proj_repeat_kv_unsqueeze_1 = self.model_layers_4_attn_v_proj_repeat_kv_Unsqueeze_1(model_layers_4_attn_v_proj_repeat_kv_gather_1);  model_layers_4_attn_v_proj_repeat_kv_gather_1 = None
        model_layers_4_attn_v_proj_repeat_kv_unsqueeze_3 = self.model_layers_4_attn_v_proj_repeat_kv_Unsqueeze_3(model_layers_4_attn_v_proj_repeat_kv_gather_3);  model_layers_4_attn_v_proj_repeat_kv_gather_3 = None
        model_layers_4_attn_k_proj_repeat_kv_unsqueeze_1 = self.model_layers_4_attn_k_proj_repeat_kv_Unsqueeze_1(model_layers_4_attn_k_proj_repeat_kv_gather_1);  model_layers_4_attn_k_proj_repeat_kv_gather_1 = None
        model_layers_4_attn_k_proj_repeat_kv_unsqueeze_3 = self.model_layers_4_attn_k_proj_repeat_kv_Unsqueeze_3(model_layers_4_attn_k_proj_repeat_kv_gather_3);  model_layers_4_attn_k_proj_repeat_kv_gather_3 = None
        initializers_onnx_initializer_177 = self.initializers.onnx_initializer_177
        initializers_onnx_initializer_178 = self.initializers.onnx_initializer_178
        initializers_onnx_initializer_179 = self.initializers.onnx_initializer_179
        model_layers_4_attn_v_proj_repeat_kv_concat_2 = self.model_layers_4_attn_v_proj_repeat_kv_Concat_2(model_layers_4_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_177, initializers_onnx_initializer_178, model_layers_4_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_179);  initializers_onnx_initializer_177 = initializers_onnx_initializer_178 = initializers_onnx_initializer_179 = None
        initializers_onnx_initializer_180 = self.initializers.onnx_initializer_180
        initializers_onnx_initializer_181 = self.initializers.onnx_initializer_181
        model_layers_4_attn_v_proj_repeat_kv_concat_3 = self.model_layers_4_attn_v_proj_repeat_kv_Concat_3(model_layers_4_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_180, model_layers_4_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_181);  model_layers_4_attn_v_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_180 = model_layers_4_attn_v_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_181 = None
        initializers_onnx_initializer_182 = self.initializers.onnx_initializer_182
        initializers_onnx_initializer_183 = self.initializers.onnx_initializer_183
        initializers_onnx_initializer_184 = self.initializers.onnx_initializer_184
        model_layers_4_attn_k_proj_repeat_kv_concat_2 = self.model_layers_4_attn_k_proj_repeat_kv_Concat_2(model_layers_4_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_182, initializers_onnx_initializer_183, model_layers_4_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_184);  initializers_onnx_initializer_182 = initializers_onnx_initializer_183 = initializers_onnx_initializer_184 = None
        initializers_onnx_initializer_185 = self.initializers.onnx_initializer_185
        initializers_onnx_initializer_186 = self.initializers.onnx_initializer_186
        model_layers_4_attn_k_proj_repeat_kv_concat_3 = self.model_layers_4_attn_k_proj_repeat_kv_Concat_3(model_layers_4_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_185, model_layers_4_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_186);  model_layers_4_attn_k_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_185 = model_layers_4_attn_k_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_186 = None
        initializers_onnx_initializer_187 = self.initializers.onnx_initializer_187
        model_layers_4_attn_v_proj_repeat_kv_equal = self.model_layers_4_attn_v_proj_repeat_kv_Equal(model_layers_4_attn_v_proj_repeat_kv_concat_2, initializers_onnx_initializer_187);  initializers_onnx_initializer_187 = None
        initializers_onnx_initializer_188 = self.initializers.onnx_initializer_188
        model_layers_4_attn_k_proj_repeat_kv_equal = self.model_layers_4_attn_k_proj_repeat_kv_Equal(model_layers_4_attn_k_proj_repeat_kv_concat_2, initializers_onnx_initializer_188);  initializers_onnx_initializer_188 = None
        initializers_onnx_initializer_189 = self.initializers.onnx_initializer_189
        model_layers_4_attn_v_proj_repeat_kv_where = self.model_layers_4_attn_v_proj_repeat_kv_Where(model_layers_4_attn_v_proj_repeat_kv_equal, initializers_onnx_initializer_189, model_layers_4_attn_v_proj_repeat_kv_concat_2);  model_layers_4_attn_v_proj_repeat_kv_equal = initializers_onnx_initializer_189 = model_layers_4_attn_v_proj_repeat_kv_concat_2 = None
        initializers_onnx_initializer_190 = self.initializers.onnx_initializer_190
        model_layers_4_attn_k_proj_repeat_kv_where = self.model_layers_4_attn_k_proj_repeat_kv_Where(model_layers_4_attn_k_proj_repeat_kv_equal, initializers_onnx_initializer_190, model_layers_4_attn_k_proj_repeat_kv_concat_2);  model_layers_4_attn_k_proj_repeat_kv_equal = initializers_onnx_initializer_190 = model_layers_4_attn_k_proj_repeat_kv_concat_2 = None
        model_layers_4_attn_v_proj_repeat_kv_expand = self.model_layers_4_attn_v_proj_repeat_kv_Expand(model_layers_4_attn_v_proj_repeat_kv_unsqueeze_5, model_layers_4_attn_v_proj_repeat_kv_where);  model_layers_4_attn_v_proj_repeat_kv_unsqueeze_5 = model_layers_4_attn_v_proj_repeat_kv_where = None
        model_layers_4_attn_k_proj_repeat_kv_expand = self.model_layers_4_attn_k_proj_repeat_kv_Expand(model_layers_4_attn_k_proj_repeat_kv_unsqueeze_5, model_layers_4_attn_k_proj_repeat_kv_where);  model_layers_4_attn_k_proj_repeat_kv_unsqueeze_5 = model_layers_4_attn_k_proj_repeat_kv_where = None
        model_layers_4_attn_v_proj_repeat_kv_reshape_3 = self.model_layers_4_attn_v_proj_repeat_kv_Reshape_3(model_layers_4_attn_v_proj_repeat_kv_expand, model_layers_4_attn_v_proj_repeat_kv_concat_3);  model_layers_4_attn_v_proj_repeat_kv_expand = model_layers_4_attn_v_proj_repeat_kv_concat_3 = None
        model_layers_4_attn_k_proj_repeat_kv_reshape_3 = self.model_layers_4_attn_k_proj_repeat_kv_Reshape_3(model_layers_4_attn_k_proj_repeat_kv_expand, model_layers_4_attn_k_proj_repeat_kv_concat_3);  model_layers_4_attn_k_proj_repeat_kv_expand = model_layers_4_attn_k_proj_repeat_kv_concat_3 = None
        model_layers_4_attn_v_proj_repeat_kv_transpose_2 = self.model_layers_4_attn_v_proj_repeat_kv_Transpose_2(model_layers_4_attn_v_proj_repeat_kv_reshape_3);  model_layers_4_attn_v_proj_repeat_kv_reshape_3 = None
        model_layers_4_attn_k_proj_repeat_kv_transpose_2 = self.model_layers_4_attn_k_proj_repeat_kv_Transpose_2(model_layers_4_attn_k_proj_repeat_kv_reshape_3);  model_layers_4_attn_k_proj_repeat_kv_reshape_3 = None
        initializers_onnx_initializer_191 = self.initializers.onnx_initializer_191
        model_layers_4_attn_v_proj_repeat_kv_reshape_4 = self.model_layers_4_attn_v_proj_repeat_kv_Reshape_4(model_layers_4_attn_v_proj_repeat_kv_transpose_2, initializers_onnx_initializer_191);  model_layers_4_attn_v_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_191 = None
        initializers_onnx_initializer_192 = self.initializers.onnx_initializer_192
        model_layers_4_attn_k_proj_repeat_kv_reshape_4 = self.model_layers_4_attn_k_proj_repeat_kv_Reshape_4(model_layers_4_attn_k_proj_repeat_kv_transpose_2, initializers_onnx_initializer_192);  model_layers_4_attn_k_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_192 = None
        com_microsoft__model_layers_4_attn_multi_head_attention = self.com_microsoft__model_layers_4_attn_MultiHeadAttention(com_microsoft__model_layers_4_attn_q_rotary_rotary_embedding, model_layers_4_attn_k_proj_repeat_kv_reshape_4, model_layers_4_attn_v_proj_repeat_kv_reshape_4, attention_mask = model_attn_mask_reformat_tile);  com_microsoft__model_layers_4_attn_q_rotary_rotary_embedding = model_layers_4_attn_k_proj_repeat_kv_reshape_4 = model_layers_4_attn_v_proj_repeat_kv_reshape_4 = None
        getitem_31 = com_microsoft__model_layers_4_attn_multi_head_attention[0];  com_microsoft__model_layers_4_attn_multi_head_attention = None
        initializers_onnx_initializer_193 = self.initializers.onnx_initializer_193
        model_layers_4_attn_o_proj_mat_mul = self.model_layers_4_attn_o_proj_MatMul(getitem_31, initializers_onnx_initializer_193);  getitem_31 = initializers_onnx_initializer_193 = None
        getitem_32 = com_microsoft__model_layers_4_input_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_4_input_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_194 = self.initializers.onnx_initializer_194
        com_microsoft__model_layers_4_post_attention_layernorm_skip_layer_norm = self.com_microsoft__model_layers_4_post_attention_layernorm_SkipLayerNorm(getitem_32, model_layers_4_attn_o_proj_mat_mul, initializers_onnx_initializer_194);  getitem_32 = model_layers_4_attn_o_proj_mat_mul = initializers_onnx_initializer_194 = None
        getitem_33 = com_microsoft__model_layers_4_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_195 = self.initializers.onnx_initializer_195
        model_layers_4_mlp_gate_proj_mat_mul = self.model_layers_4_mlp_gate_proj_MatMul(getitem_33, initializers_onnx_initializer_195);  getitem_33 = initializers_onnx_initializer_195 = None
        getitem_34 = com_microsoft__model_layers_4_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_196 = self.initializers.onnx_initializer_196
        model_layers_4_mlp_up_proj_mat_mul = self.model_layers_4_mlp_up_proj_MatMul(getitem_34, initializers_onnx_initializer_196);  getitem_34 = initializers_onnx_initializer_196 = None
        model_layers_4_mlp_act_fn_sigmoid = self.model_layers_4_mlp_act_fn_Sigmoid(model_layers_4_mlp_gate_proj_mat_mul)
        model_layers_4_mlp_act_fn_mul = self.model_layers_4_mlp_act_fn_Mul(model_layers_4_mlp_gate_proj_mat_mul, model_layers_4_mlp_act_fn_sigmoid);  model_layers_4_mlp_gate_proj_mat_mul = model_layers_4_mlp_act_fn_sigmoid = None
        model_layers_4_mlp_mul = self.model_layers_4_mlp_Mul(model_layers_4_mlp_act_fn_mul, model_layers_4_mlp_up_proj_mat_mul);  model_layers_4_mlp_act_fn_mul = model_layers_4_mlp_up_proj_mat_mul = None
        initializers_onnx_initializer_197 = self.initializers.onnx_initializer_197
        model_layers_4_mlp_down_proj_mat_mul = self.model_layers_4_mlp_down_proj_MatMul(model_layers_4_mlp_mul, initializers_onnx_initializer_197);  model_layers_4_mlp_mul = initializers_onnx_initializer_197 = None
        getitem_35 = com_microsoft__model_layers_4_post_attention_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_4_post_attention_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_198 = self.initializers.onnx_initializer_198
        com_microsoft__model_layers_5_input_layernorm_skip_layer_norm = self.com_microsoft__model_layers_5_input_layernorm_SkipLayerNorm(getitem_35, model_layers_4_mlp_down_proj_mat_mul, initializers_onnx_initializer_198);  getitem_35 = model_layers_4_mlp_down_proj_mat_mul = initializers_onnx_initializer_198 = None
        getitem_36 = com_microsoft__model_layers_5_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_199 = self.initializers.onnx_initializer_199
        model_layers_5_attn_q_proj_mat_mul = self.model_layers_5_attn_q_proj_MatMul(getitem_36, initializers_onnx_initializer_199);  getitem_36 = initializers_onnx_initializer_199 = None
        getitem_37 = com_microsoft__model_layers_5_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_200 = self.initializers.onnx_initializer_200
        model_layers_5_attn_k_proj_mat_mul = self.model_layers_5_attn_k_proj_MatMul(getitem_37, initializers_onnx_initializer_200);  getitem_37 = initializers_onnx_initializer_200 = None
        getitem_38 = com_microsoft__model_layers_5_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_201 = self.initializers.onnx_initializer_201
        model_layers_5_attn_v_proj_mat_mul = self.model_layers_5_attn_v_proj_MatMul(getitem_38, initializers_onnx_initializer_201);  getitem_38 = initializers_onnx_initializer_201 = None
        initializers_onnx_initializer_202 = self.initializers.onnx_initializer_202
        initializers_onnx_initializer_203 = self.initializers.onnx_initializer_203
        com_microsoft__model_layers_5_attn_q_rotary_rotary_embedding = self.com_microsoft__model_layers_5_attn_q_rotary_RotaryEmbedding(model_layers_5_attn_q_proj_mat_mul, input_3, initializers_onnx_initializer_202, initializers_onnx_initializer_203);  model_layers_5_attn_q_proj_mat_mul = initializers_onnx_initializer_202 = initializers_onnx_initializer_203 = None
        initializers_onnx_initializer_204 = self.initializers.onnx_initializer_204
        initializers_onnx_initializer_205 = self.initializers.onnx_initializer_205
        com_microsoft__model_layers_5_attn_k_rotary_rotary_embedding = self.com_microsoft__model_layers_5_attn_k_rotary_RotaryEmbedding(model_layers_5_attn_k_proj_mat_mul, input_3, initializers_onnx_initializer_204, initializers_onnx_initializer_205);  model_layers_5_attn_k_proj_mat_mul = initializers_onnx_initializer_204 = initializers_onnx_initializer_205 = None
        initializers_onnx_initializer_206 = self.initializers.onnx_initializer_206
        model_layers_5_attn_v_proj_repeat_kv_reshape_1 = self.model_layers_5_attn_v_proj_repeat_kv_Reshape_1(model_layers_5_attn_v_proj_mat_mul, initializers_onnx_initializer_206);  model_layers_5_attn_v_proj_mat_mul = initializers_onnx_initializer_206 = None
        initializers_onnx_initializer_207 = self.initializers.onnx_initializer_207
        model_layers_5_attn_k_proj_repeat_kv_reshape_1 = self.model_layers_5_attn_k_proj_repeat_kv_Reshape_1(com_microsoft__model_layers_5_attn_k_rotary_rotary_embedding, initializers_onnx_initializer_207);  com_microsoft__model_layers_5_attn_k_rotary_rotary_embedding = initializers_onnx_initializer_207 = None
        model_layers_5_attn_v_proj_repeat_kv_transpose_1 = self.model_layers_5_attn_v_proj_repeat_kv_Transpose_1(model_layers_5_attn_v_proj_repeat_kv_reshape_1);  model_layers_5_attn_v_proj_repeat_kv_reshape_1 = None
        model_layers_5_attn_k_proj_repeat_kv_transpose_1 = self.model_layers_5_attn_k_proj_repeat_kv_Transpose_1(model_layers_5_attn_k_proj_repeat_kv_reshape_1);  model_layers_5_attn_k_proj_repeat_kv_reshape_1 = None
        model_layers_5_attn_v_proj_repeat_kv_concat_1 = self.model_layers_5_attn_v_proj_repeat_kv_Concat_1(input_15, model_layers_5_attn_v_proj_repeat_kv_transpose_1);  input_15 = model_layers_5_attn_v_proj_repeat_kv_transpose_1 = None
        model_layers_5_attn_k_proj_repeat_kv_concat_1 = self.model_layers_5_attn_k_proj_repeat_kv_Concat_1(input_14, model_layers_5_attn_k_proj_repeat_kv_transpose_1);  input_14 = model_layers_5_attn_k_proj_repeat_kv_transpose_1 = None
        model_layers_5_attn_v_proj_repeat_kv_shape_1 = self.model_layers_5_attn_v_proj_repeat_kv_Shape_1(model_layers_5_attn_v_proj_repeat_kv_concat_1)
        model_layers_5_attn_v_proj_repeat_kv_unsqueeze_5 = self.model_layers_5_attn_v_proj_repeat_kv_Unsqueeze_5(model_layers_5_attn_v_proj_repeat_kv_concat_1)
        model_layers_5_attn_k_proj_repeat_kv_shape_1 = self.model_layers_5_attn_k_proj_repeat_kv_Shape_1(model_layers_5_attn_k_proj_repeat_kv_concat_1)
        model_layers_5_attn_k_proj_repeat_kv_unsqueeze_5 = self.model_layers_5_attn_k_proj_repeat_kv_Unsqueeze_5(model_layers_5_attn_k_proj_repeat_kv_concat_1)
        initializers_onnx_initializer_208 = self.initializers.onnx_initializer_208
        model_layers_5_attn_v_proj_repeat_kv_gather_1 = self.model_layers_5_attn_v_proj_repeat_kv_Gather_1(model_layers_5_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_208);  initializers_onnx_initializer_208 = None
        initializers_onnx_initializer_209 = self.initializers.onnx_initializer_209
        model_layers_5_attn_v_proj_repeat_kv_gather_3 = self.model_layers_5_attn_v_proj_repeat_kv_Gather_3(model_layers_5_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_209);  model_layers_5_attn_v_proj_repeat_kv_shape_1 = initializers_onnx_initializer_209 = None
        initializers_onnx_initializer_210 = self.initializers.onnx_initializer_210
        model_layers_5_attn_k_proj_repeat_kv_gather_1 = self.model_layers_5_attn_k_proj_repeat_kv_Gather_1(model_layers_5_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_210);  initializers_onnx_initializer_210 = None
        initializers_onnx_initializer_211 = self.initializers.onnx_initializer_211
        model_layers_5_attn_k_proj_repeat_kv_gather_3 = self.model_layers_5_attn_k_proj_repeat_kv_Gather_3(model_layers_5_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_211);  model_layers_5_attn_k_proj_repeat_kv_shape_1 = initializers_onnx_initializer_211 = None
        model_layers_5_attn_v_proj_repeat_kv_unsqueeze_1 = self.model_layers_5_attn_v_proj_repeat_kv_Unsqueeze_1(model_layers_5_attn_v_proj_repeat_kv_gather_1);  model_layers_5_attn_v_proj_repeat_kv_gather_1 = None
        model_layers_5_attn_v_proj_repeat_kv_unsqueeze_3 = self.model_layers_5_attn_v_proj_repeat_kv_Unsqueeze_3(model_layers_5_attn_v_proj_repeat_kv_gather_3);  model_layers_5_attn_v_proj_repeat_kv_gather_3 = None
        model_layers_5_attn_k_proj_repeat_kv_unsqueeze_1 = self.model_layers_5_attn_k_proj_repeat_kv_Unsqueeze_1(model_layers_5_attn_k_proj_repeat_kv_gather_1);  model_layers_5_attn_k_proj_repeat_kv_gather_1 = None
        model_layers_5_attn_k_proj_repeat_kv_unsqueeze_3 = self.model_layers_5_attn_k_proj_repeat_kv_Unsqueeze_3(model_layers_5_attn_k_proj_repeat_kv_gather_3);  model_layers_5_attn_k_proj_repeat_kv_gather_3 = None
        initializers_onnx_initializer_212 = self.initializers.onnx_initializer_212
        initializers_onnx_initializer_213 = self.initializers.onnx_initializer_213
        initializers_onnx_initializer_214 = self.initializers.onnx_initializer_214
        model_layers_5_attn_v_proj_repeat_kv_concat_2 = self.model_layers_5_attn_v_proj_repeat_kv_Concat_2(model_layers_5_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_212, initializers_onnx_initializer_213, model_layers_5_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_214);  initializers_onnx_initializer_212 = initializers_onnx_initializer_213 = initializers_onnx_initializer_214 = None
        initializers_onnx_initializer_215 = self.initializers.onnx_initializer_215
        initializers_onnx_initializer_216 = self.initializers.onnx_initializer_216
        model_layers_5_attn_v_proj_repeat_kv_concat_3 = self.model_layers_5_attn_v_proj_repeat_kv_Concat_3(model_layers_5_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_215, model_layers_5_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_216);  model_layers_5_attn_v_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_215 = model_layers_5_attn_v_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_216 = None
        initializers_onnx_initializer_217 = self.initializers.onnx_initializer_217
        initializers_onnx_initializer_218 = self.initializers.onnx_initializer_218
        initializers_onnx_initializer_219 = self.initializers.onnx_initializer_219
        model_layers_5_attn_k_proj_repeat_kv_concat_2 = self.model_layers_5_attn_k_proj_repeat_kv_Concat_2(model_layers_5_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_217, initializers_onnx_initializer_218, model_layers_5_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_219);  initializers_onnx_initializer_217 = initializers_onnx_initializer_218 = initializers_onnx_initializer_219 = None
        initializers_onnx_initializer_220 = self.initializers.onnx_initializer_220
        initializers_onnx_initializer_221 = self.initializers.onnx_initializer_221
        model_layers_5_attn_k_proj_repeat_kv_concat_3 = self.model_layers_5_attn_k_proj_repeat_kv_Concat_3(model_layers_5_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_220, model_layers_5_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_221);  model_layers_5_attn_k_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_220 = model_layers_5_attn_k_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_221 = None
        initializers_onnx_initializer_222 = self.initializers.onnx_initializer_222
        model_layers_5_attn_v_proj_repeat_kv_equal = self.model_layers_5_attn_v_proj_repeat_kv_Equal(model_layers_5_attn_v_proj_repeat_kv_concat_2, initializers_onnx_initializer_222);  initializers_onnx_initializer_222 = None
        initializers_onnx_initializer_223 = self.initializers.onnx_initializer_223
        model_layers_5_attn_k_proj_repeat_kv_equal = self.model_layers_5_attn_k_proj_repeat_kv_Equal(model_layers_5_attn_k_proj_repeat_kv_concat_2, initializers_onnx_initializer_223);  initializers_onnx_initializer_223 = None
        initializers_onnx_initializer_224 = self.initializers.onnx_initializer_224
        model_layers_5_attn_v_proj_repeat_kv_where = self.model_layers_5_attn_v_proj_repeat_kv_Where(model_layers_5_attn_v_proj_repeat_kv_equal, initializers_onnx_initializer_224, model_layers_5_attn_v_proj_repeat_kv_concat_2);  model_layers_5_attn_v_proj_repeat_kv_equal = initializers_onnx_initializer_224 = model_layers_5_attn_v_proj_repeat_kv_concat_2 = None
        initializers_onnx_initializer_225 = self.initializers.onnx_initializer_225
        model_layers_5_attn_k_proj_repeat_kv_where = self.model_layers_5_attn_k_proj_repeat_kv_Where(model_layers_5_attn_k_proj_repeat_kv_equal, initializers_onnx_initializer_225, model_layers_5_attn_k_proj_repeat_kv_concat_2);  model_layers_5_attn_k_proj_repeat_kv_equal = initializers_onnx_initializer_225 = model_layers_5_attn_k_proj_repeat_kv_concat_2 = None
        model_layers_5_attn_v_proj_repeat_kv_expand = self.model_layers_5_attn_v_proj_repeat_kv_Expand(model_layers_5_attn_v_proj_repeat_kv_unsqueeze_5, model_layers_5_attn_v_proj_repeat_kv_where);  model_layers_5_attn_v_proj_repeat_kv_unsqueeze_5 = model_layers_5_attn_v_proj_repeat_kv_where = None
        model_layers_5_attn_k_proj_repeat_kv_expand = self.model_layers_5_attn_k_proj_repeat_kv_Expand(model_layers_5_attn_k_proj_repeat_kv_unsqueeze_5, model_layers_5_attn_k_proj_repeat_kv_where);  model_layers_5_attn_k_proj_repeat_kv_unsqueeze_5 = model_layers_5_attn_k_proj_repeat_kv_where = None
        model_layers_5_attn_v_proj_repeat_kv_reshape_3 = self.model_layers_5_attn_v_proj_repeat_kv_Reshape_3(model_layers_5_attn_v_proj_repeat_kv_expand, model_layers_5_attn_v_proj_repeat_kv_concat_3);  model_layers_5_attn_v_proj_repeat_kv_expand = model_layers_5_attn_v_proj_repeat_kv_concat_3 = None
        model_layers_5_attn_k_proj_repeat_kv_reshape_3 = self.model_layers_5_attn_k_proj_repeat_kv_Reshape_3(model_layers_5_attn_k_proj_repeat_kv_expand, model_layers_5_attn_k_proj_repeat_kv_concat_3);  model_layers_5_attn_k_proj_repeat_kv_expand = model_layers_5_attn_k_proj_repeat_kv_concat_3 = None
        model_layers_5_attn_v_proj_repeat_kv_transpose_2 = self.model_layers_5_attn_v_proj_repeat_kv_Transpose_2(model_layers_5_attn_v_proj_repeat_kv_reshape_3);  model_layers_5_attn_v_proj_repeat_kv_reshape_3 = None
        model_layers_5_attn_k_proj_repeat_kv_transpose_2 = self.model_layers_5_attn_k_proj_repeat_kv_Transpose_2(model_layers_5_attn_k_proj_repeat_kv_reshape_3);  model_layers_5_attn_k_proj_repeat_kv_reshape_3 = None
        initializers_onnx_initializer_226 = self.initializers.onnx_initializer_226
        model_layers_5_attn_v_proj_repeat_kv_reshape_4 = self.model_layers_5_attn_v_proj_repeat_kv_Reshape_4(model_layers_5_attn_v_proj_repeat_kv_transpose_2, initializers_onnx_initializer_226);  model_layers_5_attn_v_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_226 = None
        initializers_onnx_initializer_227 = self.initializers.onnx_initializer_227
        model_layers_5_attn_k_proj_repeat_kv_reshape_4 = self.model_layers_5_attn_k_proj_repeat_kv_Reshape_4(model_layers_5_attn_k_proj_repeat_kv_transpose_2, initializers_onnx_initializer_227);  model_layers_5_attn_k_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_227 = None
        com_microsoft__model_layers_5_attn_multi_head_attention = self.com_microsoft__model_layers_5_attn_MultiHeadAttention(com_microsoft__model_layers_5_attn_q_rotary_rotary_embedding, model_layers_5_attn_k_proj_repeat_kv_reshape_4, model_layers_5_attn_v_proj_repeat_kv_reshape_4, attention_mask = model_attn_mask_reformat_tile);  com_microsoft__model_layers_5_attn_q_rotary_rotary_embedding = model_layers_5_attn_k_proj_repeat_kv_reshape_4 = model_layers_5_attn_v_proj_repeat_kv_reshape_4 = None
        getitem_39 = com_microsoft__model_layers_5_attn_multi_head_attention[0];  com_microsoft__model_layers_5_attn_multi_head_attention = None
        initializers_onnx_initializer_228 = self.initializers.onnx_initializer_228
        model_layers_5_attn_o_proj_mat_mul = self.model_layers_5_attn_o_proj_MatMul(getitem_39, initializers_onnx_initializer_228);  getitem_39 = initializers_onnx_initializer_228 = None
        getitem_40 = com_microsoft__model_layers_5_input_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_5_input_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_229 = self.initializers.onnx_initializer_229
        com_microsoft__model_layers_5_post_attention_layernorm_skip_layer_norm = self.com_microsoft__model_layers_5_post_attention_layernorm_SkipLayerNorm(getitem_40, model_layers_5_attn_o_proj_mat_mul, initializers_onnx_initializer_229);  getitem_40 = model_layers_5_attn_o_proj_mat_mul = initializers_onnx_initializer_229 = None
        getitem_41 = com_microsoft__model_layers_5_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_230 = self.initializers.onnx_initializer_230
        model_layers_5_mlp_gate_proj_mat_mul = self.model_layers_5_mlp_gate_proj_MatMul(getitem_41, initializers_onnx_initializer_230);  getitem_41 = initializers_onnx_initializer_230 = None
        getitem_42 = com_microsoft__model_layers_5_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_231 = self.initializers.onnx_initializer_231
        model_layers_5_mlp_up_proj_mat_mul = self.model_layers_5_mlp_up_proj_MatMul(getitem_42, initializers_onnx_initializer_231);  getitem_42 = initializers_onnx_initializer_231 = None
        model_layers_5_mlp_act_fn_sigmoid = self.model_layers_5_mlp_act_fn_Sigmoid(model_layers_5_mlp_gate_proj_mat_mul)
        model_layers_5_mlp_act_fn_mul = self.model_layers_5_mlp_act_fn_Mul(model_layers_5_mlp_gate_proj_mat_mul, model_layers_5_mlp_act_fn_sigmoid);  model_layers_5_mlp_gate_proj_mat_mul = model_layers_5_mlp_act_fn_sigmoid = None
        model_layers_5_mlp_mul = self.model_layers_5_mlp_Mul(model_layers_5_mlp_act_fn_mul, model_layers_5_mlp_up_proj_mat_mul);  model_layers_5_mlp_act_fn_mul = model_layers_5_mlp_up_proj_mat_mul = None
        initializers_onnx_initializer_232 = self.initializers.onnx_initializer_232
        model_layers_5_mlp_down_proj_mat_mul = self.model_layers_5_mlp_down_proj_MatMul(model_layers_5_mlp_mul, initializers_onnx_initializer_232);  model_layers_5_mlp_mul = initializers_onnx_initializer_232 = None
        getitem_43 = com_microsoft__model_layers_5_post_attention_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_5_post_attention_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_233 = self.initializers.onnx_initializer_233
        com_microsoft__model_layers_6_input_layernorm_skip_layer_norm = self.com_microsoft__model_layers_6_input_layernorm_SkipLayerNorm(getitem_43, model_layers_5_mlp_down_proj_mat_mul, initializers_onnx_initializer_233);  getitem_43 = model_layers_5_mlp_down_proj_mat_mul = initializers_onnx_initializer_233 = None
        getitem_44 = com_microsoft__model_layers_6_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_234 = self.initializers.onnx_initializer_234
        model_layers_6_attn_q_proj_mat_mul = self.model_layers_6_attn_q_proj_MatMul(getitem_44, initializers_onnx_initializer_234);  getitem_44 = initializers_onnx_initializer_234 = None
        getitem_45 = com_microsoft__model_layers_6_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_235 = self.initializers.onnx_initializer_235
        model_layers_6_attn_k_proj_mat_mul = self.model_layers_6_attn_k_proj_MatMul(getitem_45, initializers_onnx_initializer_235);  getitem_45 = initializers_onnx_initializer_235 = None
        getitem_46 = com_microsoft__model_layers_6_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_236 = self.initializers.onnx_initializer_236
        model_layers_6_attn_v_proj_mat_mul = self.model_layers_6_attn_v_proj_MatMul(getitem_46, initializers_onnx_initializer_236);  getitem_46 = initializers_onnx_initializer_236 = None
        initializers_onnx_initializer_237 = self.initializers.onnx_initializer_237
        initializers_onnx_initializer_238 = self.initializers.onnx_initializer_238
        com_microsoft__model_layers_6_attn_q_rotary_rotary_embedding = self.com_microsoft__model_layers_6_attn_q_rotary_RotaryEmbedding(model_layers_6_attn_q_proj_mat_mul, input_3, initializers_onnx_initializer_237, initializers_onnx_initializer_238);  model_layers_6_attn_q_proj_mat_mul = initializers_onnx_initializer_237 = initializers_onnx_initializer_238 = None
        initializers_onnx_initializer_239 = self.initializers.onnx_initializer_239
        initializers_onnx_initializer_240 = self.initializers.onnx_initializer_240
        com_microsoft__model_layers_6_attn_k_rotary_rotary_embedding = self.com_microsoft__model_layers_6_attn_k_rotary_RotaryEmbedding(model_layers_6_attn_k_proj_mat_mul, input_3, initializers_onnx_initializer_239, initializers_onnx_initializer_240);  model_layers_6_attn_k_proj_mat_mul = initializers_onnx_initializer_239 = initializers_onnx_initializer_240 = None
        initializers_onnx_initializer_241 = self.initializers.onnx_initializer_241
        model_layers_6_attn_v_proj_repeat_kv_reshape_1 = self.model_layers_6_attn_v_proj_repeat_kv_Reshape_1(model_layers_6_attn_v_proj_mat_mul, initializers_onnx_initializer_241);  model_layers_6_attn_v_proj_mat_mul = initializers_onnx_initializer_241 = None
        initializers_onnx_initializer_242 = self.initializers.onnx_initializer_242
        model_layers_6_attn_k_proj_repeat_kv_reshape_1 = self.model_layers_6_attn_k_proj_repeat_kv_Reshape_1(com_microsoft__model_layers_6_attn_k_rotary_rotary_embedding, initializers_onnx_initializer_242);  com_microsoft__model_layers_6_attn_k_rotary_rotary_embedding = initializers_onnx_initializer_242 = None
        model_layers_6_attn_v_proj_repeat_kv_transpose_1 = self.model_layers_6_attn_v_proj_repeat_kv_Transpose_1(model_layers_6_attn_v_proj_repeat_kv_reshape_1);  model_layers_6_attn_v_proj_repeat_kv_reshape_1 = None
        model_layers_6_attn_k_proj_repeat_kv_transpose_1 = self.model_layers_6_attn_k_proj_repeat_kv_Transpose_1(model_layers_6_attn_k_proj_repeat_kv_reshape_1);  model_layers_6_attn_k_proj_repeat_kv_reshape_1 = None
        model_layers_6_attn_v_proj_repeat_kv_concat_1 = self.model_layers_6_attn_v_proj_repeat_kv_Concat_1(input_17, model_layers_6_attn_v_proj_repeat_kv_transpose_1);  input_17 = model_layers_6_attn_v_proj_repeat_kv_transpose_1 = None
        model_layers_6_attn_k_proj_repeat_kv_concat_1 = self.model_layers_6_attn_k_proj_repeat_kv_Concat_1(input_16, model_layers_6_attn_k_proj_repeat_kv_transpose_1);  input_16 = model_layers_6_attn_k_proj_repeat_kv_transpose_1 = None
        model_layers_6_attn_v_proj_repeat_kv_shape_1 = self.model_layers_6_attn_v_proj_repeat_kv_Shape_1(model_layers_6_attn_v_proj_repeat_kv_concat_1)
        model_layers_6_attn_v_proj_repeat_kv_unsqueeze_5 = self.model_layers_6_attn_v_proj_repeat_kv_Unsqueeze_5(model_layers_6_attn_v_proj_repeat_kv_concat_1)
        model_layers_6_attn_k_proj_repeat_kv_shape_1 = self.model_layers_6_attn_k_proj_repeat_kv_Shape_1(model_layers_6_attn_k_proj_repeat_kv_concat_1)
        model_layers_6_attn_k_proj_repeat_kv_unsqueeze_5 = self.model_layers_6_attn_k_proj_repeat_kv_Unsqueeze_5(model_layers_6_attn_k_proj_repeat_kv_concat_1)
        initializers_onnx_initializer_243 = self.initializers.onnx_initializer_243
        model_layers_6_attn_v_proj_repeat_kv_gather_1 = self.model_layers_6_attn_v_proj_repeat_kv_Gather_1(model_layers_6_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_243);  initializers_onnx_initializer_243 = None
        initializers_onnx_initializer_244 = self.initializers.onnx_initializer_244
        model_layers_6_attn_v_proj_repeat_kv_gather_3 = self.model_layers_6_attn_v_proj_repeat_kv_Gather_3(model_layers_6_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_244);  model_layers_6_attn_v_proj_repeat_kv_shape_1 = initializers_onnx_initializer_244 = None
        initializers_onnx_initializer_245 = self.initializers.onnx_initializer_245
        model_layers_6_attn_k_proj_repeat_kv_gather_1 = self.model_layers_6_attn_k_proj_repeat_kv_Gather_1(model_layers_6_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_245);  initializers_onnx_initializer_245 = None
        initializers_onnx_initializer_246 = self.initializers.onnx_initializer_246
        model_layers_6_attn_k_proj_repeat_kv_gather_3 = self.model_layers_6_attn_k_proj_repeat_kv_Gather_3(model_layers_6_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_246);  model_layers_6_attn_k_proj_repeat_kv_shape_1 = initializers_onnx_initializer_246 = None
        model_layers_6_attn_v_proj_repeat_kv_unsqueeze_1 = self.model_layers_6_attn_v_proj_repeat_kv_Unsqueeze_1(model_layers_6_attn_v_proj_repeat_kv_gather_1);  model_layers_6_attn_v_proj_repeat_kv_gather_1 = None
        model_layers_6_attn_v_proj_repeat_kv_unsqueeze_3 = self.model_layers_6_attn_v_proj_repeat_kv_Unsqueeze_3(model_layers_6_attn_v_proj_repeat_kv_gather_3);  model_layers_6_attn_v_proj_repeat_kv_gather_3 = None
        model_layers_6_attn_k_proj_repeat_kv_unsqueeze_1 = self.model_layers_6_attn_k_proj_repeat_kv_Unsqueeze_1(model_layers_6_attn_k_proj_repeat_kv_gather_1);  model_layers_6_attn_k_proj_repeat_kv_gather_1 = None
        model_layers_6_attn_k_proj_repeat_kv_unsqueeze_3 = self.model_layers_6_attn_k_proj_repeat_kv_Unsqueeze_3(model_layers_6_attn_k_proj_repeat_kv_gather_3);  model_layers_6_attn_k_proj_repeat_kv_gather_3 = None
        initializers_onnx_initializer_247 = self.initializers.onnx_initializer_247
        initializers_onnx_initializer_248 = self.initializers.onnx_initializer_248
        initializers_onnx_initializer_249 = self.initializers.onnx_initializer_249
        model_layers_6_attn_v_proj_repeat_kv_concat_2 = self.model_layers_6_attn_v_proj_repeat_kv_Concat_2(model_layers_6_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_247, initializers_onnx_initializer_248, model_layers_6_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_249);  initializers_onnx_initializer_247 = initializers_onnx_initializer_248 = initializers_onnx_initializer_249 = None
        initializers_onnx_initializer_250 = self.initializers.onnx_initializer_250
        initializers_onnx_initializer_251 = self.initializers.onnx_initializer_251
        model_layers_6_attn_v_proj_repeat_kv_concat_3 = self.model_layers_6_attn_v_proj_repeat_kv_Concat_3(model_layers_6_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_250, model_layers_6_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_251);  model_layers_6_attn_v_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_250 = model_layers_6_attn_v_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_251 = None
        initializers_onnx_initializer_252 = self.initializers.onnx_initializer_252
        initializers_onnx_initializer_253 = self.initializers.onnx_initializer_253
        initializers_onnx_initializer_254 = self.initializers.onnx_initializer_254
        model_layers_6_attn_k_proj_repeat_kv_concat_2 = self.model_layers_6_attn_k_proj_repeat_kv_Concat_2(model_layers_6_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_252, initializers_onnx_initializer_253, model_layers_6_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_254);  initializers_onnx_initializer_252 = initializers_onnx_initializer_253 = initializers_onnx_initializer_254 = None
        initializers_onnx_initializer_255 = self.initializers.onnx_initializer_255
        initializers_onnx_initializer_256 = self.initializers.onnx_initializer_256
        model_layers_6_attn_k_proj_repeat_kv_concat_3 = self.model_layers_6_attn_k_proj_repeat_kv_Concat_3(model_layers_6_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_255, model_layers_6_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_256);  model_layers_6_attn_k_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_255 = model_layers_6_attn_k_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_256 = None
        initializers_onnx_initializer_257 = self.initializers.onnx_initializer_257
        model_layers_6_attn_v_proj_repeat_kv_equal = self.model_layers_6_attn_v_proj_repeat_kv_Equal(model_layers_6_attn_v_proj_repeat_kv_concat_2, initializers_onnx_initializer_257);  initializers_onnx_initializer_257 = None
        initializers_onnx_initializer_258 = self.initializers.onnx_initializer_258
        model_layers_6_attn_k_proj_repeat_kv_equal = self.model_layers_6_attn_k_proj_repeat_kv_Equal(model_layers_6_attn_k_proj_repeat_kv_concat_2, initializers_onnx_initializer_258);  initializers_onnx_initializer_258 = None
        initializers_onnx_initializer_259 = self.initializers.onnx_initializer_259
        model_layers_6_attn_v_proj_repeat_kv_where = self.model_layers_6_attn_v_proj_repeat_kv_Where(model_layers_6_attn_v_proj_repeat_kv_equal, initializers_onnx_initializer_259, model_layers_6_attn_v_proj_repeat_kv_concat_2);  model_layers_6_attn_v_proj_repeat_kv_equal = initializers_onnx_initializer_259 = model_layers_6_attn_v_proj_repeat_kv_concat_2 = None
        initializers_onnx_initializer_260 = self.initializers.onnx_initializer_260
        model_layers_6_attn_k_proj_repeat_kv_where = self.model_layers_6_attn_k_proj_repeat_kv_Where(model_layers_6_attn_k_proj_repeat_kv_equal, initializers_onnx_initializer_260, model_layers_6_attn_k_proj_repeat_kv_concat_2);  model_layers_6_attn_k_proj_repeat_kv_equal = initializers_onnx_initializer_260 = model_layers_6_attn_k_proj_repeat_kv_concat_2 = None
        model_layers_6_attn_v_proj_repeat_kv_expand = self.model_layers_6_attn_v_proj_repeat_kv_Expand(model_layers_6_attn_v_proj_repeat_kv_unsqueeze_5, model_layers_6_attn_v_proj_repeat_kv_where);  model_layers_6_attn_v_proj_repeat_kv_unsqueeze_5 = model_layers_6_attn_v_proj_repeat_kv_where = None
        model_layers_6_attn_k_proj_repeat_kv_expand = self.model_layers_6_attn_k_proj_repeat_kv_Expand(model_layers_6_attn_k_proj_repeat_kv_unsqueeze_5, model_layers_6_attn_k_proj_repeat_kv_where);  model_layers_6_attn_k_proj_repeat_kv_unsqueeze_5 = model_layers_6_attn_k_proj_repeat_kv_where = None
        model_layers_6_attn_v_proj_repeat_kv_reshape_3 = self.model_layers_6_attn_v_proj_repeat_kv_Reshape_3(model_layers_6_attn_v_proj_repeat_kv_expand, model_layers_6_attn_v_proj_repeat_kv_concat_3);  model_layers_6_attn_v_proj_repeat_kv_expand = model_layers_6_attn_v_proj_repeat_kv_concat_3 = None
        model_layers_6_attn_k_proj_repeat_kv_reshape_3 = self.model_layers_6_attn_k_proj_repeat_kv_Reshape_3(model_layers_6_attn_k_proj_repeat_kv_expand, model_layers_6_attn_k_proj_repeat_kv_concat_3);  model_layers_6_attn_k_proj_repeat_kv_expand = model_layers_6_attn_k_proj_repeat_kv_concat_3 = None
        model_layers_6_attn_v_proj_repeat_kv_transpose_2 = self.model_layers_6_attn_v_proj_repeat_kv_Transpose_2(model_layers_6_attn_v_proj_repeat_kv_reshape_3);  model_layers_6_attn_v_proj_repeat_kv_reshape_3 = None
        model_layers_6_attn_k_proj_repeat_kv_transpose_2 = self.model_layers_6_attn_k_proj_repeat_kv_Transpose_2(model_layers_6_attn_k_proj_repeat_kv_reshape_3);  model_layers_6_attn_k_proj_repeat_kv_reshape_3 = None
        initializers_onnx_initializer_261 = self.initializers.onnx_initializer_261
        model_layers_6_attn_v_proj_repeat_kv_reshape_4 = self.model_layers_6_attn_v_proj_repeat_kv_Reshape_4(model_layers_6_attn_v_proj_repeat_kv_transpose_2, initializers_onnx_initializer_261);  model_layers_6_attn_v_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_261 = None
        initializers_onnx_initializer_262 = self.initializers.onnx_initializer_262
        model_layers_6_attn_k_proj_repeat_kv_reshape_4 = self.model_layers_6_attn_k_proj_repeat_kv_Reshape_4(model_layers_6_attn_k_proj_repeat_kv_transpose_2, initializers_onnx_initializer_262);  model_layers_6_attn_k_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_262 = None
        com_microsoft__model_layers_6_attn_multi_head_attention = self.com_microsoft__model_layers_6_attn_MultiHeadAttention(com_microsoft__model_layers_6_attn_q_rotary_rotary_embedding, model_layers_6_attn_k_proj_repeat_kv_reshape_4, model_layers_6_attn_v_proj_repeat_kv_reshape_4, attention_mask = model_attn_mask_reformat_tile);  com_microsoft__model_layers_6_attn_q_rotary_rotary_embedding = model_layers_6_attn_k_proj_repeat_kv_reshape_4 = model_layers_6_attn_v_proj_repeat_kv_reshape_4 = None
        getitem_47 = com_microsoft__model_layers_6_attn_multi_head_attention[0];  com_microsoft__model_layers_6_attn_multi_head_attention = None
        initializers_onnx_initializer_263 = self.initializers.onnx_initializer_263
        model_layers_6_attn_o_proj_mat_mul = self.model_layers_6_attn_o_proj_MatMul(getitem_47, initializers_onnx_initializer_263);  getitem_47 = initializers_onnx_initializer_263 = None
        getitem_48 = com_microsoft__model_layers_6_input_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_6_input_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_264 = self.initializers.onnx_initializer_264
        com_microsoft__model_layers_6_post_attention_layernorm_skip_layer_norm = self.com_microsoft__model_layers_6_post_attention_layernorm_SkipLayerNorm(getitem_48, model_layers_6_attn_o_proj_mat_mul, initializers_onnx_initializer_264);  getitem_48 = model_layers_6_attn_o_proj_mat_mul = initializers_onnx_initializer_264 = None
        getitem_49 = com_microsoft__model_layers_6_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_265 = self.initializers.onnx_initializer_265
        model_layers_6_mlp_gate_proj_mat_mul = self.model_layers_6_mlp_gate_proj_MatMul(getitem_49, initializers_onnx_initializer_265);  getitem_49 = initializers_onnx_initializer_265 = None
        getitem_50 = com_microsoft__model_layers_6_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_266 = self.initializers.onnx_initializer_266
        model_layers_6_mlp_up_proj_mat_mul = self.model_layers_6_mlp_up_proj_MatMul(getitem_50, initializers_onnx_initializer_266);  getitem_50 = initializers_onnx_initializer_266 = None
        model_layers_6_mlp_act_fn_sigmoid = self.model_layers_6_mlp_act_fn_Sigmoid(model_layers_6_mlp_gate_proj_mat_mul)
        model_layers_6_mlp_act_fn_mul = self.model_layers_6_mlp_act_fn_Mul(model_layers_6_mlp_gate_proj_mat_mul, model_layers_6_mlp_act_fn_sigmoid);  model_layers_6_mlp_gate_proj_mat_mul = model_layers_6_mlp_act_fn_sigmoid = None
        model_layers_6_mlp_mul = self.model_layers_6_mlp_Mul(model_layers_6_mlp_act_fn_mul, model_layers_6_mlp_up_proj_mat_mul);  model_layers_6_mlp_act_fn_mul = model_layers_6_mlp_up_proj_mat_mul = None
        initializers_onnx_initializer_267 = self.initializers.onnx_initializer_267
        model_layers_6_mlp_down_proj_mat_mul = self.model_layers_6_mlp_down_proj_MatMul(model_layers_6_mlp_mul, initializers_onnx_initializer_267);  model_layers_6_mlp_mul = initializers_onnx_initializer_267 = None
        getitem_51 = com_microsoft__model_layers_6_post_attention_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_6_post_attention_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_268 = self.initializers.onnx_initializer_268
        com_microsoft__model_layers_7_input_layernorm_skip_layer_norm = self.com_microsoft__model_layers_7_input_layernorm_SkipLayerNorm(getitem_51, model_layers_6_mlp_down_proj_mat_mul, initializers_onnx_initializer_268);  getitem_51 = model_layers_6_mlp_down_proj_mat_mul = initializers_onnx_initializer_268 = None
        getitem_52 = com_microsoft__model_layers_7_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_269 = self.initializers.onnx_initializer_269
        model_layers_7_attn_q_proj_mat_mul = self.model_layers_7_attn_q_proj_MatMul(getitem_52, initializers_onnx_initializer_269);  getitem_52 = initializers_onnx_initializer_269 = None
        getitem_53 = com_microsoft__model_layers_7_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_270 = self.initializers.onnx_initializer_270
        model_layers_7_attn_k_proj_mat_mul = self.model_layers_7_attn_k_proj_MatMul(getitem_53, initializers_onnx_initializer_270);  getitem_53 = initializers_onnx_initializer_270 = None
        getitem_54 = com_microsoft__model_layers_7_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_271 = self.initializers.onnx_initializer_271
        model_layers_7_attn_v_proj_mat_mul = self.model_layers_7_attn_v_proj_MatMul(getitem_54, initializers_onnx_initializer_271);  getitem_54 = initializers_onnx_initializer_271 = None
        initializers_onnx_initializer_272 = self.initializers.onnx_initializer_272
        initializers_onnx_initializer_273 = self.initializers.onnx_initializer_273
        com_microsoft__model_layers_7_attn_q_rotary_rotary_embedding = self.com_microsoft__model_layers_7_attn_q_rotary_RotaryEmbedding(model_layers_7_attn_q_proj_mat_mul, input_3, initializers_onnx_initializer_272, initializers_onnx_initializer_273);  model_layers_7_attn_q_proj_mat_mul = initializers_onnx_initializer_272 = initializers_onnx_initializer_273 = None
        initializers_onnx_initializer_274 = self.initializers.onnx_initializer_274
        initializers_onnx_initializer_275 = self.initializers.onnx_initializer_275
        com_microsoft__model_layers_7_attn_k_rotary_rotary_embedding = self.com_microsoft__model_layers_7_attn_k_rotary_RotaryEmbedding(model_layers_7_attn_k_proj_mat_mul, input_3, initializers_onnx_initializer_274, initializers_onnx_initializer_275);  model_layers_7_attn_k_proj_mat_mul = initializers_onnx_initializer_274 = initializers_onnx_initializer_275 = None
        initializers_onnx_initializer_276 = self.initializers.onnx_initializer_276
        model_layers_7_attn_v_proj_repeat_kv_reshape_1 = self.model_layers_7_attn_v_proj_repeat_kv_Reshape_1(model_layers_7_attn_v_proj_mat_mul, initializers_onnx_initializer_276);  model_layers_7_attn_v_proj_mat_mul = initializers_onnx_initializer_276 = None
        initializers_onnx_initializer_277 = self.initializers.onnx_initializer_277
        model_layers_7_attn_k_proj_repeat_kv_reshape_1 = self.model_layers_7_attn_k_proj_repeat_kv_Reshape_1(com_microsoft__model_layers_7_attn_k_rotary_rotary_embedding, initializers_onnx_initializer_277);  com_microsoft__model_layers_7_attn_k_rotary_rotary_embedding = initializers_onnx_initializer_277 = None
        model_layers_7_attn_v_proj_repeat_kv_transpose_1 = self.model_layers_7_attn_v_proj_repeat_kv_Transpose_1(model_layers_7_attn_v_proj_repeat_kv_reshape_1);  model_layers_7_attn_v_proj_repeat_kv_reshape_1 = None
        model_layers_7_attn_k_proj_repeat_kv_transpose_1 = self.model_layers_7_attn_k_proj_repeat_kv_Transpose_1(model_layers_7_attn_k_proj_repeat_kv_reshape_1);  model_layers_7_attn_k_proj_repeat_kv_reshape_1 = None
        model_layers_7_attn_v_proj_repeat_kv_concat_1 = self.model_layers_7_attn_v_proj_repeat_kv_Concat_1(input_19, model_layers_7_attn_v_proj_repeat_kv_transpose_1);  input_19 = model_layers_7_attn_v_proj_repeat_kv_transpose_1 = None
        model_layers_7_attn_k_proj_repeat_kv_concat_1 = self.model_layers_7_attn_k_proj_repeat_kv_Concat_1(input_18, model_layers_7_attn_k_proj_repeat_kv_transpose_1);  input_18 = model_layers_7_attn_k_proj_repeat_kv_transpose_1 = None
        model_layers_7_attn_v_proj_repeat_kv_shape_1 = self.model_layers_7_attn_v_proj_repeat_kv_Shape_1(model_layers_7_attn_v_proj_repeat_kv_concat_1)
        model_layers_7_attn_v_proj_repeat_kv_unsqueeze_5 = self.model_layers_7_attn_v_proj_repeat_kv_Unsqueeze_5(model_layers_7_attn_v_proj_repeat_kv_concat_1)
        model_layers_7_attn_k_proj_repeat_kv_shape_1 = self.model_layers_7_attn_k_proj_repeat_kv_Shape_1(model_layers_7_attn_k_proj_repeat_kv_concat_1)
        model_layers_7_attn_k_proj_repeat_kv_unsqueeze_5 = self.model_layers_7_attn_k_proj_repeat_kv_Unsqueeze_5(model_layers_7_attn_k_proj_repeat_kv_concat_1)
        initializers_onnx_initializer_278 = self.initializers.onnx_initializer_278
        model_layers_7_attn_v_proj_repeat_kv_gather_1 = self.model_layers_7_attn_v_proj_repeat_kv_Gather_1(model_layers_7_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_278);  initializers_onnx_initializer_278 = None
        initializers_onnx_initializer_279 = self.initializers.onnx_initializer_279
        model_layers_7_attn_v_proj_repeat_kv_gather_3 = self.model_layers_7_attn_v_proj_repeat_kv_Gather_3(model_layers_7_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_279);  model_layers_7_attn_v_proj_repeat_kv_shape_1 = initializers_onnx_initializer_279 = None
        initializers_onnx_initializer_280 = self.initializers.onnx_initializer_280
        model_layers_7_attn_k_proj_repeat_kv_gather_1 = self.model_layers_7_attn_k_proj_repeat_kv_Gather_1(model_layers_7_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_280);  initializers_onnx_initializer_280 = None
        initializers_onnx_initializer_281 = self.initializers.onnx_initializer_281
        model_layers_7_attn_k_proj_repeat_kv_gather_3 = self.model_layers_7_attn_k_proj_repeat_kv_Gather_3(model_layers_7_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_281);  model_layers_7_attn_k_proj_repeat_kv_shape_1 = initializers_onnx_initializer_281 = None
        model_layers_7_attn_v_proj_repeat_kv_unsqueeze_1 = self.model_layers_7_attn_v_proj_repeat_kv_Unsqueeze_1(model_layers_7_attn_v_proj_repeat_kv_gather_1);  model_layers_7_attn_v_proj_repeat_kv_gather_1 = None
        model_layers_7_attn_v_proj_repeat_kv_unsqueeze_3 = self.model_layers_7_attn_v_proj_repeat_kv_Unsqueeze_3(model_layers_7_attn_v_proj_repeat_kv_gather_3);  model_layers_7_attn_v_proj_repeat_kv_gather_3 = None
        model_layers_7_attn_k_proj_repeat_kv_unsqueeze_1 = self.model_layers_7_attn_k_proj_repeat_kv_Unsqueeze_1(model_layers_7_attn_k_proj_repeat_kv_gather_1);  model_layers_7_attn_k_proj_repeat_kv_gather_1 = None
        model_layers_7_attn_k_proj_repeat_kv_unsqueeze_3 = self.model_layers_7_attn_k_proj_repeat_kv_Unsqueeze_3(model_layers_7_attn_k_proj_repeat_kv_gather_3);  model_layers_7_attn_k_proj_repeat_kv_gather_3 = None
        initializers_onnx_initializer_282 = self.initializers.onnx_initializer_282
        initializers_onnx_initializer_283 = self.initializers.onnx_initializer_283
        initializers_onnx_initializer_284 = self.initializers.onnx_initializer_284
        model_layers_7_attn_v_proj_repeat_kv_concat_2 = self.model_layers_7_attn_v_proj_repeat_kv_Concat_2(model_layers_7_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_282, initializers_onnx_initializer_283, model_layers_7_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_284);  initializers_onnx_initializer_282 = initializers_onnx_initializer_283 = initializers_onnx_initializer_284 = None
        initializers_onnx_initializer_285 = self.initializers.onnx_initializer_285
        initializers_onnx_initializer_286 = self.initializers.onnx_initializer_286
        model_layers_7_attn_v_proj_repeat_kv_concat_3 = self.model_layers_7_attn_v_proj_repeat_kv_Concat_3(model_layers_7_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_285, model_layers_7_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_286);  model_layers_7_attn_v_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_285 = model_layers_7_attn_v_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_286 = None
        initializers_onnx_initializer_287 = self.initializers.onnx_initializer_287
        initializers_onnx_initializer_288 = self.initializers.onnx_initializer_288
        initializers_onnx_initializer_289 = self.initializers.onnx_initializer_289
        model_layers_7_attn_k_proj_repeat_kv_concat_2 = self.model_layers_7_attn_k_proj_repeat_kv_Concat_2(model_layers_7_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_287, initializers_onnx_initializer_288, model_layers_7_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_289);  initializers_onnx_initializer_287 = initializers_onnx_initializer_288 = initializers_onnx_initializer_289 = None
        initializers_onnx_initializer_290 = self.initializers.onnx_initializer_290
        initializers_onnx_initializer_291 = self.initializers.onnx_initializer_291
        model_layers_7_attn_k_proj_repeat_kv_concat_3 = self.model_layers_7_attn_k_proj_repeat_kv_Concat_3(model_layers_7_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_290, model_layers_7_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_291);  model_layers_7_attn_k_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_290 = model_layers_7_attn_k_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_291 = None
        initializers_onnx_initializer_292 = self.initializers.onnx_initializer_292
        model_layers_7_attn_v_proj_repeat_kv_equal = self.model_layers_7_attn_v_proj_repeat_kv_Equal(model_layers_7_attn_v_proj_repeat_kv_concat_2, initializers_onnx_initializer_292);  initializers_onnx_initializer_292 = None
        initializers_onnx_initializer_293 = self.initializers.onnx_initializer_293
        model_layers_7_attn_k_proj_repeat_kv_equal = self.model_layers_7_attn_k_proj_repeat_kv_Equal(model_layers_7_attn_k_proj_repeat_kv_concat_2, initializers_onnx_initializer_293);  initializers_onnx_initializer_293 = None
        initializers_onnx_initializer_294 = self.initializers.onnx_initializer_294
        model_layers_7_attn_v_proj_repeat_kv_where = self.model_layers_7_attn_v_proj_repeat_kv_Where(model_layers_7_attn_v_proj_repeat_kv_equal, initializers_onnx_initializer_294, model_layers_7_attn_v_proj_repeat_kv_concat_2);  model_layers_7_attn_v_proj_repeat_kv_equal = initializers_onnx_initializer_294 = model_layers_7_attn_v_proj_repeat_kv_concat_2 = None
        initializers_onnx_initializer_295 = self.initializers.onnx_initializer_295
        model_layers_7_attn_k_proj_repeat_kv_where = self.model_layers_7_attn_k_proj_repeat_kv_Where(model_layers_7_attn_k_proj_repeat_kv_equal, initializers_onnx_initializer_295, model_layers_7_attn_k_proj_repeat_kv_concat_2);  model_layers_7_attn_k_proj_repeat_kv_equal = initializers_onnx_initializer_295 = model_layers_7_attn_k_proj_repeat_kv_concat_2 = None
        model_layers_7_attn_v_proj_repeat_kv_expand = self.model_layers_7_attn_v_proj_repeat_kv_Expand(model_layers_7_attn_v_proj_repeat_kv_unsqueeze_5, model_layers_7_attn_v_proj_repeat_kv_where);  model_layers_7_attn_v_proj_repeat_kv_unsqueeze_5 = model_layers_7_attn_v_proj_repeat_kv_where = None
        model_layers_7_attn_k_proj_repeat_kv_expand = self.model_layers_7_attn_k_proj_repeat_kv_Expand(model_layers_7_attn_k_proj_repeat_kv_unsqueeze_5, model_layers_7_attn_k_proj_repeat_kv_where);  model_layers_7_attn_k_proj_repeat_kv_unsqueeze_5 = model_layers_7_attn_k_proj_repeat_kv_where = None
        model_layers_7_attn_v_proj_repeat_kv_reshape_3 = self.model_layers_7_attn_v_proj_repeat_kv_Reshape_3(model_layers_7_attn_v_proj_repeat_kv_expand, model_layers_7_attn_v_proj_repeat_kv_concat_3);  model_layers_7_attn_v_proj_repeat_kv_expand = model_layers_7_attn_v_proj_repeat_kv_concat_3 = None
        model_layers_7_attn_k_proj_repeat_kv_reshape_3 = self.model_layers_7_attn_k_proj_repeat_kv_Reshape_3(model_layers_7_attn_k_proj_repeat_kv_expand, model_layers_7_attn_k_proj_repeat_kv_concat_3);  model_layers_7_attn_k_proj_repeat_kv_expand = model_layers_7_attn_k_proj_repeat_kv_concat_3 = None
        model_layers_7_attn_v_proj_repeat_kv_transpose_2 = self.model_layers_7_attn_v_proj_repeat_kv_Transpose_2(model_layers_7_attn_v_proj_repeat_kv_reshape_3);  model_layers_7_attn_v_proj_repeat_kv_reshape_3 = None
        model_layers_7_attn_k_proj_repeat_kv_transpose_2 = self.model_layers_7_attn_k_proj_repeat_kv_Transpose_2(model_layers_7_attn_k_proj_repeat_kv_reshape_3);  model_layers_7_attn_k_proj_repeat_kv_reshape_3 = None
        initializers_onnx_initializer_296 = self.initializers.onnx_initializer_296
        model_layers_7_attn_v_proj_repeat_kv_reshape_4 = self.model_layers_7_attn_v_proj_repeat_kv_Reshape_4(model_layers_7_attn_v_proj_repeat_kv_transpose_2, initializers_onnx_initializer_296);  model_layers_7_attn_v_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_296 = None
        initializers_onnx_initializer_297 = self.initializers.onnx_initializer_297
        model_layers_7_attn_k_proj_repeat_kv_reshape_4 = self.model_layers_7_attn_k_proj_repeat_kv_Reshape_4(model_layers_7_attn_k_proj_repeat_kv_transpose_2, initializers_onnx_initializer_297);  model_layers_7_attn_k_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_297 = None
        com_microsoft__model_layers_7_attn_multi_head_attention = self.com_microsoft__model_layers_7_attn_MultiHeadAttention(com_microsoft__model_layers_7_attn_q_rotary_rotary_embedding, model_layers_7_attn_k_proj_repeat_kv_reshape_4, model_layers_7_attn_v_proj_repeat_kv_reshape_4, attention_mask = model_attn_mask_reformat_tile);  com_microsoft__model_layers_7_attn_q_rotary_rotary_embedding = model_layers_7_attn_k_proj_repeat_kv_reshape_4 = model_layers_7_attn_v_proj_repeat_kv_reshape_4 = None
        getitem_55 = com_microsoft__model_layers_7_attn_multi_head_attention[0];  com_microsoft__model_layers_7_attn_multi_head_attention = None
        initializers_onnx_initializer_298 = self.initializers.onnx_initializer_298
        model_layers_7_attn_o_proj_mat_mul = self.model_layers_7_attn_o_proj_MatMul(getitem_55, initializers_onnx_initializer_298);  getitem_55 = initializers_onnx_initializer_298 = None
        getitem_56 = com_microsoft__model_layers_7_input_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_7_input_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_299 = self.initializers.onnx_initializer_299
        com_microsoft__model_layers_7_post_attention_layernorm_skip_layer_norm = self.com_microsoft__model_layers_7_post_attention_layernorm_SkipLayerNorm(getitem_56, model_layers_7_attn_o_proj_mat_mul, initializers_onnx_initializer_299);  getitem_56 = model_layers_7_attn_o_proj_mat_mul = initializers_onnx_initializer_299 = None
        getitem_57 = com_microsoft__model_layers_7_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_300 = self.initializers.onnx_initializer_300
        model_layers_7_mlp_gate_proj_mat_mul = self.model_layers_7_mlp_gate_proj_MatMul(getitem_57, initializers_onnx_initializer_300);  getitem_57 = initializers_onnx_initializer_300 = None
        getitem_58 = com_microsoft__model_layers_7_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_301 = self.initializers.onnx_initializer_301
        model_layers_7_mlp_up_proj_mat_mul = self.model_layers_7_mlp_up_proj_MatMul(getitem_58, initializers_onnx_initializer_301);  getitem_58 = initializers_onnx_initializer_301 = None
        model_layers_7_mlp_act_fn_sigmoid = self.model_layers_7_mlp_act_fn_Sigmoid(model_layers_7_mlp_gate_proj_mat_mul)
        model_layers_7_mlp_act_fn_mul = self.model_layers_7_mlp_act_fn_Mul(model_layers_7_mlp_gate_proj_mat_mul, model_layers_7_mlp_act_fn_sigmoid);  model_layers_7_mlp_gate_proj_mat_mul = model_layers_7_mlp_act_fn_sigmoid = None
        model_layers_7_mlp_mul = self.model_layers_7_mlp_Mul(model_layers_7_mlp_act_fn_mul, model_layers_7_mlp_up_proj_mat_mul);  model_layers_7_mlp_act_fn_mul = model_layers_7_mlp_up_proj_mat_mul = None
        initializers_onnx_initializer_302 = self.initializers.onnx_initializer_302
        model_layers_7_mlp_down_proj_mat_mul = self.model_layers_7_mlp_down_proj_MatMul(model_layers_7_mlp_mul, initializers_onnx_initializer_302);  model_layers_7_mlp_mul = initializers_onnx_initializer_302 = None
        getitem_59 = com_microsoft__model_layers_7_post_attention_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_7_post_attention_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_303 = self.initializers.onnx_initializer_303
        com_microsoft__model_layers_8_input_layernorm_skip_layer_norm = self.com_microsoft__model_layers_8_input_layernorm_SkipLayerNorm(getitem_59, model_layers_7_mlp_down_proj_mat_mul, initializers_onnx_initializer_303);  getitem_59 = model_layers_7_mlp_down_proj_mat_mul = initializers_onnx_initializer_303 = None
        getitem_60 = com_microsoft__model_layers_8_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_304 = self.initializers.onnx_initializer_304
        model_layers_8_attn_q_proj_mat_mul = self.model_layers_8_attn_q_proj_MatMul(getitem_60, initializers_onnx_initializer_304);  getitem_60 = initializers_onnx_initializer_304 = None
        getitem_61 = com_microsoft__model_layers_8_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_305 = self.initializers.onnx_initializer_305
        model_layers_8_attn_k_proj_mat_mul = self.model_layers_8_attn_k_proj_MatMul(getitem_61, initializers_onnx_initializer_305);  getitem_61 = initializers_onnx_initializer_305 = None
        getitem_62 = com_microsoft__model_layers_8_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_306 = self.initializers.onnx_initializer_306
        model_layers_8_attn_v_proj_mat_mul = self.model_layers_8_attn_v_proj_MatMul(getitem_62, initializers_onnx_initializer_306);  getitem_62 = initializers_onnx_initializer_306 = None
        initializers_onnx_initializer_307 = self.initializers.onnx_initializer_307
        initializers_onnx_initializer_308 = self.initializers.onnx_initializer_308
        com_microsoft__model_layers_8_attn_q_rotary_rotary_embedding = self.com_microsoft__model_layers_8_attn_q_rotary_RotaryEmbedding(model_layers_8_attn_q_proj_mat_mul, input_3, initializers_onnx_initializer_307, initializers_onnx_initializer_308);  model_layers_8_attn_q_proj_mat_mul = initializers_onnx_initializer_307 = initializers_onnx_initializer_308 = None
        initializers_onnx_initializer_309 = self.initializers.onnx_initializer_309
        initializers_onnx_initializer_310 = self.initializers.onnx_initializer_310
        com_microsoft__model_layers_8_attn_k_rotary_rotary_embedding = self.com_microsoft__model_layers_8_attn_k_rotary_RotaryEmbedding(model_layers_8_attn_k_proj_mat_mul, input_3, initializers_onnx_initializer_309, initializers_onnx_initializer_310);  model_layers_8_attn_k_proj_mat_mul = initializers_onnx_initializer_309 = initializers_onnx_initializer_310 = None
        initializers_onnx_initializer_311 = self.initializers.onnx_initializer_311
        model_layers_8_attn_v_proj_repeat_kv_reshape_1 = self.model_layers_8_attn_v_proj_repeat_kv_Reshape_1(model_layers_8_attn_v_proj_mat_mul, initializers_onnx_initializer_311);  model_layers_8_attn_v_proj_mat_mul = initializers_onnx_initializer_311 = None
        initializers_onnx_initializer_312 = self.initializers.onnx_initializer_312
        model_layers_8_attn_k_proj_repeat_kv_reshape_1 = self.model_layers_8_attn_k_proj_repeat_kv_Reshape_1(com_microsoft__model_layers_8_attn_k_rotary_rotary_embedding, initializers_onnx_initializer_312);  com_microsoft__model_layers_8_attn_k_rotary_rotary_embedding = initializers_onnx_initializer_312 = None
        model_layers_8_attn_v_proj_repeat_kv_transpose_1 = self.model_layers_8_attn_v_proj_repeat_kv_Transpose_1(model_layers_8_attn_v_proj_repeat_kv_reshape_1);  model_layers_8_attn_v_proj_repeat_kv_reshape_1 = None
        model_layers_8_attn_k_proj_repeat_kv_transpose_1 = self.model_layers_8_attn_k_proj_repeat_kv_Transpose_1(model_layers_8_attn_k_proj_repeat_kv_reshape_1);  model_layers_8_attn_k_proj_repeat_kv_reshape_1 = None
        model_layers_8_attn_v_proj_repeat_kv_concat_1 = self.model_layers_8_attn_v_proj_repeat_kv_Concat_1(input_21, model_layers_8_attn_v_proj_repeat_kv_transpose_1);  input_21 = model_layers_8_attn_v_proj_repeat_kv_transpose_1 = None
        model_layers_8_attn_k_proj_repeat_kv_concat_1 = self.model_layers_8_attn_k_proj_repeat_kv_Concat_1(input_20, model_layers_8_attn_k_proj_repeat_kv_transpose_1);  input_20 = model_layers_8_attn_k_proj_repeat_kv_transpose_1 = None
        model_layers_8_attn_v_proj_repeat_kv_shape_1 = self.model_layers_8_attn_v_proj_repeat_kv_Shape_1(model_layers_8_attn_v_proj_repeat_kv_concat_1)
        model_layers_8_attn_v_proj_repeat_kv_unsqueeze_5 = self.model_layers_8_attn_v_proj_repeat_kv_Unsqueeze_5(model_layers_8_attn_v_proj_repeat_kv_concat_1)
        model_layers_8_attn_k_proj_repeat_kv_shape_1 = self.model_layers_8_attn_k_proj_repeat_kv_Shape_1(model_layers_8_attn_k_proj_repeat_kv_concat_1)
        model_layers_8_attn_k_proj_repeat_kv_unsqueeze_5 = self.model_layers_8_attn_k_proj_repeat_kv_Unsqueeze_5(model_layers_8_attn_k_proj_repeat_kv_concat_1)
        initializers_onnx_initializer_313 = self.initializers.onnx_initializer_313
        model_layers_8_attn_v_proj_repeat_kv_gather_1 = self.model_layers_8_attn_v_proj_repeat_kv_Gather_1(model_layers_8_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_313);  initializers_onnx_initializer_313 = None
        initializers_onnx_initializer_314 = self.initializers.onnx_initializer_314
        model_layers_8_attn_v_proj_repeat_kv_gather_3 = self.model_layers_8_attn_v_proj_repeat_kv_Gather_3(model_layers_8_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_314);  model_layers_8_attn_v_proj_repeat_kv_shape_1 = initializers_onnx_initializer_314 = None
        initializers_onnx_initializer_315 = self.initializers.onnx_initializer_315
        model_layers_8_attn_k_proj_repeat_kv_gather_1 = self.model_layers_8_attn_k_proj_repeat_kv_Gather_1(model_layers_8_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_315);  initializers_onnx_initializer_315 = None
        initializers_onnx_initializer_316 = self.initializers.onnx_initializer_316
        model_layers_8_attn_k_proj_repeat_kv_gather_3 = self.model_layers_8_attn_k_proj_repeat_kv_Gather_3(model_layers_8_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_316);  model_layers_8_attn_k_proj_repeat_kv_shape_1 = initializers_onnx_initializer_316 = None
        model_layers_8_attn_v_proj_repeat_kv_unsqueeze_1 = self.model_layers_8_attn_v_proj_repeat_kv_Unsqueeze_1(model_layers_8_attn_v_proj_repeat_kv_gather_1);  model_layers_8_attn_v_proj_repeat_kv_gather_1 = None
        model_layers_8_attn_v_proj_repeat_kv_unsqueeze_3 = self.model_layers_8_attn_v_proj_repeat_kv_Unsqueeze_3(model_layers_8_attn_v_proj_repeat_kv_gather_3);  model_layers_8_attn_v_proj_repeat_kv_gather_3 = None
        model_layers_8_attn_k_proj_repeat_kv_unsqueeze_1 = self.model_layers_8_attn_k_proj_repeat_kv_Unsqueeze_1(model_layers_8_attn_k_proj_repeat_kv_gather_1);  model_layers_8_attn_k_proj_repeat_kv_gather_1 = None
        model_layers_8_attn_k_proj_repeat_kv_unsqueeze_3 = self.model_layers_8_attn_k_proj_repeat_kv_Unsqueeze_3(model_layers_8_attn_k_proj_repeat_kv_gather_3);  model_layers_8_attn_k_proj_repeat_kv_gather_3 = None
        initializers_onnx_initializer_317 = self.initializers.onnx_initializer_317
        initializers_onnx_initializer_318 = self.initializers.onnx_initializer_318
        initializers_onnx_initializer_319 = self.initializers.onnx_initializer_319
        model_layers_8_attn_v_proj_repeat_kv_concat_2 = self.model_layers_8_attn_v_proj_repeat_kv_Concat_2(model_layers_8_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_317, initializers_onnx_initializer_318, model_layers_8_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_319);  initializers_onnx_initializer_317 = initializers_onnx_initializer_318 = initializers_onnx_initializer_319 = None
        initializers_onnx_initializer_320 = self.initializers.onnx_initializer_320
        initializers_onnx_initializer_321 = self.initializers.onnx_initializer_321
        model_layers_8_attn_v_proj_repeat_kv_concat_3 = self.model_layers_8_attn_v_proj_repeat_kv_Concat_3(model_layers_8_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_320, model_layers_8_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_321);  model_layers_8_attn_v_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_320 = model_layers_8_attn_v_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_321 = None
        initializers_onnx_initializer_322 = self.initializers.onnx_initializer_322
        initializers_onnx_initializer_323 = self.initializers.onnx_initializer_323
        initializers_onnx_initializer_324 = self.initializers.onnx_initializer_324
        model_layers_8_attn_k_proj_repeat_kv_concat_2 = self.model_layers_8_attn_k_proj_repeat_kv_Concat_2(model_layers_8_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_322, initializers_onnx_initializer_323, model_layers_8_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_324);  initializers_onnx_initializer_322 = initializers_onnx_initializer_323 = initializers_onnx_initializer_324 = None
        initializers_onnx_initializer_325 = self.initializers.onnx_initializer_325
        initializers_onnx_initializer_326 = self.initializers.onnx_initializer_326
        model_layers_8_attn_k_proj_repeat_kv_concat_3 = self.model_layers_8_attn_k_proj_repeat_kv_Concat_3(model_layers_8_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_325, model_layers_8_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_326);  model_layers_8_attn_k_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_325 = model_layers_8_attn_k_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_326 = None
        initializers_onnx_initializer_327 = self.initializers.onnx_initializer_327
        model_layers_8_attn_v_proj_repeat_kv_equal = self.model_layers_8_attn_v_proj_repeat_kv_Equal(model_layers_8_attn_v_proj_repeat_kv_concat_2, initializers_onnx_initializer_327);  initializers_onnx_initializer_327 = None
        initializers_onnx_initializer_328 = self.initializers.onnx_initializer_328
        model_layers_8_attn_k_proj_repeat_kv_equal = self.model_layers_8_attn_k_proj_repeat_kv_Equal(model_layers_8_attn_k_proj_repeat_kv_concat_2, initializers_onnx_initializer_328);  initializers_onnx_initializer_328 = None
        initializers_onnx_initializer_329 = self.initializers.onnx_initializer_329
        model_layers_8_attn_v_proj_repeat_kv_where = self.model_layers_8_attn_v_proj_repeat_kv_Where(model_layers_8_attn_v_proj_repeat_kv_equal, initializers_onnx_initializer_329, model_layers_8_attn_v_proj_repeat_kv_concat_2);  model_layers_8_attn_v_proj_repeat_kv_equal = initializers_onnx_initializer_329 = model_layers_8_attn_v_proj_repeat_kv_concat_2 = None
        initializers_onnx_initializer_330 = self.initializers.onnx_initializer_330
        model_layers_8_attn_k_proj_repeat_kv_where = self.model_layers_8_attn_k_proj_repeat_kv_Where(model_layers_8_attn_k_proj_repeat_kv_equal, initializers_onnx_initializer_330, model_layers_8_attn_k_proj_repeat_kv_concat_2);  model_layers_8_attn_k_proj_repeat_kv_equal = initializers_onnx_initializer_330 = model_layers_8_attn_k_proj_repeat_kv_concat_2 = None
        model_layers_8_attn_v_proj_repeat_kv_expand = self.model_layers_8_attn_v_proj_repeat_kv_Expand(model_layers_8_attn_v_proj_repeat_kv_unsqueeze_5, model_layers_8_attn_v_proj_repeat_kv_where);  model_layers_8_attn_v_proj_repeat_kv_unsqueeze_5 = model_layers_8_attn_v_proj_repeat_kv_where = None
        model_layers_8_attn_k_proj_repeat_kv_expand = self.model_layers_8_attn_k_proj_repeat_kv_Expand(model_layers_8_attn_k_proj_repeat_kv_unsqueeze_5, model_layers_8_attn_k_proj_repeat_kv_where);  model_layers_8_attn_k_proj_repeat_kv_unsqueeze_5 = model_layers_8_attn_k_proj_repeat_kv_where = None
        model_layers_8_attn_v_proj_repeat_kv_reshape_3 = self.model_layers_8_attn_v_proj_repeat_kv_Reshape_3(model_layers_8_attn_v_proj_repeat_kv_expand, model_layers_8_attn_v_proj_repeat_kv_concat_3);  model_layers_8_attn_v_proj_repeat_kv_expand = model_layers_8_attn_v_proj_repeat_kv_concat_3 = None
        model_layers_8_attn_k_proj_repeat_kv_reshape_3 = self.model_layers_8_attn_k_proj_repeat_kv_Reshape_3(model_layers_8_attn_k_proj_repeat_kv_expand, model_layers_8_attn_k_proj_repeat_kv_concat_3);  model_layers_8_attn_k_proj_repeat_kv_expand = model_layers_8_attn_k_proj_repeat_kv_concat_3 = None
        model_layers_8_attn_v_proj_repeat_kv_transpose_2 = self.model_layers_8_attn_v_proj_repeat_kv_Transpose_2(model_layers_8_attn_v_proj_repeat_kv_reshape_3);  model_layers_8_attn_v_proj_repeat_kv_reshape_3 = None
        model_layers_8_attn_k_proj_repeat_kv_transpose_2 = self.model_layers_8_attn_k_proj_repeat_kv_Transpose_2(model_layers_8_attn_k_proj_repeat_kv_reshape_3);  model_layers_8_attn_k_proj_repeat_kv_reshape_3 = None
        initializers_onnx_initializer_331 = self.initializers.onnx_initializer_331
        model_layers_8_attn_v_proj_repeat_kv_reshape_4 = self.model_layers_8_attn_v_proj_repeat_kv_Reshape_4(model_layers_8_attn_v_proj_repeat_kv_transpose_2, initializers_onnx_initializer_331);  model_layers_8_attn_v_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_331 = None
        initializers_onnx_initializer_332 = self.initializers.onnx_initializer_332
        model_layers_8_attn_k_proj_repeat_kv_reshape_4 = self.model_layers_8_attn_k_proj_repeat_kv_Reshape_4(model_layers_8_attn_k_proj_repeat_kv_transpose_2, initializers_onnx_initializer_332);  model_layers_8_attn_k_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_332 = None
        com_microsoft__model_layers_8_attn_multi_head_attention = self.com_microsoft__model_layers_8_attn_MultiHeadAttention(com_microsoft__model_layers_8_attn_q_rotary_rotary_embedding, model_layers_8_attn_k_proj_repeat_kv_reshape_4, model_layers_8_attn_v_proj_repeat_kv_reshape_4, attention_mask = model_attn_mask_reformat_tile);  com_microsoft__model_layers_8_attn_q_rotary_rotary_embedding = model_layers_8_attn_k_proj_repeat_kv_reshape_4 = model_layers_8_attn_v_proj_repeat_kv_reshape_4 = None
        getitem_63 = com_microsoft__model_layers_8_attn_multi_head_attention[0];  com_microsoft__model_layers_8_attn_multi_head_attention = None
        initializers_onnx_initializer_333 = self.initializers.onnx_initializer_333
        model_layers_8_attn_o_proj_mat_mul = self.model_layers_8_attn_o_proj_MatMul(getitem_63, initializers_onnx_initializer_333);  getitem_63 = initializers_onnx_initializer_333 = None
        getitem_64 = com_microsoft__model_layers_8_input_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_8_input_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_334 = self.initializers.onnx_initializer_334
        com_microsoft__model_layers_8_post_attention_layernorm_skip_layer_norm = self.com_microsoft__model_layers_8_post_attention_layernorm_SkipLayerNorm(getitem_64, model_layers_8_attn_o_proj_mat_mul, initializers_onnx_initializer_334);  getitem_64 = model_layers_8_attn_o_proj_mat_mul = initializers_onnx_initializer_334 = None
        getitem_65 = com_microsoft__model_layers_8_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_335 = self.initializers.onnx_initializer_335
        model_layers_8_mlp_gate_proj_mat_mul = self.model_layers_8_mlp_gate_proj_MatMul(getitem_65, initializers_onnx_initializer_335);  getitem_65 = initializers_onnx_initializer_335 = None
        getitem_66 = com_microsoft__model_layers_8_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_336 = self.initializers.onnx_initializer_336
        model_layers_8_mlp_up_proj_mat_mul = self.model_layers_8_mlp_up_proj_MatMul(getitem_66, initializers_onnx_initializer_336);  getitem_66 = initializers_onnx_initializer_336 = None
        model_layers_8_mlp_act_fn_sigmoid = self.model_layers_8_mlp_act_fn_Sigmoid(model_layers_8_mlp_gate_proj_mat_mul)
        model_layers_8_mlp_act_fn_mul = self.model_layers_8_mlp_act_fn_Mul(model_layers_8_mlp_gate_proj_mat_mul, model_layers_8_mlp_act_fn_sigmoid);  model_layers_8_mlp_gate_proj_mat_mul = model_layers_8_mlp_act_fn_sigmoid = None
        model_layers_8_mlp_mul = self.model_layers_8_mlp_Mul(model_layers_8_mlp_act_fn_mul, model_layers_8_mlp_up_proj_mat_mul);  model_layers_8_mlp_act_fn_mul = model_layers_8_mlp_up_proj_mat_mul = None
        initializers_onnx_initializer_337 = self.initializers.onnx_initializer_337
        model_layers_8_mlp_down_proj_mat_mul = self.model_layers_8_mlp_down_proj_MatMul(model_layers_8_mlp_mul, initializers_onnx_initializer_337);  model_layers_8_mlp_mul = initializers_onnx_initializer_337 = None
        getitem_67 = com_microsoft__model_layers_8_post_attention_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_8_post_attention_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_338 = self.initializers.onnx_initializer_338
        com_microsoft__model_layers_9_input_layernorm_skip_layer_norm = self.com_microsoft__model_layers_9_input_layernorm_SkipLayerNorm(getitem_67, model_layers_8_mlp_down_proj_mat_mul, initializers_onnx_initializer_338);  getitem_67 = model_layers_8_mlp_down_proj_mat_mul = initializers_onnx_initializer_338 = None
        getitem_68 = com_microsoft__model_layers_9_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_339 = self.initializers.onnx_initializer_339
        model_layers_9_attn_q_proj_mat_mul = self.model_layers_9_attn_q_proj_MatMul(getitem_68, initializers_onnx_initializer_339);  getitem_68 = initializers_onnx_initializer_339 = None
        getitem_69 = com_microsoft__model_layers_9_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_340 = self.initializers.onnx_initializer_340
        model_layers_9_attn_k_proj_mat_mul = self.model_layers_9_attn_k_proj_MatMul(getitem_69, initializers_onnx_initializer_340);  getitem_69 = initializers_onnx_initializer_340 = None
        getitem_70 = com_microsoft__model_layers_9_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_341 = self.initializers.onnx_initializer_341
        model_layers_9_attn_v_proj_mat_mul = self.model_layers_9_attn_v_proj_MatMul(getitem_70, initializers_onnx_initializer_341);  getitem_70 = initializers_onnx_initializer_341 = None
        initializers_onnx_initializer_342 = self.initializers.onnx_initializer_342
        initializers_onnx_initializer_343 = self.initializers.onnx_initializer_343
        com_microsoft__model_layers_9_attn_q_rotary_rotary_embedding = self.com_microsoft__model_layers_9_attn_q_rotary_RotaryEmbedding(model_layers_9_attn_q_proj_mat_mul, input_3, initializers_onnx_initializer_342, initializers_onnx_initializer_343);  model_layers_9_attn_q_proj_mat_mul = initializers_onnx_initializer_342 = initializers_onnx_initializer_343 = None
        initializers_onnx_initializer_344 = self.initializers.onnx_initializer_344
        initializers_onnx_initializer_345 = self.initializers.onnx_initializer_345
        com_microsoft__model_layers_9_attn_k_rotary_rotary_embedding = self.com_microsoft__model_layers_9_attn_k_rotary_RotaryEmbedding(model_layers_9_attn_k_proj_mat_mul, input_3, initializers_onnx_initializer_344, initializers_onnx_initializer_345);  model_layers_9_attn_k_proj_mat_mul = initializers_onnx_initializer_344 = initializers_onnx_initializer_345 = None
        initializers_onnx_initializer_346 = self.initializers.onnx_initializer_346
        model_layers_9_attn_v_proj_repeat_kv_reshape_1 = self.model_layers_9_attn_v_proj_repeat_kv_Reshape_1(model_layers_9_attn_v_proj_mat_mul, initializers_onnx_initializer_346);  model_layers_9_attn_v_proj_mat_mul = initializers_onnx_initializer_346 = None
        initializers_onnx_initializer_347 = self.initializers.onnx_initializer_347
        model_layers_9_attn_k_proj_repeat_kv_reshape_1 = self.model_layers_9_attn_k_proj_repeat_kv_Reshape_1(com_microsoft__model_layers_9_attn_k_rotary_rotary_embedding, initializers_onnx_initializer_347);  com_microsoft__model_layers_9_attn_k_rotary_rotary_embedding = initializers_onnx_initializer_347 = None
        model_layers_9_attn_v_proj_repeat_kv_transpose_1 = self.model_layers_9_attn_v_proj_repeat_kv_Transpose_1(model_layers_9_attn_v_proj_repeat_kv_reshape_1);  model_layers_9_attn_v_proj_repeat_kv_reshape_1 = None
        model_layers_9_attn_k_proj_repeat_kv_transpose_1 = self.model_layers_9_attn_k_proj_repeat_kv_Transpose_1(model_layers_9_attn_k_proj_repeat_kv_reshape_1);  model_layers_9_attn_k_proj_repeat_kv_reshape_1 = None
        model_layers_9_attn_v_proj_repeat_kv_concat_1 = self.model_layers_9_attn_v_proj_repeat_kv_Concat_1(input_23, model_layers_9_attn_v_proj_repeat_kv_transpose_1);  input_23 = model_layers_9_attn_v_proj_repeat_kv_transpose_1 = None
        model_layers_9_attn_k_proj_repeat_kv_concat_1 = self.model_layers_9_attn_k_proj_repeat_kv_Concat_1(input_22, model_layers_9_attn_k_proj_repeat_kv_transpose_1);  input_22 = model_layers_9_attn_k_proj_repeat_kv_transpose_1 = None
        model_layers_9_attn_v_proj_repeat_kv_shape_1 = self.model_layers_9_attn_v_proj_repeat_kv_Shape_1(model_layers_9_attn_v_proj_repeat_kv_concat_1)
        model_layers_9_attn_v_proj_repeat_kv_unsqueeze_5 = self.model_layers_9_attn_v_proj_repeat_kv_Unsqueeze_5(model_layers_9_attn_v_proj_repeat_kv_concat_1)
        model_layers_9_attn_k_proj_repeat_kv_shape_1 = self.model_layers_9_attn_k_proj_repeat_kv_Shape_1(model_layers_9_attn_k_proj_repeat_kv_concat_1)
        model_layers_9_attn_k_proj_repeat_kv_unsqueeze_5 = self.model_layers_9_attn_k_proj_repeat_kv_Unsqueeze_5(model_layers_9_attn_k_proj_repeat_kv_concat_1)
        initializers_onnx_initializer_348 = self.initializers.onnx_initializer_348
        model_layers_9_attn_v_proj_repeat_kv_gather_1 = self.model_layers_9_attn_v_proj_repeat_kv_Gather_1(model_layers_9_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_348);  initializers_onnx_initializer_348 = None
        initializers_onnx_initializer_349 = self.initializers.onnx_initializer_349
        model_layers_9_attn_v_proj_repeat_kv_gather_3 = self.model_layers_9_attn_v_proj_repeat_kv_Gather_3(model_layers_9_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_349);  model_layers_9_attn_v_proj_repeat_kv_shape_1 = initializers_onnx_initializer_349 = None
        initializers_onnx_initializer_350 = self.initializers.onnx_initializer_350
        model_layers_9_attn_k_proj_repeat_kv_gather_1 = self.model_layers_9_attn_k_proj_repeat_kv_Gather_1(model_layers_9_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_350);  initializers_onnx_initializer_350 = None
        initializers_onnx_initializer_351 = self.initializers.onnx_initializer_351
        model_layers_9_attn_k_proj_repeat_kv_gather_3 = self.model_layers_9_attn_k_proj_repeat_kv_Gather_3(model_layers_9_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_351);  model_layers_9_attn_k_proj_repeat_kv_shape_1 = initializers_onnx_initializer_351 = None
        model_layers_9_attn_v_proj_repeat_kv_unsqueeze_1 = self.model_layers_9_attn_v_proj_repeat_kv_Unsqueeze_1(model_layers_9_attn_v_proj_repeat_kv_gather_1);  model_layers_9_attn_v_proj_repeat_kv_gather_1 = None
        model_layers_9_attn_v_proj_repeat_kv_unsqueeze_3 = self.model_layers_9_attn_v_proj_repeat_kv_Unsqueeze_3(model_layers_9_attn_v_proj_repeat_kv_gather_3);  model_layers_9_attn_v_proj_repeat_kv_gather_3 = None
        model_layers_9_attn_k_proj_repeat_kv_unsqueeze_1 = self.model_layers_9_attn_k_proj_repeat_kv_Unsqueeze_1(model_layers_9_attn_k_proj_repeat_kv_gather_1);  model_layers_9_attn_k_proj_repeat_kv_gather_1 = None
        model_layers_9_attn_k_proj_repeat_kv_unsqueeze_3 = self.model_layers_9_attn_k_proj_repeat_kv_Unsqueeze_3(model_layers_9_attn_k_proj_repeat_kv_gather_3);  model_layers_9_attn_k_proj_repeat_kv_gather_3 = None
        initializers_onnx_initializer_352 = self.initializers.onnx_initializer_352
        initializers_onnx_initializer_353 = self.initializers.onnx_initializer_353
        initializers_onnx_initializer_354 = self.initializers.onnx_initializer_354
        model_layers_9_attn_v_proj_repeat_kv_concat_2 = self.model_layers_9_attn_v_proj_repeat_kv_Concat_2(model_layers_9_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_352, initializers_onnx_initializer_353, model_layers_9_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_354);  initializers_onnx_initializer_352 = initializers_onnx_initializer_353 = initializers_onnx_initializer_354 = None
        initializers_onnx_initializer_355 = self.initializers.onnx_initializer_355
        initializers_onnx_initializer_356 = self.initializers.onnx_initializer_356
        model_layers_9_attn_v_proj_repeat_kv_concat_3 = self.model_layers_9_attn_v_proj_repeat_kv_Concat_3(model_layers_9_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_355, model_layers_9_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_356);  model_layers_9_attn_v_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_355 = model_layers_9_attn_v_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_356 = None
        initializers_onnx_initializer_357 = self.initializers.onnx_initializer_357
        initializers_onnx_initializer_358 = self.initializers.onnx_initializer_358
        initializers_onnx_initializer_359 = self.initializers.onnx_initializer_359
        model_layers_9_attn_k_proj_repeat_kv_concat_2 = self.model_layers_9_attn_k_proj_repeat_kv_Concat_2(model_layers_9_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_357, initializers_onnx_initializer_358, model_layers_9_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_359);  initializers_onnx_initializer_357 = initializers_onnx_initializer_358 = initializers_onnx_initializer_359 = None
        initializers_onnx_initializer_360 = self.initializers.onnx_initializer_360
        initializers_onnx_initializer_361 = self.initializers.onnx_initializer_361
        model_layers_9_attn_k_proj_repeat_kv_concat_3 = self.model_layers_9_attn_k_proj_repeat_kv_Concat_3(model_layers_9_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_360, model_layers_9_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_361);  model_layers_9_attn_k_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_360 = model_layers_9_attn_k_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_361 = None
        initializers_onnx_initializer_362 = self.initializers.onnx_initializer_362
        model_layers_9_attn_v_proj_repeat_kv_equal = self.model_layers_9_attn_v_proj_repeat_kv_Equal(model_layers_9_attn_v_proj_repeat_kv_concat_2, initializers_onnx_initializer_362);  initializers_onnx_initializer_362 = None
        initializers_onnx_initializer_363 = self.initializers.onnx_initializer_363
        model_layers_9_attn_k_proj_repeat_kv_equal = self.model_layers_9_attn_k_proj_repeat_kv_Equal(model_layers_9_attn_k_proj_repeat_kv_concat_2, initializers_onnx_initializer_363);  initializers_onnx_initializer_363 = None
        initializers_onnx_initializer_364 = self.initializers.onnx_initializer_364
        model_layers_9_attn_v_proj_repeat_kv_where = self.model_layers_9_attn_v_proj_repeat_kv_Where(model_layers_9_attn_v_proj_repeat_kv_equal, initializers_onnx_initializer_364, model_layers_9_attn_v_proj_repeat_kv_concat_2);  model_layers_9_attn_v_proj_repeat_kv_equal = initializers_onnx_initializer_364 = model_layers_9_attn_v_proj_repeat_kv_concat_2 = None
        initializers_onnx_initializer_365 = self.initializers.onnx_initializer_365
        model_layers_9_attn_k_proj_repeat_kv_where = self.model_layers_9_attn_k_proj_repeat_kv_Where(model_layers_9_attn_k_proj_repeat_kv_equal, initializers_onnx_initializer_365, model_layers_9_attn_k_proj_repeat_kv_concat_2);  model_layers_9_attn_k_proj_repeat_kv_equal = initializers_onnx_initializer_365 = model_layers_9_attn_k_proj_repeat_kv_concat_2 = None
        model_layers_9_attn_v_proj_repeat_kv_expand = self.model_layers_9_attn_v_proj_repeat_kv_Expand(model_layers_9_attn_v_proj_repeat_kv_unsqueeze_5, model_layers_9_attn_v_proj_repeat_kv_where);  model_layers_9_attn_v_proj_repeat_kv_unsqueeze_5 = model_layers_9_attn_v_proj_repeat_kv_where = None
        model_layers_9_attn_k_proj_repeat_kv_expand = self.model_layers_9_attn_k_proj_repeat_kv_Expand(model_layers_9_attn_k_proj_repeat_kv_unsqueeze_5, model_layers_9_attn_k_proj_repeat_kv_where);  model_layers_9_attn_k_proj_repeat_kv_unsqueeze_5 = model_layers_9_attn_k_proj_repeat_kv_where = None
        model_layers_9_attn_v_proj_repeat_kv_reshape_3 = self.model_layers_9_attn_v_proj_repeat_kv_Reshape_3(model_layers_9_attn_v_proj_repeat_kv_expand, model_layers_9_attn_v_proj_repeat_kv_concat_3);  model_layers_9_attn_v_proj_repeat_kv_expand = model_layers_9_attn_v_proj_repeat_kv_concat_3 = None
        model_layers_9_attn_k_proj_repeat_kv_reshape_3 = self.model_layers_9_attn_k_proj_repeat_kv_Reshape_3(model_layers_9_attn_k_proj_repeat_kv_expand, model_layers_9_attn_k_proj_repeat_kv_concat_3);  model_layers_9_attn_k_proj_repeat_kv_expand = model_layers_9_attn_k_proj_repeat_kv_concat_3 = None
        model_layers_9_attn_v_proj_repeat_kv_transpose_2 = self.model_layers_9_attn_v_proj_repeat_kv_Transpose_2(model_layers_9_attn_v_proj_repeat_kv_reshape_3);  model_layers_9_attn_v_proj_repeat_kv_reshape_3 = None
        model_layers_9_attn_k_proj_repeat_kv_transpose_2 = self.model_layers_9_attn_k_proj_repeat_kv_Transpose_2(model_layers_9_attn_k_proj_repeat_kv_reshape_3);  model_layers_9_attn_k_proj_repeat_kv_reshape_3 = None
        initializers_onnx_initializer_366 = self.initializers.onnx_initializer_366
        model_layers_9_attn_v_proj_repeat_kv_reshape_4 = self.model_layers_9_attn_v_proj_repeat_kv_Reshape_4(model_layers_9_attn_v_proj_repeat_kv_transpose_2, initializers_onnx_initializer_366);  model_layers_9_attn_v_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_366 = None
        initializers_onnx_initializer_367 = self.initializers.onnx_initializer_367
        model_layers_9_attn_k_proj_repeat_kv_reshape_4 = self.model_layers_9_attn_k_proj_repeat_kv_Reshape_4(model_layers_9_attn_k_proj_repeat_kv_transpose_2, initializers_onnx_initializer_367);  model_layers_9_attn_k_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_367 = None
        com_microsoft__model_layers_9_attn_multi_head_attention = self.com_microsoft__model_layers_9_attn_MultiHeadAttention(com_microsoft__model_layers_9_attn_q_rotary_rotary_embedding, model_layers_9_attn_k_proj_repeat_kv_reshape_4, model_layers_9_attn_v_proj_repeat_kv_reshape_4, attention_mask = model_attn_mask_reformat_tile);  com_microsoft__model_layers_9_attn_q_rotary_rotary_embedding = model_layers_9_attn_k_proj_repeat_kv_reshape_4 = model_layers_9_attn_v_proj_repeat_kv_reshape_4 = None
        getitem_71 = com_microsoft__model_layers_9_attn_multi_head_attention[0];  com_microsoft__model_layers_9_attn_multi_head_attention = None
        initializers_onnx_initializer_368 = self.initializers.onnx_initializer_368
        model_layers_9_attn_o_proj_mat_mul = self.model_layers_9_attn_o_proj_MatMul(getitem_71, initializers_onnx_initializer_368);  getitem_71 = initializers_onnx_initializer_368 = None
        getitem_72 = com_microsoft__model_layers_9_input_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_9_input_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_369 = self.initializers.onnx_initializer_369
        com_microsoft__model_layers_9_post_attention_layernorm_skip_layer_norm = self.com_microsoft__model_layers_9_post_attention_layernorm_SkipLayerNorm(getitem_72, model_layers_9_attn_o_proj_mat_mul, initializers_onnx_initializer_369);  getitem_72 = model_layers_9_attn_o_proj_mat_mul = initializers_onnx_initializer_369 = None
        getitem_73 = com_microsoft__model_layers_9_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_370 = self.initializers.onnx_initializer_370
        model_layers_9_mlp_gate_proj_mat_mul = self.model_layers_9_mlp_gate_proj_MatMul(getitem_73, initializers_onnx_initializer_370);  getitem_73 = initializers_onnx_initializer_370 = None
        getitem_74 = com_microsoft__model_layers_9_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_371 = self.initializers.onnx_initializer_371
        model_layers_9_mlp_up_proj_mat_mul = self.model_layers_9_mlp_up_proj_MatMul(getitem_74, initializers_onnx_initializer_371);  getitem_74 = initializers_onnx_initializer_371 = None
        model_layers_9_mlp_act_fn_sigmoid = self.model_layers_9_mlp_act_fn_Sigmoid(model_layers_9_mlp_gate_proj_mat_mul)
        model_layers_9_mlp_act_fn_mul = self.model_layers_9_mlp_act_fn_Mul(model_layers_9_mlp_gate_proj_mat_mul, model_layers_9_mlp_act_fn_sigmoid);  model_layers_9_mlp_gate_proj_mat_mul = model_layers_9_mlp_act_fn_sigmoid = None
        model_layers_9_mlp_mul = self.model_layers_9_mlp_Mul(model_layers_9_mlp_act_fn_mul, model_layers_9_mlp_up_proj_mat_mul);  model_layers_9_mlp_act_fn_mul = model_layers_9_mlp_up_proj_mat_mul = None
        initializers_onnx_initializer_372 = self.initializers.onnx_initializer_372
        model_layers_9_mlp_down_proj_mat_mul = self.model_layers_9_mlp_down_proj_MatMul(model_layers_9_mlp_mul, initializers_onnx_initializer_372);  model_layers_9_mlp_mul = initializers_onnx_initializer_372 = None
        getitem_75 = com_microsoft__model_layers_9_post_attention_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_9_post_attention_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_373 = self.initializers.onnx_initializer_373
        com_microsoft__model_layers_10_input_layernorm_skip_layer_norm = self.com_microsoft__model_layers_10_input_layernorm_SkipLayerNorm(getitem_75, model_layers_9_mlp_down_proj_mat_mul, initializers_onnx_initializer_373);  getitem_75 = model_layers_9_mlp_down_proj_mat_mul = initializers_onnx_initializer_373 = None
        getitem_76 = com_microsoft__model_layers_10_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_374 = self.initializers.onnx_initializer_374
        model_layers_10_attn_q_proj_mat_mul = self.model_layers_10_attn_q_proj_MatMul(getitem_76, initializers_onnx_initializer_374);  getitem_76 = initializers_onnx_initializer_374 = None
        getitem_77 = com_microsoft__model_layers_10_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_375 = self.initializers.onnx_initializer_375
        model_layers_10_attn_k_proj_mat_mul = self.model_layers_10_attn_k_proj_MatMul(getitem_77, initializers_onnx_initializer_375);  getitem_77 = initializers_onnx_initializer_375 = None
        getitem_78 = com_microsoft__model_layers_10_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_376 = self.initializers.onnx_initializer_376
        model_layers_10_attn_v_proj_mat_mul = self.model_layers_10_attn_v_proj_MatMul(getitem_78, initializers_onnx_initializer_376);  getitem_78 = initializers_onnx_initializer_376 = None
        initializers_onnx_initializer_377 = self.initializers.onnx_initializer_377
        initializers_onnx_initializer_378 = self.initializers.onnx_initializer_378
        com_microsoft__model_layers_10_attn_q_rotary_rotary_embedding = self.com_microsoft__model_layers_10_attn_q_rotary_RotaryEmbedding(model_layers_10_attn_q_proj_mat_mul, input_3, initializers_onnx_initializer_377, initializers_onnx_initializer_378);  model_layers_10_attn_q_proj_mat_mul = initializers_onnx_initializer_377 = initializers_onnx_initializer_378 = None
        initializers_onnx_initializer_379 = self.initializers.onnx_initializer_379
        initializers_onnx_initializer_380 = self.initializers.onnx_initializer_380
        com_microsoft__model_layers_10_attn_k_rotary_rotary_embedding = self.com_microsoft__model_layers_10_attn_k_rotary_RotaryEmbedding(model_layers_10_attn_k_proj_mat_mul, input_3, initializers_onnx_initializer_379, initializers_onnx_initializer_380);  model_layers_10_attn_k_proj_mat_mul = initializers_onnx_initializer_379 = initializers_onnx_initializer_380 = None
        initializers_onnx_initializer_381 = self.initializers.onnx_initializer_381
        model_layers_10_attn_v_proj_repeat_kv_reshape_1 = self.model_layers_10_attn_v_proj_repeat_kv_Reshape_1(model_layers_10_attn_v_proj_mat_mul, initializers_onnx_initializer_381);  model_layers_10_attn_v_proj_mat_mul = initializers_onnx_initializer_381 = None
        initializers_onnx_initializer_382 = self.initializers.onnx_initializer_382
        model_layers_10_attn_k_proj_repeat_kv_reshape_1 = self.model_layers_10_attn_k_proj_repeat_kv_Reshape_1(com_microsoft__model_layers_10_attn_k_rotary_rotary_embedding, initializers_onnx_initializer_382);  com_microsoft__model_layers_10_attn_k_rotary_rotary_embedding = initializers_onnx_initializer_382 = None
        model_layers_10_attn_v_proj_repeat_kv_transpose_1 = self.model_layers_10_attn_v_proj_repeat_kv_Transpose_1(model_layers_10_attn_v_proj_repeat_kv_reshape_1);  model_layers_10_attn_v_proj_repeat_kv_reshape_1 = None
        model_layers_10_attn_k_proj_repeat_kv_transpose_1 = self.model_layers_10_attn_k_proj_repeat_kv_Transpose_1(model_layers_10_attn_k_proj_repeat_kv_reshape_1);  model_layers_10_attn_k_proj_repeat_kv_reshape_1 = None
        model_layers_10_attn_v_proj_repeat_kv_concat_1 = self.model_layers_10_attn_v_proj_repeat_kv_Concat_1(input_25, model_layers_10_attn_v_proj_repeat_kv_transpose_1);  input_25 = model_layers_10_attn_v_proj_repeat_kv_transpose_1 = None
        model_layers_10_attn_k_proj_repeat_kv_concat_1 = self.model_layers_10_attn_k_proj_repeat_kv_Concat_1(input_24, model_layers_10_attn_k_proj_repeat_kv_transpose_1);  input_24 = model_layers_10_attn_k_proj_repeat_kv_transpose_1 = None
        model_layers_10_attn_v_proj_repeat_kv_shape_1 = self.model_layers_10_attn_v_proj_repeat_kv_Shape_1(model_layers_10_attn_v_proj_repeat_kv_concat_1)
        model_layers_10_attn_v_proj_repeat_kv_unsqueeze_5 = self.model_layers_10_attn_v_proj_repeat_kv_Unsqueeze_5(model_layers_10_attn_v_proj_repeat_kv_concat_1)
        model_layers_10_attn_k_proj_repeat_kv_shape_1 = self.model_layers_10_attn_k_proj_repeat_kv_Shape_1(model_layers_10_attn_k_proj_repeat_kv_concat_1)
        model_layers_10_attn_k_proj_repeat_kv_unsqueeze_5 = self.model_layers_10_attn_k_proj_repeat_kv_Unsqueeze_5(model_layers_10_attn_k_proj_repeat_kv_concat_1)
        initializers_onnx_initializer_383 = self.initializers.onnx_initializer_383
        model_layers_10_attn_v_proj_repeat_kv_gather_1 = self.model_layers_10_attn_v_proj_repeat_kv_Gather_1(model_layers_10_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_383);  initializers_onnx_initializer_383 = None
        initializers_onnx_initializer_384 = self.initializers.onnx_initializer_384
        model_layers_10_attn_v_proj_repeat_kv_gather_3 = self.model_layers_10_attn_v_proj_repeat_kv_Gather_3(model_layers_10_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_384);  model_layers_10_attn_v_proj_repeat_kv_shape_1 = initializers_onnx_initializer_384 = None
        initializers_onnx_initializer_385 = self.initializers.onnx_initializer_385
        model_layers_10_attn_k_proj_repeat_kv_gather_1 = self.model_layers_10_attn_k_proj_repeat_kv_Gather_1(model_layers_10_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_385);  initializers_onnx_initializer_385 = None
        initializers_onnx_initializer_386 = self.initializers.onnx_initializer_386
        model_layers_10_attn_k_proj_repeat_kv_gather_3 = self.model_layers_10_attn_k_proj_repeat_kv_Gather_3(model_layers_10_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_386);  model_layers_10_attn_k_proj_repeat_kv_shape_1 = initializers_onnx_initializer_386 = None
        model_layers_10_attn_v_proj_repeat_kv_unsqueeze_1 = self.model_layers_10_attn_v_proj_repeat_kv_Unsqueeze_1(model_layers_10_attn_v_proj_repeat_kv_gather_1);  model_layers_10_attn_v_proj_repeat_kv_gather_1 = None
        model_layers_10_attn_v_proj_repeat_kv_unsqueeze_3 = self.model_layers_10_attn_v_proj_repeat_kv_Unsqueeze_3(model_layers_10_attn_v_proj_repeat_kv_gather_3);  model_layers_10_attn_v_proj_repeat_kv_gather_3 = None
        model_layers_10_attn_k_proj_repeat_kv_unsqueeze_1 = self.model_layers_10_attn_k_proj_repeat_kv_Unsqueeze_1(model_layers_10_attn_k_proj_repeat_kv_gather_1);  model_layers_10_attn_k_proj_repeat_kv_gather_1 = None
        model_layers_10_attn_k_proj_repeat_kv_unsqueeze_3 = self.model_layers_10_attn_k_proj_repeat_kv_Unsqueeze_3(model_layers_10_attn_k_proj_repeat_kv_gather_3);  model_layers_10_attn_k_proj_repeat_kv_gather_3 = None
        initializers_onnx_initializer_387 = self.initializers.onnx_initializer_387
        initializers_onnx_initializer_388 = self.initializers.onnx_initializer_388
        initializers_onnx_initializer_389 = self.initializers.onnx_initializer_389
        model_layers_10_attn_v_proj_repeat_kv_concat_2 = self.model_layers_10_attn_v_proj_repeat_kv_Concat_2(model_layers_10_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_387, initializers_onnx_initializer_388, model_layers_10_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_389);  initializers_onnx_initializer_387 = initializers_onnx_initializer_388 = initializers_onnx_initializer_389 = None
        initializers_onnx_initializer_390 = self.initializers.onnx_initializer_390
        initializers_onnx_initializer_391 = self.initializers.onnx_initializer_391
        model_layers_10_attn_v_proj_repeat_kv_concat_3 = self.model_layers_10_attn_v_proj_repeat_kv_Concat_3(model_layers_10_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_390, model_layers_10_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_391);  model_layers_10_attn_v_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_390 = model_layers_10_attn_v_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_391 = None
        initializers_onnx_initializer_392 = self.initializers.onnx_initializer_392
        initializers_onnx_initializer_393 = self.initializers.onnx_initializer_393
        initializers_onnx_initializer_394 = self.initializers.onnx_initializer_394
        model_layers_10_attn_k_proj_repeat_kv_concat_2 = self.model_layers_10_attn_k_proj_repeat_kv_Concat_2(model_layers_10_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_392, initializers_onnx_initializer_393, model_layers_10_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_394);  initializers_onnx_initializer_392 = initializers_onnx_initializer_393 = initializers_onnx_initializer_394 = None
        initializers_onnx_initializer_395 = self.initializers.onnx_initializer_395
        initializers_onnx_initializer_396 = self.initializers.onnx_initializer_396
        model_layers_10_attn_k_proj_repeat_kv_concat_3 = self.model_layers_10_attn_k_proj_repeat_kv_Concat_3(model_layers_10_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_395, model_layers_10_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_396);  model_layers_10_attn_k_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_395 = model_layers_10_attn_k_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_396 = None
        initializers_onnx_initializer_397 = self.initializers.onnx_initializer_397
        model_layers_10_attn_v_proj_repeat_kv_equal = self.model_layers_10_attn_v_proj_repeat_kv_Equal(model_layers_10_attn_v_proj_repeat_kv_concat_2, initializers_onnx_initializer_397);  initializers_onnx_initializer_397 = None
        initializers_onnx_initializer_398 = self.initializers.onnx_initializer_398
        model_layers_10_attn_k_proj_repeat_kv_equal = self.model_layers_10_attn_k_proj_repeat_kv_Equal(model_layers_10_attn_k_proj_repeat_kv_concat_2, initializers_onnx_initializer_398);  initializers_onnx_initializer_398 = None
        initializers_onnx_initializer_399 = self.initializers.onnx_initializer_399
        model_layers_10_attn_v_proj_repeat_kv_where = self.model_layers_10_attn_v_proj_repeat_kv_Where(model_layers_10_attn_v_proj_repeat_kv_equal, initializers_onnx_initializer_399, model_layers_10_attn_v_proj_repeat_kv_concat_2);  model_layers_10_attn_v_proj_repeat_kv_equal = initializers_onnx_initializer_399 = model_layers_10_attn_v_proj_repeat_kv_concat_2 = None
        initializers_onnx_initializer_400 = self.initializers.onnx_initializer_400
        model_layers_10_attn_k_proj_repeat_kv_where = self.model_layers_10_attn_k_proj_repeat_kv_Where(model_layers_10_attn_k_proj_repeat_kv_equal, initializers_onnx_initializer_400, model_layers_10_attn_k_proj_repeat_kv_concat_2);  model_layers_10_attn_k_proj_repeat_kv_equal = initializers_onnx_initializer_400 = model_layers_10_attn_k_proj_repeat_kv_concat_2 = None
        model_layers_10_attn_v_proj_repeat_kv_expand = self.model_layers_10_attn_v_proj_repeat_kv_Expand(model_layers_10_attn_v_proj_repeat_kv_unsqueeze_5, model_layers_10_attn_v_proj_repeat_kv_where);  model_layers_10_attn_v_proj_repeat_kv_unsqueeze_5 = model_layers_10_attn_v_proj_repeat_kv_where = None
        model_layers_10_attn_k_proj_repeat_kv_expand = self.model_layers_10_attn_k_proj_repeat_kv_Expand(model_layers_10_attn_k_proj_repeat_kv_unsqueeze_5, model_layers_10_attn_k_proj_repeat_kv_where);  model_layers_10_attn_k_proj_repeat_kv_unsqueeze_5 = model_layers_10_attn_k_proj_repeat_kv_where = None
        model_layers_10_attn_v_proj_repeat_kv_reshape_3 = self.model_layers_10_attn_v_proj_repeat_kv_Reshape_3(model_layers_10_attn_v_proj_repeat_kv_expand, model_layers_10_attn_v_proj_repeat_kv_concat_3);  model_layers_10_attn_v_proj_repeat_kv_expand = model_layers_10_attn_v_proj_repeat_kv_concat_3 = None
        model_layers_10_attn_k_proj_repeat_kv_reshape_3 = self.model_layers_10_attn_k_proj_repeat_kv_Reshape_3(model_layers_10_attn_k_proj_repeat_kv_expand, model_layers_10_attn_k_proj_repeat_kv_concat_3);  model_layers_10_attn_k_proj_repeat_kv_expand = model_layers_10_attn_k_proj_repeat_kv_concat_3 = None
        model_layers_10_attn_v_proj_repeat_kv_transpose_2 = self.model_layers_10_attn_v_proj_repeat_kv_Transpose_2(model_layers_10_attn_v_proj_repeat_kv_reshape_3);  model_layers_10_attn_v_proj_repeat_kv_reshape_3 = None
        model_layers_10_attn_k_proj_repeat_kv_transpose_2 = self.model_layers_10_attn_k_proj_repeat_kv_Transpose_2(model_layers_10_attn_k_proj_repeat_kv_reshape_3);  model_layers_10_attn_k_proj_repeat_kv_reshape_3 = None
        initializers_onnx_initializer_401 = self.initializers.onnx_initializer_401
        model_layers_10_attn_v_proj_repeat_kv_reshape_4 = self.model_layers_10_attn_v_proj_repeat_kv_Reshape_4(model_layers_10_attn_v_proj_repeat_kv_transpose_2, initializers_onnx_initializer_401);  model_layers_10_attn_v_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_401 = None
        initializers_onnx_initializer_402 = self.initializers.onnx_initializer_402
        model_layers_10_attn_k_proj_repeat_kv_reshape_4 = self.model_layers_10_attn_k_proj_repeat_kv_Reshape_4(model_layers_10_attn_k_proj_repeat_kv_transpose_2, initializers_onnx_initializer_402);  model_layers_10_attn_k_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_402 = None
        com_microsoft__model_layers_10_attn_multi_head_attention = self.com_microsoft__model_layers_10_attn_MultiHeadAttention(com_microsoft__model_layers_10_attn_q_rotary_rotary_embedding, model_layers_10_attn_k_proj_repeat_kv_reshape_4, model_layers_10_attn_v_proj_repeat_kv_reshape_4, attention_mask = model_attn_mask_reformat_tile);  com_microsoft__model_layers_10_attn_q_rotary_rotary_embedding = model_layers_10_attn_k_proj_repeat_kv_reshape_4 = model_layers_10_attn_v_proj_repeat_kv_reshape_4 = None
        getitem_79 = com_microsoft__model_layers_10_attn_multi_head_attention[0];  com_microsoft__model_layers_10_attn_multi_head_attention = None
        initializers_onnx_initializer_403 = self.initializers.onnx_initializer_403
        model_layers_10_attn_o_proj_mat_mul = self.model_layers_10_attn_o_proj_MatMul(getitem_79, initializers_onnx_initializer_403);  getitem_79 = initializers_onnx_initializer_403 = None
        getitem_80 = com_microsoft__model_layers_10_input_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_10_input_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_404 = self.initializers.onnx_initializer_404
        com_microsoft__model_layers_10_post_attention_layernorm_skip_layer_norm = self.com_microsoft__model_layers_10_post_attention_layernorm_SkipLayerNorm(getitem_80, model_layers_10_attn_o_proj_mat_mul, initializers_onnx_initializer_404);  getitem_80 = model_layers_10_attn_o_proj_mat_mul = initializers_onnx_initializer_404 = None
        getitem_81 = com_microsoft__model_layers_10_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_405 = self.initializers.onnx_initializer_405
        model_layers_10_mlp_gate_proj_mat_mul = self.model_layers_10_mlp_gate_proj_MatMul(getitem_81, initializers_onnx_initializer_405);  getitem_81 = initializers_onnx_initializer_405 = None
        getitem_82 = com_microsoft__model_layers_10_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_406 = self.initializers.onnx_initializer_406
        model_layers_10_mlp_up_proj_mat_mul = self.model_layers_10_mlp_up_proj_MatMul(getitem_82, initializers_onnx_initializer_406);  getitem_82 = initializers_onnx_initializer_406 = None
        model_layers_10_mlp_act_fn_sigmoid = self.model_layers_10_mlp_act_fn_Sigmoid(model_layers_10_mlp_gate_proj_mat_mul)
        model_layers_10_mlp_act_fn_mul = self.model_layers_10_mlp_act_fn_Mul(model_layers_10_mlp_gate_proj_mat_mul, model_layers_10_mlp_act_fn_sigmoid);  model_layers_10_mlp_gate_proj_mat_mul = model_layers_10_mlp_act_fn_sigmoid = None
        model_layers_10_mlp_mul = self.model_layers_10_mlp_Mul(model_layers_10_mlp_act_fn_mul, model_layers_10_mlp_up_proj_mat_mul);  model_layers_10_mlp_act_fn_mul = model_layers_10_mlp_up_proj_mat_mul = None
        initializers_onnx_initializer_407 = self.initializers.onnx_initializer_407
        model_layers_10_mlp_down_proj_mat_mul = self.model_layers_10_mlp_down_proj_MatMul(model_layers_10_mlp_mul, initializers_onnx_initializer_407);  model_layers_10_mlp_mul = initializers_onnx_initializer_407 = None
        getitem_83 = com_microsoft__model_layers_10_post_attention_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_10_post_attention_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_408 = self.initializers.onnx_initializer_408
        com_microsoft__model_layers_11_input_layernorm_skip_layer_norm = self.com_microsoft__model_layers_11_input_layernorm_SkipLayerNorm(getitem_83, model_layers_10_mlp_down_proj_mat_mul, initializers_onnx_initializer_408);  getitem_83 = model_layers_10_mlp_down_proj_mat_mul = initializers_onnx_initializer_408 = None
        getitem_84 = com_microsoft__model_layers_11_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_409 = self.initializers.onnx_initializer_409
        model_layers_11_attn_q_proj_mat_mul = self.model_layers_11_attn_q_proj_MatMul(getitem_84, initializers_onnx_initializer_409);  getitem_84 = initializers_onnx_initializer_409 = None
        getitem_85 = com_microsoft__model_layers_11_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_410 = self.initializers.onnx_initializer_410
        model_layers_11_attn_k_proj_mat_mul = self.model_layers_11_attn_k_proj_MatMul(getitem_85, initializers_onnx_initializer_410);  getitem_85 = initializers_onnx_initializer_410 = None
        getitem_86 = com_microsoft__model_layers_11_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_411 = self.initializers.onnx_initializer_411
        model_layers_11_attn_v_proj_mat_mul = self.model_layers_11_attn_v_proj_MatMul(getitem_86, initializers_onnx_initializer_411);  getitem_86 = initializers_onnx_initializer_411 = None
        initializers_onnx_initializer_412 = self.initializers.onnx_initializer_412
        initializers_onnx_initializer_413 = self.initializers.onnx_initializer_413
        com_microsoft__model_layers_11_attn_q_rotary_rotary_embedding = self.com_microsoft__model_layers_11_attn_q_rotary_RotaryEmbedding(model_layers_11_attn_q_proj_mat_mul, input_3, initializers_onnx_initializer_412, initializers_onnx_initializer_413);  model_layers_11_attn_q_proj_mat_mul = initializers_onnx_initializer_412 = initializers_onnx_initializer_413 = None
        initializers_onnx_initializer_414 = self.initializers.onnx_initializer_414
        initializers_onnx_initializer_415 = self.initializers.onnx_initializer_415
        com_microsoft__model_layers_11_attn_k_rotary_rotary_embedding = self.com_microsoft__model_layers_11_attn_k_rotary_RotaryEmbedding(model_layers_11_attn_k_proj_mat_mul, input_3, initializers_onnx_initializer_414, initializers_onnx_initializer_415);  model_layers_11_attn_k_proj_mat_mul = initializers_onnx_initializer_414 = initializers_onnx_initializer_415 = None
        initializers_onnx_initializer_416 = self.initializers.onnx_initializer_416
        model_layers_11_attn_v_proj_repeat_kv_reshape_1 = self.model_layers_11_attn_v_proj_repeat_kv_Reshape_1(model_layers_11_attn_v_proj_mat_mul, initializers_onnx_initializer_416);  model_layers_11_attn_v_proj_mat_mul = initializers_onnx_initializer_416 = None
        initializers_onnx_initializer_417 = self.initializers.onnx_initializer_417
        model_layers_11_attn_k_proj_repeat_kv_reshape_1 = self.model_layers_11_attn_k_proj_repeat_kv_Reshape_1(com_microsoft__model_layers_11_attn_k_rotary_rotary_embedding, initializers_onnx_initializer_417);  com_microsoft__model_layers_11_attn_k_rotary_rotary_embedding = initializers_onnx_initializer_417 = None
        model_layers_11_attn_v_proj_repeat_kv_transpose_1 = self.model_layers_11_attn_v_proj_repeat_kv_Transpose_1(model_layers_11_attn_v_proj_repeat_kv_reshape_1);  model_layers_11_attn_v_proj_repeat_kv_reshape_1 = None
        model_layers_11_attn_k_proj_repeat_kv_transpose_1 = self.model_layers_11_attn_k_proj_repeat_kv_Transpose_1(model_layers_11_attn_k_proj_repeat_kv_reshape_1);  model_layers_11_attn_k_proj_repeat_kv_reshape_1 = None
        model_layers_11_attn_v_proj_repeat_kv_concat_1 = self.model_layers_11_attn_v_proj_repeat_kv_Concat_1(input_27, model_layers_11_attn_v_proj_repeat_kv_transpose_1);  input_27 = model_layers_11_attn_v_proj_repeat_kv_transpose_1 = None
        model_layers_11_attn_k_proj_repeat_kv_concat_1 = self.model_layers_11_attn_k_proj_repeat_kv_Concat_1(input_26, model_layers_11_attn_k_proj_repeat_kv_transpose_1);  input_26 = model_layers_11_attn_k_proj_repeat_kv_transpose_1 = None
        model_layers_11_attn_v_proj_repeat_kv_shape_1 = self.model_layers_11_attn_v_proj_repeat_kv_Shape_1(model_layers_11_attn_v_proj_repeat_kv_concat_1)
        model_layers_11_attn_v_proj_repeat_kv_unsqueeze_5 = self.model_layers_11_attn_v_proj_repeat_kv_Unsqueeze_5(model_layers_11_attn_v_proj_repeat_kv_concat_1)
        model_layers_11_attn_k_proj_repeat_kv_shape_1 = self.model_layers_11_attn_k_proj_repeat_kv_Shape_1(model_layers_11_attn_k_proj_repeat_kv_concat_1)
        model_layers_11_attn_k_proj_repeat_kv_unsqueeze_5 = self.model_layers_11_attn_k_proj_repeat_kv_Unsqueeze_5(model_layers_11_attn_k_proj_repeat_kv_concat_1)
        initializers_onnx_initializer_418 = self.initializers.onnx_initializer_418
        model_layers_11_attn_v_proj_repeat_kv_gather_1 = self.model_layers_11_attn_v_proj_repeat_kv_Gather_1(model_layers_11_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_418);  initializers_onnx_initializer_418 = None
        initializers_onnx_initializer_419 = self.initializers.onnx_initializer_419
        model_layers_11_attn_v_proj_repeat_kv_gather_3 = self.model_layers_11_attn_v_proj_repeat_kv_Gather_3(model_layers_11_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_419);  model_layers_11_attn_v_proj_repeat_kv_shape_1 = initializers_onnx_initializer_419 = None
        initializers_onnx_initializer_420 = self.initializers.onnx_initializer_420
        model_layers_11_attn_k_proj_repeat_kv_gather_1 = self.model_layers_11_attn_k_proj_repeat_kv_Gather_1(model_layers_11_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_420);  initializers_onnx_initializer_420 = None
        initializers_onnx_initializer_421 = self.initializers.onnx_initializer_421
        model_layers_11_attn_k_proj_repeat_kv_gather_3 = self.model_layers_11_attn_k_proj_repeat_kv_Gather_3(model_layers_11_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_421);  model_layers_11_attn_k_proj_repeat_kv_shape_1 = initializers_onnx_initializer_421 = None
        model_layers_11_attn_v_proj_repeat_kv_unsqueeze_1 = self.model_layers_11_attn_v_proj_repeat_kv_Unsqueeze_1(model_layers_11_attn_v_proj_repeat_kv_gather_1);  model_layers_11_attn_v_proj_repeat_kv_gather_1 = None
        model_layers_11_attn_v_proj_repeat_kv_unsqueeze_3 = self.model_layers_11_attn_v_proj_repeat_kv_Unsqueeze_3(model_layers_11_attn_v_proj_repeat_kv_gather_3);  model_layers_11_attn_v_proj_repeat_kv_gather_3 = None
        model_layers_11_attn_k_proj_repeat_kv_unsqueeze_1 = self.model_layers_11_attn_k_proj_repeat_kv_Unsqueeze_1(model_layers_11_attn_k_proj_repeat_kv_gather_1);  model_layers_11_attn_k_proj_repeat_kv_gather_1 = None
        model_layers_11_attn_k_proj_repeat_kv_unsqueeze_3 = self.model_layers_11_attn_k_proj_repeat_kv_Unsqueeze_3(model_layers_11_attn_k_proj_repeat_kv_gather_3);  model_layers_11_attn_k_proj_repeat_kv_gather_3 = None
        initializers_onnx_initializer_422 = self.initializers.onnx_initializer_422
        initializers_onnx_initializer_423 = self.initializers.onnx_initializer_423
        initializers_onnx_initializer_424 = self.initializers.onnx_initializer_424
        model_layers_11_attn_v_proj_repeat_kv_concat_2 = self.model_layers_11_attn_v_proj_repeat_kv_Concat_2(model_layers_11_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_422, initializers_onnx_initializer_423, model_layers_11_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_424);  initializers_onnx_initializer_422 = initializers_onnx_initializer_423 = initializers_onnx_initializer_424 = None
        initializers_onnx_initializer_425 = self.initializers.onnx_initializer_425
        initializers_onnx_initializer_426 = self.initializers.onnx_initializer_426
        model_layers_11_attn_v_proj_repeat_kv_concat_3 = self.model_layers_11_attn_v_proj_repeat_kv_Concat_3(model_layers_11_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_425, model_layers_11_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_426);  model_layers_11_attn_v_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_425 = model_layers_11_attn_v_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_426 = None
        initializers_onnx_initializer_427 = self.initializers.onnx_initializer_427
        initializers_onnx_initializer_428 = self.initializers.onnx_initializer_428
        initializers_onnx_initializer_429 = self.initializers.onnx_initializer_429
        model_layers_11_attn_k_proj_repeat_kv_concat_2 = self.model_layers_11_attn_k_proj_repeat_kv_Concat_2(model_layers_11_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_427, initializers_onnx_initializer_428, model_layers_11_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_429);  initializers_onnx_initializer_427 = initializers_onnx_initializer_428 = initializers_onnx_initializer_429 = None
        initializers_onnx_initializer_430 = self.initializers.onnx_initializer_430
        initializers_onnx_initializer_431 = self.initializers.onnx_initializer_431
        model_layers_11_attn_k_proj_repeat_kv_concat_3 = self.model_layers_11_attn_k_proj_repeat_kv_Concat_3(model_layers_11_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_430, model_layers_11_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_431);  model_layers_11_attn_k_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_430 = model_layers_11_attn_k_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_431 = None
        initializers_onnx_initializer_432 = self.initializers.onnx_initializer_432
        model_layers_11_attn_v_proj_repeat_kv_equal = self.model_layers_11_attn_v_proj_repeat_kv_Equal(model_layers_11_attn_v_proj_repeat_kv_concat_2, initializers_onnx_initializer_432);  initializers_onnx_initializer_432 = None
        initializers_onnx_initializer_433 = self.initializers.onnx_initializer_433
        model_layers_11_attn_k_proj_repeat_kv_equal = self.model_layers_11_attn_k_proj_repeat_kv_Equal(model_layers_11_attn_k_proj_repeat_kv_concat_2, initializers_onnx_initializer_433);  initializers_onnx_initializer_433 = None
        initializers_onnx_initializer_434 = self.initializers.onnx_initializer_434
        model_layers_11_attn_v_proj_repeat_kv_where = self.model_layers_11_attn_v_proj_repeat_kv_Where(model_layers_11_attn_v_proj_repeat_kv_equal, initializers_onnx_initializer_434, model_layers_11_attn_v_proj_repeat_kv_concat_2);  model_layers_11_attn_v_proj_repeat_kv_equal = initializers_onnx_initializer_434 = model_layers_11_attn_v_proj_repeat_kv_concat_2 = None
        initializers_onnx_initializer_435 = self.initializers.onnx_initializer_435
        model_layers_11_attn_k_proj_repeat_kv_where = self.model_layers_11_attn_k_proj_repeat_kv_Where(model_layers_11_attn_k_proj_repeat_kv_equal, initializers_onnx_initializer_435, model_layers_11_attn_k_proj_repeat_kv_concat_2);  model_layers_11_attn_k_proj_repeat_kv_equal = initializers_onnx_initializer_435 = model_layers_11_attn_k_proj_repeat_kv_concat_2 = None
        model_layers_11_attn_v_proj_repeat_kv_expand = self.model_layers_11_attn_v_proj_repeat_kv_Expand(model_layers_11_attn_v_proj_repeat_kv_unsqueeze_5, model_layers_11_attn_v_proj_repeat_kv_where);  model_layers_11_attn_v_proj_repeat_kv_unsqueeze_5 = model_layers_11_attn_v_proj_repeat_kv_where = None
        model_layers_11_attn_k_proj_repeat_kv_expand = self.model_layers_11_attn_k_proj_repeat_kv_Expand(model_layers_11_attn_k_proj_repeat_kv_unsqueeze_5, model_layers_11_attn_k_proj_repeat_kv_where);  model_layers_11_attn_k_proj_repeat_kv_unsqueeze_5 = model_layers_11_attn_k_proj_repeat_kv_where = None
        model_layers_11_attn_v_proj_repeat_kv_reshape_3 = self.model_layers_11_attn_v_proj_repeat_kv_Reshape_3(model_layers_11_attn_v_proj_repeat_kv_expand, model_layers_11_attn_v_proj_repeat_kv_concat_3);  model_layers_11_attn_v_proj_repeat_kv_expand = model_layers_11_attn_v_proj_repeat_kv_concat_3 = None
        model_layers_11_attn_k_proj_repeat_kv_reshape_3 = self.model_layers_11_attn_k_proj_repeat_kv_Reshape_3(model_layers_11_attn_k_proj_repeat_kv_expand, model_layers_11_attn_k_proj_repeat_kv_concat_3);  model_layers_11_attn_k_proj_repeat_kv_expand = model_layers_11_attn_k_proj_repeat_kv_concat_3 = None
        model_layers_11_attn_v_proj_repeat_kv_transpose_2 = self.model_layers_11_attn_v_proj_repeat_kv_Transpose_2(model_layers_11_attn_v_proj_repeat_kv_reshape_3);  model_layers_11_attn_v_proj_repeat_kv_reshape_3 = None
        model_layers_11_attn_k_proj_repeat_kv_transpose_2 = self.model_layers_11_attn_k_proj_repeat_kv_Transpose_2(model_layers_11_attn_k_proj_repeat_kv_reshape_3);  model_layers_11_attn_k_proj_repeat_kv_reshape_3 = None
        initializers_onnx_initializer_436 = self.initializers.onnx_initializer_436
        model_layers_11_attn_v_proj_repeat_kv_reshape_4 = self.model_layers_11_attn_v_proj_repeat_kv_Reshape_4(model_layers_11_attn_v_proj_repeat_kv_transpose_2, initializers_onnx_initializer_436);  model_layers_11_attn_v_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_436 = None
        initializers_onnx_initializer_437 = self.initializers.onnx_initializer_437
        model_layers_11_attn_k_proj_repeat_kv_reshape_4 = self.model_layers_11_attn_k_proj_repeat_kv_Reshape_4(model_layers_11_attn_k_proj_repeat_kv_transpose_2, initializers_onnx_initializer_437);  model_layers_11_attn_k_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_437 = None
        com_microsoft__model_layers_11_attn_multi_head_attention = self.com_microsoft__model_layers_11_attn_MultiHeadAttention(com_microsoft__model_layers_11_attn_q_rotary_rotary_embedding, model_layers_11_attn_k_proj_repeat_kv_reshape_4, model_layers_11_attn_v_proj_repeat_kv_reshape_4, attention_mask = model_attn_mask_reformat_tile);  com_microsoft__model_layers_11_attn_q_rotary_rotary_embedding = model_layers_11_attn_k_proj_repeat_kv_reshape_4 = model_layers_11_attn_v_proj_repeat_kv_reshape_4 = None
        getitem_87 = com_microsoft__model_layers_11_attn_multi_head_attention[0];  com_microsoft__model_layers_11_attn_multi_head_attention = None
        initializers_onnx_initializer_438 = self.initializers.onnx_initializer_438
        model_layers_11_attn_o_proj_mat_mul = self.model_layers_11_attn_o_proj_MatMul(getitem_87, initializers_onnx_initializer_438);  getitem_87 = initializers_onnx_initializer_438 = None
        getitem_88 = com_microsoft__model_layers_11_input_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_11_input_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_439 = self.initializers.onnx_initializer_439
        com_microsoft__model_layers_11_post_attention_layernorm_skip_layer_norm = self.com_microsoft__model_layers_11_post_attention_layernorm_SkipLayerNorm(getitem_88, model_layers_11_attn_o_proj_mat_mul, initializers_onnx_initializer_439);  getitem_88 = model_layers_11_attn_o_proj_mat_mul = initializers_onnx_initializer_439 = None
        getitem_89 = com_microsoft__model_layers_11_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_440 = self.initializers.onnx_initializer_440
        model_layers_11_mlp_gate_proj_mat_mul = self.model_layers_11_mlp_gate_proj_MatMul(getitem_89, initializers_onnx_initializer_440);  getitem_89 = initializers_onnx_initializer_440 = None
        getitem_90 = com_microsoft__model_layers_11_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_441 = self.initializers.onnx_initializer_441
        model_layers_11_mlp_up_proj_mat_mul = self.model_layers_11_mlp_up_proj_MatMul(getitem_90, initializers_onnx_initializer_441);  getitem_90 = initializers_onnx_initializer_441 = None
        model_layers_11_mlp_act_fn_sigmoid = self.model_layers_11_mlp_act_fn_Sigmoid(model_layers_11_mlp_gate_proj_mat_mul)
        model_layers_11_mlp_act_fn_mul = self.model_layers_11_mlp_act_fn_Mul(model_layers_11_mlp_gate_proj_mat_mul, model_layers_11_mlp_act_fn_sigmoid);  model_layers_11_mlp_gate_proj_mat_mul = model_layers_11_mlp_act_fn_sigmoid = None
        model_layers_11_mlp_mul = self.model_layers_11_mlp_Mul(model_layers_11_mlp_act_fn_mul, model_layers_11_mlp_up_proj_mat_mul);  model_layers_11_mlp_act_fn_mul = model_layers_11_mlp_up_proj_mat_mul = None
        initializers_onnx_initializer_442 = self.initializers.onnx_initializer_442
        model_layers_11_mlp_down_proj_mat_mul = self.model_layers_11_mlp_down_proj_MatMul(model_layers_11_mlp_mul, initializers_onnx_initializer_442);  model_layers_11_mlp_mul = initializers_onnx_initializer_442 = None
        getitem_91 = com_microsoft__model_layers_11_post_attention_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_11_post_attention_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_443 = self.initializers.onnx_initializer_443
        com_microsoft__model_layers_12_input_layernorm_skip_layer_norm = self.com_microsoft__model_layers_12_input_layernorm_SkipLayerNorm(getitem_91, model_layers_11_mlp_down_proj_mat_mul, initializers_onnx_initializer_443);  getitem_91 = model_layers_11_mlp_down_proj_mat_mul = initializers_onnx_initializer_443 = None
        getitem_92 = com_microsoft__model_layers_12_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_444 = self.initializers.onnx_initializer_444
        model_layers_12_attn_q_proj_mat_mul = self.model_layers_12_attn_q_proj_MatMul(getitem_92, initializers_onnx_initializer_444);  getitem_92 = initializers_onnx_initializer_444 = None
        getitem_93 = com_microsoft__model_layers_12_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_445 = self.initializers.onnx_initializer_445
        model_layers_12_attn_k_proj_mat_mul = self.model_layers_12_attn_k_proj_MatMul(getitem_93, initializers_onnx_initializer_445);  getitem_93 = initializers_onnx_initializer_445 = None
        getitem_94 = com_microsoft__model_layers_12_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_446 = self.initializers.onnx_initializer_446
        model_layers_12_attn_v_proj_mat_mul = self.model_layers_12_attn_v_proj_MatMul(getitem_94, initializers_onnx_initializer_446);  getitem_94 = initializers_onnx_initializer_446 = None
        initializers_onnx_initializer_447 = self.initializers.onnx_initializer_447
        initializers_onnx_initializer_448 = self.initializers.onnx_initializer_448
        com_microsoft__model_layers_12_attn_q_rotary_rotary_embedding = self.com_microsoft__model_layers_12_attn_q_rotary_RotaryEmbedding(model_layers_12_attn_q_proj_mat_mul, input_3, initializers_onnx_initializer_447, initializers_onnx_initializer_448);  model_layers_12_attn_q_proj_mat_mul = initializers_onnx_initializer_447 = initializers_onnx_initializer_448 = None
        initializers_onnx_initializer_449 = self.initializers.onnx_initializer_449
        initializers_onnx_initializer_450 = self.initializers.onnx_initializer_450
        com_microsoft__model_layers_12_attn_k_rotary_rotary_embedding = self.com_microsoft__model_layers_12_attn_k_rotary_RotaryEmbedding(model_layers_12_attn_k_proj_mat_mul, input_3, initializers_onnx_initializer_449, initializers_onnx_initializer_450);  model_layers_12_attn_k_proj_mat_mul = initializers_onnx_initializer_449 = initializers_onnx_initializer_450 = None
        initializers_onnx_initializer_451 = self.initializers.onnx_initializer_451
        model_layers_12_attn_v_proj_repeat_kv_reshape_1 = self.model_layers_12_attn_v_proj_repeat_kv_Reshape_1(model_layers_12_attn_v_proj_mat_mul, initializers_onnx_initializer_451);  model_layers_12_attn_v_proj_mat_mul = initializers_onnx_initializer_451 = None
        initializers_onnx_initializer_452 = self.initializers.onnx_initializer_452
        model_layers_12_attn_k_proj_repeat_kv_reshape_1 = self.model_layers_12_attn_k_proj_repeat_kv_Reshape_1(com_microsoft__model_layers_12_attn_k_rotary_rotary_embedding, initializers_onnx_initializer_452);  com_microsoft__model_layers_12_attn_k_rotary_rotary_embedding = initializers_onnx_initializer_452 = None
        model_layers_12_attn_v_proj_repeat_kv_transpose_1 = self.model_layers_12_attn_v_proj_repeat_kv_Transpose_1(model_layers_12_attn_v_proj_repeat_kv_reshape_1);  model_layers_12_attn_v_proj_repeat_kv_reshape_1 = None
        model_layers_12_attn_k_proj_repeat_kv_transpose_1 = self.model_layers_12_attn_k_proj_repeat_kv_Transpose_1(model_layers_12_attn_k_proj_repeat_kv_reshape_1);  model_layers_12_attn_k_proj_repeat_kv_reshape_1 = None
        model_layers_12_attn_v_proj_repeat_kv_concat_1 = self.model_layers_12_attn_v_proj_repeat_kv_Concat_1(input_29, model_layers_12_attn_v_proj_repeat_kv_transpose_1);  input_29 = model_layers_12_attn_v_proj_repeat_kv_transpose_1 = None
        model_layers_12_attn_k_proj_repeat_kv_concat_1 = self.model_layers_12_attn_k_proj_repeat_kv_Concat_1(input_28, model_layers_12_attn_k_proj_repeat_kv_transpose_1);  input_28 = model_layers_12_attn_k_proj_repeat_kv_transpose_1 = None
        model_layers_12_attn_v_proj_repeat_kv_shape_1 = self.model_layers_12_attn_v_proj_repeat_kv_Shape_1(model_layers_12_attn_v_proj_repeat_kv_concat_1)
        model_layers_12_attn_v_proj_repeat_kv_unsqueeze_5 = self.model_layers_12_attn_v_proj_repeat_kv_Unsqueeze_5(model_layers_12_attn_v_proj_repeat_kv_concat_1)
        model_layers_12_attn_k_proj_repeat_kv_shape_1 = self.model_layers_12_attn_k_proj_repeat_kv_Shape_1(model_layers_12_attn_k_proj_repeat_kv_concat_1)
        model_layers_12_attn_k_proj_repeat_kv_unsqueeze_5 = self.model_layers_12_attn_k_proj_repeat_kv_Unsqueeze_5(model_layers_12_attn_k_proj_repeat_kv_concat_1)
        initializers_onnx_initializer_453 = self.initializers.onnx_initializer_453
        model_layers_12_attn_v_proj_repeat_kv_gather_1 = self.model_layers_12_attn_v_proj_repeat_kv_Gather_1(model_layers_12_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_453);  initializers_onnx_initializer_453 = None
        initializers_onnx_initializer_454 = self.initializers.onnx_initializer_454
        model_layers_12_attn_v_proj_repeat_kv_gather_3 = self.model_layers_12_attn_v_proj_repeat_kv_Gather_3(model_layers_12_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_454);  model_layers_12_attn_v_proj_repeat_kv_shape_1 = initializers_onnx_initializer_454 = None
        initializers_onnx_initializer_455 = self.initializers.onnx_initializer_455
        model_layers_12_attn_k_proj_repeat_kv_gather_1 = self.model_layers_12_attn_k_proj_repeat_kv_Gather_1(model_layers_12_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_455);  initializers_onnx_initializer_455 = None
        initializers_onnx_initializer_456 = self.initializers.onnx_initializer_456
        model_layers_12_attn_k_proj_repeat_kv_gather_3 = self.model_layers_12_attn_k_proj_repeat_kv_Gather_3(model_layers_12_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_456);  model_layers_12_attn_k_proj_repeat_kv_shape_1 = initializers_onnx_initializer_456 = None
        model_layers_12_attn_v_proj_repeat_kv_unsqueeze_1 = self.model_layers_12_attn_v_proj_repeat_kv_Unsqueeze_1(model_layers_12_attn_v_proj_repeat_kv_gather_1);  model_layers_12_attn_v_proj_repeat_kv_gather_1 = None
        model_layers_12_attn_v_proj_repeat_kv_unsqueeze_3 = self.model_layers_12_attn_v_proj_repeat_kv_Unsqueeze_3(model_layers_12_attn_v_proj_repeat_kv_gather_3);  model_layers_12_attn_v_proj_repeat_kv_gather_3 = None
        model_layers_12_attn_k_proj_repeat_kv_unsqueeze_1 = self.model_layers_12_attn_k_proj_repeat_kv_Unsqueeze_1(model_layers_12_attn_k_proj_repeat_kv_gather_1);  model_layers_12_attn_k_proj_repeat_kv_gather_1 = None
        model_layers_12_attn_k_proj_repeat_kv_unsqueeze_3 = self.model_layers_12_attn_k_proj_repeat_kv_Unsqueeze_3(model_layers_12_attn_k_proj_repeat_kv_gather_3);  model_layers_12_attn_k_proj_repeat_kv_gather_3 = None
        initializers_onnx_initializer_457 = self.initializers.onnx_initializer_457
        initializers_onnx_initializer_458 = self.initializers.onnx_initializer_458
        initializers_onnx_initializer_459 = self.initializers.onnx_initializer_459
        model_layers_12_attn_v_proj_repeat_kv_concat_2 = self.model_layers_12_attn_v_proj_repeat_kv_Concat_2(model_layers_12_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_457, initializers_onnx_initializer_458, model_layers_12_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_459);  initializers_onnx_initializer_457 = initializers_onnx_initializer_458 = initializers_onnx_initializer_459 = None
        initializers_onnx_initializer_460 = self.initializers.onnx_initializer_460
        initializers_onnx_initializer_461 = self.initializers.onnx_initializer_461
        model_layers_12_attn_v_proj_repeat_kv_concat_3 = self.model_layers_12_attn_v_proj_repeat_kv_Concat_3(model_layers_12_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_460, model_layers_12_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_461);  model_layers_12_attn_v_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_460 = model_layers_12_attn_v_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_461 = None
        initializers_onnx_initializer_462 = self.initializers.onnx_initializer_462
        initializers_onnx_initializer_463 = self.initializers.onnx_initializer_463
        initializers_onnx_initializer_464 = self.initializers.onnx_initializer_464
        model_layers_12_attn_k_proj_repeat_kv_concat_2 = self.model_layers_12_attn_k_proj_repeat_kv_Concat_2(model_layers_12_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_462, initializers_onnx_initializer_463, model_layers_12_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_464);  initializers_onnx_initializer_462 = initializers_onnx_initializer_463 = initializers_onnx_initializer_464 = None
        initializers_onnx_initializer_465 = self.initializers.onnx_initializer_465
        initializers_onnx_initializer_466 = self.initializers.onnx_initializer_466
        model_layers_12_attn_k_proj_repeat_kv_concat_3 = self.model_layers_12_attn_k_proj_repeat_kv_Concat_3(model_layers_12_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_465, model_layers_12_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_466);  model_layers_12_attn_k_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_465 = model_layers_12_attn_k_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_466 = None
        initializers_onnx_initializer_467 = self.initializers.onnx_initializer_467
        model_layers_12_attn_v_proj_repeat_kv_equal = self.model_layers_12_attn_v_proj_repeat_kv_Equal(model_layers_12_attn_v_proj_repeat_kv_concat_2, initializers_onnx_initializer_467);  initializers_onnx_initializer_467 = None
        initializers_onnx_initializer_468 = self.initializers.onnx_initializer_468
        model_layers_12_attn_k_proj_repeat_kv_equal = self.model_layers_12_attn_k_proj_repeat_kv_Equal(model_layers_12_attn_k_proj_repeat_kv_concat_2, initializers_onnx_initializer_468);  initializers_onnx_initializer_468 = None
        initializers_onnx_initializer_469 = self.initializers.onnx_initializer_469
        model_layers_12_attn_v_proj_repeat_kv_where = self.model_layers_12_attn_v_proj_repeat_kv_Where(model_layers_12_attn_v_proj_repeat_kv_equal, initializers_onnx_initializer_469, model_layers_12_attn_v_proj_repeat_kv_concat_2);  model_layers_12_attn_v_proj_repeat_kv_equal = initializers_onnx_initializer_469 = model_layers_12_attn_v_proj_repeat_kv_concat_2 = None
        initializers_onnx_initializer_470 = self.initializers.onnx_initializer_470
        model_layers_12_attn_k_proj_repeat_kv_where = self.model_layers_12_attn_k_proj_repeat_kv_Where(model_layers_12_attn_k_proj_repeat_kv_equal, initializers_onnx_initializer_470, model_layers_12_attn_k_proj_repeat_kv_concat_2);  model_layers_12_attn_k_proj_repeat_kv_equal = initializers_onnx_initializer_470 = model_layers_12_attn_k_proj_repeat_kv_concat_2 = None
        model_layers_12_attn_v_proj_repeat_kv_expand = self.model_layers_12_attn_v_proj_repeat_kv_Expand(model_layers_12_attn_v_proj_repeat_kv_unsqueeze_5, model_layers_12_attn_v_proj_repeat_kv_where);  model_layers_12_attn_v_proj_repeat_kv_unsqueeze_5 = model_layers_12_attn_v_proj_repeat_kv_where = None
        model_layers_12_attn_k_proj_repeat_kv_expand = self.model_layers_12_attn_k_proj_repeat_kv_Expand(model_layers_12_attn_k_proj_repeat_kv_unsqueeze_5, model_layers_12_attn_k_proj_repeat_kv_where);  model_layers_12_attn_k_proj_repeat_kv_unsqueeze_5 = model_layers_12_attn_k_proj_repeat_kv_where = None
        model_layers_12_attn_v_proj_repeat_kv_reshape_3 = self.model_layers_12_attn_v_proj_repeat_kv_Reshape_3(model_layers_12_attn_v_proj_repeat_kv_expand, model_layers_12_attn_v_proj_repeat_kv_concat_3);  model_layers_12_attn_v_proj_repeat_kv_expand = model_layers_12_attn_v_proj_repeat_kv_concat_3 = None
        model_layers_12_attn_k_proj_repeat_kv_reshape_3 = self.model_layers_12_attn_k_proj_repeat_kv_Reshape_3(model_layers_12_attn_k_proj_repeat_kv_expand, model_layers_12_attn_k_proj_repeat_kv_concat_3);  model_layers_12_attn_k_proj_repeat_kv_expand = model_layers_12_attn_k_proj_repeat_kv_concat_3 = None
        model_layers_12_attn_v_proj_repeat_kv_transpose_2 = self.model_layers_12_attn_v_proj_repeat_kv_Transpose_2(model_layers_12_attn_v_proj_repeat_kv_reshape_3);  model_layers_12_attn_v_proj_repeat_kv_reshape_3 = None
        model_layers_12_attn_k_proj_repeat_kv_transpose_2 = self.model_layers_12_attn_k_proj_repeat_kv_Transpose_2(model_layers_12_attn_k_proj_repeat_kv_reshape_3);  model_layers_12_attn_k_proj_repeat_kv_reshape_3 = None
        initializers_onnx_initializer_471 = self.initializers.onnx_initializer_471
        model_layers_12_attn_v_proj_repeat_kv_reshape_4 = self.model_layers_12_attn_v_proj_repeat_kv_Reshape_4(model_layers_12_attn_v_proj_repeat_kv_transpose_2, initializers_onnx_initializer_471);  model_layers_12_attn_v_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_471 = None
        initializers_onnx_initializer_472 = self.initializers.onnx_initializer_472
        model_layers_12_attn_k_proj_repeat_kv_reshape_4 = self.model_layers_12_attn_k_proj_repeat_kv_Reshape_4(model_layers_12_attn_k_proj_repeat_kv_transpose_2, initializers_onnx_initializer_472);  model_layers_12_attn_k_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_472 = None
        com_microsoft__model_layers_12_attn_multi_head_attention = self.com_microsoft__model_layers_12_attn_MultiHeadAttention(com_microsoft__model_layers_12_attn_q_rotary_rotary_embedding, model_layers_12_attn_k_proj_repeat_kv_reshape_4, model_layers_12_attn_v_proj_repeat_kv_reshape_4, attention_mask = model_attn_mask_reformat_tile);  com_microsoft__model_layers_12_attn_q_rotary_rotary_embedding = model_layers_12_attn_k_proj_repeat_kv_reshape_4 = model_layers_12_attn_v_proj_repeat_kv_reshape_4 = None
        getitem_95 = com_microsoft__model_layers_12_attn_multi_head_attention[0];  com_microsoft__model_layers_12_attn_multi_head_attention = None
        initializers_onnx_initializer_473 = self.initializers.onnx_initializer_473
        model_layers_12_attn_o_proj_mat_mul = self.model_layers_12_attn_o_proj_MatMul(getitem_95, initializers_onnx_initializer_473);  getitem_95 = initializers_onnx_initializer_473 = None
        getitem_96 = com_microsoft__model_layers_12_input_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_12_input_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_474 = self.initializers.onnx_initializer_474
        com_microsoft__model_layers_12_post_attention_layernorm_skip_layer_norm = self.com_microsoft__model_layers_12_post_attention_layernorm_SkipLayerNorm(getitem_96, model_layers_12_attn_o_proj_mat_mul, initializers_onnx_initializer_474);  getitem_96 = model_layers_12_attn_o_proj_mat_mul = initializers_onnx_initializer_474 = None
        getitem_97 = com_microsoft__model_layers_12_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_475 = self.initializers.onnx_initializer_475
        model_layers_12_mlp_gate_proj_mat_mul = self.model_layers_12_mlp_gate_proj_MatMul(getitem_97, initializers_onnx_initializer_475);  getitem_97 = initializers_onnx_initializer_475 = None
        getitem_98 = com_microsoft__model_layers_12_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_476 = self.initializers.onnx_initializer_476
        model_layers_12_mlp_up_proj_mat_mul = self.model_layers_12_mlp_up_proj_MatMul(getitem_98, initializers_onnx_initializer_476);  getitem_98 = initializers_onnx_initializer_476 = None
        model_layers_12_mlp_act_fn_sigmoid = self.model_layers_12_mlp_act_fn_Sigmoid(model_layers_12_mlp_gate_proj_mat_mul)
        model_layers_12_mlp_act_fn_mul = self.model_layers_12_mlp_act_fn_Mul(model_layers_12_mlp_gate_proj_mat_mul, model_layers_12_mlp_act_fn_sigmoid);  model_layers_12_mlp_gate_proj_mat_mul = model_layers_12_mlp_act_fn_sigmoid = None
        model_layers_12_mlp_mul = self.model_layers_12_mlp_Mul(model_layers_12_mlp_act_fn_mul, model_layers_12_mlp_up_proj_mat_mul);  model_layers_12_mlp_act_fn_mul = model_layers_12_mlp_up_proj_mat_mul = None
        initializers_onnx_initializer_477 = self.initializers.onnx_initializer_477
        model_layers_12_mlp_down_proj_mat_mul = self.model_layers_12_mlp_down_proj_MatMul(model_layers_12_mlp_mul, initializers_onnx_initializer_477);  model_layers_12_mlp_mul = initializers_onnx_initializer_477 = None
        getitem_99 = com_microsoft__model_layers_12_post_attention_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_12_post_attention_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_478 = self.initializers.onnx_initializer_478
        com_microsoft__model_layers_13_input_layernorm_skip_layer_norm = self.com_microsoft__model_layers_13_input_layernorm_SkipLayerNorm(getitem_99, model_layers_12_mlp_down_proj_mat_mul, initializers_onnx_initializer_478);  getitem_99 = model_layers_12_mlp_down_proj_mat_mul = initializers_onnx_initializer_478 = None
        getitem_100 = com_microsoft__model_layers_13_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_479 = self.initializers.onnx_initializer_479
        model_layers_13_attn_q_proj_mat_mul = self.model_layers_13_attn_q_proj_MatMul(getitem_100, initializers_onnx_initializer_479);  getitem_100 = initializers_onnx_initializer_479 = None
        getitem_101 = com_microsoft__model_layers_13_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_480 = self.initializers.onnx_initializer_480
        model_layers_13_attn_k_proj_mat_mul = self.model_layers_13_attn_k_proj_MatMul(getitem_101, initializers_onnx_initializer_480);  getitem_101 = initializers_onnx_initializer_480 = None
        getitem_102 = com_microsoft__model_layers_13_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_481 = self.initializers.onnx_initializer_481
        model_layers_13_attn_v_proj_mat_mul = self.model_layers_13_attn_v_proj_MatMul(getitem_102, initializers_onnx_initializer_481);  getitem_102 = initializers_onnx_initializer_481 = None
        initializers_onnx_initializer_482 = self.initializers.onnx_initializer_482
        initializers_onnx_initializer_483 = self.initializers.onnx_initializer_483
        com_microsoft__model_layers_13_attn_q_rotary_rotary_embedding = self.com_microsoft__model_layers_13_attn_q_rotary_RotaryEmbedding(model_layers_13_attn_q_proj_mat_mul, input_3, initializers_onnx_initializer_482, initializers_onnx_initializer_483);  model_layers_13_attn_q_proj_mat_mul = initializers_onnx_initializer_482 = initializers_onnx_initializer_483 = None
        initializers_onnx_initializer_484 = self.initializers.onnx_initializer_484
        initializers_onnx_initializer_485 = self.initializers.onnx_initializer_485
        com_microsoft__model_layers_13_attn_k_rotary_rotary_embedding = self.com_microsoft__model_layers_13_attn_k_rotary_RotaryEmbedding(model_layers_13_attn_k_proj_mat_mul, input_3, initializers_onnx_initializer_484, initializers_onnx_initializer_485);  model_layers_13_attn_k_proj_mat_mul = initializers_onnx_initializer_484 = initializers_onnx_initializer_485 = None
        initializers_onnx_initializer_486 = self.initializers.onnx_initializer_486
        model_layers_13_attn_v_proj_repeat_kv_reshape_1 = self.model_layers_13_attn_v_proj_repeat_kv_Reshape_1(model_layers_13_attn_v_proj_mat_mul, initializers_onnx_initializer_486);  model_layers_13_attn_v_proj_mat_mul = initializers_onnx_initializer_486 = None
        initializers_onnx_initializer_487 = self.initializers.onnx_initializer_487
        model_layers_13_attn_k_proj_repeat_kv_reshape_1 = self.model_layers_13_attn_k_proj_repeat_kv_Reshape_1(com_microsoft__model_layers_13_attn_k_rotary_rotary_embedding, initializers_onnx_initializer_487);  com_microsoft__model_layers_13_attn_k_rotary_rotary_embedding = initializers_onnx_initializer_487 = None
        model_layers_13_attn_v_proj_repeat_kv_transpose_1 = self.model_layers_13_attn_v_proj_repeat_kv_Transpose_1(model_layers_13_attn_v_proj_repeat_kv_reshape_1);  model_layers_13_attn_v_proj_repeat_kv_reshape_1 = None
        model_layers_13_attn_k_proj_repeat_kv_transpose_1 = self.model_layers_13_attn_k_proj_repeat_kv_Transpose_1(model_layers_13_attn_k_proj_repeat_kv_reshape_1);  model_layers_13_attn_k_proj_repeat_kv_reshape_1 = None
        model_layers_13_attn_v_proj_repeat_kv_concat_1 = self.model_layers_13_attn_v_proj_repeat_kv_Concat_1(input_31, model_layers_13_attn_v_proj_repeat_kv_transpose_1);  input_31 = model_layers_13_attn_v_proj_repeat_kv_transpose_1 = None
        model_layers_13_attn_k_proj_repeat_kv_concat_1 = self.model_layers_13_attn_k_proj_repeat_kv_Concat_1(input_30, model_layers_13_attn_k_proj_repeat_kv_transpose_1);  input_30 = model_layers_13_attn_k_proj_repeat_kv_transpose_1 = None
        model_layers_13_attn_v_proj_repeat_kv_shape_1 = self.model_layers_13_attn_v_proj_repeat_kv_Shape_1(model_layers_13_attn_v_proj_repeat_kv_concat_1)
        model_layers_13_attn_v_proj_repeat_kv_unsqueeze_5 = self.model_layers_13_attn_v_proj_repeat_kv_Unsqueeze_5(model_layers_13_attn_v_proj_repeat_kv_concat_1)
        model_layers_13_attn_k_proj_repeat_kv_shape_1 = self.model_layers_13_attn_k_proj_repeat_kv_Shape_1(model_layers_13_attn_k_proj_repeat_kv_concat_1)
        model_layers_13_attn_k_proj_repeat_kv_unsqueeze_5 = self.model_layers_13_attn_k_proj_repeat_kv_Unsqueeze_5(model_layers_13_attn_k_proj_repeat_kv_concat_1)
        initializers_onnx_initializer_488 = self.initializers.onnx_initializer_488
        model_layers_13_attn_v_proj_repeat_kv_gather_1 = self.model_layers_13_attn_v_proj_repeat_kv_Gather_1(model_layers_13_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_488);  initializers_onnx_initializer_488 = None
        initializers_onnx_initializer_489 = self.initializers.onnx_initializer_489
        model_layers_13_attn_v_proj_repeat_kv_gather_3 = self.model_layers_13_attn_v_proj_repeat_kv_Gather_3(model_layers_13_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_489);  model_layers_13_attn_v_proj_repeat_kv_shape_1 = initializers_onnx_initializer_489 = None
        initializers_onnx_initializer_490 = self.initializers.onnx_initializer_490
        model_layers_13_attn_k_proj_repeat_kv_gather_1 = self.model_layers_13_attn_k_proj_repeat_kv_Gather_1(model_layers_13_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_490);  initializers_onnx_initializer_490 = None
        initializers_onnx_initializer_491 = self.initializers.onnx_initializer_491
        model_layers_13_attn_k_proj_repeat_kv_gather_3 = self.model_layers_13_attn_k_proj_repeat_kv_Gather_3(model_layers_13_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_491);  model_layers_13_attn_k_proj_repeat_kv_shape_1 = initializers_onnx_initializer_491 = None
        model_layers_13_attn_v_proj_repeat_kv_unsqueeze_1 = self.model_layers_13_attn_v_proj_repeat_kv_Unsqueeze_1(model_layers_13_attn_v_proj_repeat_kv_gather_1);  model_layers_13_attn_v_proj_repeat_kv_gather_1 = None
        model_layers_13_attn_v_proj_repeat_kv_unsqueeze_3 = self.model_layers_13_attn_v_proj_repeat_kv_Unsqueeze_3(model_layers_13_attn_v_proj_repeat_kv_gather_3);  model_layers_13_attn_v_proj_repeat_kv_gather_3 = None
        model_layers_13_attn_k_proj_repeat_kv_unsqueeze_1 = self.model_layers_13_attn_k_proj_repeat_kv_Unsqueeze_1(model_layers_13_attn_k_proj_repeat_kv_gather_1);  model_layers_13_attn_k_proj_repeat_kv_gather_1 = None
        model_layers_13_attn_k_proj_repeat_kv_unsqueeze_3 = self.model_layers_13_attn_k_proj_repeat_kv_Unsqueeze_3(model_layers_13_attn_k_proj_repeat_kv_gather_3);  model_layers_13_attn_k_proj_repeat_kv_gather_3 = None
        initializers_onnx_initializer_492 = self.initializers.onnx_initializer_492
        initializers_onnx_initializer_493 = self.initializers.onnx_initializer_493
        initializers_onnx_initializer_494 = self.initializers.onnx_initializer_494
        model_layers_13_attn_v_proj_repeat_kv_concat_2 = self.model_layers_13_attn_v_proj_repeat_kv_Concat_2(model_layers_13_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_492, initializers_onnx_initializer_493, model_layers_13_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_494);  initializers_onnx_initializer_492 = initializers_onnx_initializer_493 = initializers_onnx_initializer_494 = None
        initializers_onnx_initializer_495 = self.initializers.onnx_initializer_495
        initializers_onnx_initializer_496 = self.initializers.onnx_initializer_496
        model_layers_13_attn_v_proj_repeat_kv_concat_3 = self.model_layers_13_attn_v_proj_repeat_kv_Concat_3(model_layers_13_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_495, model_layers_13_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_496);  model_layers_13_attn_v_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_495 = model_layers_13_attn_v_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_496 = None
        initializers_onnx_initializer_497 = self.initializers.onnx_initializer_497
        initializers_onnx_initializer_498 = self.initializers.onnx_initializer_498
        initializers_onnx_initializer_499 = self.initializers.onnx_initializer_499
        model_layers_13_attn_k_proj_repeat_kv_concat_2 = self.model_layers_13_attn_k_proj_repeat_kv_Concat_2(model_layers_13_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_497, initializers_onnx_initializer_498, model_layers_13_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_499);  initializers_onnx_initializer_497 = initializers_onnx_initializer_498 = initializers_onnx_initializer_499 = None
        initializers_onnx_initializer_500 = self.initializers.onnx_initializer_500
        initializers_onnx_initializer_501 = self.initializers.onnx_initializer_501
        model_layers_13_attn_k_proj_repeat_kv_concat_3 = self.model_layers_13_attn_k_proj_repeat_kv_Concat_3(model_layers_13_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_500, model_layers_13_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_501);  model_layers_13_attn_k_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_500 = model_layers_13_attn_k_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_501 = None
        initializers_onnx_initializer_502 = self.initializers.onnx_initializer_502
        model_layers_13_attn_v_proj_repeat_kv_equal = self.model_layers_13_attn_v_proj_repeat_kv_Equal(model_layers_13_attn_v_proj_repeat_kv_concat_2, initializers_onnx_initializer_502);  initializers_onnx_initializer_502 = None
        initializers_onnx_initializer_503 = self.initializers.onnx_initializer_503
        model_layers_13_attn_k_proj_repeat_kv_equal = self.model_layers_13_attn_k_proj_repeat_kv_Equal(model_layers_13_attn_k_proj_repeat_kv_concat_2, initializers_onnx_initializer_503);  initializers_onnx_initializer_503 = None
        initializers_onnx_initializer_504 = self.initializers.onnx_initializer_504
        model_layers_13_attn_v_proj_repeat_kv_where = self.model_layers_13_attn_v_proj_repeat_kv_Where(model_layers_13_attn_v_proj_repeat_kv_equal, initializers_onnx_initializer_504, model_layers_13_attn_v_proj_repeat_kv_concat_2);  model_layers_13_attn_v_proj_repeat_kv_equal = initializers_onnx_initializer_504 = model_layers_13_attn_v_proj_repeat_kv_concat_2 = None
        initializers_onnx_initializer_505 = self.initializers.onnx_initializer_505
        model_layers_13_attn_k_proj_repeat_kv_where = self.model_layers_13_attn_k_proj_repeat_kv_Where(model_layers_13_attn_k_proj_repeat_kv_equal, initializers_onnx_initializer_505, model_layers_13_attn_k_proj_repeat_kv_concat_2);  model_layers_13_attn_k_proj_repeat_kv_equal = initializers_onnx_initializer_505 = model_layers_13_attn_k_proj_repeat_kv_concat_2 = None
        model_layers_13_attn_v_proj_repeat_kv_expand = self.model_layers_13_attn_v_proj_repeat_kv_Expand(model_layers_13_attn_v_proj_repeat_kv_unsqueeze_5, model_layers_13_attn_v_proj_repeat_kv_where);  model_layers_13_attn_v_proj_repeat_kv_unsqueeze_5 = model_layers_13_attn_v_proj_repeat_kv_where = None
        model_layers_13_attn_k_proj_repeat_kv_expand = self.model_layers_13_attn_k_proj_repeat_kv_Expand(model_layers_13_attn_k_proj_repeat_kv_unsqueeze_5, model_layers_13_attn_k_proj_repeat_kv_where);  model_layers_13_attn_k_proj_repeat_kv_unsqueeze_5 = model_layers_13_attn_k_proj_repeat_kv_where = None
        model_layers_13_attn_v_proj_repeat_kv_reshape_3 = self.model_layers_13_attn_v_proj_repeat_kv_Reshape_3(model_layers_13_attn_v_proj_repeat_kv_expand, model_layers_13_attn_v_proj_repeat_kv_concat_3);  model_layers_13_attn_v_proj_repeat_kv_expand = model_layers_13_attn_v_proj_repeat_kv_concat_3 = None
        model_layers_13_attn_k_proj_repeat_kv_reshape_3 = self.model_layers_13_attn_k_proj_repeat_kv_Reshape_3(model_layers_13_attn_k_proj_repeat_kv_expand, model_layers_13_attn_k_proj_repeat_kv_concat_3);  model_layers_13_attn_k_proj_repeat_kv_expand = model_layers_13_attn_k_proj_repeat_kv_concat_3 = None
        model_layers_13_attn_v_proj_repeat_kv_transpose_2 = self.model_layers_13_attn_v_proj_repeat_kv_Transpose_2(model_layers_13_attn_v_proj_repeat_kv_reshape_3);  model_layers_13_attn_v_proj_repeat_kv_reshape_3 = None
        model_layers_13_attn_k_proj_repeat_kv_transpose_2 = self.model_layers_13_attn_k_proj_repeat_kv_Transpose_2(model_layers_13_attn_k_proj_repeat_kv_reshape_3);  model_layers_13_attn_k_proj_repeat_kv_reshape_3 = None
        initializers_onnx_initializer_506 = self.initializers.onnx_initializer_506
        model_layers_13_attn_v_proj_repeat_kv_reshape_4 = self.model_layers_13_attn_v_proj_repeat_kv_Reshape_4(model_layers_13_attn_v_proj_repeat_kv_transpose_2, initializers_onnx_initializer_506);  model_layers_13_attn_v_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_506 = None
        initializers_onnx_initializer_507 = self.initializers.onnx_initializer_507
        model_layers_13_attn_k_proj_repeat_kv_reshape_4 = self.model_layers_13_attn_k_proj_repeat_kv_Reshape_4(model_layers_13_attn_k_proj_repeat_kv_transpose_2, initializers_onnx_initializer_507);  model_layers_13_attn_k_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_507 = None
        com_microsoft__model_layers_13_attn_multi_head_attention = self.com_microsoft__model_layers_13_attn_MultiHeadAttention(com_microsoft__model_layers_13_attn_q_rotary_rotary_embedding, model_layers_13_attn_k_proj_repeat_kv_reshape_4, model_layers_13_attn_v_proj_repeat_kv_reshape_4, attention_mask = model_attn_mask_reformat_tile);  com_microsoft__model_layers_13_attn_q_rotary_rotary_embedding = model_layers_13_attn_k_proj_repeat_kv_reshape_4 = model_layers_13_attn_v_proj_repeat_kv_reshape_4 = None
        getitem_103 = com_microsoft__model_layers_13_attn_multi_head_attention[0];  com_microsoft__model_layers_13_attn_multi_head_attention = None
        initializers_onnx_initializer_508 = self.initializers.onnx_initializer_508
        model_layers_13_attn_o_proj_mat_mul = self.model_layers_13_attn_o_proj_MatMul(getitem_103, initializers_onnx_initializer_508);  getitem_103 = initializers_onnx_initializer_508 = None
        getitem_104 = com_microsoft__model_layers_13_input_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_13_input_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_509 = self.initializers.onnx_initializer_509
        com_microsoft__model_layers_13_post_attention_layernorm_skip_layer_norm = self.com_microsoft__model_layers_13_post_attention_layernorm_SkipLayerNorm(getitem_104, model_layers_13_attn_o_proj_mat_mul, initializers_onnx_initializer_509);  getitem_104 = model_layers_13_attn_o_proj_mat_mul = initializers_onnx_initializer_509 = None
        getitem_105 = com_microsoft__model_layers_13_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_510 = self.initializers.onnx_initializer_510
        model_layers_13_mlp_gate_proj_mat_mul = self.model_layers_13_mlp_gate_proj_MatMul(getitem_105, initializers_onnx_initializer_510);  getitem_105 = initializers_onnx_initializer_510 = None
        getitem_106 = com_microsoft__model_layers_13_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_511 = self.initializers.onnx_initializer_511
        model_layers_13_mlp_up_proj_mat_mul = self.model_layers_13_mlp_up_proj_MatMul(getitem_106, initializers_onnx_initializer_511);  getitem_106 = initializers_onnx_initializer_511 = None
        model_layers_13_mlp_act_fn_sigmoid = self.model_layers_13_mlp_act_fn_Sigmoid(model_layers_13_mlp_gate_proj_mat_mul)
        model_layers_13_mlp_act_fn_mul = self.model_layers_13_mlp_act_fn_Mul(model_layers_13_mlp_gate_proj_mat_mul, model_layers_13_mlp_act_fn_sigmoid);  model_layers_13_mlp_gate_proj_mat_mul = model_layers_13_mlp_act_fn_sigmoid = None
        model_layers_13_mlp_mul = self.model_layers_13_mlp_Mul(model_layers_13_mlp_act_fn_mul, model_layers_13_mlp_up_proj_mat_mul);  model_layers_13_mlp_act_fn_mul = model_layers_13_mlp_up_proj_mat_mul = None
        initializers_onnx_initializer_512 = self.initializers.onnx_initializer_512
        model_layers_13_mlp_down_proj_mat_mul = self.model_layers_13_mlp_down_proj_MatMul(model_layers_13_mlp_mul, initializers_onnx_initializer_512);  model_layers_13_mlp_mul = initializers_onnx_initializer_512 = None
        getitem_107 = com_microsoft__model_layers_13_post_attention_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_13_post_attention_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_513 = self.initializers.onnx_initializer_513
        com_microsoft__model_layers_14_input_layernorm_skip_layer_norm = self.com_microsoft__model_layers_14_input_layernorm_SkipLayerNorm(getitem_107, model_layers_13_mlp_down_proj_mat_mul, initializers_onnx_initializer_513);  getitem_107 = model_layers_13_mlp_down_proj_mat_mul = initializers_onnx_initializer_513 = None
        getitem_108 = com_microsoft__model_layers_14_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_514 = self.initializers.onnx_initializer_514
        model_layers_14_attn_q_proj_mat_mul = self.model_layers_14_attn_q_proj_MatMul(getitem_108, initializers_onnx_initializer_514);  getitem_108 = initializers_onnx_initializer_514 = None
        getitem_109 = com_microsoft__model_layers_14_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_515 = self.initializers.onnx_initializer_515
        model_layers_14_attn_k_proj_mat_mul = self.model_layers_14_attn_k_proj_MatMul(getitem_109, initializers_onnx_initializer_515);  getitem_109 = initializers_onnx_initializer_515 = None
        getitem_110 = com_microsoft__model_layers_14_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_516 = self.initializers.onnx_initializer_516
        model_layers_14_attn_v_proj_mat_mul = self.model_layers_14_attn_v_proj_MatMul(getitem_110, initializers_onnx_initializer_516);  getitem_110 = initializers_onnx_initializer_516 = None
        initializers_onnx_initializer_517 = self.initializers.onnx_initializer_517
        initializers_onnx_initializer_518 = self.initializers.onnx_initializer_518
        com_microsoft__model_layers_14_attn_q_rotary_rotary_embedding = self.com_microsoft__model_layers_14_attn_q_rotary_RotaryEmbedding(model_layers_14_attn_q_proj_mat_mul, input_3, initializers_onnx_initializer_517, initializers_onnx_initializer_518);  model_layers_14_attn_q_proj_mat_mul = initializers_onnx_initializer_517 = initializers_onnx_initializer_518 = None
        initializers_onnx_initializer_519 = self.initializers.onnx_initializer_519
        initializers_onnx_initializer_520 = self.initializers.onnx_initializer_520
        com_microsoft__model_layers_14_attn_k_rotary_rotary_embedding = self.com_microsoft__model_layers_14_attn_k_rotary_RotaryEmbedding(model_layers_14_attn_k_proj_mat_mul, input_3, initializers_onnx_initializer_519, initializers_onnx_initializer_520);  model_layers_14_attn_k_proj_mat_mul = initializers_onnx_initializer_519 = initializers_onnx_initializer_520 = None
        initializers_onnx_initializer_521 = self.initializers.onnx_initializer_521
        model_layers_14_attn_v_proj_repeat_kv_reshape_1 = self.model_layers_14_attn_v_proj_repeat_kv_Reshape_1(model_layers_14_attn_v_proj_mat_mul, initializers_onnx_initializer_521);  model_layers_14_attn_v_proj_mat_mul = initializers_onnx_initializer_521 = None
        initializers_onnx_initializer_522 = self.initializers.onnx_initializer_522
        model_layers_14_attn_k_proj_repeat_kv_reshape_1 = self.model_layers_14_attn_k_proj_repeat_kv_Reshape_1(com_microsoft__model_layers_14_attn_k_rotary_rotary_embedding, initializers_onnx_initializer_522);  com_microsoft__model_layers_14_attn_k_rotary_rotary_embedding = initializers_onnx_initializer_522 = None
        model_layers_14_attn_v_proj_repeat_kv_transpose_1 = self.model_layers_14_attn_v_proj_repeat_kv_Transpose_1(model_layers_14_attn_v_proj_repeat_kv_reshape_1);  model_layers_14_attn_v_proj_repeat_kv_reshape_1 = None
        model_layers_14_attn_k_proj_repeat_kv_transpose_1 = self.model_layers_14_attn_k_proj_repeat_kv_Transpose_1(model_layers_14_attn_k_proj_repeat_kv_reshape_1);  model_layers_14_attn_k_proj_repeat_kv_reshape_1 = None
        model_layers_14_attn_v_proj_repeat_kv_concat_1 = self.model_layers_14_attn_v_proj_repeat_kv_Concat_1(input_33, model_layers_14_attn_v_proj_repeat_kv_transpose_1);  input_33 = model_layers_14_attn_v_proj_repeat_kv_transpose_1 = None
        model_layers_14_attn_k_proj_repeat_kv_concat_1 = self.model_layers_14_attn_k_proj_repeat_kv_Concat_1(input_32, model_layers_14_attn_k_proj_repeat_kv_transpose_1);  input_32 = model_layers_14_attn_k_proj_repeat_kv_transpose_1 = None
        model_layers_14_attn_v_proj_repeat_kv_shape_1 = self.model_layers_14_attn_v_proj_repeat_kv_Shape_1(model_layers_14_attn_v_proj_repeat_kv_concat_1)
        model_layers_14_attn_v_proj_repeat_kv_unsqueeze_5 = self.model_layers_14_attn_v_proj_repeat_kv_Unsqueeze_5(model_layers_14_attn_v_proj_repeat_kv_concat_1)
        model_layers_14_attn_k_proj_repeat_kv_shape_1 = self.model_layers_14_attn_k_proj_repeat_kv_Shape_1(model_layers_14_attn_k_proj_repeat_kv_concat_1)
        model_layers_14_attn_k_proj_repeat_kv_unsqueeze_5 = self.model_layers_14_attn_k_proj_repeat_kv_Unsqueeze_5(model_layers_14_attn_k_proj_repeat_kv_concat_1)
        initializers_onnx_initializer_523 = self.initializers.onnx_initializer_523
        model_layers_14_attn_v_proj_repeat_kv_gather_1 = self.model_layers_14_attn_v_proj_repeat_kv_Gather_1(model_layers_14_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_523);  initializers_onnx_initializer_523 = None
        initializers_onnx_initializer_524 = self.initializers.onnx_initializer_524
        model_layers_14_attn_v_proj_repeat_kv_gather_3 = self.model_layers_14_attn_v_proj_repeat_kv_Gather_3(model_layers_14_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_524);  model_layers_14_attn_v_proj_repeat_kv_shape_1 = initializers_onnx_initializer_524 = None
        initializers_onnx_initializer_525 = self.initializers.onnx_initializer_525
        model_layers_14_attn_k_proj_repeat_kv_gather_1 = self.model_layers_14_attn_k_proj_repeat_kv_Gather_1(model_layers_14_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_525);  initializers_onnx_initializer_525 = None
        initializers_onnx_initializer_526 = self.initializers.onnx_initializer_526
        model_layers_14_attn_k_proj_repeat_kv_gather_3 = self.model_layers_14_attn_k_proj_repeat_kv_Gather_3(model_layers_14_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_526);  model_layers_14_attn_k_proj_repeat_kv_shape_1 = initializers_onnx_initializer_526 = None
        model_layers_14_attn_v_proj_repeat_kv_unsqueeze_1 = self.model_layers_14_attn_v_proj_repeat_kv_Unsqueeze_1(model_layers_14_attn_v_proj_repeat_kv_gather_1);  model_layers_14_attn_v_proj_repeat_kv_gather_1 = None
        model_layers_14_attn_v_proj_repeat_kv_unsqueeze_3 = self.model_layers_14_attn_v_proj_repeat_kv_Unsqueeze_3(model_layers_14_attn_v_proj_repeat_kv_gather_3);  model_layers_14_attn_v_proj_repeat_kv_gather_3 = None
        model_layers_14_attn_k_proj_repeat_kv_unsqueeze_1 = self.model_layers_14_attn_k_proj_repeat_kv_Unsqueeze_1(model_layers_14_attn_k_proj_repeat_kv_gather_1);  model_layers_14_attn_k_proj_repeat_kv_gather_1 = None
        model_layers_14_attn_k_proj_repeat_kv_unsqueeze_3 = self.model_layers_14_attn_k_proj_repeat_kv_Unsqueeze_3(model_layers_14_attn_k_proj_repeat_kv_gather_3);  model_layers_14_attn_k_proj_repeat_kv_gather_3 = None
        initializers_onnx_initializer_527 = self.initializers.onnx_initializer_527
        initializers_onnx_initializer_528 = self.initializers.onnx_initializer_528
        initializers_onnx_initializer_529 = self.initializers.onnx_initializer_529
        model_layers_14_attn_v_proj_repeat_kv_concat_2 = self.model_layers_14_attn_v_proj_repeat_kv_Concat_2(model_layers_14_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_527, initializers_onnx_initializer_528, model_layers_14_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_529);  initializers_onnx_initializer_527 = initializers_onnx_initializer_528 = initializers_onnx_initializer_529 = None
        initializers_onnx_initializer_530 = self.initializers.onnx_initializer_530
        initializers_onnx_initializer_531 = self.initializers.onnx_initializer_531
        model_layers_14_attn_v_proj_repeat_kv_concat_3 = self.model_layers_14_attn_v_proj_repeat_kv_Concat_3(model_layers_14_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_530, model_layers_14_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_531);  model_layers_14_attn_v_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_530 = model_layers_14_attn_v_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_531 = None
        initializers_onnx_initializer_532 = self.initializers.onnx_initializer_532
        initializers_onnx_initializer_533 = self.initializers.onnx_initializer_533
        initializers_onnx_initializer_534 = self.initializers.onnx_initializer_534
        model_layers_14_attn_k_proj_repeat_kv_concat_2 = self.model_layers_14_attn_k_proj_repeat_kv_Concat_2(model_layers_14_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_532, initializers_onnx_initializer_533, model_layers_14_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_534);  initializers_onnx_initializer_532 = initializers_onnx_initializer_533 = initializers_onnx_initializer_534 = None
        initializers_onnx_initializer_535 = self.initializers.onnx_initializer_535
        initializers_onnx_initializer_536 = self.initializers.onnx_initializer_536
        model_layers_14_attn_k_proj_repeat_kv_concat_3 = self.model_layers_14_attn_k_proj_repeat_kv_Concat_3(model_layers_14_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_535, model_layers_14_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_536);  model_layers_14_attn_k_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_535 = model_layers_14_attn_k_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_536 = None
        initializers_onnx_initializer_537 = self.initializers.onnx_initializer_537
        model_layers_14_attn_v_proj_repeat_kv_equal = self.model_layers_14_attn_v_proj_repeat_kv_Equal(model_layers_14_attn_v_proj_repeat_kv_concat_2, initializers_onnx_initializer_537);  initializers_onnx_initializer_537 = None
        initializers_onnx_initializer_538 = self.initializers.onnx_initializer_538
        model_layers_14_attn_k_proj_repeat_kv_equal = self.model_layers_14_attn_k_proj_repeat_kv_Equal(model_layers_14_attn_k_proj_repeat_kv_concat_2, initializers_onnx_initializer_538);  initializers_onnx_initializer_538 = None
        initializers_onnx_initializer_539 = self.initializers.onnx_initializer_539
        model_layers_14_attn_v_proj_repeat_kv_where = self.model_layers_14_attn_v_proj_repeat_kv_Where(model_layers_14_attn_v_proj_repeat_kv_equal, initializers_onnx_initializer_539, model_layers_14_attn_v_proj_repeat_kv_concat_2);  model_layers_14_attn_v_proj_repeat_kv_equal = initializers_onnx_initializer_539 = model_layers_14_attn_v_proj_repeat_kv_concat_2 = None
        initializers_onnx_initializer_540 = self.initializers.onnx_initializer_540
        model_layers_14_attn_k_proj_repeat_kv_where = self.model_layers_14_attn_k_proj_repeat_kv_Where(model_layers_14_attn_k_proj_repeat_kv_equal, initializers_onnx_initializer_540, model_layers_14_attn_k_proj_repeat_kv_concat_2);  model_layers_14_attn_k_proj_repeat_kv_equal = initializers_onnx_initializer_540 = model_layers_14_attn_k_proj_repeat_kv_concat_2 = None
        model_layers_14_attn_v_proj_repeat_kv_expand = self.model_layers_14_attn_v_proj_repeat_kv_Expand(model_layers_14_attn_v_proj_repeat_kv_unsqueeze_5, model_layers_14_attn_v_proj_repeat_kv_where);  model_layers_14_attn_v_proj_repeat_kv_unsqueeze_5 = model_layers_14_attn_v_proj_repeat_kv_where = None
        model_layers_14_attn_k_proj_repeat_kv_expand = self.model_layers_14_attn_k_proj_repeat_kv_Expand(model_layers_14_attn_k_proj_repeat_kv_unsqueeze_5, model_layers_14_attn_k_proj_repeat_kv_where);  model_layers_14_attn_k_proj_repeat_kv_unsqueeze_5 = model_layers_14_attn_k_proj_repeat_kv_where = None
        model_layers_14_attn_v_proj_repeat_kv_reshape_3 = self.model_layers_14_attn_v_proj_repeat_kv_Reshape_3(model_layers_14_attn_v_proj_repeat_kv_expand, model_layers_14_attn_v_proj_repeat_kv_concat_3);  model_layers_14_attn_v_proj_repeat_kv_expand = model_layers_14_attn_v_proj_repeat_kv_concat_3 = None
        model_layers_14_attn_k_proj_repeat_kv_reshape_3 = self.model_layers_14_attn_k_proj_repeat_kv_Reshape_3(model_layers_14_attn_k_proj_repeat_kv_expand, model_layers_14_attn_k_proj_repeat_kv_concat_3);  model_layers_14_attn_k_proj_repeat_kv_expand = model_layers_14_attn_k_proj_repeat_kv_concat_3 = None
        model_layers_14_attn_v_proj_repeat_kv_transpose_2 = self.model_layers_14_attn_v_proj_repeat_kv_Transpose_2(model_layers_14_attn_v_proj_repeat_kv_reshape_3);  model_layers_14_attn_v_proj_repeat_kv_reshape_3 = None
        model_layers_14_attn_k_proj_repeat_kv_transpose_2 = self.model_layers_14_attn_k_proj_repeat_kv_Transpose_2(model_layers_14_attn_k_proj_repeat_kv_reshape_3);  model_layers_14_attn_k_proj_repeat_kv_reshape_3 = None
        initializers_onnx_initializer_541 = self.initializers.onnx_initializer_541
        model_layers_14_attn_v_proj_repeat_kv_reshape_4 = self.model_layers_14_attn_v_proj_repeat_kv_Reshape_4(model_layers_14_attn_v_proj_repeat_kv_transpose_2, initializers_onnx_initializer_541);  model_layers_14_attn_v_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_541 = None
        initializers_onnx_initializer_542 = self.initializers.onnx_initializer_542
        model_layers_14_attn_k_proj_repeat_kv_reshape_4 = self.model_layers_14_attn_k_proj_repeat_kv_Reshape_4(model_layers_14_attn_k_proj_repeat_kv_transpose_2, initializers_onnx_initializer_542);  model_layers_14_attn_k_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_542 = None
        com_microsoft__model_layers_14_attn_multi_head_attention = self.com_microsoft__model_layers_14_attn_MultiHeadAttention(com_microsoft__model_layers_14_attn_q_rotary_rotary_embedding, model_layers_14_attn_k_proj_repeat_kv_reshape_4, model_layers_14_attn_v_proj_repeat_kv_reshape_4, attention_mask = model_attn_mask_reformat_tile);  com_microsoft__model_layers_14_attn_q_rotary_rotary_embedding = model_layers_14_attn_k_proj_repeat_kv_reshape_4 = model_layers_14_attn_v_proj_repeat_kv_reshape_4 = None
        getitem_111 = com_microsoft__model_layers_14_attn_multi_head_attention[0];  com_microsoft__model_layers_14_attn_multi_head_attention = None
        initializers_onnx_initializer_543 = self.initializers.onnx_initializer_543
        model_layers_14_attn_o_proj_mat_mul = self.model_layers_14_attn_o_proj_MatMul(getitem_111, initializers_onnx_initializer_543);  getitem_111 = initializers_onnx_initializer_543 = None
        getitem_112 = com_microsoft__model_layers_14_input_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_14_input_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_544 = self.initializers.onnx_initializer_544
        com_microsoft__model_layers_14_post_attention_layernorm_skip_layer_norm = self.com_microsoft__model_layers_14_post_attention_layernorm_SkipLayerNorm(getitem_112, model_layers_14_attn_o_proj_mat_mul, initializers_onnx_initializer_544);  getitem_112 = model_layers_14_attn_o_proj_mat_mul = initializers_onnx_initializer_544 = None
        getitem_113 = com_microsoft__model_layers_14_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_545 = self.initializers.onnx_initializer_545
        model_layers_14_mlp_gate_proj_mat_mul = self.model_layers_14_mlp_gate_proj_MatMul(getitem_113, initializers_onnx_initializer_545);  getitem_113 = initializers_onnx_initializer_545 = None
        getitem_114 = com_microsoft__model_layers_14_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_546 = self.initializers.onnx_initializer_546
        model_layers_14_mlp_up_proj_mat_mul = self.model_layers_14_mlp_up_proj_MatMul(getitem_114, initializers_onnx_initializer_546);  getitem_114 = initializers_onnx_initializer_546 = None
        model_layers_14_mlp_act_fn_sigmoid = self.model_layers_14_mlp_act_fn_Sigmoid(model_layers_14_mlp_gate_proj_mat_mul)
        model_layers_14_mlp_act_fn_mul = self.model_layers_14_mlp_act_fn_Mul(model_layers_14_mlp_gate_proj_mat_mul, model_layers_14_mlp_act_fn_sigmoid);  model_layers_14_mlp_gate_proj_mat_mul = model_layers_14_mlp_act_fn_sigmoid = None
        model_layers_14_mlp_mul = self.model_layers_14_mlp_Mul(model_layers_14_mlp_act_fn_mul, model_layers_14_mlp_up_proj_mat_mul);  model_layers_14_mlp_act_fn_mul = model_layers_14_mlp_up_proj_mat_mul = None
        initializers_onnx_initializer_547 = self.initializers.onnx_initializer_547
        model_layers_14_mlp_down_proj_mat_mul = self.model_layers_14_mlp_down_proj_MatMul(model_layers_14_mlp_mul, initializers_onnx_initializer_547);  model_layers_14_mlp_mul = initializers_onnx_initializer_547 = None
        getitem_115 = com_microsoft__model_layers_14_post_attention_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_14_post_attention_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_548 = self.initializers.onnx_initializer_548
        com_microsoft__model_layers_15_input_layernorm_skip_layer_norm = self.com_microsoft__model_layers_15_input_layernorm_SkipLayerNorm(getitem_115, model_layers_14_mlp_down_proj_mat_mul, initializers_onnx_initializer_548);  getitem_115 = model_layers_14_mlp_down_proj_mat_mul = initializers_onnx_initializer_548 = None
        getitem_116 = com_microsoft__model_layers_15_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_549 = self.initializers.onnx_initializer_549
        model_layers_15_attn_q_proj_mat_mul = self.model_layers_15_attn_q_proj_MatMul(getitem_116, initializers_onnx_initializer_549);  getitem_116 = initializers_onnx_initializer_549 = None
        getitem_117 = com_microsoft__model_layers_15_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_550 = self.initializers.onnx_initializer_550
        model_layers_15_attn_k_proj_mat_mul = self.model_layers_15_attn_k_proj_MatMul(getitem_117, initializers_onnx_initializer_550);  getitem_117 = initializers_onnx_initializer_550 = None
        getitem_118 = com_microsoft__model_layers_15_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_551 = self.initializers.onnx_initializer_551
        model_layers_15_attn_v_proj_mat_mul = self.model_layers_15_attn_v_proj_MatMul(getitem_118, initializers_onnx_initializer_551);  getitem_118 = initializers_onnx_initializer_551 = None
        initializers_onnx_initializer_552 = self.initializers.onnx_initializer_552
        initializers_onnx_initializer_553 = self.initializers.onnx_initializer_553
        com_microsoft__model_layers_15_attn_q_rotary_rotary_embedding = self.com_microsoft__model_layers_15_attn_q_rotary_RotaryEmbedding(model_layers_15_attn_q_proj_mat_mul, input_3, initializers_onnx_initializer_552, initializers_onnx_initializer_553);  model_layers_15_attn_q_proj_mat_mul = initializers_onnx_initializer_552 = initializers_onnx_initializer_553 = None
        initializers_onnx_initializer_554 = self.initializers.onnx_initializer_554
        initializers_onnx_initializer_555 = self.initializers.onnx_initializer_555
        com_microsoft__model_layers_15_attn_k_rotary_rotary_embedding = self.com_microsoft__model_layers_15_attn_k_rotary_RotaryEmbedding(model_layers_15_attn_k_proj_mat_mul, input_3, initializers_onnx_initializer_554, initializers_onnx_initializer_555);  model_layers_15_attn_k_proj_mat_mul = initializers_onnx_initializer_554 = initializers_onnx_initializer_555 = None
        initializers_onnx_initializer_556 = self.initializers.onnx_initializer_556
        model_layers_15_attn_v_proj_repeat_kv_reshape_1 = self.model_layers_15_attn_v_proj_repeat_kv_Reshape_1(model_layers_15_attn_v_proj_mat_mul, initializers_onnx_initializer_556);  model_layers_15_attn_v_proj_mat_mul = initializers_onnx_initializer_556 = None
        initializers_onnx_initializer_557 = self.initializers.onnx_initializer_557
        model_layers_15_attn_k_proj_repeat_kv_reshape_1 = self.model_layers_15_attn_k_proj_repeat_kv_Reshape_1(com_microsoft__model_layers_15_attn_k_rotary_rotary_embedding, initializers_onnx_initializer_557);  com_microsoft__model_layers_15_attn_k_rotary_rotary_embedding = initializers_onnx_initializer_557 = None
        model_layers_15_attn_v_proj_repeat_kv_transpose_1 = self.model_layers_15_attn_v_proj_repeat_kv_Transpose_1(model_layers_15_attn_v_proj_repeat_kv_reshape_1);  model_layers_15_attn_v_proj_repeat_kv_reshape_1 = None
        model_layers_15_attn_k_proj_repeat_kv_transpose_1 = self.model_layers_15_attn_k_proj_repeat_kv_Transpose_1(model_layers_15_attn_k_proj_repeat_kv_reshape_1);  model_layers_15_attn_k_proj_repeat_kv_reshape_1 = None
        model_layers_15_attn_v_proj_repeat_kv_concat_1 = self.model_layers_15_attn_v_proj_repeat_kv_Concat_1(input_35, model_layers_15_attn_v_proj_repeat_kv_transpose_1);  input_35 = model_layers_15_attn_v_proj_repeat_kv_transpose_1 = None
        model_layers_15_attn_k_proj_repeat_kv_concat_1 = self.model_layers_15_attn_k_proj_repeat_kv_Concat_1(input_34, model_layers_15_attn_k_proj_repeat_kv_transpose_1);  input_34 = model_layers_15_attn_k_proj_repeat_kv_transpose_1 = None
        model_layers_15_attn_v_proj_repeat_kv_shape_1 = self.model_layers_15_attn_v_proj_repeat_kv_Shape_1(model_layers_15_attn_v_proj_repeat_kv_concat_1)
        model_layers_15_attn_v_proj_repeat_kv_unsqueeze_5 = self.model_layers_15_attn_v_proj_repeat_kv_Unsqueeze_5(model_layers_15_attn_v_proj_repeat_kv_concat_1)
        model_layers_15_attn_k_proj_repeat_kv_shape_1 = self.model_layers_15_attn_k_proj_repeat_kv_Shape_1(model_layers_15_attn_k_proj_repeat_kv_concat_1)
        model_layers_15_attn_k_proj_repeat_kv_unsqueeze_5 = self.model_layers_15_attn_k_proj_repeat_kv_Unsqueeze_5(model_layers_15_attn_k_proj_repeat_kv_concat_1)
        initializers_onnx_initializer_558 = self.initializers.onnx_initializer_558
        model_layers_15_attn_v_proj_repeat_kv_gather_1 = self.model_layers_15_attn_v_proj_repeat_kv_Gather_1(model_layers_15_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_558);  initializers_onnx_initializer_558 = None
        initializers_onnx_initializer_559 = self.initializers.onnx_initializer_559
        model_layers_15_attn_v_proj_repeat_kv_gather_3 = self.model_layers_15_attn_v_proj_repeat_kv_Gather_3(model_layers_15_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_559);  model_layers_15_attn_v_proj_repeat_kv_shape_1 = initializers_onnx_initializer_559 = None
        initializers_onnx_initializer_560 = self.initializers.onnx_initializer_560
        model_layers_15_attn_k_proj_repeat_kv_gather_1 = self.model_layers_15_attn_k_proj_repeat_kv_Gather_1(model_layers_15_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_560);  initializers_onnx_initializer_560 = None
        initializers_onnx_initializer_561 = self.initializers.onnx_initializer_561
        model_layers_15_attn_k_proj_repeat_kv_gather_3 = self.model_layers_15_attn_k_proj_repeat_kv_Gather_3(model_layers_15_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_561);  model_layers_15_attn_k_proj_repeat_kv_shape_1 = initializers_onnx_initializer_561 = None
        model_layers_15_attn_v_proj_repeat_kv_unsqueeze_1 = self.model_layers_15_attn_v_proj_repeat_kv_Unsqueeze_1(model_layers_15_attn_v_proj_repeat_kv_gather_1);  model_layers_15_attn_v_proj_repeat_kv_gather_1 = None
        model_layers_15_attn_v_proj_repeat_kv_unsqueeze_3 = self.model_layers_15_attn_v_proj_repeat_kv_Unsqueeze_3(model_layers_15_attn_v_proj_repeat_kv_gather_3);  model_layers_15_attn_v_proj_repeat_kv_gather_3 = None
        model_layers_15_attn_k_proj_repeat_kv_unsqueeze_1 = self.model_layers_15_attn_k_proj_repeat_kv_Unsqueeze_1(model_layers_15_attn_k_proj_repeat_kv_gather_1);  model_layers_15_attn_k_proj_repeat_kv_gather_1 = None
        model_layers_15_attn_k_proj_repeat_kv_unsqueeze_3 = self.model_layers_15_attn_k_proj_repeat_kv_Unsqueeze_3(model_layers_15_attn_k_proj_repeat_kv_gather_3);  model_layers_15_attn_k_proj_repeat_kv_gather_3 = None
        initializers_onnx_initializer_562 = self.initializers.onnx_initializer_562
        initializers_onnx_initializer_563 = self.initializers.onnx_initializer_563
        initializers_onnx_initializer_564 = self.initializers.onnx_initializer_564
        model_layers_15_attn_v_proj_repeat_kv_concat_2 = self.model_layers_15_attn_v_proj_repeat_kv_Concat_2(model_layers_15_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_562, initializers_onnx_initializer_563, model_layers_15_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_564);  initializers_onnx_initializer_562 = initializers_onnx_initializer_563 = initializers_onnx_initializer_564 = None
        initializers_onnx_initializer_565 = self.initializers.onnx_initializer_565
        initializers_onnx_initializer_566 = self.initializers.onnx_initializer_566
        model_layers_15_attn_v_proj_repeat_kv_concat_3 = self.model_layers_15_attn_v_proj_repeat_kv_Concat_3(model_layers_15_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_565, model_layers_15_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_566);  model_layers_15_attn_v_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_565 = model_layers_15_attn_v_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_566 = None
        initializers_onnx_initializer_567 = self.initializers.onnx_initializer_567
        initializers_onnx_initializer_568 = self.initializers.onnx_initializer_568
        initializers_onnx_initializer_569 = self.initializers.onnx_initializer_569
        model_layers_15_attn_k_proj_repeat_kv_concat_2 = self.model_layers_15_attn_k_proj_repeat_kv_Concat_2(model_layers_15_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_567, initializers_onnx_initializer_568, model_layers_15_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_569);  initializers_onnx_initializer_567 = initializers_onnx_initializer_568 = initializers_onnx_initializer_569 = None
        initializers_onnx_initializer_570 = self.initializers.onnx_initializer_570
        initializers_onnx_initializer_571 = self.initializers.onnx_initializer_571
        model_layers_15_attn_k_proj_repeat_kv_concat_3 = self.model_layers_15_attn_k_proj_repeat_kv_Concat_3(model_layers_15_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_570, model_layers_15_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_571);  model_layers_15_attn_k_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_570 = model_layers_15_attn_k_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_571 = None
        initializers_onnx_initializer_572 = self.initializers.onnx_initializer_572
        model_layers_15_attn_v_proj_repeat_kv_equal = self.model_layers_15_attn_v_proj_repeat_kv_Equal(model_layers_15_attn_v_proj_repeat_kv_concat_2, initializers_onnx_initializer_572);  initializers_onnx_initializer_572 = None
        initializers_onnx_initializer_573 = self.initializers.onnx_initializer_573
        model_layers_15_attn_k_proj_repeat_kv_equal = self.model_layers_15_attn_k_proj_repeat_kv_Equal(model_layers_15_attn_k_proj_repeat_kv_concat_2, initializers_onnx_initializer_573);  initializers_onnx_initializer_573 = None
        initializers_onnx_initializer_574 = self.initializers.onnx_initializer_574
        model_layers_15_attn_v_proj_repeat_kv_where = self.model_layers_15_attn_v_proj_repeat_kv_Where(model_layers_15_attn_v_proj_repeat_kv_equal, initializers_onnx_initializer_574, model_layers_15_attn_v_proj_repeat_kv_concat_2);  model_layers_15_attn_v_proj_repeat_kv_equal = initializers_onnx_initializer_574 = model_layers_15_attn_v_proj_repeat_kv_concat_2 = None
        initializers_onnx_initializer_575 = self.initializers.onnx_initializer_575
        model_layers_15_attn_k_proj_repeat_kv_where = self.model_layers_15_attn_k_proj_repeat_kv_Where(model_layers_15_attn_k_proj_repeat_kv_equal, initializers_onnx_initializer_575, model_layers_15_attn_k_proj_repeat_kv_concat_2);  model_layers_15_attn_k_proj_repeat_kv_equal = initializers_onnx_initializer_575 = model_layers_15_attn_k_proj_repeat_kv_concat_2 = None
        model_layers_15_attn_v_proj_repeat_kv_expand = self.model_layers_15_attn_v_proj_repeat_kv_Expand(model_layers_15_attn_v_proj_repeat_kv_unsqueeze_5, model_layers_15_attn_v_proj_repeat_kv_where);  model_layers_15_attn_v_proj_repeat_kv_unsqueeze_5 = model_layers_15_attn_v_proj_repeat_kv_where = None
        model_layers_15_attn_k_proj_repeat_kv_expand = self.model_layers_15_attn_k_proj_repeat_kv_Expand(model_layers_15_attn_k_proj_repeat_kv_unsqueeze_5, model_layers_15_attn_k_proj_repeat_kv_where);  model_layers_15_attn_k_proj_repeat_kv_unsqueeze_5 = model_layers_15_attn_k_proj_repeat_kv_where = None
        model_layers_15_attn_v_proj_repeat_kv_reshape_3 = self.model_layers_15_attn_v_proj_repeat_kv_Reshape_3(model_layers_15_attn_v_proj_repeat_kv_expand, model_layers_15_attn_v_proj_repeat_kv_concat_3);  model_layers_15_attn_v_proj_repeat_kv_expand = model_layers_15_attn_v_proj_repeat_kv_concat_3 = None
        model_layers_15_attn_k_proj_repeat_kv_reshape_3 = self.model_layers_15_attn_k_proj_repeat_kv_Reshape_3(model_layers_15_attn_k_proj_repeat_kv_expand, model_layers_15_attn_k_proj_repeat_kv_concat_3);  model_layers_15_attn_k_proj_repeat_kv_expand = model_layers_15_attn_k_proj_repeat_kv_concat_3 = None
        model_layers_15_attn_v_proj_repeat_kv_transpose_2 = self.model_layers_15_attn_v_proj_repeat_kv_Transpose_2(model_layers_15_attn_v_proj_repeat_kv_reshape_3);  model_layers_15_attn_v_proj_repeat_kv_reshape_3 = None
        model_layers_15_attn_k_proj_repeat_kv_transpose_2 = self.model_layers_15_attn_k_proj_repeat_kv_Transpose_2(model_layers_15_attn_k_proj_repeat_kv_reshape_3);  model_layers_15_attn_k_proj_repeat_kv_reshape_3 = None
        initializers_onnx_initializer_576 = self.initializers.onnx_initializer_576
        model_layers_15_attn_v_proj_repeat_kv_reshape_4 = self.model_layers_15_attn_v_proj_repeat_kv_Reshape_4(model_layers_15_attn_v_proj_repeat_kv_transpose_2, initializers_onnx_initializer_576);  model_layers_15_attn_v_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_576 = None
        initializers_onnx_initializer_577 = self.initializers.onnx_initializer_577
        model_layers_15_attn_k_proj_repeat_kv_reshape_4 = self.model_layers_15_attn_k_proj_repeat_kv_Reshape_4(model_layers_15_attn_k_proj_repeat_kv_transpose_2, initializers_onnx_initializer_577);  model_layers_15_attn_k_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_577 = None
        com_microsoft__model_layers_15_attn_multi_head_attention = self.com_microsoft__model_layers_15_attn_MultiHeadAttention(com_microsoft__model_layers_15_attn_q_rotary_rotary_embedding, model_layers_15_attn_k_proj_repeat_kv_reshape_4, model_layers_15_attn_v_proj_repeat_kv_reshape_4, attention_mask = model_attn_mask_reformat_tile);  com_microsoft__model_layers_15_attn_q_rotary_rotary_embedding = model_layers_15_attn_k_proj_repeat_kv_reshape_4 = model_layers_15_attn_v_proj_repeat_kv_reshape_4 = None
        getitem_119 = com_microsoft__model_layers_15_attn_multi_head_attention[0];  com_microsoft__model_layers_15_attn_multi_head_attention = None
        initializers_onnx_initializer_578 = self.initializers.onnx_initializer_578
        model_layers_15_attn_o_proj_mat_mul = self.model_layers_15_attn_o_proj_MatMul(getitem_119, initializers_onnx_initializer_578);  getitem_119 = initializers_onnx_initializer_578 = None
        getitem_120 = com_microsoft__model_layers_15_input_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_15_input_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_579 = self.initializers.onnx_initializer_579
        com_microsoft__model_layers_15_post_attention_layernorm_skip_layer_norm = self.com_microsoft__model_layers_15_post_attention_layernorm_SkipLayerNorm(getitem_120, model_layers_15_attn_o_proj_mat_mul, initializers_onnx_initializer_579);  getitem_120 = model_layers_15_attn_o_proj_mat_mul = initializers_onnx_initializer_579 = None
        getitem_121 = com_microsoft__model_layers_15_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_580 = self.initializers.onnx_initializer_580
        model_layers_15_mlp_gate_proj_mat_mul = self.model_layers_15_mlp_gate_proj_MatMul(getitem_121, initializers_onnx_initializer_580);  getitem_121 = initializers_onnx_initializer_580 = None
        getitem_122 = com_microsoft__model_layers_15_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_581 = self.initializers.onnx_initializer_581
        model_layers_15_mlp_up_proj_mat_mul = self.model_layers_15_mlp_up_proj_MatMul(getitem_122, initializers_onnx_initializer_581);  getitem_122 = initializers_onnx_initializer_581 = None
        model_layers_15_mlp_act_fn_sigmoid = self.model_layers_15_mlp_act_fn_Sigmoid(model_layers_15_mlp_gate_proj_mat_mul)
        model_layers_15_mlp_act_fn_mul = self.model_layers_15_mlp_act_fn_Mul(model_layers_15_mlp_gate_proj_mat_mul, model_layers_15_mlp_act_fn_sigmoid);  model_layers_15_mlp_gate_proj_mat_mul = model_layers_15_mlp_act_fn_sigmoid = None
        model_layers_15_mlp_mul = self.model_layers_15_mlp_Mul(model_layers_15_mlp_act_fn_mul, model_layers_15_mlp_up_proj_mat_mul);  model_layers_15_mlp_act_fn_mul = model_layers_15_mlp_up_proj_mat_mul = None
        initializers_onnx_initializer_582 = self.initializers.onnx_initializer_582
        model_layers_15_mlp_down_proj_mat_mul = self.model_layers_15_mlp_down_proj_MatMul(model_layers_15_mlp_mul, initializers_onnx_initializer_582);  model_layers_15_mlp_mul = initializers_onnx_initializer_582 = None
        getitem_123 = com_microsoft__model_layers_15_post_attention_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_15_post_attention_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_583 = self.initializers.onnx_initializer_583
        com_microsoft__model_layers_16_input_layernorm_skip_layer_norm = self.com_microsoft__model_layers_16_input_layernorm_SkipLayerNorm(getitem_123, model_layers_15_mlp_down_proj_mat_mul, initializers_onnx_initializer_583);  getitem_123 = model_layers_15_mlp_down_proj_mat_mul = initializers_onnx_initializer_583 = None
        getitem_124 = com_microsoft__model_layers_16_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_584 = self.initializers.onnx_initializer_584
        model_layers_16_attn_q_proj_mat_mul = self.model_layers_16_attn_q_proj_MatMul(getitem_124, initializers_onnx_initializer_584);  getitem_124 = initializers_onnx_initializer_584 = None
        getitem_125 = com_microsoft__model_layers_16_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_585 = self.initializers.onnx_initializer_585
        model_layers_16_attn_k_proj_mat_mul = self.model_layers_16_attn_k_proj_MatMul(getitem_125, initializers_onnx_initializer_585);  getitem_125 = initializers_onnx_initializer_585 = None
        getitem_126 = com_microsoft__model_layers_16_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_586 = self.initializers.onnx_initializer_586
        model_layers_16_attn_v_proj_mat_mul = self.model_layers_16_attn_v_proj_MatMul(getitem_126, initializers_onnx_initializer_586);  getitem_126 = initializers_onnx_initializer_586 = None
        initializers_onnx_initializer_587 = self.initializers.onnx_initializer_587
        initializers_onnx_initializer_588 = self.initializers.onnx_initializer_588
        com_microsoft__model_layers_16_attn_q_rotary_rotary_embedding = self.com_microsoft__model_layers_16_attn_q_rotary_RotaryEmbedding(model_layers_16_attn_q_proj_mat_mul, input_3, initializers_onnx_initializer_587, initializers_onnx_initializer_588);  model_layers_16_attn_q_proj_mat_mul = initializers_onnx_initializer_587 = initializers_onnx_initializer_588 = None
        initializers_onnx_initializer_589 = self.initializers.onnx_initializer_589
        initializers_onnx_initializer_590 = self.initializers.onnx_initializer_590
        com_microsoft__model_layers_16_attn_k_rotary_rotary_embedding = self.com_microsoft__model_layers_16_attn_k_rotary_RotaryEmbedding(model_layers_16_attn_k_proj_mat_mul, input_3, initializers_onnx_initializer_589, initializers_onnx_initializer_590);  model_layers_16_attn_k_proj_mat_mul = initializers_onnx_initializer_589 = initializers_onnx_initializer_590 = None
        initializers_onnx_initializer_591 = self.initializers.onnx_initializer_591
        model_layers_16_attn_v_proj_repeat_kv_reshape_1 = self.model_layers_16_attn_v_proj_repeat_kv_Reshape_1(model_layers_16_attn_v_proj_mat_mul, initializers_onnx_initializer_591);  model_layers_16_attn_v_proj_mat_mul = initializers_onnx_initializer_591 = None
        initializers_onnx_initializer_592 = self.initializers.onnx_initializer_592
        model_layers_16_attn_k_proj_repeat_kv_reshape_1 = self.model_layers_16_attn_k_proj_repeat_kv_Reshape_1(com_microsoft__model_layers_16_attn_k_rotary_rotary_embedding, initializers_onnx_initializer_592);  com_microsoft__model_layers_16_attn_k_rotary_rotary_embedding = initializers_onnx_initializer_592 = None
        model_layers_16_attn_v_proj_repeat_kv_transpose_1 = self.model_layers_16_attn_v_proj_repeat_kv_Transpose_1(model_layers_16_attn_v_proj_repeat_kv_reshape_1);  model_layers_16_attn_v_proj_repeat_kv_reshape_1 = None
        model_layers_16_attn_k_proj_repeat_kv_transpose_1 = self.model_layers_16_attn_k_proj_repeat_kv_Transpose_1(model_layers_16_attn_k_proj_repeat_kv_reshape_1);  model_layers_16_attn_k_proj_repeat_kv_reshape_1 = None
        model_layers_16_attn_v_proj_repeat_kv_concat_1 = self.model_layers_16_attn_v_proj_repeat_kv_Concat_1(input_37, model_layers_16_attn_v_proj_repeat_kv_transpose_1);  input_37 = model_layers_16_attn_v_proj_repeat_kv_transpose_1 = None
        model_layers_16_attn_k_proj_repeat_kv_concat_1 = self.model_layers_16_attn_k_proj_repeat_kv_Concat_1(input_36, model_layers_16_attn_k_proj_repeat_kv_transpose_1);  input_36 = model_layers_16_attn_k_proj_repeat_kv_transpose_1 = None
        model_layers_16_attn_v_proj_repeat_kv_shape_1 = self.model_layers_16_attn_v_proj_repeat_kv_Shape_1(model_layers_16_attn_v_proj_repeat_kv_concat_1)
        model_layers_16_attn_v_proj_repeat_kv_unsqueeze_5 = self.model_layers_16_attn_v_proj_repeat_kv_Unsqueeze_5(model_layers_16_attn_v_proj_repeat_kv_concat_1)
        model_layers_16_attn_k_proj_repeat_kv_shape_1 = self.model_layers_16_attn_k_proj_repeat_kv_Shape_1(model_layers_16_attn_k_proj_repeat_kv_concat_1)
        model_layers_16_attn_k_proj_repeat_kv_unsqueeze_5 = self.model_layers_16_attn_k_proj_repeat_kv_Unsqueeze_5(model_layers_16_attn_k_proj_repeat_kv_concat_1)
        initializers_onnx_initializer_593 = self.initializers.onnx_initializer_593
        model_layers_16_attn_v_proj_repeat_kv_gather_1 = self.model_layers_16_attn_v_proj_repeat_kv_Gather_1(model_layers_16_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_593);  initializers_onnx_initializer_593 = None
        initializers_onnx_initializer_594 = self.initializers.onnx_initializer_594
        model_layers_16_attn_v_proj_repeat_kv_gather_3 = self.model_layers_16_attn_v_proj_repeat_kv_Gather_3(model_layers_16_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_594);  model_layers_16_attn_v_proj_repeat_kv_shape_1 = initializers_onnx_initializer_594 = None
        initializers_onnx_initializer_595 = self.initializers.onnx_initializer_595
        model_layers_16_attn_k_proj_repeat_kv_gather_1 = self.model_layers_16_attn_k_proj_repeat_kv_Gather_1(model_layers_16_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_595);  initializers_onnx_initializer_595 = None
        initializers_onnx_initializer_596 = self.initializers.onnx_initializer_596
        model_layers_16_attn_k_proj_repeat_kv_gather_3 = self.model_layers_16_attn_k_proj_repeat_kv_Gather_3(model_layers_16_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_596);  model_layers_16_attn_k_proj_repeat_kv_shape_1 = initializers_onnx_initializer_596 = None
        model_layers_16_attn_v_proj_repeat_kv_unsqueeze_1 = self.model_layers_16_attn_v_proj_repeat_kv_Unsqueeze_1(model_layers_16_attn_v_proj_repeat_kv_gather_1);  model_layers_16_attn_v_proj_repeat_kv_gather_1 = None
        model_layers_16_attn_v_proj_repeat_kv_unsqueeze_3 = self.model_layers_16_attn_v_proj_repeat_kv_Unsqueeze_3(model_layers_16_attn_v_proj_repeat_kv_gather_3);  model_layers_16_attn_v_proj_repeat_kv_gather_3 = None
        model_layers_16_attn_k_proj_repeat_kv_unsqueeze_1 = self.model_layers_16_attn_k_proj_repeat_kv_Unsqueeze_1(model_layers_16_attn_k_proj_repeat_kv_gather_1);  model_layers_16_attn_k_proj_repeat_kv_gather_1 = None
        model_layers_16_attn_k_proj_repeat_kv_unsqueeze_3 = self.model_layers_16_attn_k_proj_repeat_kv_Unsqueeze_3(model_layers_16_attn_k_proj_repeat_kv_gather_3);  model_layers_16_attn_k_proj_repeat_kv_gather_3 = None
        initializers_onnx_initializer_597 = self.initializers.onnx_initializer_597
        initializers_onnx_initializer_598 = self.initializers.onnx_initializer_598
        initializers_onnx_initializer_599 = self.initializers.onnx_initializer_599
        model_layers_16_attn_v_proj_repeat_kv_concat_2 = self.model_layers_16_attn_v_proj_repeat_kv_Concat_2(model_layers_16_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_597, initializers_onnx_initializer_598, model_layers_16_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_599);  initializers_onnx_initializer_597 = initializers_onnx_initializer_598 = initializers_onnx_initializer_599 = None
        initializers_onnx_initializer_600 = self.initializers.onnx_initializer_600
        initializers_onnx_initializer_601 = self.initializers.onnx_initializer_601
        model_layers_16_attn_v_proj_repeat_kv_concat_3 = self.model_layers_16_attn_v_proj_repeat_kv_Concat_3(model_layers_16_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_600, model_layers_16_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_601);  model_layers_16_attn_v_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_600 = model_layers_16_attn_v_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_601 = None
        initializers_onnx_initializer_602 = self.initializers.onnx_initializer_602
        initializers_onnx_initializer_603 = self.initializers.onnx_initializer_603
        initializers_onnx_initializer_604 = self.initializers.onnx_initializer_604
        model_layers_16_attn_k_proj_repeat_kv_concat_2 = self.model_layers_16_attn_k_proj_repeat_kv_Concat_2(model_layers_16_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_602, initializers_onnx_initializer_603, model_layers_16_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_604);  initializers_onnx_initializer_602 = initializers_onnx_initializer_603 = initializers_onnx_initializer_604 = None
        initializers_onnx_initializer_605 = self.initializers.onnx_initializer_605
        initializers_onnx_initializer_606 = self.initializers.onnx_initializer_606
        model_layers_16_attn_k_proj_repeat_kv_concat_3 = self.model_layers_16_attn_k_proj_repeat_kv_Concat_3(model_layers_16_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_605, model_layers_16_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_606);  model_layers_16_attn_k_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_605 = model_layers_16_attn_k_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_606 = None
        initializers_onnx_initializer_607 = self.initializers.onnx_initializer_607
        model_layers_16_attn_v_proj_repeat_kv_equal = self.model_layers_16_attn_v_proj_repeat_kv_Equal(model_layers_16_attn_v_proj_repeat_kv_concat_2, initializers_onnx_initializer_607);  initializers_onnx_initializer_607 = None
        initializers_onnx_initializer_608 = self.initializers.onnx_initializer_608
        model_layers_16_attn_k_proj_repeat_kv_equal = self.model_layers_16_attn_k_proj_repeat_kv_Equal(model_layers_16_attn_k_proj_repeat_kv_concat_2, initializers_onnx_initializer_608);  initializers_onnx_initializer_608 = None
        initializers_onnx_initializer_609 = self.initializers.onnx_initializer_609
        model_layers_16_attn_v_proj_repeat_kv_where = self.model_layers_16_attn_v_proj_repeat_kv_Where(model_layers_16_attn_v_proj_repeat_kv_equal, initializers_onnx_initializer_609, model_layers_16_attn_v_proj_repeat_kv_concat_2);  model_layers_16_attn_v_proj_repeat_kv_equal = initializers_onnx_initializer_609 = model_layers_16_attn_v_proj_repeat_kv_concat_2 = None
        initializers_onnx_initializer_610 = self.initializers.onnx_initializer_610
        model_layers_16_attn_k_proj_repeat_kv_where = self.model_layers_16_attn_k_proj_repeat_kv_Where(model_layers_16_attn_k_proj_repeat_kv_equal, initializers_onnx_initializer_610, model_layers_16_attn_k_proj_repeat_kv_concat_2);  model_layers_16_attn_k_proj_repeat_kv_equal = initializers_onnx_initializer_610 = model_layers_16_attn_k_proj_repeat_kv_concat_2 = None
        model_layers_16_attn_v_proj_repeat_kv_expand = self.model_layers_16_attn_v_proj_repeat_kv_Expand(model_layers_16_attn_v_proj_repeat_kv_unsqueeze_5, model_layers_16_attn_v_proj_repeat_kv_where);  model_layers_16_attn_v_proj_repeat_kv_unsqueeze_5 = model_layers_16_attn_v_proj_repeat_kv_where = None
        model_layers_16_attn_k_proj_repeat_kv_expand = self.model_layers_16_attn_k_proj_repeat_kv_Expand(model_layers_16_attn_k_proj_repeat_kv_unsqueeze_5, model_layers_16_attn_k_proj_repeat_kv_where);  model_layers_16_attn_k_proj_repeat_kv_unsqueeze_5 = model_layers_16_attn_k_proj_repeat_kv_where = None
        model_layers_16_attn_v_proj_repeat_kv_reshape_3 = self.model_layers_16_attn_v_proj_repeat_kv_Reshape_3(model_layers_16_attn_v_proj_repeat_kv_expand, model_layers_16_attn_v_proj_repeat_kv_concat_3);  model_layers_16_attn_v_proj_repeat_kv_expand = model_layers_16_attn_v_proj_repeat_kv_concat_3 = None
        model_layers_16_attn_k_proj_repeat_kv_reshape_3 = self.model_layers_16_attn_k_proj_repeat_kv_Reshape_3(model_layers_16_attn_k_proj_repeat_kv_expand, model_layers_16_attn_k_proj_repeat_kv_concat_3);  model_layers_16_attn_k_proj_repeat_kv_expand = model_layers_16_attn_k_proj_repeat_kv_concat_3 = None
        model_layers_16_attn_v_proj_repeat_kv_transpose_2 = self.model_layers_16_attn_v_proj_repeat_kv_Transpose_2(model_layers_16_attn_v_proj_repeat_kv_reshape_3);  model_layers_16_attn_v_proj_repeat_kv_reshape_3 = None
        model_layers_16_attn_k_proj_repeat_kv_transpose_2 = self.model_layers_16_attn_k_proj_repeat_kv_Transpose_2(model_layers_16_attn_k_proj_repeat_kv_reshape_3);  model_layers_16_attn_k_proj_repeat_kv_reshape_3 = None
        initializers_onnx_initializer_611 = self.initializers.onnx_initializer_611
        model_layers_16_attn_v_proj_repeat_kv_reshape_4 = self.model_layers_16_attn_v_proj_repeat_kv_Reshape_4(model_layers_16_attn_v_proj_repeat_kv_transpose_2, initializers_onnx_initializer_611);  model_layers_16_attn_v_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_611 = None
        initializers_onnx_initializer_612 = self.initializers.onnx_initializer_612
        model_layers_16_attn_k_proj_repeat_kv_reshape_4 = self.model_layers_16_attn_k_proj_repeat_kv_Reshape_4(model_layers_16_attn_k_proj_repeat_kv_transpose_2, initializers_onnx_initializer_612);  model_layers_16_attn_k_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_612 = None
        com_microsoft__model_layers_16_attn_multi_head_attention = self.com_microsoft__model_layers_16_attn_MultiHeadAttention(com_microsoft__model_layers_16_attn_q_rotary_rotary_embedding, model_layers_16_attn_k_proj_repeat_kv_reshape_4, model_layers_16_attn_v_proj_repeat_kv_reshape_4, attention_mask = model_attn_mask_reformat_tile);  com_microsoft__model_layers_16_attn_q_rotary_rotary_embedding = model_layers_16_attn_k_proj_repeat_kv_reshape_4 = model_layers_16_attn_v_proj_repeat_kv_reshape_4 = None
        getitem_127 = com_microsoft__model_layers_16_attn_multi_head_attention[0];  com_microsoft__model_layers_16_attn_multi_head_attention = None
        initializers_onnx_initializer_613 = self.initializers.onnx_initializer_613
        model_layers_16_attn_o_proj_mat_mul = self.model_layers_16_attn_o_proj_MatMul(getitem_127, initializers_onnx_initializer_613);  getitem_127 = initializers_onnx_initializer_613 = None
        getitem_128 = com_microsoft__model_layers_16_input_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_16_input_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_614 = self.initializers.onnx_initializer_614
        com_microsoft__model_layers_16_post_attention_layernorm_skip_layer_norm = self.com_microsoft__model_layers_16_post_attention_layernorm_SkipLayerNorm(getitem_128, model_layers_16_attn_o_proj_mat_mul, initializers_onnx_initializer_614);  getitem_128 = model_layers_16_attn_o_proj_mat_mul = initializers_onnx_initializer_614 = None
        getitem_129 = com_microsoft__model_layers_16_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_615 = self.initializers.onnx_initializer_615
        model_layers_16_mlp_gate_proj_mat_mul = self.model_layers_16_mlp_gate_proj_MatMul(getitem_129, initializers_onnx_initializer_615);  getitem_129 = initializers_onnx_initializer_615 = None
        getitem_130 = com_microsoft__model_layers_16_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_616 = self.initializers.onnx_initializer_616
        model_layers_16_mlp_up_proj_mat_mul = self.model_layers_16_mlp_up_proj_MatMul(getitem_130, initializers_onnx_initializer_616);  getitem_130 = initializers_onnx_initializer_616 = None
        model_layers_16_mlp_act_fn_sigmoid = self.model_layers_16_mlp_act_fn_Sigmoid(model_layers_16_mlp_gate_proj_mat_mul)
        model_layers_16_mlp_act_fn_mul = self.model_layers_16_mlp_act_fn_Mul(model_layers_16_mlp_gate_proj_mat_mul, model_layers_16_mlp_act_fn_sigmoid);  model_layers_16_mlp_gate_proj_mat_mul = model_layers_16_mlp_act_fn_sigmoid = None
        model_layers_16_mlp_mul = self.model_layers_16_mlp_Mul(model_layers_16_mlp_act_fn_mul, model_layers_16_mlp_up_proj_mat_mul);  model_layers_16_mlp_act_fn_mul = model_layers_16_mlp_up_proj_mat_mul = None
        initializers_onnx_initializer_617 = self.initializers.onnx_initializer_617
        model_layers_16_mlp_down_proj_mat_mul = self.model_layers_16_mlp_down_proj_MatMul(model_layers_16_mlp_mul, initializers_onnx_initializer_617);  model_layers_16_mlp_mul = initializers_onnx_initializer_617 = None
        getitem_131 = com_microsoft__model_layers_16_post_attention_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_16_post_attention_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_618 = self.initializers.onnx_initializer_618
        com_microsoft__model_layers_17_input_layernorm_skip_layer_norm = self.com_microsoft__model_layers_17_input_layernorm_SkipLayerNorm(getitem_131, model_layers_16_mlp_down_proj_mat_mul, initializers_onnx_initializer_618);  getitem_131 = model_layers_16_mlp_down_proj_mat_mul = initializers_onnx_initializer_618 = None
        getitem_132 = com_microsoft__model_layers_17_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_619 = self.initializers.onnx_initializer_619
        model_layers_17_attn_q_proj_mat_mul = self.model_layers_17_attn_q_proj_MatMul(getitem_132, initializers_onnx_initializer_619);  getitem_132 = initializers_onnx_initializer_619 = None
        getitem_133 = com_microsoft__model_layers_17_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_620 = self.initializers.onnx_initializer_620
        model_layers_17_attn_k_proj_mat_mul = self.model_layers_17_attn_k_proj_MatMul(getitem_133, initializers_onnx_initializer_620);  getitem_133 = initializers_onnx_initializer_620 = None
        getitem_134 = com_microsoft__model_layers_17_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_621 = self.initializers.onnx_initializer_621
        model_layers_17_attn_v_proj_mat_mul = self.model_layers_17_attn_v_proj_MatMul(getitem_134, initializers_onnx_initializer_621);  getitem_134 = initializers_onnx_initializer_621 = None
        initializers_onnx_initializer_622 = self.initializers.onnx_initializer_622
        initializers_onnx_initializer_623 = self.initializers.onnx_initializer_623
        com_microsoft__model_layers_17_attn_q_rotary_rotary_embedding = self.com_microsoft__model_layers_17_attn_q_rotary_RotaryEmbedding(model_layers_17_attn_q_proj_mat_mul, input_3, initializers_onnx_initializer_622, initializers_onnx_initializer_623);  model_layers_17_attn_q_proj_mat_mul = initializers_onnx_initializer_622 = initializers_onnx_initializer_623 = None
        initializers_onnx_initializer_624 = self.initializers.onnx_initializer_624
        initializers_onnx_initializer_625 = self.initializers.onnx_initializer_625
        com_microsoft__model_layers_17_attn_k_rotary_rotary_embedding = self.com_microsoft__model_layers_17_attn_k_rotary_RotaryEmbedding(model_layers_17_attn_k_proj_mat_mul, input_3, initializers_onnx_initializer_624, initializers_onnx_initializer_625);  model_layers_17_attn_k_proj_mat_mul = initializers_onnx_initializer_624 = initializers_onnx_initializer_625 = None
        initializers_onnx_initializer_626 = self.initializers.onnx_initializer_626
        model_layers_17_attn_v_proj_repeat_kv_reshape_1 = self.model_layers_17_attn_v_proj_repeat_kv_Reshape_1(model_layers_17_attn_v_proj_mat_mul, initializers_onnx_initializer_626);  model_layers_17_attn_v_proj_mat_mul = initializers_onnx_initializer_626 = None
        initializers_onnx_initializer_627 = self.initializers.onnx_initializer_627
        model_layers_17_attn_k_proj_repeat_kv_reshape_1 = self.model_layers_17_attn_k_proj_repeat_kv_Reshape_1(com_microsoft__model_layers_17_attn_k_rotary_rotary_embedding, initializers_onnx_initializer_627);  com_microsoft__model_layers_17_attn_k_rotary_rotary_embedding = initializers_onnx_initializer_627 = None
        model_layers_17_attn_v_proj_repeat_kv_transpose_1 = self.model_layers_17_attn_v_proj_repeat_kv_Transpose_1(model_layers_17_attn_v_proj_repeat_kv_reshape_1);  model_layers_17_attn_v_proj_repeat_kv_reshape_1 = None
        model_layers_17_attn_k_proj_repeat_kv_transpose_1 = self.model_layers_17_attn_k_proj_repeat_kv_Transpose_1(model_layers_17_attn_k_proj_repeat_kv_reshape_1);  model_layers_17_attn_k_proj_repeat_kv_reshape_1 = None
        model_layers_17_attn_v_proj_repeat_kv_concat_1 = self.model_layers_17_attn_v_proj_repeat_kv_Concat_1(input_39, model_layers_17_attn_v_proj_repeat_kv_transpose_1);  input_39 = model_layers_17_attn_v_proj_repeat_kv_transpose_1 = None
        model_layers_17_attn_k_proj_repeat_kv_concat_1 = self.model_layers_17_attn_k_proj_repeat_kv_Concat_1(input_38, model_layers_17_attn_k_proj_repeat_kv_transpose_1);  input_38 = model_layers_17_attn_k_proj_repeat_kv_transpose_1 = None
        model_layers_17_attn_v_proj_repeat_kv_shape_1 = self.model_layers_17_attn_v_proj_repeat_kv_Shape_1(model_layers_17_attn_v_proj_repeat_kv_concat_1)
        model_layers_17_attn_v_proj_repeat_kv_unsqueeze_5 = self.model_layers_17_attn_v_proj_repeat_kv_Unsqueeze_5(model_layers_17_attn_v_proj_repeat_kv_concat_1)
        model_layers_17_attn_k_proj_repeat_kv_shape_1 = self.model_layers_17_attn_k_proj_repeat_kv_Shape_1(model_layers_17_attn_k_proj_repeat_kv_concat_1)
        model_layers_17_attn_k_proj_repeat_kv_unsqueeze_5 = self.model_layers_17_attn_k_proj_repeat_kv_Unsqueeze_5(model_layers_17_attn_k_proj_repeat_kv_concat_1)
        initializers_onnx_initializer_628 = self.initializers.onnx_initializer_628
        model_layers_17_attn_v_proj_repeat_kv_gather_1 = self.model_layers_17_attn_v_proj_repeat_kv_Gather_1(model_layers_17_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_628);  initializers_onnx_initializer_628 = None
        initializers_onnx_initializer_629 = self.initializers.onnx_initializer_629
        model_layers_17_attn_v_proj_repeat_kv_gather_3 = self.model_layers_17_attn_v_proj_repeat_kv_Gather_3(model_layers_17_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_629);  model_layers_17_attn_v_proj_repeat_kv_shape_1 = initializers_onnx_initializer_629 = None
        initializers_onnx_initializer_630 = self.initializers.onnx_initializer_630
        model_layers_17_attn_k_proj_repeat_kv_gather_1 = self.model_layers_17_attn_k_proj_repeat_kv_Gather_1(model_layers_17_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_630);  initializers_onnx_initializer_630 = None
        initializers_onnx_initializer_631 = self.initializers.onnx_initializer_631
        model_layers_17_attn_k_proj_repeat_kv_gather_3 = self.model_layers_17_attn_k_proj_repeat_kv_Gather_3(model_layers_17_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_631);  model_layers_17_attn_k_proj_repeat_kv_shape_1 = initializers_onnx_initializer_631 = None
        model_layers_17_attn_v_proj_repeat_kv_unsqueeze_1 = self.model_layers_17_attn_v_proj_repeat_kv_Unsqueeze_1(model_layers_17_attn_v_proj_repeat_kv_gather_1);  model_layers_17_attn_v_proj_repeat_kv_gather_1 = None
        model_layers_17_attn_v_proj_repeat_kv_unsqueeze_3 = self.model_layers_17_attn_v_proj_repeat_kv_Unsqueeze_3(model_layers_17_attn_v_proj_repeat_kv_gather_3);  model_layers_17_attn_v_proj_repeat_kv_gather_3 = None
        model_layers_17_attn_k_proj_repeat_kv_unsqueeze_1 = self.model_layers_17_attn_k_proj_repeat_kv_Unsqueeze_1(model_layers_17_attn_k_proj_repeat_kv_gather_1);  model_layers_17_attn_k_proj_repeat_kv_gather_1 = None
        model_layers_17_attn_k_proj_repeat_kv_unsqueeze_3 = self.model_layers_17_attn_k_proj_repeat_kv_Unsqueeze_3(model_layers_17_attn_k_proj_repeat_kv_gather_3);  model_layers_17_attn_k_proj_repeat_kv_gather_3 = None
        initializers_onnx_initializer_632 = self.initializers.onnx_initializer_632
        initializers_onnx_initializer_633 = self.initializers.onnx_initializer_633
        initializers_onnx_initializer_634 = self.initializers.onnx_initializer_634
        model_layers_17_attn_v_proj_repeat_kv_concat_2 = self.model_layers_17_attn_v_proj_repeat_kv_Concat_2(model_layers_17_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_632, initializers_onnx_initializer_633, model_layers_17_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_634);  initializers_onnx_initializer_632 = initializers_onnx_initializer_633 = initializers_onnx_initializer_634 = None
        initializers_onnx_initializer_635 = self.initializers.onnx_initializer_635
        initializers_onnx_initializer_636 = self.initializers.onnx_initializer_636
        model_layers_17_attn_v_proj_repeat_kv_concat_3 = self.model_layers_17_attn_v_proj_repeat_kv_Concat_3(model_layers_17_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_635, model_layers_17_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_636);  model_layers_17_attn_v_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_635 = model_layers_17_attn_v_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_636 = None
        initializers_onnx_initializer_637 = self.initializers.onnx_initializer_637
        initializers_onnx_initializer_638 = self.initializers.onnx_initializer_638
        initializers_onnx_initializer_639 = self.initializers.onnx_initializer_639
        model_layers_17_attn_k_proj_repeat_kv_concat_2 = self.model_layers_17_attn_k_proj_repeat_kv_Concat_2(model_layers_17_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_637, initializers_onnx_initializer_638, model_layers_17_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_639);  initializers_onnx_initializer_637 = initializers_onnx_initializer_638 = initializers_onnx_initializer_639 = None
        initializers_onnx_initializer_640 = self.initializers.onnx_initializer_640
        initializers_onnx_initializer_641 = self.initializers.onnx_initializer_641
        model_layers_17_attn_k_proj_repeat_kv_concat_3 = self.model_layers_17_attn_k_proj_repeat_kv_Concat_3(model_layers_17_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_640, model_layers_17_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_641);  model_layers_17_attn_k_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_640 = model_layers_17_attn_k_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_641 = None
        initializers_onnx_initializer_642 = self.initializers.onnx_initializer_642
        model_layers_17_attn_v_proj_repeat_kv_equal = self.model_layers_17_attn_v_proj_repeat_kv_Equal(model_layers_17_attn_v_proj_repeat_kv_concat_2, initializers_onnx_initializer_642);  initializers_onnx_initializer_642 = None
        initializers_onnx_initializer_643 = self.initializers.onnx_initializer_643
        model_layers_17_attn_k_proj_repeat_kv_equal = self.model_layers_17_attn_k_proj_repeat_kv_Equal(model_layers_17_attn_k_proj_repeat_kv_concat_2, initializers_onnx_initializer_643);  initializers_onnx_initializer_643 = None
        initializers_onnx_initializer_644 = self.initializers.onnx_initializer_644
        model_layers_17_attn_v_proj_repeat_kv_where = self.model_layers_17_attn_v_proj_repeat_kv_Where(model_layers_17_attn_v_proj_repeat_kv_equal, initializers_onnx_initializer_644, model_layers_17_attn_v_proj_repeat_kv_concat_2);  model_layers_17_attn_v_proj_repeat_kv_equal = initializers_onnx_initializer_644 = model_layers_17_attn_v_proj_repeat_kv_concat_2 = None
        initializers_onnx_initializer_645 = self.initializers.onnx_initializer_645
        model_layers_17_attn_k_proj_repeat_kv_where = self.model_layers_17_attn_k_proj_repeat_kv_Where(model_layers_17_attn_k_proj_repeat_kv_equal, initializers_onnx_initializer_645, model_layers_17_attn_k_proj_repeat_kv_concat_2);  model_layers_17_attn_k_proj_repeat_kv_equal = initializers_onnx_initializer_645 = model_layers_17_attn_k_proj_repeat_kv_concat_2 = None
        model_layers_17_attn_v_proj_repeat_kv_expand = self.model_layers_17_attn_v_proj_repeat_kv_Expand(model_layers_17_attn_v_proj_repeat_kv_unsqueeze_5, model_layers_17_attn_v_proj_repeat_kv_where);  model_layers_17_attn_v_proj_repeat_kv_unsqueeze_5 = model_layers_17_attn_v_proj_repeat_kv_where = None
        model_layers_17_attn_k_proj_repeat_kv_expand = self.model_layers_17_attn_k_proj_repeat_kv_Expand(model_layers_17_attn_k_proj_repeat_kv_unsqueeze_5, model_layers_17_attn_k_proj_repeat_kv_where);  model_layers_17_attn_k_proj_repeat_kv_unsqueeze_5 = model_layers_17_attn_k_proj_repeat_kv_where = None
        model_layers_17_attn_v_proj_repeat_kv_reshape_3 = self.model_layers_17_attn_v_proj_repeat_kv_Reshape_3(model_layers_17_attn_v_proj_repeat_kv_expand, model_layers_17_attn_v_proj_repeat_kv_concat_3);  model_layers_17_attn_v_proj_repeat_kv_expand = model_layers_17_attn_v_proj_repeat_kv_concat_3 = None
        model_layers_17_attn_k_proj_repeat_kv_reshape_3 = self.model_layers_17_attn_k_proj_repeat_kv_Reshape_3(model_layers_17_attn_k_proj_repeat_kv_expand, model_layers_17_attn_k_proj_repeat_kv_concat_3);  model_layers_17_attn_k_proj_repeat_kv_expand = model_layers_17_attn_k_proj_repeat_kv_concat_3 = None
        model_layers_17_attn_v_proj_repeat_kv_transpose_2 = self.model_layers_17_attn_v_proj_repeat_kv_Transpose_2(model_layers_17_attn_v_proj_repeat_kv_reshape_3);  model_layers_17_attn_v_proj_repeat_kv_reshape_3 = None
        model_layers_17_attn_k_proj_repeat_kv_transpose_2 = self.model_layers_17_attn_k_proj_repeat_kv_Transpose_2(model_layers_17_attn_k_proj_repeat_kv_reshape_3);  model_layers_17_attn_k_proj_repeat_kv_reshape_3 = None
        initializers_onnx_initializer_646 = self.initializers.onnx_initializer_646
        model_layers_17_attn_v_proj_repeat_kv_reshape_4 = self.model_layers_17_attn_v_proj_repeat_kv_Reshape_4(model_layers_17_attn_v_proj_repeat_kv_transpose_2, initializers_onnx_initializer_646);  model_layers_17_attn_v_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_646 = None
        initializers_onnx_initializer_647 = self.initializers.onnx_initializer_647
        model_layers_17_attn_k_proj_repeat_kv_reshape_4 = self.model_layers_17_attn_k_proj_repeat_kv_Reshape_4(model_layers_17_attn_k_proj_repeat_kv_transpose_2, initializers_onnx_initializer_647);  model_layers_17_attn_k_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_647 = None
        com_microsoft__model_layers_17_attn_multi_head_attention = self.com_microsoft__model_layers_17_attn_MultiHeadAttention(com_microsoft__model_layers_17_attn_q_rotary_rotary_embedding, model_layers_17_attn_k_proj_repeat_kv_reshape_4, model_layers_17_attn_v_proj_repeat_kv_reshape_4, attention_mask = model_attn_mask_reformat_tile);  com_microsoft__model_layers_17_attn_q_rotary_rotary_embedding = model_layers_17_attn_k_proj_repeat_kv_reshape_4 = model_layers_17_attn_v_proj_repeat_kv_reshape_4 = None
        getitem_135 = com_microsoft__model_layers_17_attn_multi_head_attention[0];  com_microsoft__model_layers_17_attn_multi_head_attention = None
        initializers_onnx_initializer_648 = self.initializers.onnx_initializer_648
        model_layers_17_attn_o_proj_mat_mul = self.model_layers_17_attn_o_proj_MatMul(getitem_135, initializers_onnx_initializer_648);  getitem_135 = initializers_onnx_initializer_648 = None
        getitem_136 = com_microsoft__model_layers_17_input_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_17_input_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_649 = self.initializers.onnx_initializer_649
        com_microsoft__model_layers_17_post_attention_layernorm_skip_layer_norm = self.com_microsoft__model_layers_17_post_attention_layernorm_SkipLayerNorm(getitem_136, model_layers_17_attn_o_proj_mat_mul, initializers_onnx_initializer_649);  getitem_136 = model_layers_17_attn_o_proj_mat_mul = initializers_onnx_initializer_649 = None
        getitem_137 = com_microsoft__model_layers_17_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_650 = self.initializers.onnx_initializer_650
        model_layers_17_mlp_gate_proj_mat_mul = self.model_layers_17_mlp_gate_proj_MatMul(getitem_137, initializers_onnx_initializer_650);  getitem_137 = initializers_onnx_initializer_650 = None
        getitem_138 = com_microsoft__model_layers_17_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_651 = self.initializers.onnx_initializer_651
        model_layers_17_mlp_up_proj_mat_mul = self.model_layers_17_mlp_up_proj_MatMul(getitem_138, initializers_onnx_initializer_651);  getitem_138 = initializers_onnx_initializer_651 = None
        model_layers_17_mlp_act_fn_sigmoid = self.model_layers_17_mlp_act_fn_Sigmoid(model_layers_17_mlp_gate_proj_mat_mul)
        model_layers_17_mlp_act_fn_mul = self.model_layers_17_mlp_act_fn_Mul(model_layers_17_mlp_gate_proj_mat_mul, model_layers_17_mlp_act_fn_sigmoid);  model_layers_17_mlp_gate_proj_mat_mul = model_layers_17_mlp_act_fn_sigmoid = None
        model_layers_17_mlp_mul = self.model_layers_17_mlp_Mul(model_layers_17_mlp_act_fn_mul, model_layers_17_mlp_up_proj_mat_mul);  model_layers_17_mlp_act_fn_mul = model_layers_17_mlp_up_proj_mat_mul = None
        initializers_onnx_initializer_652 = self.initializers.onnx_initializer_652
        model_layers_17_mlp_down_proj_mat_mul = self.model_layers_17_mlp_down_proj_MatMul(model_layers_17_mlp_mul, initializers_onnx_initializer_652);  model_layers_17_mlp_mul = initializers_onnx_initializer_652 = None
        getitem_139 = com_microsoft__model_layers_17_post_attention_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_17_post_attention_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_653 = self.initializers.onnx_initializer_653
        com_microsoft__model_layers_18_input_layernorm_skip_layer_norm = self.com_microsoft__model_layers_18_input_layernorm_SkipLayerNorm(getitem_139, model_layers_17_mlp_down_proj_mat_mul, initializers_onnx_initializer_653);  getitem_139 = model_layers_17_mlp_down_proj_mat_mul = initializers_onnx_initializer_653 = None
        getitem_140 = com_microsoft__model_layers_18_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_654 = self.initializers.onnx_initializer_654
        model_layers_18_attn_q_proj_mat_mul = self.model_layers_18_attn_q_proj_MatMul(getitem_140, initializers_onnx_initializer_654);  getitem_140 = initializers_onnx_initializer_654 = None
        getitem_141 = com_microsoft__model_layers_18_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_655 = self.initializers.onnx_initializer_655
        model_layers_18_attn_k_proj_mat_mul = self.model_layers_18_attn_k_proj_MatMul(getitem_141, initializers_onnx_initializer_655);  getitem_141 = initializers_onnx_initializer_655 = None
        getitem_142 = com_microsoft__model_layers_18_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_656 = self.initializers.onnx_initializer_656
        model_layers_18_attn_v_proj_mat_mul = self.model_layers_18_attn_v_proj_MatMul(getitem_142, initializers_onnx_initializer_656);  getitem_142 = initializers_onnx_initializer_656 = None
        initializers_onnx_initializer_657 = self.initializers.onnx_initializer_657
        initializers_onnx_initializer_658 = self.initializers.onnx_initializer_658
        com_microsoft__model_layers_18_attn_q_rotary_rotary_embedding = self.com_microsoft__model_layers_18_attn_q_rotary_RotaryEmbedding(model_layers_18_attn_q_proj_mat_mul, input_3, initializers_onnx_initializer_657, initializers_onnx_initializer_658);  model_layers_18_attn_q_proj_mat_mul = initializers_onnx_initializer_657 = initializers_onnx_initializer_658 = None
        initializers_onnx_initializer_659 = self.initializers.onnx_initializer_659
        initializers_onnx_initializer_660 = self.initializers.onnx_initializer_660
        com_microsoft__model_layers_18_attn_k_rotary_rotary_embedding = self.com_microsoft__model_layers_18_attn_k_rotary_RotaryEmbedding(model_layers_18_attn_k_proj_mat_mul, input_3, initializers_onnx_initializer_659, initializers_onnx_initializer_660);  model_layers_18_attn_k_proj_mat_mul = initializers_onnx_initializer_659 = initializers_onnx_initializer_660 = None
        initializers_onnx_initializer_661 = self.initializers.onnx_initializer_661
        model_layers_18_attn_v_proj_repeat_kv_reshape_1 = self.model_layers_18_attn_v_proj_repeat_kv_Reshape_1(model_layers_18_attn_v_proj_mat_mul, initializers_onnx_initializer_661);  model_layers_18_attn_v_proj_mat_mul = initializers_onnx_initializer_661 = None
        initializers_onnx_initializer_662 = self.initializers.onnx_initializer_662
        model_layers_18_attn_k_proj_repeat_kv_reshape_1 = self.model_layers_18_attn_k_proj_repeat_kv_Reshape_1(com_microsoft__model_layers_18_attn_k_rotary_rotary_embedding, initializers_onnx_initializer_662);  com_microsoft__model_layers_18_attn_k_rotary_rotary_embedding = initializers_onnx_initializer_662 = None
        model_layers_18_attn_v_proj_repeat_kv_transpose_1 = self.model_layers_18_attn_v_proj_repeat_kv_Transpose_1(model_layers_18_attn_v_proj_repeat_kv_reshape_1);  model_layers_18_attn_v_proj_repeat_kv_reshape_1 = None
        model_layers_18_attn_k_proj_repeat_kv_transpose_1 = self.model_layers_18_attn_k_proj_repeat_kv_Transpose_1(model_layers_18_attn_k_proj_repeat_kv_reshape_1);  model_layers_18_attn_k_proj_repeat_kv_reshape_1 = None
        model_layers_18_attn_v_proj_repeat_kv_concat_1 = self.model_layers_18_attn_v_proj_repeat_kv_Concat_1(input_41, model_layers_18_attn_v_proj_repeat_kv_transpose_1);  input_41 = model_layers_18_attn_v_proj_repeat_kv_transpose_1 = None
        model_layers_18_attn_k_proj_repeat_kv_concat_1 = self.model_layers_18_attn_k_proj_repeat_kv_Concat_1(input_40, model_layers_18_attn_k_proj_repeat_kv_transpose_1);  input_40 = model_layers_18_attn_k_proj_repeat_kv_transpose_1 = None
        model_layers_18_attn_v_proj_repeat_kv_shape_1 = self.model_layers_18_attn_v_proj_repeat_kv_Shape_1(model_layers_18_attn_v_proj_repeat_kv_concat_1)
        model_layers_18_attn_v_proj_repeat_kv_unsqueeze_5 = self.model_layers_18_attn_v_proj_repeat_kv_Unsqueeze_5(model_layers_18_attn_v_proj_repeat_kv_concat_1)
        model_layers_18_attn_k_proj_repeat_kv_shape_1 = self.model_layers_18_attn_k_proj_repeat_kv_Shape_1(model_layers_18_attn_k_proj_repeat_kv_concat_1)
        model_layers_18_attn_k_proj_repeat_kv_unsqueeze_5 = self.model_layers_18_attn_k_proj_repeat_kv_Unsqueeze_5(model_layers_18_attn_k_proj_repeat_kv_concat_1)
        initializers_onnx_initializer_663 = self.initializers.onnx_initializer_663
        model_layers_18_attn_v_proj_repeat_kv_gather_1 = self.model_layers_18_attn_v_proj_repeat_kv_Gather_1(model_layers_18_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_663);  initializers_onnx_initializer_663 = None
        initializers_onnx_initializer_664 = self.initializers.onnx_initializer_664
        model_layers_18_attn_v_proj_repeat_kv_gather_3 = self.model_layers_18_attn_v_proj_repeat_kv_Gather_3(model_layers_18_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_664);  model_layers_18_attn_v_proj_repeat_kv_shape_1 = initializers_onnx_initializer_664 = None
        initializers_onnx_initializer_665 = self.initializers.onnx_initializer_665
        model_layers_18_attn_k_proj_repeat_kv_gather_1 = self.model_layers_18_attn_k_proj_repeat_kv_Gather_1(model_layers_18_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_665);  initializers_onnx_initializer_665 = None
        initializers_onnx_initializer_666 = self.initializers.onnx_initializer_666
        model_layers_18_attn_k_proj_repeat_kv_gather_3 = self.model_layers_18_attn_k_proj_repeat_kv_Gather_3(model_layers_18_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_666);  model_layers_18_attn_k_proj_repeat_kv_shape_1 = initializers_onnx_initializer_666 = None
        model_layers_18_attn_v_proj_repeat_kv_unsqueeze_1 = self.model_layers_18_attn_v_proj_repeat_kv_Unsqueeze_1(model_layers_18_attn_v_proj_repeat_kv_gather_1);  model_layers_18_attn_v_proj_repeat_kv_gather_1 = None
        model_layers_18_attn_v_proj_repeat_kv_unsqueeze_3 = self.model_layers_18_attn_v_proj_repeat_kv_Unsqueeze_3(model_layers_18_attn_v_proj_repeat_kv_gather_3);  model_layers_18_attn_v_proj_repeat_kv_gather_3 = None
        model_layers_18_attn_k_proj_repeat_kv_unsqueeze_1 = self.model_layers_18_attn_k_proj_repeat_kv_Unsqueeze_1(model_layers_18_attn_k_proj_repeat_kv_gather_1);  model_layers_18_attn_k_proj_repeat_kv_gather_1 = None
        model_layers_18_attn_k_proj_repeat_kv_unsqueeze_3 = self.model_layers_18_attn_k_proj_repeat_kv_Unsqueeze_3(model_layers_18_attn_k_proj_repeat_kv_gather_3);  model_layers_18_attn_k_proj_repeat_kv_gather_3 = None
        initializers_onnx_initializer_667 = self.initializers.onnx_initializer_667
        initializers_onnx_initializer_668 = self.initializers.onnx_initializer_668
        initializers_onnx_initializer_669 = self.initializers.onnx_initializer_669
        model_layers_18_attn_v_proj_repeat_kv_concat_2 = self.model_layers_18_attn_v_proj_repeat_kv_Concat_2(model_layers_18_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_667, initializers_onnx_initializer_668, model_layers_18_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_669);  initializers_onnx_initializer_667 = initializers_onnx_initializer_668 = initializers_onnx_initializer_669 = None
        initializers_onnx_initializer_670 = self.initializers.onnx_initializer_670
        initializers_onnx_initializer_671 = self.initializers.onnx_initializer_671
        model_layers_18_attn_v_proj_repeat_kv_concat_3 = self.model_layers_18_attn_v_proj_repeat_kv_Concat_3(model_layers_18_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_670, model_layers_18_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_671);  model_layers_18_attn_v_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_670 = model_layers_18_attn_v_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_671 = None
        initializers_onnx_initializer_672 = self.initializers.onnx_initializer_672
        initializers_onnx_initializer_673 = self.initializers.onnx_initializer_673
        initializers_onnx_initializer_674 = self.initializers.onnx_initializer_674
        model_layers_18_attn_k_proj_repeat_kv_concat_2 = self.model_layers_18_attn_k_proj_repeat_kv_Concat_2(model_layers_18_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_672, initializers_onnx_initializer_673, model_layers_18_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_674);  initializers_onnx_initializer_672 = initializers_onnx_initializer_673 = initializers_onnx_initializer_674 = None
        initializers_onnx_initializer_675 = self.initializers.onnx_initializer_675
        initializers_onnx_initializer_676 = self.initializers.onnx_initializer_676
        model_layers_18_attn_k_proj_repeat_kv_concat_3 = self.model_layers_18_attn_k_proj_repeat_kv_Concat_3(model_layers_18_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_675, model_layers_18_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_676);  model_layers_18_attn_k_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_675 = model_layers_18_attn_k_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_676 = None
        initializers_onnx_initializer_677 = self.initializers.onnx_initializer_677
        model_layers_18_attn_v_proj_repeat_kv_equal = self.model_layers_18_attn_v_proj_repeat_kv_Equal(model_layers_18_attn_v_proj_repeat_kv_concat_2, initializers_onnx_initializer_677);  initializers_onnx_initializer_677 = None
        initializers_onnx_initializer_678 = self.initializers.onnx_initializer_678
        model_layers_18_attn_k_proj_repeat_kv_equal = self.model_layers_18_attn_k_proj_repeat_kv_Equal(model_layers_18_attn_k_proj_repeat_kv_concat_2, initializers_onnx_initializer_678);  initializers_onnx_initializer_678 = None
        initializers_onnx_initializer_679 = self.initializers.onnx_initializer_679
        model_layers_18_attn_v_proj_repeat_kv_where = self.model_layers_18_attn_v_proj_repeat_kv_Where(model_layers_18_attn_v_proj_repeat_kv_equal, initializers_onnx_initializer_679, model_layers_18_attn_v_proj_repeat_kv_concat_2);  model_layers_18_attn_v_proj_repeat_kv_equal = initializers_onnx_initializer_679 = model_layers_18_attn_v_proj_repeat_kv_concat_2 = None
        initializers_onnx_initializer_680 = self.initializers.onnx_initializer_680
        model_layers_18_attn_k_proj_repeat_kv_where = self.model_layers_18_attn_k_proj_repeat_kv_Where(model_layers_18_attn_k_proj_repeat_kv_equal, initializers_onnx_initializer_680, model_layers_18_attn_k_proj_repeat_kv_concat_2);  model_layers_18_attn_k_proj_repeat_kv_equal = initializers_onnx_initializer_680 = model_layers_18_attn_k_proj_repeat_kv_concat_2 = None
        model_layers_18_attn_v_proj_repeat_kv_expand = self.model_layers_18_attn_v_proj_repeat_kv_Expand(model_layers_18_attn_v_proj_repeat_kv_unsqueeze_5, model_layers_18_attn_v_proj_repeat_kv_where);  model_layers_18_attn_v_proj_repeat_kv_unsqueeze_5 = model_layers_18_attn_v_proj_repeat_kv_where = None
        model_layers_18_attn_k_proj_repeat_kv_expand = self.model_layers_18_attn_k_proj_repeat_kv_Expand(model_layers_18_attn_k_proj_repeat_kv_unsqueeze_5, model_layers_18_attn_k_proj_repeat_kv_where);  model_layers_18_attn_k_proj_repeat_kv_unsqueeze_5 = model_layers_18_attn_k_proj_repeat_kv_where = None
        model_layers_18_attn_v_proj_repeat_kv_reshape_3 = self.model_layers_18_attn_v_proj_repeat_kv_Reshape_3(model_layers_18_attn_v_proj_repeat_kv_expand, model_layers_18_attn_v_proj_repeat_kv_concat_3);  model_layers_18_attn_v_proj_repeat_kv_expand = model_layers_18_attn_v_proj_repeat_kv_concat_3 = None
        model_layers_18_attn_k_proj_repeat_kv_reshape_3 = self.model_layers_18_attn_k_proj_repeat_kv_Reshape_3(model_layers_18_attn_k_proj_repeat_kv_expand, model_layers_18_attn_k_proj_repeat_kv_concat_3);  model_layers_18_attn_k_proj_repeat_kv_expand = model_layers_18_attn_k_proj_repeat_kv_concat_3 = None
        model_layers_18_attn_v_proj_repeat_kv_transpose_2 = self.model_layers_18_attn_v_proj_repeat_kv_Transpose_2(model_layers_18_attn_v_proj_repeat_kv_reshape_3);  model_layers_18_attn_v_proj_repeat_kv_reshape_3 = None
        model_layers_18_attn_k_proj_repeat_kv_transpose_2 = self.model_layers_18_attn_k_proj_repeat_kv_Transpose_2(model_layers_18_attn_k_proj_repeat_kv_reshape_3);  model_layers_18_attn_k_proj_repeat_kv_reshape_3 = None
        initializers_onnx_initializer_681 = self.initializers.onnx_initializer_681
        model_layers_18_attn_v_proj_repeat_kv_reshape_4 = self.model_layers_18_attn_v_proj_repeat_kv_Reshape_4(model_layers_18_attn_v_proj_repeat_kv_transpose_2, initializers_onnx_initializer_681);  model_layers_18_attn_v_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_681 = None
        initializers_onnx_initializer_682 = self.initializers.onnx_initializer_682
        model_layers_18_attn_k_proj_repeat_kv_reshape_4 = self.model_layers_18_attn_k_proj_repeat_kv_Reshape_4(model_layers_18_attn_k_proj_repeat_kv_transpose_2, initializers_onnx_initializer_682);  model_layers_18_attn_k_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_682 = None
        com_microsoft__model_layers_18_attn_multi_head_attention = self.com_microsoft__model_layers_18_attn_MultiHeadAttention(com_microsoft__model_layers_18_attn_q_rotary_rotary_embedding, model_layers_18_attn_k_proj_repeat_kv_reshape_4, model_layers_18_attn_v_proj_repeat_kv_reshape_4, attention_mask = model_attn_mask_reformat_tile);  com_microsoft__model_layers_18_attn_q_rotary_rotary_embedding = model_layers_18_attn_k_proj_repeat_kv_reshape_4 = model_layers_18_attn_v_proj_repeat_kv_reshape_4 = None
        getitem_143 = com_microsoft__model_layers_18_attn_multi_head_attention[0];  com_microsoft__model_layers_18_attn_multi_head_attention = None
        initializers_onnx_initializer_683 = self.initializers.onnx_initializer_683
        model_layers_18_attn_o_proj_mat_mul = self.model_layers_18_attn_o_proj_MatMul(getitem_143, initializers_onnx_initializer_683);  getitem_143 = initializers_onnx_initializer_683 = None
        getitem_144 = com_microsoft__model_layers_18_input_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_18_input_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_684 = self.initializers.onnx_initializer_684
        com_microsoft__model_layers_18_post_attention_layernorm_skip_layer_norm = self.com_microsoft__model_layers_18_post_attention_layernorm_SkipLayerNorm(getitem_144, model_layers_18_attn_o_proj_mat_mul, initializers_onnx_initializer_684);  getitem_144 = model_layers_18_attn_o_proj_mat_mul = initializers_onnx_initializer_684 = None
        getitem_145 = com_microsoft__model_layers_18_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_685 = self.initializers.onnx_initializer_685
        model_layers_18_mlp_gate_proj_mat_mul = self.model_layers_18_mlp_gate_proj_MatMul(getitem_145, initializers_onnx_initializer_685);  getitem_145 = initializers_onnx_initializer_685 = None
        getitem_146 = com_microsoft__model_layers_18_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_686 = self.initializers.onnx_initializer_686
        model_layers_18_mlp_up_proj_mat_mul = self.model_layers_18_mlp_up_proj_MatMul(getitem_146, initializers_onnx_initializer_686);  getitem_146 = initializers_onnx_initializer_686 = None
        model_layers_18_mlp_act_fn_sigmoid = self.model_layers_18_mlp_act_fn_Sigmoid(model_layers_18_mlp_gate_proj_mat_mul)
        model_layers_18_mlp_act_fn_mul = self.model_layers_18_mlp_act_fn_Mul(model_layers_18_mlp_gate_proj_mat_mul, model_layers_18_mlp_act_fn_sigmoid);  model_layers_18_mlp_gate_proj_mat_mul = model_layers_18_mlp_act_fn_sigmoid = None
        model_layers_18_mlp_mul = self.model_layers_18_mlp_Mul(model_layers_18_mlp_act_fn_mul, model_layers_18_mlp_up_proj_mat_mul);  model_layers_18_mlp_act_fn_mul = model_layers_18_mlp_up_proj_mat_mul = None
        initializers_onnx_initializer_687 = self.initializers.onnx_initializer_687
        model_layers_18_mlp_down_proj_mat_mul = self.model_layers_18_mlp_down_proj_MatMul(model_layers_18_mlp_mul, initializers_onnx_initializer_687);  model_layers_18_mlp_mul = initializers_onnx_initializer_687 = None
        getitem_147 = com_microsoft__model_layers_18_post_attention_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_18_post_attention_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_688 = self.initializers.onnx_initializer_688
        com_microsoft__model_layers_19_input_layernorm_skip_layer_norm = self.com_microsoft__model_layers_19_input_layernorm_SkipLayerNorm(getitem_147, model_layers_18_mlp_down_proj_mat_mul, initializers_onnx_initializer_688);  getitem_147 = model_layers_18_mlp_down_proj_mat_mul = initializers_onnx_initializer_688 = None
        getitem_148 = com_microsoft__model_layers_19_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_689 = self.initializers.onnx_initializer_689
        model_layers_19_attn_q_proj_mat_mul = self.model_layers_19_attn_q_proj_MatMul(getitem_148, initializers_onnx_initializer_689);  getitem_148 = initializers_onnx_initializer_689 = None
        getitem_149 = com_microsoft__model_layers_19_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_690 = self.initializers.onnx_initializer_690
        model_layers_19_attn_k_proj_mat_mul = self.model_layers_19_attn_k_proj_MatMul(getitem_149, initializers_onnx_initializer_690);  getitem_149 = initializers_onnx_initializer_690 = None
        getitem_150 = com_microsoft__model_layers_19_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_691 = self.initializers.onnx_initializer_691
        model_layers_19_attn_v_proj_mat_mul = self.model_layers_19_attn_v_proj_MatMul(getitem_150, initializers_onnx_initializer_691);  getitem_150 = initializers_onnx_initializer_691 = None
        initializers_onnx_initializer_692 = self.initializers.onnx_initializer_692
        initializers_onnx_initializer_693 = self.initializers.onnx_initializer_693
        com_microsoft__model_layers_19_attn_q_rotary_rotary_embedding = self.com_microsoft__model_layers_19_attn_q_rotary_RotaryEmbedding(model_layers_19_attn_q_proj_mat_mul, input_3, initializers_onnx_initializer_692, initializers_onnx_initializer_693);  model_layers_19_attn_q_proj_mat_mul = initializers_onnx_initializer_692 = initializers_onnx_initializer_693 = None
        initializers_onnx_initializer_694 = self.initializers.onnx_initializer_694
        initializers_onnx_initializer_695 = self.initializers.onnx_initializer_695
        com_microsoft__model_layers_19_attn_k_rotary_rotary_embedding = self.com_microsoft__model_layers_19_attn_k_rotary_RotaryEmbedding(model_layers_19_attn_k_proj_mat_mul, input_3, initializers_onnx_initializer_694, initializers_onnx_initializer_695);  model_layers_19_attn_k_proj_mat_mul = initializers_onnx_initializer_694 = initializers_onnx_initializer_695 = None
        initializers_onnx_initializer_696 = self.initializers.onnx_initializer_696
        model_layers_19_attn_v_proj_repeat_kv_reshape_1 = self.model_layers_19_attn_v_proj_repeat_kv_Reshape_1(model_layers_19_attn_v_proj_mat_mul, initializers_onnx_initializer_696);  model_layers_19_attn_v_proj_mat_mul = initializers_onnx_initializer_696 = None
        initializers_onnx_initializer_697 = self.initializers.onnx_initializer_697
        model_layers_19_attn_k_proj_repeat_kv_reshape_1 = self.model_layers_19_attn_k_proj_repeat_kv_Reshape_1(com_microsoft__model_layers_19_attn_k_rotary_rotary_embedding, initializers_onnx_initializer_697);  com_microsoft__model_layers_19_attn_k_rotary_rotary_embedding = initializers_onnx_initializer_697 = None
        model_layers_19_attn_v_proj_repeat_kv_transpose_1 = self.model_layers_19_attn_v_proj_repeat_kv_Transpose_1(model_layers_19_attn_v_proj_repeat_kv_reshape_1);  model_layers_19_attn_v_proj_repeat_kv_reshape_1 = None
        model_layers_19_attn_k_proj_repeat_kv_transpose_1 = self.model_layers_19_attn_k_proj_repeat_kv_Transpose_1(model_layers_19_attn_k_proj_repeat_kv_reshape_1);  model_layers_19_attn_k_proj_repeat_kv_reshape_1 = None
        model_layers_19_attn_v_proj_repeat_kv_concat_1 = self.model_layers_19_attn_v_proj_repeat_kv_Concat_1(input_43, model_layers_19_attn_v_proj_repeat_kv_transpose_1);  input_43 = model_layers_19_attn_v_proj_repeat_kv_transpose_1 = None
        model_layers_19_attn_k_proj_repeat_kv_concat_1 = self.model_layers_19_attn_k_proj_repeat_kv_Concat_1(input_42, model_layers_19_attn_k_proj_repeat_kv_transpose_1);  input_42 = model_layers_19_attn_k_proj_repeat_kv_transpose_1 = None
        model_layers_19_attn_v_proj_repeat_kv_shape_1 = self.model_layers_19_attn_v_proj_repeat_kv_Shape_1(model_layers_19_attn_v_proj_repeat_kv_concat_1)
        model_layers_19_attn_v_proj_repeat_kv_unsqueeze_5 = self.model_layers_19_attn_v_proj_repeat_kv_Unsqueeze_5(model_layers_19_attn_v_proj_repeat_kv_concat_1)
        model_layers_19_attn_k_proj_repeat_kv_shape_1 = self.model_layers_19_attn_k_proj_repeat_kv_Shape_1(model_layers_19_attn_k_proj_repeat_kv_concat_1)
        model_layers_19_attn_k_proj_repeat_kv_unsqueeze_5 = self.model_layers_19_attn_k_proj_repeat_kv_Unsqueeze_5(model_layers_19_attn_k_proj_repeat_kv_concat_1)
        initializers_onnx_initializer_698 = self.initializers.onnx_initializer_698
        model_layers_19_attn_v_proj_repeat_kv_gather_1 = self.model_layers_19_attn_v_proj_repeat_kv_Gather_1(model_layers_19_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_698);  initializers_onnx_initializer_698 = None
        initializers_onnx_initializer_699 = self.initializers.onnx_initializer_699
        model_layers_19_attn_v_proj_repeat_kv_gather_3 = self.model_layers_19_attn_v_proj_repeat_kv_Gather_3(model_layers_19_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_699);  model_layers_19_attn_v_proj_repeat_kv_shape_1 = initializers_onnx_initializer_699 = None
        initializers_onnx_initializer_700 = self.initializers.onnx_initializer_700
        model_layers_19_attn_k_proj_repeat_kv_gather_1 = self.model_layers_19_attn_k_proj_repeat_kv_Gather_1(model_layers_19_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_700);  initializers_onnx_initializer_700 = None
        initializers_onnx_initializer_701 = self.initializers.onnx_initializer_701
        model_layers_19_attn_k_proj_repeat_kv_gather_3 = self.model_layers_19_attn_k_proj_repeat_kv_Gather_3(model_layers_19_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_701);  model_layers_19_attn_k_proj_repeat_kv_shape_1 = initializers_onnx_initializer_701 = None
        model_layers_19_attn_v_proj_repeat_kv_unsqueeze_1 = self.model_layers_19_attn_v_proj_repeat_kv_Unsqueeze_1(model_layers_19_attn_v_proj_repeat_kv_gather_1);  model_layers_19_attn_v_proj_repeat_kv_gather_1 = None
        model_layers_19_attn_v_proj_repeat_kv_unsqueeze_3 = self.model_layers_19_attn_v_proj_repeat_kv_Unsqueeze_3(model_layers_19_attn_v_proj_repeat_kv_gather_3);  model_layers_19_attn_v_proj_repeat_kv_gather_3 = None
        model_layers_19_attn_k_proj_repeat_kv_unsqueeze_1 = self.model_layers_19_attn_k_proj_repeat_kv_Unsqueeze_1(model_layers_19_attn_k_proj_repeat_kv_gather_1);  model_layers_19_attn_k_proj_repeat_kv_gather_1 = None
        model_layers_19_attn_k_proj_repeat_kv_unsqueeze_3 = self.model_layers_19_attn_k_proj_repeat_kv_Unsqueeze_3(model_layers_19_attn_k_proj_repeat_kv_gather_3);  model_layers_19_attn_k_proj_repeat_kv_gather_3 = None
        initializers_onnx_initializer_702 = self.initializers.onnx_initializer_702
        initializers_onnx_initializer_703 = self.initializers.onnx_initializer_703
        initializers_onnx_initializer_704 = self.initializers.onnx_initializer_704
        model_layers_19_attn_v_proj_repeat_kv_concat_2 = self.model_layers_19_attn_v_proj_repeat_kv_Concat_2(model_layers_19_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_702, initializers_onnx_initializer_703, model_layers_19_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_704);  initializers_onnx_initializer_702 = initializers_onnx_initializer_703 = initializers_onnx_initializer_704 = None
        initializers_onnx_initializer_705 = self.initializers.onnx_initializer_705
        initializers_onnx_initializer_706 = self.initializers.onnx_initializer_706
        model_layers_19_attn_v_proj_repeat_kv_concat_3 = self.model_layers_19_attn_v_proj_repeat_kv_Concat_3(model_layers_19_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_705, model_layers_19_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_706);  model_layers_19_attn_v_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_705 = model_layers_19_attn_v_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_706 = None
        initializers_onnx_initializer_707 = self.initializers.onnx_initializer_707
        initializers_onnx_initializer_708 = self.initializers.onnx_initializer_708
        initializers_onnx_initializer_709 = self.initializers.onnx_initializer_709
        model_layers_19_attn_k_proj_repeat_kv_concat_2 = self.model_layers_19_attn_k_proj_repeat_kv_Concat_2(model_layers_19_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_707, initializers_onnx_initializer_708, model_layers_19_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_709);  initializers_onnx_initializer_707 = initializers_onnx_initializer_708 = initializers_onnx_initializer_709 = None
        initializers_onnx_initializer_710 = self.initializers.onnx_initializer_710
        initializers_onnx_initializer_711 = self.initializers.onnx_initializer_711
        model_layers_19_attn_k_proj_repeat_kv_concat_3 = self.model_layers_19_attn_k_proj_repeat_kv_Concat_3(model_layers_19_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_710, model_layers_19_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_711);  model_layers_19_attn_k_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_710 = model_layers_19_attn_k_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_711 = None
        initializers_onnx_initializer_712 = self.initializers.onnx_initializer_712
        model_layers_19_attn_v_proj_repeat_kv_equal = self.model_layers_19_attn_v_proj_repeat_kv_Equal(model_layers_19_attn_v_proj_repeat_kv_concat_2, initializers_onnx_initializer_712);  initializers_onnx_initializer_712 = None
        initializers_onnx_initializer_713 = self.initializers.onnx_initializer_713
        model_layers_19_attn_k_proj_repeat_kv_equal = self.model_layers_19_attn_k_proj_repeat_kv_Equal(model_layers_19_attn_k_proj_repeat_kv_concat_2, initializers_onnx_initializer_713);  initializers_onnx_initializer_713 = None
        initializers_onnx_initializer_714 = self.initializers.onnx_initializer_714
        model_layers_19_attn_v_proj_repeat_kv_where = self.model_layers_19_attn_v_proj_repeat_kv_Where(model_layers_19_attn_v_proj_repeat_kv_equal, initializers_onnx_initializer_714, model_layers_19_attn_v_proj_repeat_kv_concat_2);  model_layers_19_attn_v_proj_repeat_kv_equal = initializers_onnx_initializer_714 = model_layers_19_attn_v_proj_repeat_kv_concat_2 = None
        initializers_onnx_initializer_715 = self.initializers.onnx_initializer_715
        model_layers_19_attn_k_proj_repeat_kv_where = self.model_layers_19_attn_k_proj_repeat_kv_Where(model_layers_19_attn_k_proj_repeat_kv_equal, initializers_onnx_initializer_715, model_layers_19_attn_k_proj_repeat_kv_concat_2);  model_layers_19_attn_k_proj_repeat_kv_equal = initializers_onnx_initializer_715 = model_layers_19_attn_k_proj_repeat_kv_concat_2 = None
        model_layers_19_attn_v_proj_repeat_kv_expand = self.model_layers_19_attn_v_proj_repeat_kv_Expand(model_layers_19_attn_v_proj_repeat_kv_unsqueeze_5, model_layers_19_attn_v_proj_repeat_kv_where);  model_layers_19_attn_v_proj_repeat_kv_unsqueeze_5 = model_layers_19_attn_v_proj_repeat_kv_where = None
        model_layers_19_attn_k_proj_repeat_kv_expand = self.model_layers_19_attn_k_proj_repeat_kv_Expand(model_layers_19_attn_k_proj_repeat_kv_unsqueeze_5, model_layers_19_attn_k_proj_repeat_kv_where);  model_layers_19_attn_k_proj_repeat_kv_unsqueeze_5 = model_layers_19_attn_k_proj_repeat_kv_where = None
        model_layers_19_attn_v_proj_repeat_kv_reshape_3 = self.model_layers_19_attn_v_proj_repeat_kv_Reshape_3(model_layers_19_attn_v_proj_repeat_kv_expand, model_layers_19_attn_v_proj_repeat_kv_concat_3);  model_layers_19_attn_v_proj_repeat_kv_expand = model_layers_19_attn_v_proj_repeat_kv_concat_3 = None
        model_layers_19_attn_k_proj_repeat_kv_reshape_3 = self.model_layers_19_attn_k_proj_repeat_kv_Reshape_3(model_layers_19_attn_k_proj_repeat_kv_expand, model_layers_19_attn_k_proj_repeat_kv_concat_3);  model_layers_19_attn_k_proj_repeat_kv_expand = model_layers_19_attn_k_proj_repeat_kv_concat_3 = None
        model_layers_19_attn_v_proj_repeat_kv_transpose_2 = self.model_layers_19_attn_v_proj_repeat_kv_Transpose_2(model_layers_19_attn_v_proj_repeat_kv_reshape_3);  model_layers_19_attn_v_proj_repeat_kv_reshape_3 = None
        model_layers_19_attn_k_proj_repeat_kv_transpose_2 = self.model_layers_19_attn_k_proj_repeat_kv_Transpose_2(model_layers_19_attn_k_proj_repeat_kv_reshape_3);  model_layers_19_attn_k_proj_repeat_kv_reshape_3 = None
        initializers_onnx_initializer_716 = self.initializers.onnx_initializer_716
        model_layers_19_attn_v_proj_repeat_kv_reshape_4 = self.model_layers_19_attn_v_proj_repeat_kv_Reshape_4(model_layers_19_attn_v_proj_repeat_kv_transpose_2, initializers_onnx_initializer_716);  model_layers_19_attn_v_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_716 = None
        initializers_onnx_initializer_717 = self.initializers.onnx_initializer_717
        model_layers_19_attn_k_proj_repeat_kv_reshape_4 = self.model_layers_19_attn_k_proj_repeat_kv_Reshape_4(model_layers_19_attn_k_proj_repeat_kv_transpose_2, initializers_onnx_initializer_717);  model_layers_19_attn_k_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_717 = None
        com_microsoft__model_layers_19_attn_multi_head_attention = self.com_microsoft__model_layers_19_attn_MultiHeadAttention(com_microsoft__model_layers_19_attn_q_rotary_rotary_embedding, model_layers_19_attn_k_proj_repeat_kv_reshape_4, model_layers_19_attn_v_proj_repeat_kv_reshape_4, attention_mask = model_attn_mask_reformat_tile);  com_microsoft__model_layers_19_attn_q_rotary_rotary_embedding = model_layers_19_attn_k_proj_repeat_kv_reshape_4 = model_layers_19_attn_v_proj_repeat_kv_reshape_4 = None
        getitem_151 = com_microsoft__model_layers_19_attn_multi_head_attention[0];  com_microsoft__model_layers_19_attn_multi_head_attention = None
        initializers_onnx_initializer_718 = self.initializers.onnx_initializer_718
        model_layers_19_attn_o_proj_mat_mul = self.model_layers_19_attn_o_proj_MatMul(getitem_151, initializers_onnx_initializer_718);  getitem_151 = initializers_onnx_initializer_718 = None
        getitem_152 = com_microsoft__model_layers_19_input_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_19_input_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_719 = self.initializers.onnx_initializer_719
        com_microsoft__model_layers_19_post_attention_layernorm_skip_layer_norm = self.com_microsoft__model_layers_19_post_attention_layernorm_SkipLayerNorm(getitem_152, model_layers_19_attn_o_proj_mat_mul, initializers_onnx_initializer_719);  getitem_152 = model_layers_19_attn_o_proj_mat_mul = initializers_onnx_initializer_719 = None
        getitem_153 = com_microsoft__model_layers_19_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_720 = self.initializers.onnx_initializer_720
        model_layers_19_mlp_gate_proj_mat_mul = self.model_layers_19_mlp_gate_proj_MatMul(getitem_153, initializers_onnx_initializer_720);  getitem_153 = initializers_onnx_initializer_720 = None
        getitem_154 = com_microsoft__model_layers_19_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_721 = self.initializers.onnx_initializer_721
        model_layers_19_mlp_up_proj_mat_mul = self.model_layers_19_mlp_up_proj_MatMul(getitem_154, initializers_onnx_initializer_721);  getitem_154 = initializers_onnx_initializer_721 = None
        model_layers_19_mlp_act_fn_sigmoid = self.model_layers_19_mlp_act_fn_Sigmoid(model_layers_19_mlp_gate_proj_mat_mul)
        model_layers_19_mlp_act_fn_mul = self.model_layers_19_mlp_act_fn_Mul(model_layers_19_mlp_gate_proj_mat_mul, model_layers_19_mlp_act_fn_sigmoid);  model_layers_19_mlp_gate_proj_mat_mul = model_layers_19_mlp_act_fn_sigmoid = None
        model_layers_19_mlp_mul = self.model_layers_19_mlp_Mul(model_layers_19_mlp_act_fn_mul, model_layers_19_mlp_up_proj_mat_mul);  model_layers_19_mlp_act_fn_mul = model_layers_19_mlp_up_proj_mat_mul = None
        initializers_onnx_initializer_722 = self.initializers.onnx_initializer_722
        model_layers_19_mlp_down_proj_mat_mul = self.model_layers_19_mlp_down_proj_MatMul(model_layers_19_mlp_mul, initializers_onnx_initializer_722);  model_layers_19_mlp_mul = initializers_onnx_initializer_722 = None
        getitem_155 = com_microsoft__model_layers_19_post_attention_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_19_post_attention_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_723 = self.initializers.onnx_initializer_723
        com_microsoft__model_layers_20_input_layernorm_skip_layer_norm = self.com_microsoft__model_layers_20_input_layernorm_SkipLayerNorm(getitem_155, model_layers_19_mlp_down_proj_mat_mul, initializers_onnx_initializer_723);  getitem_155 = model_layers_19_mlp_down_proj_mat_mul = initializers_onnx_initializer_723 = None
        getitem_156 = com_microsoft__model_layers_20_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_724 = self.initializers.onnx_initializer_724
        model_layers_20_attn_q_proj_mat_mul = self.model_layers_20_attn_q_proj_MatMul(getitem_156, initializers_onnx_initializer_724);  getitem_156 = initializers_onnx_initializer_724 = None
        getitem_157 = com_microsoft__model_layers_20_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_725 = self.initializers.onnx_initializer_725
        model_layers_20_attn_k_proj_mat_mul = self.model_layers_20_attn_k_proj_MatMul(getitem_157, initializers_onnx_initializer_725);  getitem_157 = initializers_onnx_initializer_725 = None
        getitem_158 = com_microsoft__model_layers_20_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_726 = self.initializers.onnx_initializer_726
        model_layers_20_attn_v_proj_mat_mul = self.model_layers_20_attn_v_proj_MatMul(getitem_158, initializers_onnx_initializer_726);  getitem_158 = initializers_onnx_initializer_726 = None
        initializers_onnx_initializer_727 = self.initializers.onnx_initializer_727
        initializers_onnx_initializer_728 = self.initializers.onnx_initializer_728
        com_microsoft__model_layers_20_attn_q_rotary_rotary_embedding = self.com_microsoft__model_layers_20_attn_q_rotary_RotaryEmbedding(model_layers_20_attn_q_proj_mat_mul, input_3, initializers_onnx_initializer_727, initializers_onnx_initializer_728);  model_layers_20_attn_q_proj_mat_mul = initializers_onnx_initializer_727 = initializers_onnx_initializer_728 = None
        initializers_onnx_initializer_729 = self.initializers.onnx_initializer_729
        initializers_onnx_initializer_730 = self.initializers.onnx_initializer_730
        com_microsoft__model_layers_20_attn_k_rotary_rotary_embedding = self.com_microsoft__model_layers_20_attn_k_rotary_RotaryEmbedding(model_layers_20_attn_k_proj_mat_mul, input_3, initializers_onnx_initializer_729, initializers_onnx_initializer_730);  model_layers_20_attn_k_proj_mat_mul = initializers_onnx_initializer_729 = initializers_onnx_initializer_730 = None
        initializers_onnx_initializer_731 = self.initializers.onnx_initializer_731
        model_layers_20_attn_v_proj_repeat_kv_reshape_1 = self.model_layers_20_attn_v_proj_repeat_kv_Reshape_1(model_layers_20_attn_v_proj_mat_mul, initializers_onnx_initializer_731);  model_layers_20_attn_v_proj_mat_mul = initializers_onnx_initializer_731 = None
        initializers_onnx_initializer_732 = self.initializers.onnx_initializer_732
        model_layers_20_attn_k_proj_repeat_kv_reshape_1 = self.model_layers_20_attn_k_proj_repeat_kv_Reshape_1(com_microsoft__model_layers_20_attn_k_rotary_rotary_embedding, initializers_onnx_initializer_732);  com_microsoft__model_layers_20_attn_k_rotary_rotary_embedding = initializers_onnx_initializer_732 = None
        model_layers_20_attn_v_proj_repeat_kv_transpose_1 = self.model_layers_20_attn_v_proj_repeat_kv_Transpose_1(model_layers_20_attn_v_proj_repeat_kv_reshape_1);  model_layers_20_attn_v_proj_repeat_kv_reshape_1 = None
        model_layers_20_attn_k_proj_repeat_kv_transpose_1 = self.model_layers_20_attn_k_proj_repeat_kv_Transpose_1(model_layers_20_attn_k_proj_repeat_kv_reshape_1);  model_layers_20_attn_k_proj_repeat_kv_reshape_1 = None
        model_layers_20_attn_v_proj_repeat_kv_concat_1 = self.model_layers_20_attn_v_proj_repeat_kv_Concat_1(input_45, model_layers_20_attn_v_proj_repeat_kv_transpose_1);  input_45 = model_layers_20_attn_v_proj_repeat_kv_transpose_1 = None
        model_layers_20_attn_k_proj_repeat_kv_concat_1 = self.model_layers_20_attn_k_proj_repeat_kv_Concat_1(input_44, model_layers_20_attn_k_proj_repeat_kv_transpose_1);  input_44 = model_layers_20_attn_k_proj_repeat_kv_transpose_1 = None
        model_layers_20_attn_v_proj_repeat_kv_shape_1 = self.model_layers_20_attn_v_proj_repeat_kv_Shape_1(model_layers_20_attn_v_proj_repeat_kv_concat_1)
        model_layers_20_attn_v_proj_repeat_kv_unsqueeze_5 = self.model_layers_20_attn_v_proj_repeat_kv_Unsqueeze_5(model_layers_20_attn_v_proj_repeat_kv_concat_1)
        model_layers_20_attn_k_proj_repeat_kv_shape_1 = self.model_layers_20_attn_k_proj_repeat_kv_Shape_1(model_layers_20_attn_k_proj_repeat_kv_concat_1)
        model_layers_20_attn_k_proj_repeat_kv_unsqueeze_5 = self.model_layers_20_attn_k_proj_repeat_kv_Unsqueeze_5(model_layers_20_attn_k_proj_repeat_kv_concat_1)
        initializers_onnx_initializer_733 = self.initializers.onnx_initializer_733
        model_layers_20_attn_v_proj_repeat_kv_gather_1 = self.model_layers_20_attn_v_proj_repeat_kv_Gather_1(model_layers_20_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_733);  initializers_onnx_initializer_733 = None
        initializers_onnx_initializer_734 = self.initializers.onnx_initializer_734
        model_layers_20_attn_v_proj_repeat_kv_gather_3 = self.model_layers_20_attn_v_proj_repeat_kv_Gather_3(model_layers_20_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_734);  model_layers_20_attn_v_proj_repeat_kv_shape_1 = initializers_onnx_initializer_734 = None
        initializers_onnx_initializer_735 = self.initializers.onnx_initializer_735
        model_layers_20_attn_k_proj_repeat_kv_gather_1 = self.model_layers_20_attn_k_proj_repeat_kv_Gather_1(model_layers_20_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_735);  initializers_onnx_initializer_735 = None
        initializers_onnx_initializer_736 = self.initializers.onnx_initializer_736
        model_layers_20_attn_k_proj_repeat_kv_gather_3 = self.model_layers_20_attn_k_proj_repeat_kv_Gather_3(model_layers_20_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_736);  model_layers_20_attn_k_proj_repeat_kv_shape_1 = initializers_onnx_initializer_736 = None
        model_layers_20_attn_v_proj_repeat_kv_unsqueeze_1 = self.model_layers_20_attn_v_proj_repeat_kv_Unsqueeze_1(model_layers_20_attn_v_proj_repeat_kv_gather_1);  model_layers_20_attn_v_proj_repeat_kv_gather_1 = None
        model_layers_20_attn_v_proj_repeat_kv_unsqueeze_3 = self.model_layers_20_attn_v_proj_repeat_kv_Unsqueeze_3(model_layers_20_attn_v_proj_repeat_kv_gather_3);  model_layers_20_attn_v_proj_repeat_kv_gather_3 = None
        model_layers_20_attn_k_proj_repeat_kv_unsqueeze_1 = self.model_layers_20_attn_k_proj_repeat_kv_Unsqueeze_1(model_layers_20_attn_k_proj_repeat_kv_gather_1);  model_layers_20_attn_k_proj_repeat_kv_gather_1 = None
        model_layers_20_attn_k_proj_repeat_kv_unsqueeze_3 = self.model_layers_20_attn_k_proj_repeat_kv_Unsqueeze_3(model_layers_20_attn_k_proj_repeat_kv_gather_3);  model_layers_20_attn_k_proj_repeat_kv_gather_3 = None
        initializers_onnx_initializer_737 = self.initializers.onnx_initializer_737
        initializers_onnx_initializer_738 = self.initializers.onnx_initializer_738
        initializers_onnx_initializer_739 = self.initializers.onnx_initializer_739
        model_layers_20_attn_v_proj_repeat_kv_concat_2 = self.model_layers_20_attn_v_proj_repeat_kv_Concat_2(model_layers_20_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_737, initializers_onnx_initializer_738, model_layers_20_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_739);  initializers_onnx_initializer_737 = initializers_onnx_initializer_738 = initializers_onnx_initializer_739 = None
        initializers_onnx_initializer_740 = self.initializers.onnx_initializer_740
        initializers_onnx_initializer_741 = self.initializers.onnx_initializer_741
        model_layers_20_attn_v_proj_repeat_kv_concat_3 = self.model_layers_20_attn_v_proj_repeat_kv_Concat_3(model_layers_20_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_740, model_layers_20_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_741);  model_layers_20_attn_v_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_740 = model_layers_20_attn_v_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_741 = None
        initializers_onnx_initializer_742 = self.initializers.onnx_initializer_742
        initializers_onnx_initializer_743 = self.initializers.onnx_initializer_743
        initializers_onnx_initializer_744 = self.initializers.onnx_initializer_744
        model_layers_20_attn_k_proj_repeat_kv_concat_2 = self.model_layers_20_attn_k_proj_repeat_kv_Concat_2(model_layers_20_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_742, initializers_onnx_initializer_743, model_layers_20_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_744);  initializers_onnx_initializer_742 = initializers_onnx_initializer_743 = initializers_onnx_initializer_744 = None
        initializers_onnx_initializer_745 = self.initializers.onnx_initializer_745
        initializers_onnx_initializer_746 = self.initializers.onnx_initializer_746
        model_layers_20_attn_k_proj_repeat_kv_concat_3 = self.model_layers_20_attn_k_proj_repeat_kv_Concat_3(model_layers_20_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_745, model_layers_20_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_746);  model_layers_20_attn_k_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_745 = model_layers_20_attn_k_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_746 = None
        initializers_onnx_initializer_747 = self.initializers.onnx_initializer_747
        model_layers_20_attn_v_proj_repeat_kv_equal = self.model_layers_20_attn_v_proj_repeat_kv_Equal(model_layers_20_attn_v_proj_repeat_kv_concat_2, initializers_onnx_initializer_747);  initializers_onnx_initializer_747 = None
        initializers_onnx_initializer_748 = self.initializers.onnx_initializer_748
        model_layers_20_attn_k_proj_repeat_kv_equal = self.model_layers_20_attn_k_proj_repeat_kv_Equal(model_layers_20_attn_k_proj_repeat_kv_concat_2, initializers_onnx_initializer_748);  initializers_onnx_initializer_748 = None
        initializers_onnx_initializer_749 = self.initializers.onnx_initializer_749
        model_layers_20_attn_v_proj_repeat_kv_where = self.model_layers_20_attn_v_proj_repeat_kv_Where(model_layers_20_attn_v_proj_repeat_kv_equal, initializers_onnx_initializer_749, model_layers_20_attn_v_proj_repeat_kv_concat_2);  model_layers_20_attn_v_proj_repeat_kv_equal = initializers_onnx_initializer_749 = model_layers_20_attn_v_proj_repeat_kv_concat_2 = None
        initializers_onnx_initializer_750 = self.initializers.onnx_initializer_750
        model_layers_20_attn_k_proj_repeat_kv_where = self.model_layers_20_attn_k_proj_repeat_kv_Where(model_layers_20_attn_k_proj_repeat_kv_equal, initializers_onnx_initializer_750, model_layers_20_attn_k_proj_repeat_kv_concat_2);  model_layers_20_attn_k_proj_repeat_kv_equal = initializers_onnx_initializer_750 = model_layers_20_attn_k_proj_repeat_kv_concat_2 = None
        model_layers_20_attn_v_proj_repeat_kv_expand = self.model_layers_20_attn_v_proj_repeat_kv_Expand(model_layers_20_attn_v_proj_repeat_kv_unsqueeze_5, model_layers_20_attn_v_proj_repeat_kv_where);  model_layers_20_attn_v_proj_repeat_kv_unsqueeze_5 = model_layers_20_attn_v_proj_repeat_kv_where = None
        model_layers_20_attn_k_proj_repeat_kv_expand = self.model_layers_20_attn_k_proj_repeat_kv_Expand(model_layers_20_attn_k_proj_repeat_kv_unsqueeze_5, model_layers_20_attn_k_proj_repeat_kv_where);  model_layers_20_attn_k_proj_repeat_kv_unsqueeze_5 = model_layers_20_attn_k_proj_repeat_kv_where = None
        model_layers_20_attn_v_proj_repeat_kv_reshape_3 = self.model_layers_20_attn_v_proj_repeat_kv_Reshape_3(model_layers_20_attn_v_proj_repeat_kv_expand, model_layers_20_attn_v_proj_repeat_kv_concat_3);  model_layers_20_attn_v_proj_repeat_kv_expand = model_layers_20_attn_v_proj_repeat_kv_concat_3 = None
        model_layers_20_attn_k_proj_repeat_kv_reshape_3 = self.model_layers_20_attn_k_proj_repeat_kv_Reshape_3(model_layers_20_attn_k_proj_repeat_kv_expand, model_layers_20_attn_k_proj_repeat_kv_concat_3);  model_layers_20_attn_k_proj_repeat_kv_expand = model_layers_20_attn_k_proj_repeat_kv_concat_3 = None
        model_layers_20_attn_v_proj_repeat_kv_transpose_2 = self.model_layers_20_attn_v_proj_repeat_kv_Transpose_2(model_layers_20_attn_v_proj_repeat_kv_reshape_3);  model_layers_20_attn_v_proj_repeat_kv_reshape_3 = None
        model_layers_20_attn_k_proj_repeat_kv_transpose_2 = self.model_layers_20_attn_k_proj_repeat_kv_Transpose_2(model_layers_20_attn_k_proj_repeat_kv_reshape_3);  model_layers_20_attn_k_proj_repeat_kv_reshape_3 = None
        initializers_onnx_initializer_751 = self.initializers.onnx_initializer_751
        model_layers_20_attn_v_proj_repeat_kv_reshape_4 = self.model_layers_20_attn_v_proj_repeat_kv_Reshape_4(model_layers_20_attn_v_proj_repeat_kv_transpose_2, initializers_onnx_initializer_751);  model_layers_20_attn_v_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_751 = None
        initializers_onnx_initializer_752 = self.initializers.onnx_initializer_752
        model_layers_20_attn_k_proj_repeat_kv_reshape_4 = self.model_layers_20_attn_k_proj_repeat_kv_Reshape_4(model_layers_20_attn_k_proj_repeat_kv_transpose_2, initializers_onnx_initializer_752);  model_layers_20_attn_k_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_752 = None
        com_microsoft__model_layers_20_attn_multi_head_attention = self.com_microsoft__model_layers_20_attn_MultiHeadAttention(com_microsoft__model_layers_20_attn_q_rotary_rotary_embedding, model_layers_20_attn_k_proj_repeat_kv_reshape_4, model_layers_20_attn_v_proj_repeat_kv_reshape_4, attention_mask = model_attn_mask_reformat_tile);  com_microsoft__model_layers_20_attn_q_rotary_rotary_embedding = model_layers_20_attn_k_proj_repeat_kv_reshape_4 = model_layers_20_attn_v_proj_repeat_kv_reshape_4 = None
        getitem_159 = com_microsoft__model_layers_20_attn_multi_head_attention[0];  com_microsoft__model_layers_20_attn_multi_head_attention = None
        initializers_onnx_initializer_753 = self.initializers.onnx_initializer_753
        model_layers_20_attn_o_proj_mat_mul = self.model_layers_20_attn_o_proj_MatMul(getitem_159, initializers_onnx_initializer_753);  getitem_159 = initializers_onnx_initializer_753 = None
        getitem_160 = com_microsoft__model_layers_20_input_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_20_input_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_754 = self.initializers.onnx_initializer_754
        com_microsoft__model_layers_20_post_attention_layernorm_skip_layer_norm = self.com_microsoft__model_layers_20_post_attention_layernorm_SkipLayerNorm(getitem_160, model_layers_20_attn_o_proj_mat_mul, initializers_onnx_initializer_754);  getitem_160 = model_layers_20_attn_o_proj_mat_mul = initializers_onnx_initializer_754 = None
        getitem_161 = com_microsoft__model_layers_20_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_755 = self.initializers.onnx_initializer_755
        model_layers_20_mlp_gate_proj_mat_mul = self.model_layers_20_mlp_gate_proj_MatMul(getitem_161, initializers_onnx_initializer_755);  getitem_161 = initializers_onnx_initializer_755 = None
        getitem_162 = com_microsoft__model_layers_20_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_756 = self.initializers.onnx_initializer_756
        model_layers_20_mlp_up_proj_mat_mul = self.model_layers_20_mlp_up_proj_MatMul(getitem_162, initializers_onnx_initializer_756);  getitem_162 = initializers_onnx_initializer_756 = None
        model_layers_20_mlp_act_fn_sigmoid = self.model_layers_20_mlp_act_fn_Sigmoid(model_layers_20_mlp_gate_proj_mat_mul)
        model_layers_20_mlp_act_fn_mul = self.model_layers_20_mlp_act_fn_Mul(model_layers_20_mlp_gate_proj_mat_mul, model_layers_20_mlp_act_fn_sigmoid);  model_layers_20_mlp_gate_proj_mat_mul = model_layers_20_mlp_act_fn_sigmoid = None
        model_layers_20_mlp_mul = self.model_layers_20_mlp_Mul(model_layers_20_mlp_act_fn_mul, model_layers_20_mlp_up_proj_mat_mul);  model_layers_20_mlp_act_fn_mul = model_layers_20_mlp_up_proj_mat_mul = None
        initializers_onnx_initializer_757 = self.initializers.onnx_initializer_757
        model_layers_20_mlp_down_proj_mat_mul = self.model_layers_20_mlp_down_proj_MatMul(model_layers_20_mlp_mul, initializers_onnx_initializer_757);  model_layers_20_mlp_mul = initializers_onnx_initializer_757 = None
        getitem_163 = com_microsoft__model_layers_20_post_attention_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_20_post_attention_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_758 = self.initializers.onnx_initializer_758
        com_microsoft__model_layers_21_input_layernorm_skip_layer_norm = self.com_microsoft__model_layers_21_input_layernorm_SkipLayerNorm(getitem_163, model_layers_20_mlp_down_proj_mat_mul, initializers_onnx_initializer_758);  getitem_163 = model_layers_20_mlp_down_proj_mat_mul = initializers_onnx_initializer_758 = None
        getitem_164 = com_microsoft__model_layers_21_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_759 = self.initializers.onnx_initializer_759
        model_layers_21_attn_q_proj_mat_mul = self.model_layers_21_attn_q_proj_MatMul(getitem_164, initializers_onnx_initializer_759);  getitem_164 = initializers_onnx_initializer_759 = None
        getitem_165 = com_microsoft__model_layers_21_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_760 = self.initializers.onnx_initializer_760
        model_layers_21_attn_k_proj_mat_mul = self.model_layers_21_attn_k_proj_MatMul(getitem_165, initializers_onnx_initializer_760);  getitem_165 = initializers_onnx_initializer_760 = None
        getitem_166 = com_microsoft__model_layers_21_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_761 = self.initializers.onnx_initializer_761
        model_layers_21_attn_v_proj_mat_mul = self.model_layers_21_attn_v_proj_MatMul(getitem_166, initializers_onnx_initializer_761);  getitem_166 = initializers_onnx_initializer_761 = None
        initializers_onnx_initializer_762 = self.initializers.onnx_initializer_762
        initializers_onnx_initializer_763 = self.initializers.onnx_initializer_763
        com_microsoft__model_layers_21_attn_q_rotary_rotary_embedding = self.com_microsoft__model_layers_21_attn_q_rotary_RotaryEmbedding(model_layers_21_attn_q_proj_mat_mul, input_3, initializers_onnx_initializer_762, initializers_onnx_initializer_763);  model_layers_21_attn_q_proj_mat_mul = initializers_onnx_initializer_762 = initializers_onnx_initializer_763 = None
        initializers_onnx_initializer_764 = self.initializers.onnx_initializer_764
        initializers_onnx_initializer_765 = self.initializers.onnx_initializer_765
        com_microsoft__model_layers_21_attn_k_rotary_rotary_embedding = self.com_microsoft__model_layers_21_attn_k_rotary_RotaryEmbedding(model_layers_21_attn_k_proj_mat_mul, input_3, initializers_onnx_initializer_764, initializers_onnx_initializer_765);  model_layers_21_attn_k_proj_mat_mul = initializers_onnx_initializer_764 = initializers_onnx_initializer_765 = None
        initializers_onnx_initializer_766 = self.initializers.onnx_initializer_766
        model_layers_21_attn_v_proj_repeat_kv_reshape_1 = self.model_layers_21_attn_v_proj_repeat_kv_Reshape_1(model_layers_21_attn_v_proj_mat_mul, initializers_onnx_initializer_766);  model_layers_21_attn_v_proj_mat_mul = initializers_onnx_initializer_766 = None
        initializers_onnx_initializer_767 = self.initializers.onnx_initializer_767
        model_layers_21_attn_k_proj_repeat_kv_reshape_1 = self.model_layers_21_attn_k_proj_repeat_kv_Reshape_1(com_microsoft__model_layers_21_attn_k_rotary_rotary_embedding, initializers_onnx_initializer_767);  com_microsoft__model_layers_21_attn_k_rotary_rotary_embedding = initializers_onnx_initializer_767 = None
        model_layers_21_attn_v_proj_repeat_kv_transpose_1 = self.model_layers_21_attn_v_proj_repeat_kv_Transpose_1(model_layers_21_attn_v_proj_repeat_kv_reshape_1);  model_layers_21_attn_v_proj_repeat_kv_reshape_1 = None
        model_layers_21_attn_k_proj_repeat_kv_transpose_1 = self.model_layers_21_attn_k_proj_repeat_kv_Transpose_1(model_layers_21_attn_k_proj_repeat_kv_reshape_1);  model_layers_21_attn_k_proj_repeat_kv_reshape_1 = None
        model_layers_21_attn_v_proj_repeat_kv_concat_1 = self.model_layers_21_attn_v_proj_repeat_kv_Concat_1(input_47, model_layers_21_attn_v_proj_repeat_kv_transpose_1);  input_47 = model_layers_21_attn_v_proj_repeat_kv_transpose_1 = None
        model_layers_21_attn_k_proj_repeat_kv_concat_1 = self.model_layers_21_attn_k_proj_repeat_kv_Concat_1(input_46, model_layers_21_attn_k_proj_repeat_kv_transpose_1);  input_46 = model_layers_21_attn_k_proj_repeat_kv_transpose_1 = None
        model_layers_21_attn_v_proj_repeat_kv_shape_1 = self.model_layers_21_attn_v_proj_repeat_kv_Shape_1(model_layers_21_attn_v_proj_repeat_kv_concat_1)
        model_layers_21_attn_v_proj_repeat_kv_unsqueeze_5 = self.model_layers_21_attn_v_proj_repeat_kv_Unsqueeze_5(model_layers_21_attn_v_proj_repeat_kv_concat_1)
        model_layers_21_attn_k_proj_repeat_kv_shape_1 = self.model_layers_21_attn_k_proj_repeat_kv_Shape_1(model_layers_21_attn_k_proj_repeat_kv_concat_1)
        model_layers_21_attn_k_proj_repeat_kv_unsqueeze_5 = self.model_layers_21_attn_k_proj_repeat_kv_Unsqueeze_5(model_layers_21_attn_k_proj_repeat_kv_concat_1)
        initializers_onnx_initializer_768 = self.initializers.onnx_initializer_768
        model_layers_21_attn_v_proj_repeat_kv_gather_1 = self.model_layers_21_attn_v_proj_repeat_kv_Gather_1(model_layers_21_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_768);  initializers_onnx_initializer_768 = None
        initializers_onnx_initializer_769 = self.initializers.onnx_initializer_769
        model_layers_21_attn_v_proj_repeat_kv_gather_3 = self.model_layers_21_attn_v_proj_repeat_kv_Gather_3(model_layers_21_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_769);  model_layers_21_attn_v_proj_repeat_kv_shape_1 = initializers_onnx_initializer_769 = None
        initializers_onnx_initializer_770 = self.initializers.onnx_initializer_770
        model_layers_21_attn_k_proj_repeat_kv_gather_1 = self.model_layers_21_attn_k_proj_repeat_kv_Gather_1(model_layers_21_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_770);  initializers_onnx_initializer_770 = None
        initializers_onnx_initializer_771 = self.initializers.onnx_initializer_771
        model_layers_21_attn_k_proj_repeat_kv_gather_3 = self.model_layers_21_attn_k_proj_repeat_kv_Gather_3(model_layers_21_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_771);  model_layers_21_attn_k_proj_repeat_kv_shape_1 = initializers_onnx_initializer_771 = None
        model_layers_21_attn_v_proj_repeat_kv_unsqueeze_1 = self.model_layers_21_attn_v_proj_repeat_kv_Unsqueeze_1(model_layers_21_attn_v_proj_repeat_kv_gather_1);  model_layers_21_attn_v_proj_repeat_kv_gather_1 = None
        model_layers_21_attn_v_proj_repeat_kv_unsqueeze_3 = self.model_layers_21_attn_v_proj_repeat_kv_Unsqueeze_3(model_layers_21_attn_v_proj_repeat_kv_gather_3);  model_layers_21_attn_v_proj_repeat_kv_gather_3 = None
        model_layers_21_attn_k_proj_repeat_kv_unsqueeze_1 = self.model_layers_21_attn_k_proj_repeat_kv_Unsqueeze_1(model_layers_21_attn_k_proj_repeat_kv_gather_1);  model_layers_21_attn_k_proj_repeat_kv_gather_1 = None
        model_layers_21_attn_k_proj_repeat_kv_unsqueeze_3 = self.model_layers_21_attn_k_proj_repeat_kv_Unsqueeze_3(model_layers_21_attn_k_proj_repeat_kv_gather_3);  model_layers_21_attn_k_proj_repeat_kv_gather_3 = None
        initializers_onnx_initializer_772 = self.initializers.onnx_initializer_772
        initializers_onnx_initializer_773 = self.initializers.onnx_initializer_773
        initializers_onnx_initializer_774 = self.initializers.onnx_initializer_774
        model_layers_21_attn_v_proj_repeat_kv_concat_2 = self.model_layers_21_attn_v_proj_repeat_kv_Concat_2(model_layers_21_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_772, initializers_onnx_initializer_773, model_layers_21_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_774);  initializers_onnx_initializer_772 = initializers_onnx_initializer_773 = initializers_onnx_initializer_774 = None
        initializers_onnx_initializer_775 = self.initializers.onnx_initializer_775
        initializers_onnx_initializer_776 = self.initializers.onnx_initializer_776
        model_layers_21_attn_v_proj_repeat_kv_concat_3 = self.model_layers_21_attn_v_proj_repeat_kv_Concat_3(model_layers_21_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_775, model_layers_21_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_776);  model_layers_21_attn_v_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_775 = model_layers_21_attn_v_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_776 = None
        initializers_onnx_initializer_777 = self.initializers.onnx_initializer_777
        initializers_onnx_initializer_778 = self.initializers.onnx_initializer_778
        initializers_onnx_initializer_779 = self.initializers.onnx_initializer_779
        model_layers_21_attn_k_proj_repeat_kv_concat_2 = self.model_layers_21_attn_k_proj_repeat_kv_Concat_2(model_layers_21_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_777, initializers_onnx_initializer_778, model_layers_21_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_779);  initializers_onnx_initializer_777 = initializers_onnx_initializer_778 = initializers_onnx_initializer_779 = None
        initializers_onnx_initializer_780 = self.initializers.onnx_initializer_780
        initializers_onnx_initializer_781 = self.initializers.onnx_initializer_781
        model_layers_21_attn_k_proj_repeat_kv_concat_3 = self.model_layers_21_attn_k_proj_repeat_kv_Concat_3(model_layers_21_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_780, model_layers_21_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_781);  model_layers_21_attn_k_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_780 = model_layers_21_attn_k_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_781 = None
        initializers_onnx_initializer_782 = self.initializers.onnx_initializer_782
        model_layers_21_attn_v_proj_repeat_kv_equal = self.model_layers_21_attn_v_proj_repeat_kv_Equal(model_layers_21_attn_v_proj_repeat_kv_concat_2, initializers_onnx_initializer_782);  initializers_onnx_initializer_782 = None
        initializers_onnx_initializer_783 = self.initializers.onnx_initializer_783
        model_layers_21_attn_k_proj_repeat_kv_equal = self.model_layers_21_attn_k_proj_repeat_kv_Equal(model_layers_21_attn_k_proj_repeat_kv_concat_2, initializers_onnx_initializer_783);  initializers_onnx_initializer_783 = None
        initializers_onnx_initializer_784 = self.initializers.onnx_initializer_784
        model_layers_21_attn_v_proj_repeat_kv_where = self.model_layers_21_attn_v_proj_repeat_kv_Where(model_layers_21_attn_v_proj_repeat_kv_equal, initializers_onnx_initializer_784, model_layers_21_attn_v_proj_repeat_kv_concat_2);  model_layers_21_attn_v_proj_repeat_kv_equal = initializers_onnx_initializer_784 = model_layers_21_attn_v_proj_repeat_kv_concat_2 = None
        initializers_onnx_initializer_785 = self.initializers.onnx_initializer_785
        model_layers_21_attn_k_proj_repeat_kv_where = self.model_layers_21_attn_k_proj_repeat_kv_Where(model_layers_21_attn_k_proj_repeat_kv_equal, initializers_onnx_initializer_785, model_layers_21_attn_k_proj_repeat_kv_concat_2);  model_layers_21_attn_k_proj_repeat_kv_equal = initializers_onnx_initializer_785 = model_layers_21_attn_k_proj_repeat_kv_concat_2 = None
        model_layers_21_attn_v_proj_repeat_kv_expand = self.model_layers_21_attn_v_proj_repeat_kv_Expand(model_layers_21_attn_v_proj_repeat_kv_unsqueeze_5, model_layers_21_attn_v_proj_repeat_kv_where);  model_layers_21_attn_v_proj_repeat_kv_unsqueeze_5 = model_layers_21_attn_v_proj_repeat_kv_where = None
        model_layers_21_attn_k_proj_repeat_kv_expand = self.model_layers_21_attn_k_proj_repeat_kv_Expand(model_layers_21_attn_k_proj_repeat_kv_unsqueeze_5, model_layers_21_attn_k_proj_repeat_kv_where);  model_layers_21_attn_k_proj_repeat_kv_unsqueeze_5 = model_layers_21_attn_k_proj_repeat_kv_where = None
        model_layers_21_attn_v_proj_repeat_kv_reshape_3 = self.model_layers_21_attn_v_proj_repeat_kv_Reshape_3(model_layers_21_attn_v_proj_repeat_kv_expand, model_layers_21_attn_v_proj_repeat_kv_concat_3);  model_layers_21_attn_v_proj_repeat_kv_expand = model_layers_21_attn_v_proj_repeat_kv_concat_3 = None
        model_layers_21_attn_k_proj_repeat_kv_reshape_3 = self.model_layers_21_attn_k_proj_repeat_kv_Reshape_3(model_layers_21_attn_k_proj_repeat_kv_expand, model_layers_21_attn_k_proj_repeat_kv_concat_3);  model_layers_21_attn_k_proj_repeat_kv_expand = model_layers_21_attn_k_proj_repeat_kv_concat_3 = None
        model_layers_21_attn_v_proj_repeat_kv_transpose_2 = self.model_layers_21_attn_v_proj_repeat_kv_Transpose_2(model_layers_21_attn_v_proj_repeat_kv_reshape_3);  model_layers_21_attn_v_proj_repeat_kv_reshape_3 = None
        model_layers_21_attn_k_proj_repeat_kv_transpose_2 = self.model_layers_21_attn_k_proj_repeat_kv_Transpose_2(model_layers_21_attn_k_proj_repeat_kv_reshape_3);  model_layers_21_attn_k_proj_repeat_kv_reshape_3 = None
        initializers_onnx_initializer_786 = self.initializers.onnx_initializer_786
        model_layers_21_attn_v_proj_repeat_kv_reshape_4 = self.model_layers_21_attn_v_proj_repeat_kv_Reshape_4(model_layers_21_attn_v_proj_repeat_kv_transpose_2, initializers_onnx_initializer_786);  model_layers_21_attn_v_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_786 = None
        initializers_onnx_initializer_787 = self.initializers.onnx_initializer_787
        model_layers_21_attn_k_proj_repeat_kv_reshape_4 = self.model_layers_21_attn_k_proj_repeat_kv_Reshape_4(model_layers_21_attn_k_proj_repeat_kv_transpose_2, initializers_onnx_initializer_787);  model_layers_21_attn_k_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_787 = None
        com_microsoft__model_layers_21_attn_multi_head_attention = self.com_microsoft__model_layers_21_attn_MultiHeadAttention(com_microsoft__model_layers_21_attn_q_rotary_rotary_embedding, model_layers_21_attn_k_proj_repeat_kv_reshape_4, model_layers_21_attn_v_proj_repeat_kv_reshape_4, attention_mask = model_attn_mask_reformat_tile);  com_microsoft__model_layers_21_attn_q_rotary_rotary_embedding = model_layers_21_attn_k_proj_repeat_kv_reshape_4 = model_layers_21_attn_v_proj_repeat_kv_reshape_4 = None
        getitem_167 = com_microsoft__model_layers_21_attn_multi_head_attention[0];  com_microsoft__model_layers_21_attn_multi_head_attention = None
        initializers_onnx_initializer_788 = self.initializers.onnx_initializer_788
        model_layers_21_attn_o_proj_mat_mul = self.model_layers_21_attn_o_proj_MatMul(getitem_167, initializers_onnx_initializer_788);  getitem_167 = initializers_onnx_initializer_788 = None
        getitem_168 = com_microsoft__model_layers_21_input_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_21_input_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_789 = self.initializers.onnx_initializer_789
        com_microsoft__model_layers_21_post_attention_layernorm_skip_layer_norm = self.com_microsoft__model_layers_21_post_attention_layernorm_SkipLayerNorm(getitem_168, model_layers_21_attn_o_proj_mat_mul, initializers_onnx_initializer_789);  getitem_168 = model_layers_21_attn_o_proj_mat_mul = initializers_onnx_initializer_789 = None
        getitem_169 = com_microsoft__model_layers_21_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_790 = self.initializers.onnx_initializer_790
        model_layers_21_mlp_gate_proj_mat_mul = self.model_layers_21_mlp_gate_proj_MatMul(getitem_169, initializers_onnx_initializer_790);  getitem_169 = initializers_onnx_initializer_790 = None
        getitem_170 = com_microsoft__model_layers_21_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_791 = self.initializers.onnx_initializer_791
        model_layers_21_mlp_up_proj_mat_mul = self.model_layers_21_mlp_up_proj_MatMul(getitem_170, initializers_onnx_initializer_791);  getitem_170 = initializers_onnx_initializer_791 = None
        model_layers_21_mlp_act_fn_sigmoid = self.model_layers_21_mlp_act_fn_Sigmoid(model_layers_21_mlp_gate_proj_mat_mul)
        model_layers_21_mlp_act_fn_mul = self.model_layers_21_mlp_act_fn_Mul(model_layers_21_mlp_gate_proj_mat_mul, model_layers_21_mlp_act_fn_sigmoid);  model_layers_21_mlp_gate_proj_mat_mul = model_layers_21_mlp_act_fn_sigmoid = None
        model_layers_21_mlp_mul = self.model_layers_21_mlp_Mul(model_layers_21_mlp_act_fn_mul, model_layers_21_mlp_up_proj_mat_mul);  model_layers_21_mlp_act_fn_mul = model_layers_21_mlp_up_proj_mat_mul = None
        initializers_onnx_initializer_792 = self.initializers.onnx_initializer_792
        model_layers_21_mlp_down_proj_mat_mul = self.model_layers_21_mlp_down_proj_MatMul(model_layers_21_mlp_mul, initializers_onnx_initializer_792);  model_layers_21_mlp_mul = initializers_onnx_initializer_792 = None
        getitem_171 = com_microsoft__model_layers_21_post_attention_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_21_post_attention_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_793 = self.initializers.onnx_initializer_793
        com_microsoft__model_layers_22_input_layernorm_skip_layer_norm = self.com_microsoft__model_layers_22_input_layernorm_SkipLayerNorm(getitem_171, model_layers_21_mlp_down_proj_mat_mul, initializers_onnx_initializer_793);  getitem_171 = model_layers_21_mlp_down_proj_mat_mul = initializers_onnx_initializer_793 = None
        getitem_172 = com_microsoft__model_layers_22_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_794 = self.initializers.onnx_initializer_794
        model_layers_22_attn_q_proj_mat_mul = self.model_layers_22_attn_q_proj_MatMul(getitem_172, initializers_onnx_initializer_794);  getitem_172 = initializers_onnx_initializer_794 = None
        getitem_173 = com_microsoft__model_layers_22_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_795 = self.initializers.onnx_initializer_795
        model_layers_22_attn_k_proj_mat_mul = self.model_layers_22_attn_k_proj_MatMul(getitem_173, initializers_onnx_initializer_795);  getitem_173 = initializers_onnx_initializer_795 = None
        getitem_174 = com_microsoft__model_layers_22_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_796 = self.initializers.onnx_initializer_796
        model_layers_22_attn_v_proj_mat_mul = self.model_layers_22_attn_v_proj_MatMul(getitem_174, initializers_onnx_initializer_796);  getitem_174 = initializers_onnx_initializer_796 = None
        initializers_onnx_initializer_797 = self.initializers.onnx_initializer_797
        initializers_onnx_initializer_798 = self.initializers.onnx_initializer_798
        com_microsoft__model_layers_22_attn_q_rotary_rotary_embedding = self.com_microsoft__model_layers_22_attn_q_rotary_RotaryEmbedding(model_layers_22_attn_q_proj_mat_mul, input_3, initializers_onnx_initializer_797, initializers_onnx_initializer_798);  model_layers_22_attn_q_proj_mat_mul = initializers_onnx_initializer_797 = initializers_onnx_initializer_798 = None
        initializers_onnx_initializer_799 = self.initializers.onnx_initializer_799
        initializers_onnx_initializer_800 = self.initializers.onnx_initializer_800
        com_microsoft__model_layers_22_attn_k_rotary_rotary_embedding = self.com_microsoft__model_layers_22_attn_k_rotary_RotaryEmbedding(model_layers_22_attn_k_proj_mat_mul, input_3, initializers_onnx_initializer_799, initializers_onnx_initializer_800);  model_layers_22_attn_k_proj_mat_mul = initializers_onnx_initializer_799 = initializers_onnx_initializer_800 = None
        initializers_onnx_initializer_801 = self.initializers.onnx_initializer_801
        model_layers_22_attn_v_proj_repeat_kv_reshape_1 = self.model_layers_22_attn_v_proj_repeat_kv_Reshape_1(model_layers_22_attn_v_proj_mat_mul, initializers_onnx_initializer_801);  model_layers_22_attn_v_proj_mat_mul = initializers_onnx_initializer_801 = None
        initializers_onnx_initializer_802 = self.initializers.onnx_initializer_802
        model_layers_22_attn_k_proj_repeat_kv_reshape_1 = self.model_layers_22_attn_k_proj_repeat_kv_Reshape_1(com_microsoft__model_layers_22_attn_k_rotary_rotary_embedding, initializers_onnx_initializer_802);  com_microsoft__model_layers_22_attn_k_rotary_rotary_embedding = initializers_onnx_initializer_802 = None
        model_layers_22_attn_v_proj_repeat_kv_transpose_1 = self.model_layers_22_attn_v_proj_repeat_kv_Transpose_1(model_layers_22_attn_v_proj_repeat_kv_reshape_1);  model_layers_22_attn_v_proj_repeat_kv_reshape_1 = None
        model_layers_22_attn_k_proj_repeat_kv_transpose_1 = self.model_layers_22_attn_k_proj_repeat_kv_Transpose_1(model_layers_22_attn_k_proj_repeat_kv_reshape_1);  model_layers_22_attn_k_proj_repeat_kv_reshape_1 = None
        model_layers_22_attn_v_proj_repeat_kv_concat_1 = self.model_layers_22_attn_v_proj_repeat_kv_Concat_1(input_49, model_layers_22_attn_v_proj_repeat_kv_transpose_1);  input_49 = model_layers_22_attn_v_proj_repeat_kv_transpose_1 = None
        model_layers_22_attn_k_proj_repeat_kv_concat_1 = self.model_layers_22_attn_k_proj_repeat_kv_Concat_1(input_48, model_layers_22_attn_k_proj_repeat_kv_transpose_1);  input_48 = model_layers_22_attn_k_proj_repeat_kv_transpose_1 = None
        model_layers_22_attn_v_proj_repeat_kv_shape_1 = self.model_layers_22_attn_v_proj_repeat_kv_Shape_1(model_layers_22_attn_v_proj_repeat_kv_concat_1)
        model_layers_22_attn_v_proj_repeat_kv_unsqueeze_5 = self.model_layers_22_attn_v_proj_repeat_kv_Unsqueeze_5(model_layers_22_attn_v_proj_repeat_kv_concat_1)
        model_layers_22_attn_k_proj_repeat_kv_shape_1 = self.model_layers_22_attn_k_proj_repeat_kv_Shape_1(model_layers_22_attn_k_proj_repeat_kv_concat_1)
        model_layers_22_attn_k_proj_repeat_kv_unsqueeze_5 = self.model_layers_22_attn_k_proj_repeat_kv_Unsqueeze_5(model_layers_22_attn_k_proj_repeat_kv_concat_1)
        initializers_onnx_initializer_803 = self.initializers.onnx_initializer_803
        model_layers_22_attn_v_proj_repeat_kv_gather_1 = self.model_layers_22_attn_v_proj_repeat_kv_Gather_1(model_layers_22_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_803);  initializers_onnx_initializer_803 = None
        initializers_onnx_initializer_804 = self.initializers.onnx_initializer_804
        model_layers_22_attn_v_proj_repeat_kv_gather_3 = self.model_layers_22_attn_v_proj_repeat_kv_Gather_3(model_layers_22_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_804);  model_layers_22_attn_v_proj_repeat_kv_shape_1 = initializers_onnx_initializer_804 = None
        initializers_onnx_initializer_805 = self.initializers.onnx_initializer_805
        model_layers_22_attn_k_proj_repeat_kv_gather_1 = self.model_layers_22_attn_k_proj_repeat_kv_Gather_1(model_layers_22_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_805);  initializers_onnx_initializer_805 = None
        initializers_onnx_initializer_806 = self.initializers.onnx_initializer_806
        model_layers_22_attn_k_proj_repeat_kv_gather_3 = self.model_layers_22_attn_k_proj_repeat_kv_Gather_3(model_layers_22_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_806);  model_layers_22_attn_k_proj_repeat_kv_shape_1 = initializers_onnx_initializer_806 = None
        model_layers_22_attn_v_proj_repeat_kv_unsqueeze_1 = self.model_layers_22_attn_v_proj_repeat_kv_Unsqueeze_1(model_layers_22_attn_v_proj_repeat_kv_gather_1);  model_layers_22_attn_v_proj_repeat_kv_gather_1 = None
        model_layers_22_attn_v_proj_repeat_kv_unsqueeze_3 = self.model_layers_22_attn_v_proj_repeat_kv_Unsqueeze_3(model_layers_22_attn_v_proj_repeat_kv_gather_3);  model_layers_22_attn_v_proj_repeat_kv_gather_3 = None
        model_layers_22_attn_k_proj_repeat_kv_unsqueeze_1 = self.model_layers_22_attn_k_proj_repeat_kv_Unsqueeze_1(model_layers_22_attn_k_proj_repeat_kv_gather_1);  model_layers_22_attn_k_proj_repeat_kv_gather_1 = None
        model_layers_22_attn_k_proj_repeat_kv_unsqueeze_3 = self.model_layers_22_attn_k_proj_repeat_kv_Unsqueeze_3(model_layers_22_attn_k_proj_repeat_kv_gather_3);  model_layers_22_attn_k_proj_repeat_kv_gather_3 = None
        initializers_onnx_initializer_807 = self.initializers.onnx_initializer_807
        initializers_onnx_initializer_808 = self.initializers.onnx_initializer_808
        initializers_onnx_initializer_809 = self.initializers.onnx_initializer_809
        model_layers_22_attn_v_proj_repeat_kv_concat_2 = self.model_layers_22_attn_v_proj_repeat_kv_Concat_2(model_layers_22_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_807, initializers_onnx_initializer_808, model_layers_22_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_809);  initializers_onnx_initializer_807 = initializers_onnx_initializer_808 = initializers_onnx_initializer_809 = None
        initializers_onnx_initializer_810 = self.initializers.onnx_initializer_810
        initializers_onnx_initializer_811 = self.initializers.onnx_initializer_811
        model_layers_22_attn_v_proj_repeat_kv_concat_3 = self.model_layers_22_attn_v_proj_repeat_kv_Concat_3(model_layers_22_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_810, model_layers_22_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_811);  model_layers_22_attn_v_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_810 = model_layers_22_attn_v_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_811 = None
        initializers_onnx_initializer_812 = self.initializers.onnx_initializer_812
        initializers_onnx_initializer_813 = self.initializers.onnx_initializer_813
        initializers_onnx_initializer_814 = self.initializers.onnx_initializer_814
        model_layers_22_attn_k_proj_repeat_kv_concat_2 = self.model_layers_22_attn_k_proj_repeat_kv_Concat_2(model_layers_22_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_812, initializers_onnx_initializer_813, model_layers_22_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_814);  initializers_onnx_initializer_812 = initializers_onnx_initializer_813 = initializers_onnx_initializer_814 = None
        initializers_onnx_initializer_815 = self.initializers.onnx_initializer_815
        initializers_onnx_initializer_816 = self.initializers.onnx_initializer_816
        model_layers_22_attn_k_proj_repeat_kv_concat_3 = self.model_layers_22_attn_k_proj_repeat_kv_Concat_3(model_layers_22_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_815, model_layers_22_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_816);  model_layers_22_attn_k_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_815 = model_layers_22_attn_k_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_816 = None
        initializers_onnx_initializer_817 = self.initializers.onnx_initializer_817
        model_layers_22_attn_v_proj_repeat_kv_equal = self.model_layers_22_attn_v_proj_repeat_kv_Equal(model_layers_22_attn_v_proj_repeat_kv_concat_2, initializers_onnx_initializer_817);  initializers_onnx_initializer_817 = None
        initializers_onnx_initializer_818 = self.initializers.onnx_initializer_818
        model_layers_22_attn_k_proj_repeat_kv_equal = self.model_layers_22_attn_k_proj_repeat_kv_Equal(model_layers_22_attn_k_proj_repeat_kv_concat_2, initializers_onnx_initializer_818);  initializers_onnx_initializer_818 = None
        initializers_onnx_initializer_819 = self.initializers.onnx_initializer_819
        model_layers_22_attn_v_proj_repeat_kv_where = self.model_layers_22_attn_v_proj_repeat_kv_Where(model_layers_22_attn_v_proj_repeat_kv_equal, initializers_onnx_initializer_819, model_layers_22_attn_v_proj_repeat_kv_concat_2);  model_layers_22_attn_v_proj_repeat_kv_equal = initializers_onnx_initializer_819 = model_layers_22_attn_v_proj_repeat_kv_concat_2 = None
        initializers_onnx_initializer_820 = self.initializers.onnx_initializer_820
        model_layers_22_attn_k_proj_repeat_kv_where = self.model_layers_22_attn_k_proj_repeat_kv_Where(model_layers_22_attn_k_proj_repeat_kv_equal, initializers_onnx_initializer_820, model_layers_22_attn_k_proj_repeat_kv_concat_2);  model_layers_22_attn_k_proj_repeat_kv_equal = initializers_onnx_initializer_820 = model_layers_22_attn_k_proj_repeat_kv_concat_2 = None
        model_layers_22_attn_v_proj_repeat_kv_expand = self.model_layers_22_attn_v_proj_repeat_kv_Expand(model_layers_22_attn_v_proj_repeat_kv_unsqueeze_5, model_layers_22_attn_v_proj_repeat_kv_where);  model_layers_22_attn_v_proj_repeat_kv_unsqueeze_5 = model_layers_22_attn_v_proj_repeat_kv_where = None
        model_layers_22_attn_k_proj_repeat_kv_expand = self.model_layers_22_attn_k_proj_repeat_kv_Expand(model_layers_22_attn_k_proj_repeat_kv_unsqueeze_5, model_layers_22_attn_k_proj_repeat_kv_where);  model_layers_22_attn_k_proj_repeat_kv_unsqueeze_5 = model_layers_22_attn_k_proj_repeat_kv_where = None
        model_layers_22_attn_v_proj_repeat_kv_reshape_3 = self.model_layers_22_attn_v_proj_repeat_kv_Reshape_3(model_layers_22_attn_v_proj_repeat_kv_expand, model_layers_22_attn_v_proj_repeat_kv_concat_3);  model_layers_22_attn_v_proj_repeat_kv_expand = model_layers_22_attn_v_proj_repeat_kv_concat_3 = None
        model_layers_22_attn_k_proj_repeat_kv_reshape_3 = self.model_layers_22_attn_k_proj_repeat_kv_Reshape_3(model_layers_22_attn_k_proj_repeat_kv_expand, model_layers_22_attn_k_proj_repeat_kv_concat_3);  model_layers_22_attn_k_proj_repeat_kv_expand = model_layers_22_attn_k_proj_repeat_kv_concat_3 = None
        model_layers_22_attn_v_proj_repeat_kv_transpose_2 = self.model_layers_22_attn_v_proj_repeat_kv_Transpose_2(model_layers_22_attn_v_proj_repeat_kv_reshape_3);  model_layers_22_attn_v_proj_repeat_kv_reshape_3 = None
        model_layers_22_attn_k_proj_repeat_kv_transpose_2 = self.model_layers_22_attn_k_proj_repeat_kv_Transpose_2(model_layers_22_attn_k_proj_repeat_kv_reshape_3);  model_layers_22_attn_k_proj_repeat_kv_reshape_3 = None
        initializers_onnx_initializer_821 = self.initializers.onnx_initializer_821
        model_layers_22_attn_v_proj_repeat_kv_reshape_4 = self.model_layers_22_attn_v_proj_repeat_kv_Reshape_4(model_layers_22_attn_v_proj_repeat_kv_transpose_2, initializers_onnx_initializer_821);  model_layers_22_attn_v_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_821 = None
        initializers_onnx_initializer_822 = self.initializers.onnx_initializer_822
        model_layers_22_attn_k_proj_repeat_kv_reshape_4 = self.model_layers_22_attn_k_proj_repeat_kv_Reshape_4(model_layers_22_attn_k_proj_repeat_kv_transpose_2, initializers_onnx_initializer_822);  model_layers_22_attn_k_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_822 = None
        com_microsoft__model_layers_22_attn_multi_head_attention = self.com_microsoft__model_layers_22_attn_MultiHeadAttention(com_microsoft__model_layers_22_attn_q_rotary_rotary_embedding, model_layers_22_attn_k_proj_repeat_kv_reshape_4, model_layers_22_attn_v_proj_repeat_kv_reshape_4, attention_mask = model_attn_mask_reformat_tile);  com_microsoft__model_layers_22_attn_q_rotary_rotary_embedding = model_layers_22_attn_k_proj_repeat_kv_reshape_4 = model_layers_22_attn_v_proj_repeat_kv_reshape_4 = None
        getitem_175 = com_microsoft__model_layers_22_attn_multi_head_attention[0];  com_microsoft__model_layers_22_attn_multi_head_attention = None
        initializers_onnx_initializer_823 = self.initializers.onnx_initializer_823
        model_layers_22_attn_o_proj_mat_mul = self.model_layers_22_attn_o_proj_MatMul(getitem_175, initializers_onnx_initializer_823);  getitem_175 = initializers_onnx_initializer_823 = None
        getitem_176 = com_microsoft__model_layers_22_input_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_22_input_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_824 = self.initializers.onnx_initializer_824
        com_microsoft__model_layers_22_post_attention_layernorm_skip_layer_norm = self.com_microsoft__model_layers_22_post_attention_layernorm_SkipLayerNorm(getitem_176, model_layers_22_attn_o_proj_mat_mul, initializers_onnx_initializer_824);  getitem_176 = model_layers_22_attn_o_proj_mat_mul = initializers_onnx_initializer_824 = None
        getitem_177 = com_microsoft__model_layers_22_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_825 = self.initializers.onnx_initializer_825
        model_layers_22_mlp_gate_proj_mat_mul = self.model_layers_22_mlp_gate_proj_MatMul(getitem_177, initializers_onnx_initializer_825);  getitem_177 = initializers_onnx_initializer_825 = None
        getitem_178 = com_microsoft__model_layers_22_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_826 = self.initializers.onnx_initializer_826
        model_layers_22_mlp_up_proj_mat_mul = self.model_layers_22_mlp_up_proj_MatMul(getitem_178, initializers_onnx_initializer_826);  getitem_178 = initializers_onnx_initializer_826 = None
        model_layers_22_mlp_act_fn_sigmoid = self.model_layers_22_mlp_act_fn_Sigmoid(model_layers_22_mlp_gate_proj_mat_mul)
        model_layers_22_mlp_act_fn_mul = self.model_layers_22_mlp_act_fn_Mul(model_layers_22_mlp_gate_proj_mat_mul, model_layers_22_mlp_act_fn_sigmoid);  model_layers_22_mlp_gate_proj_mat_mul = model_layers_22_mlp_act_fn_sigmoid = None
        model_layers_22_mlp_mul = self.model_layers_22_mlp_Mul(model_layers_22_mlp_act_fn_mul, model_layers_22_mlp_up_proj_mat_mul);  model_layers_22_mlp_act_fn_mul = model_layers_22_mlp_up_proj_mat_mul = None
        initializers_onnx_initializer_827 = self.initializers.onnx_initializer_827
        model_layers_22_mlp_down_proj_mat_mul = self.model_layers_22_mlp_down_proj_MatMul(model_layers_22_mlp_mul, initializers_onnx_initializer_827);  model_layers_22_mlp_mul = initializers_onnx_initializer_827 = None
        getitem_179 = com_microsoft__model_layers_22_post_attention_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_22_post_attention_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_828 = self.initializers.onnx_initializer_828
        com_microsoft__model_layers_23_input_layernorm_skip_layer_norm = self.com_microsoft__model_layers_23_input_layernorm_SkipLayerNorm(getitem_179, model_layers_22_mlp_down_proj_mat_mul, initializers_onnx_initializer_828);  getitem_179 = model_layers_22_mlp_down_proj_mat_mul = initializers_onnx_initializer_828 = None
        getitem_180 = com_microsoft__model_layers_23_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_829 = self.initializers.onnx_initializer_829
        model_layers_23_attn_q_proj_mat_mul = self.model_layers_23_attn_q_proj_MatMul(getitem_180, initializers_onnx_initializer_829);  getitem_180 = initializers_onnx_initializer_829 = None
        getitem_181 = com_microsoft__model_layers_23_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_830 = self.initializers.onnx_initializer_830
        model_layers_23_attn_k_proj_mat_mul = self.model_layers_23_attn_k_proj_MatMul(getitem_181, initializers_onnx_initializer_830);  getitem_181 = initializers_onnx_initializer_830 = None
        getitem_182 = com_microsoft__model_layers_23_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_831 = self.initializers.onnx_initializer_831
        model_layers_23_attn_v_proj_mat_mul = self.model_layers_23_attn_v_proj_MatMul(getitem_182, initializers_onnx_initializer_831);  getitem_182 = initializers_onnx_initializer_831 = None
        initializers_onnx_initializer_832 = self.initializers.onnx_initializer_832
        initializers_onnx_initializer_833 = self.initializers.onnx_initializer_833
        com_microsoft__model_layers_23_attn_q_rotary_rotary_embedding = self.com_microsoft__model_layers_23_attn_q_rotary_RotaryEmbedding(model_layers_23_attn_q_proj_mat_mul, input_3, initializers_onnx_initializer_832, initializers_onnx_initializer_833);  model_layers_23_attn_q_proj_mat_mul = initializers_onnx_initializer_832 = initializers_onnx_initializer_833 = None
        initializers_onnx_initializer_834 = self.initializers.onnx_initializer_834
        initializers_onnx_initializer_835 = self.initializers.onnx_initializer_835
        com_microsoft__model_layers_23_attn_k_rotary_rotary_embedding = self.com_microsoft__model_layers_23_attn_k_rotary_RotaryEmbedding(model_layers_23_attn_k_proj_mat_mul, input_3, initializers_onnx_initializer_834, initializers_onnx_initializer_835);  model_layers_23_attn_k_proj_mat_mul = initializers_onnx_initializer_834 = initializers_onnx_initializer_835 = None
        initializers_onnx_initializer_836 = self.initializers.onnx_initializer_836
        model_layers_23_attn_v_proj_repeat_kv_reshape_1 = self.model_layers_23_attn_v_proj_repeat_kv_Reshape_1(model_layers_23_attn_v_proj_mat_mul, initializers_onnx_initializer_836);  model_layers_23_attn_v_proj_mat_mul = initializers_onnx_initializer_836 = None
        initializers_onnx_initializer_837 = self.initializers.onnx_initializer_837
        model_layers_23_attn_k_proj_repeat_kv_reshape_1 = self.model_layers_23_attn_k_proj_repeat_kv_Reshape_1(com_microsoft__model_layers_23_attn_k_rotary_rotary_embedding, initializers_onnx_initializer_837);  com_microsoft__model_layers_23_attn_k_rotary_rotary_embedding = initializers_onnx_initializer_837 = None
        model_layers_23_attn_v_proj_repeat_kv_transpose_1 = self.model_layers_23_attn_v_proj_repeat_kv_Transpose_1(model_layers_23_attn_v_proj_repeat_kv_reshape_1);  model_layers_23_attn_v_proj_repeat_kv_reshape_1 = None
        model_layers_23_attn_k_proj_repeat_kv_transpose_1 = self.model_layers_23_attn_k_proj_repeat_kv_Transpose_1(model_layers_23_attn_k_proj_repeat_kv_reshape_1);  model_layers_23_attn_k_proj_repeat_kv_reshape_1 = None
        model_layers_23_attn_v_proj_repeat_kv_concat_1 = self.model_layers_23_attn_v_proj_repeat_kv_Concat_1(input_51, model_layers_23_attn_v_proj_repeat_kv_transpose_1);  input_51 = model_layers_23_attn_v_proj_repeat_kv_transpose_1 = None
        model_layers_23_attn_k_proj_repeat_kv_concat_1 = self.model_layers_23_attn_k_proj_repeat_kv_Concat_1(input_50, model_layers_23_attn_k_proj_repeat_kv_transpose_1);  input_50 = model_layers_23_attn_k_proj_repeat_kv_transpose_1 = None
        model_layers_23_attn_v_proj_repeat_kv_shape_1 = self.model_layers_23_attn_v_proj_repeat_kv_Shape_1(model_layers_23_attn_v_proj_repeat_kv_concat_1)
        model_layers_23_attn_v_proj_repeat_kv_unsqueeze_5 = self.model_layers_23_attn_v_proj_repeat_kv_Unsqueeze_5(model_layers_23_attn_v_proj_repeat_kv_concat_1)
        model_layers_23_attn_k_proj_repeat_kv_shape_1 = self.model_layers_23_attn_k_proj_repeat_kv_Shape_1(model_layers_23_attn_k_proj_repeat_kv_concat_1)
        model_layers_23_attn_k_proj_repeat_kv_unsqueeze_5 = self.model_layers_23_attn_k_proj_repeat_kv_Unsqueeze_5(model_layers_23_attn_k_proj_repeat_kv_concat_1)
        initializers_onnx_initializer_838 = self.initializers.onnx_initializer_838
        model_layers_23_attn_v_proj_repeat_kv_gather_1 = self.model_layers_23_attn_v_proj_repeat_kv_Gather_1(model_layers_23_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_838);  initializers_onnx_initializer_838 = None
        initializers_onnx_initializer_839 = self.initializers.onnx_initializer_839
        model_layers_23_attn_v_proj_repeat_kv_gather_3 = self.model_layers_23_attn_v_proj_repeat_kv_Gather_3(model_layers_23_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_839);  model_layers_23_attn_v_proj_repeat_kv_shape_1 = initializers_onnx_initializer_839 = None
        initializers_onnx_initializer_840 = self.initializers.onnx_initializer_840
        model_layers_23_attn_k_proj_repeat_kv_gather_1 = self.model_layers_23_attn_k_proj_repeat_kv_Gather_1(model_layers_23_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_840);  initializers_onnx_initializer_840 = None
        initializers_onnx_initializer_841 = self.initializers.onnx_initializer_841
        model_layers_23_attn_k_proj_repeat_kv_gather_3 = self.model_layers_23_attn_k_proj_repeat_kv_Gather_3(model_layers_23_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_841);  model_layers_23_attn_k_proj_repeat_kv_shape_1 = initializers_onnx_initializer_841 = None
        model_layers_23_attn_v_proj_repeat_kv_unsqueeze_1 = self.model_layers_23_attn_v_proj_repeat_kv_Unsqueeze_1(model_layers_23_attn_v_proj_repeat_kv_gather_1);  model_layers_23_attn_v_proj_repeat_kv_gather_1 = None
        model_layers_23_attn_v_proj_repeat_kv_unsqueeze_3 = self.model_layers_23_attn_v_proj_repeat_kv_Unsqueeze_3(model_layers_23_attn_v_proj_repeat_kv_gather_3);  model_layers_23_attn_v_proj_repeat_kv_gather_3 = None
        model_layers_23_attn_k_proj_repeat_kv_unsqueeze_1 = self.model_layers_23_attn_k_proj_repeat_kv_Unsqueeze_1(model_layers_23_attn_k_proj_repeat_kv_gather_1);  model_layers_23_attn_k_proj_repeat_kv_gather_1 = None
        model_layers_23_attn_k_proj_repeat_kv_unsqueeze_3 = self.model_layers_23_attn_k_proj_repeat_kv_Unsqueeze_3(model_layers_23_attn_k_proj_repeat_kv_gather_3);  model_layers_23_attn_k_proj_repeat_kv_gather_3 = None
        initializers_onnx_initializer_842 = self.initializers.onnx_initializer_842
        initializers_onnx_initializer_843 = self.initializers.onnx_initializer_843
        initializers_onnx_initializer_844 = self.initializers.onnx_initializer_844
        model_layers_23_attn_v_proj_repeat_kv_concat_2 = self.model_layers_23_attn_v_proj_repeat_kv_Concat_2(model_layers_23_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_842, initializers_onnx_initializer_843, model_layers_23_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_844);  initializers_onnx_initializer_842 = initializers_onnx_initializer_843 = initializers_onnx_initializer_844 = None
        initializers_onnx_initializer_845 = self.initializers.onnx_initializer_845
        initializers_onnx_initializer_846 = self.initializers.onnx_initializer_846
        model_layers_23_attn_v_proj_repeat_kv_concat_3 = self.model_layers_23_attn_v_proj_repeat_kv_Concat_3(model_layers_23_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_845, model_layers_23_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_846);  model_layers_23_attn_v_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_845 = model_layers_23_attn_v_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_846 = None
        initializers_onnx_initializer_847 = self.initializers.onnx_initializer_847
        initializers_onnx_initializer_848 = self.initializers.onnx_initializer_848
        initializers_onnx_initializer_849 = self.initializers.onnx_initializer_849
        model_layers_23_attn_k_proj_repeat_kv_concat_2 = self.model_layers_23_attn_k_proj_repeat_kv_Concat_2(model_layers_23_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_847, initializers_onnx_initializer_848, model_layers_23_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_849);  initializers_onnx_initializer_847 = initializers_onnx_initializer_848 = initializers_onnx_initializer_849 = None
        initializers_onnx_initializer_850 = self.initializers.onnx_initializer_850
        initializers_onnx_initializer_851 = self.initializers.onnx_initializer_851
        model_layers_23_attn_k_proj_repeat_kv_concat_3 = self.model_layers_23_attn_k_proj_repeat_kv_Concat_3(model_layers_23_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_850, model_layers_23_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_851);  model_layers_23_attn_k_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_850 = model_layers_23_attn_k_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_851 = None
        initializers_onnx_initializer_852 = self.initializers.onnx_initializer_852
        model_layers_23_attn_v_proj_repeat_kv_equal = self.model_layers_23_attn_v_proj_repeat_kv_Equal(model_layers_23_attn_v_proj_repeat_kv_concat_2, initializers_onnx_initializer_852);  initializers_onnx_initializer_852 = None
        initializers_onnx_initializer_853 = self.initializers.onnx_initializer_853
        model_layers_23_attn_k_proj_repeat_kv_equal = self.model_layers_23_attn_k_proj_repeat_kv_Equal(model_layers_23_attn_k_proj_repeat_kv_concat_2, initializers_onnx_initializer_853);  initializers_onnx_initializer_853 = None
        initializers_onnx_initializer_854 = self.initializers.onnx_initializer_854
        model_layers_23_attn_v_proj_repeat_kv_where = self.model_layers_23_attn_v_proj_repeat_kv_Where(model_layers_23_attn_v_proj_repeat_kv_equal, initializers_onnx_initializer_854, model_layers_23_attn_v_proj_repeat_kv_concat_2);  model_layers_23_attn_v_proj_repeat_kv_equal = initializers_onnx_initializer_854 = model_layers_23_attn_v_proj_repeat_kv_concat_2 = None
        initializers_onnx_initializer_855 = self.initializers.onnx_initializer_855
        model_layers_23_attn_k_proj_repeat_kv_where = self.model_layers_23_attn_k_proj_repeat_kv_Where(model_layers_23_attn_k_proj_repeat_kv_equal, initializers_onnx_initializer_855, model_layers_23_attn_k_proj_repeat_kv_concat_2);  model_layers_23_attn_k_proj_repeat_kv_equal = initializers_onnx_initializer_855 = model_layers_23_attn_k_proj_repeat_kv_concat_2 = None
        model_layers_23_attn_v_proj_repeat_kv_expand = self.model_layers_23_attn_v_proj_repeat_kv_Expand(model_layers_23_attn_v_proj_repeat_kv_unsqueeze_5, model_layers_23_attn_v_proj_repeat_kv_where);  model_layers_23_attn_v_proj_repeat_kv_unsqueeze_5 = model_layers_23_attn_v_proj_repeat_kv_where = None
        model_layers_23_attn_k_proj_repeat_kv_expand = self.model_layers_23_attn_k_proj_repeat_kv_Expand(model_layers_23_attn_k_proj_repeat_kv_unsqueeze_5, model_layers_23_attn_k_proj_repeat_kv_where);  model_layers_23_attn_k_proj_repeat_kv_unsqueeze_5 = model_layers_23_attn_k_proj_repeat_kv_where = None
        model_layers_23_attn_v_proj_repeat_kv_reshape_3 = self.model_layers_23_attn_v_proj_repeat_kv_Reshape_3(model_layers_23_attn_v_proj_repeat_kv_expand, model_layers_23_attn_v_proj_repeat_kv_concat_3);  model_layers_23_attn_v_proj_repeat_kv_expand = model_layers_23_attn_v_proj_repeat_kv_concat_3 = None
        model_layers_23_attn_k_proj_repeat_kv_reshape_3 = self.model_layers_23_attn_k_proj_repeat_kv_Reshape_3(model_layers_23_attn_k_proj_repeat_kv_expand, model_layers_23_attn_k_proj_repeat_kv_concat_3);  model_layers_23_attn_k_proj_repeat_kv_expand = model_layers_23_attn_k_proj_repeat_kv_concat_3 = None
        model_layers_23_attn_v_proj_repeat_kv_transpose_2 = self.model_layers_23_attn_v_proj_repeat_kv_Transpose_2(model_layers_23_attn_v_proj_repeat_kv_reshape_3);  model_layers_23_attn_v_proj_repeat_kv_reshape_3 = None
        model_layers_23_attn_k_proj_repeat_kv_transpose_2 = self.model_layers_23_attn_k_proj_repeat_kv_Transpose_2(model_layers_23_attn_k_proj_repeat_kv_reshape_3);  model_layers_23_attn_k_proj_repeat_kv_reshape_3 = None
        initializers_onnx_initializer_856 = self.initializers.onnx_initializer_856
        model_layers_23_attn_v_proj_repeat_kv_reshape_4 = self.model_layers_23_attn_v_proj_repeat_kv_Reshape_4(model_layers_23_attn_v_proj_repeat_kv_transpose_2, initializers_onnx_initializer_856);  model_layers_23_attn_v_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_856 = None
        initializers_onnx_initializer_857 = self.initializers.onnx_initializer_857
        model_layers_23_attn_k_proj_repeat_kv_reshape_4 = self.model_layers_23_attn_k_proj_repeat_kv_Reshape_4(model_layers_23_attn_k_proj_repeat_kv_transpose_2, initializers_onnx_initializer_857);  model_layers_23_attn_k_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_857 = None
        com_microsoft__model_layers_23_attn_multi_head_attention = self.com_microsoft__model_layers_23_attn_MultiHeadAttention(com_microsoft__model_layers_23_attn_q_rotary_rotary_embedding, model_layers_23_attn_k_proj_repeat_kv_reshape_4, model_layers_23_attn_v_proj_repeat_kv_reshape_4, attention_mask = model_attn_mask_reformat_tile);  com_microsoft__model_layers_23_attn_q_rotary_rotary_embedding = model_layers_23_attn_k_proj_repeat_kv_reshape_4 = model_layers_23_attn_v_proj_repeat_kv_reshape_4 = None
        getitem_183 = com_microsoft__model_layers_23_attn_multi_head_attention[0];  com_microsoft__model_layers_23_attn_multi_head_attention = None
        initializers_onnx_initializer_858 = self.initializers.onnx_initializer_858
        model_layers_23_attn_o_proj_mat_mul = self.model_layers_23_attn_o_proj_MatMul(getitem_183, initializers_onnx_initializer_858);  getitem_183 = initializers_onnx_initializer_858 = None
        getitem_184 = com_microsoft__model_layers_23_input_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_23_input_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_859 = self.initializers.onnx_initializer_859
        com_microsoft__model_layers_23_post_attention_layernorm_skip_layer_norm = self.com_microsoft__model_layers_23_post_attention_layernorm_SkipLayerNorm(getitem_184, model_layers_23_attn_o_proj_mat_mul, initializers_onnx_initializer_859);  getitem_184 = model_layers_23_attn_o_proj_mat_mul = initializers_onnx_initializer_859 = None
        getitem_185 = com_microsoft__model_layers_23_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_860 = self.initializers.onnx_initializer_860
        model_layers_23_mlp_gate_proj_mat_mul = self.model_layers_23_mlp_gate_proj_MatMul(getitem_185, initializers_onnx_initializer_860);  getitem_185 = initializers_onnx_initializer_860 = None
        getitem_186 = com_microsoft__model_layers_23_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_861 = self.initializers.onnx_initializer_861
        model_layers_23_mlp_up_proj_mat_mul = self.model_layers_23_mlp_up_proj_MatMul(getitem_186, initializers_onnx_initializer_861);  getitem_186 = initializers_onnx_initializer_861 = None
        model_layers_23_mlp_act_fn_sigmoid = self.model_layers_23_mlp_act_fn_Sigmoid(model_layers_23_mlp_gate_proj_mat_mul)
        model_layers_23_mlp_act_fn_mul = self.model_layers_23_mlp_act_fn_Mul(model_layers_23_mlp_gate_proj_mat_mul, model_layers_23_mlp_act_fn_sigmoid);  model_layers_23_mlp_gate_proj_mat_mul = model_layers_23_mlp_act_fn_sigmoid = None
        model_layers_23_mlp_mul = self.model_layers_23_mlp_Mul(model_layers_23_mlp_act_fn_mul, model_layers_23_mlp_up_proj_mat_mul);  model_layers_23_mlp_act_fn_mul = model_layers_23_mlp_up_proj_mat_mul = None
        initializers_onnx_initializer_862 = self.initializers.onnx_initializer_862
        model_layers_23_mlp_down_proj_mat_mul = self.model_layers_23_mlp_down_proj_MatMul(model_layers_23_mlp_mul, initializers_onnx_initializer_862);  model_layers_23_mlp_mul = initializers_onnx_initializer_862 = None
        getitem_187 = com_microsoft__model_layers_23_post_attention_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_23_post_attention_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_863 = self.initializers.onnx_initializer_863
        com_microsoft__model_layers_24_input_layernorm_skip_layer_norm = self.com_microsoft__model_layers_24_input_layernorm_SkipLayerNorm(getitem_187, model_layers_23_mlp_down_proj_mat_mul, initializers_onnx_initializer_863);  getitem_187 = model_layers_23_mlp_down_proj_mat_mul = initializers_onnx_initializer_863 = None
        getitem_188 = com_microsoft__model_layers_24_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_864 = self.initializers.onnx_initializer_864
        model_layers_24_attn_q_proj_mat_mul = self.model_layers_24_attn_q_proj_MatMul(getitem_188, initializers_onnx_initializer_864);  getitem_188 = initializers_onnx_initializer_864 = None
        getitem_189 = com_microsoft__model_layers_24_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_865 = self.initializers.onnx_initializer_865
        model_layers_24_attn_k_proj_mat_mul = self.model_layers_24_attn_k_proj_MatMul(getitem_189, initializers_onnx_initializer_865);  getitem_189 = initializers_onnx_initializer_865 = None
        getitem_190 = com_microsoft__model_layers_24_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_866 = self.initializers.onnx_initializer_866
        model_layers_24_attn_v_proj_mat_mul = self.model_layers_24_attn_v_proj_MatMul(getitem_190, initializers_onnx_initializer_866);  getitem_190 = initializers_onnx_initializer_866 = None
        initializers_onnx_initializer_867 = self.initializers.onnx_initializer_867
        initializers_onnx_initializer_868 = self.initializers.onnx_initializer_868
        com_microsoft__model_layers_24_attn_q_rotary_rotary_embedding = self.com_microsoft__model_layers_24_attn_q_rotary_RotaryEmbedding(model_layers_24_attn_q_proj_mat_mul, input_3, initializers_onnx_initializer_867, initializers_onnx_initializer_868);  model_layers_24_attn_q_proj_mat_mul = initializers_onnx_initializer_867 = initializers_onnx_initializer_868 = None
        initializers_onnx_initializer_869 = self.initializers.onnx_initializer_869
        initializers_onnx_initializer_870 = self.initializers.onnx_initializer_870
        com_microsoft__model_layers_24_attn_k_rotary_rotary_embedding = self.com_microsoft__model_layers_24_attn_k_rotary_RotaryEmbedding(model_layers_24_attn_k_proj_mat_mul, input_3, initializers_onnx_initializer_869, initializers_onnx_initializer_870);  model_layers_24_attn_k_proj_mat_mul = initializers_onnx_initializer_869 = initializers_onnx_initializer_870 = None
        initializers_onnx_initializer_871 = self.initializers.onnx_initializer_871
        model_layers_24_attn_v_proj_repeat_kv_reshape_1 = self.model_layers_24_attn_v_proj_repeat_kv_Reshape_1(model_layers_24_attn_v_proj_mat_mul, initializers_onnx_initializer_871);  model_layers_24_attn_v_proj_mat_mul = initializers_onnx_initializer_871 = None
        initializers_onnx_initializer_872 = self.initializers.onnx_initializer_872
        model_layers_24_attn_k_proj_repeat_kv_reshape_1 = self.model_layers_24_attn_k_proj_repeat_kv_Reshape_1(com_microsoft__model_layers_24_attn_k_rotary_rotary_embedding, initializers_onnx_initializer_872);  com_microsoft__model_layers_24_attn_k_rotary_rotary_embedding = initializers_onnx_initializer_872 = None
        model_layers_24_attn_v_proj_repeat_kv_transpose_1 = self.model_layers_24_attn_v_proj_repeat_kv_Transpose_1(model_layers_24_attn_v_proj_repeat_kv_reshape_1);  model_layers_24_attn_v_proj_repeat_kv_reshape_1 = None
        model_layers_24_attn_k_proj_repeat_kv_transpose_1 = self.model_layers_24_attn_k_proj_repeat_kv_Transpose_1(model_layers_24_attn_k_proj_repeat_kv_reshape_1);  model_layers_24_attn_k_proj_repeat_kv_reshape_1 = None
        model_layers_24_attn_v_proj_repeat_kv_concat_1 = self.model_layers_24_attn_v_proj_repeat_kv_Concat_1(input_53, model_layers_24_attn_v_proj_repeat_kv_transpose_1);  input_53 = model_layers_24_attn_v_proj_repeat_kv_transpose_1 = None
        model_layers_24_attn_k_proj_repeat_kv_concat_1 = self.model_layers_24_attn_k_proj_repeat_kv_Concat_1(input_52, model_layers_24_attn_k_proj_repeat_kv_transpose_1);  input_52 = model_layers_24_attn_k_proj_repeat_kv_transpose_1 = None
        model_layers_24_attn_v_proj_repeat_kv_shape_1 = self.model_layers_24_attn_v_proj_repeat_kv_Shape_1(model_layers_24_attn_v_proj_repeat_kv_concat_1)
        model_layers_24_attn_v_proj_repeat_kv_unsqueeze_5 = self.model_layers_24_attn_v_proj_repeat_kv_Unsqueeze_5(model_layers_24_attn_v_proj_repeat_kv_concat_1)
        model_layers_24_attn_k_proj_repeat_kv_shape_1 = self.model_layers_24_attn_k_proj_repeat_kv_Shape_1(model_layers_24_attn_k_proj_repeat_kv_concat_1)
        model_layers_24_attn_k_proj_repeat_kv_unsqueeze_5 = self.model_layers_24_attn_k_proj_repeat_kv_Unsqueeze_5(model_layers_24_attn_k_proj_repeat_kv_concat_1)
        initializers_onnx_initializer_873 = self.initializers.onnx_initializer_873
        model_layers_24_attn_v_proj_repeat_kv_gather_1 = self.model_layers_24_attn_v_proj_repeat_kv_Gather_1(model_layers_24_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_873);  initializers_onnx_initializer_873 = None
        initializers_onnx_initializer_874 = self.initializers.onnx_initializer_874
        model_layers_24_attn_v_proj_repeat_kv_gather_3 = self.model_layers_24_attn_v_proj_repeat_kv_Gather_3(model_layers_24_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_874);  model_layers_24_attn_v_proj_repeat_kv_shape_1 = initializers_onnx_initializer_874 = None
        initializers_onnx_initializer_875 = self.initializers.onnx_initializer_875
        model_layers_24_attn_k_proj_repeat_kv_gather_1 = self.model_layers_24_attn_k_proj_repeat_kv_Gather_1(model_layers_24_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_875);  initializers_onnx_initializer_875 = None
        initializers_onnx_initializer_876 = self.initializers.onnx_initializer_876
        model_layers_24_attn_k_proj_repeat_kv_gather_3 = self.model_layers_24_attn_k_proj_repeat_kv_Gather_3(model_layers_24_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_876);  model_layers_24_attn_k_proj_repeat_kv_shape_1 = initializers_onnx_initializer_876 = None
        model_layers_24_attn_v_proj_repeat_kv_unsqueeze_1 = self.model_layers_24_attn_v_proj_repeat_kv_Unsqueeze_1(model_layers_24_attn_v_proj_repeat_kv_gather_1);  model_layers_24_attn_v_proj_repeat_kv_gather_1 = None
        model_layers_24_attn_v_proj_repeat_kv_unsqueeze_3 = self.model_layers_24_attn_v_proj_repeat_kv_Unsqueeze_3(model_layers_24_attn_v_proj_repeat_kv_gather_3);  model_layers_24_attn_v_proj_repeat_kv_gather_3 = None
        model_layers_24_attn_k_proj_repeat_kv_unsqueeze_1 = self.model_layers_24_attn_k_proj_repeat_kv_Unsqueeze_1(model_layers_24_attn_k_proj_repeat_kv_gather_1);  model_layers_24_attn_k_proj_repeat_kv_gather_1 = None
        model_layers_24_attn_k_proj_repeat_kv_unsqueeze_3 = self.model_layers_24_attn_k_proj_repeat_kv_Unsqueeze_3(model_layers_24_attn_k_proj_repeat_kv_gather_3);  model_layers_24_attn_k_proj_repeat_kv_gather_3 = None
        initializers_onnx_initializer_877 = self.initializers.onnx_initializer_877
        initializers_onnx_initializer_878 = self.initializers.onnx_initializer_878
        initializers_onnx_initializer_879 = self.initializers.onnx_initializer_879
        model_layers_24_attn_v_proj_repeat_kv_concat_2 = self.model_layers_24_attn_v_proj_repeat_kv_Concat_2(model_layers_24_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_877, initializers_onnx_initializer_878, model_layers_24_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_879);  initializers_onnx_initializer_877 = initializers_onnx_initializer_878 = initializers_onnx_initializer_879 = None
        initializers_onnx_initializer_880 = self.initializers.onnx_initializer_880
        initializers_onnx_initializer_881 = self.initializers.onnx_initializer_881
        model_layers_24_attn_v_proj_repeat_kv_concat_3 = self.model_layers_24_attn_v_proj_repeat_kv_Concat_3(model_layers_24_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_880, model_layers_24_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_881);  model_layers_24_attn_v_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_880 = model_layers_24_attn_v_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_881 = None
        initializers_onnx_initializer_882 = self.initializers.onnx_initializer_882
        initializers_onnx_initializer_883 = self.initializers.onnx_initializer_883
        initializers_onnx_initializer_884 = self.initializers.onnx_initializer_884
        model_layers_24_attn_k_proj_repeat_kv_concat_2 = self.model_layers_24_attn_k_proj_repeat_kv_Concat_2(model_layers_24_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_882, initializers_onnx_initializer_883, model_layers_24_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_884);  initializers_onnx_initializer_882 = initializers_onnx_initializer_883 = initializers_onnx_initializer_884 = None
        initializers_onnx_initializer_885 = self.initializers.onnx_initializer_885
        initializers_onnx_initializer_886 = self.initializers.onnx_initializer_886
        model_layers_24_attn_k_proj_repeat_kv_concat_3 = self.model_layers_24_attn_k_proj_repeat_kv_Concat_3(model_layers_24_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_885, model_layers_24_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_886);  model_layers_24_attn_k_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_885 = model_layers_24_attn_k_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_886 = None
        initializers_onnx_initializer_887 = self.initializers.onnx_initializer_887
        model_layers_24_attn_v_proj_repeat_kv_equal = self.model_layers_24_attn_v_proj_repeat_kv_Equal(model_layers_24_attn_v_proj_repeat_kv_concat_2, initializers_onnx_initializer_887);  initializers_onnx_initializer_887 = None
        initializers_onnx_initializer_888 = self.initializers.onnx_initializer_888
        model_layers_24_attn_k_proj_repeat_kv_equal = self.model_layers_24_attn_k_proj_repeat_kv_Equal(model_layers_24_attn_k_proj_repeat_kv_concat_2, initializers_onnx_initializer_888);  initializers_onnx_initializer_888 = None
        initializers_onnx_initializer_889 = self.initializers.onnx_initializer_889
        model_layers_24_attn_v_proj_repeat_kv_where = self.model_layers_24_attn_v_proj_repeat_kv_Where(model_layers_24_attn_v_proj_repeat_kv_equal, initializers_onnx_initializer_889, model_layers_24_attn_v_proj_repeat_kv_concat_2);  model_layers_24_attn_v_proj_repeat_kv_equal = initializers_onnx_initializer_889 = model_layers_24_attn_v_proj_repeat_kv_concat_2 = None
        initializers_onnx_initializer_890 = self.initializers.onnx_initializer_890
        model_layers_24_attn_k_proj_repeat_kv_where = self.model_layers_24_attn_k_proj_repeat_kv_Where(model_layers_24_attn_k_proj_repeat_kv_equal, initializers_onnx_initializer_890, model_layers_24_attn_k_proj_repeat_kv_concat_2);  model_layers_24_attn_k_proj_repeat_kv_equal = initializers_onnx_initializer_890 = model_layers_24_attn_k_proj_repeat_kv_concat_2 = None
        model_layers_24_attn_v_proj_repeat_kv_expand = self.model_layers_24_attn_v_proj_repeat_kv_Expand(model_layers_24_attn_v_proj_repeat_kv_unsqueeze_5, model_layers_24_attn_v_proj_repeat_kv_where);  model_layers_24_attn_v_proj_repeat_kv_unsqueeze_5 = model_layers_24_attn_v_proj_repeat_kv_where = None
        model_layers_24_attn_k_proj_repeat_kv_expand = self.model_layers_24_attn_k_proj_repeat_kv_Expand(model_layers_24_attn_k_proj_repeat_kv_unsqueeze_5, model_layers_24_attn_k_proj_repeat_kv_where);  model_layers_24_attn_k_proj_repeat_kv_unsqueeze_5 = model_layers_24_attn_k_proj_repeat_kv_where = None
        model_layers_24_attn_v_proj_repeat_kv_reshape_3 = self.model_layers_24_attn_v_proj_repeat_kv_Reshape_3(model_layers_24_attn_v_proj_repeat_kv_expand, model_layers_24_attn_v_proj_repeat_kv_concat_3);  model_layers_24_attn_v_proj_repeat_kv_expand = model_layers_24_attn_v_proj_repeat_kv_concat_3 = None
        model_layers_24_attn_k_proj_repeat_kv_reshape_3 = self.model_layers_24_attn_k_proj_repeat_kv_Reshape_3(model_layers_24_attn_k_proj_repeat_kv_expand, model_layers_24_attn_k_proj_repeat_kv_concat_3);  model_layers_24_attn_k_proj_repeat_kv_expand = model_layers_24_attn_k_proj_repeat_kv_concat_3 = None
        model_layers_24_attn_v_proj_repeat_kv_transpose_2 = self.model_layers_24_attn_v_proj_repeat_kv_Transpose_2(model_layers_24_attn_v_proj_repeat_kv_reshape_3);  model_layers_24_attn_v_proj_repeat_kv_reshape_3 = None
        model_layers_24_attn_k_proj_repeat_kv_transpose_2 = self.model_layers_24_attn_k_proj_repeat_kv_Transpose_2(model_layers_24_attn_k_proj_repeat_kv_reshape_3);  model_layers_24_attn_k_proj_repeat_kv_reshape_3 = None
        initializers_onnx_initializer_891 = self.initializers.onnx_initializer_891
        model_layers_24_attn_v_proj_repeat_kv_reshape_4 = self.model_layers_24_attn_v_proj_repeat_kv_Reshape_4(model_layers_24_attn_v_proj_repeat_kv_transpose_2, initializers_onnx_initializer_891);  model_layers_24_attn_v_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_891 = None
        initializers_onnx_initializer_892 = self.initializers.onnx_initializer_892
        model_layers_24_attn_k_proj_repeat_kv_reshape_4 = self.model_layers_24_attn_k_proj_repeat_kv_Reshape_4(model_layers_24_attn_k_proj_repeat_kv_transpose_2, initializers_onnx_initializer_892);  model_layers_24_attn_k_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_892 = None
        com_microsoft__model_layers_24_attn_multi_head_attention = self.com_microsoft__model_layers_24_attn_MultiHeadAttention(com_microsoft__model_layers_24_attn_q_rotary_rotary_embedding, model_layers_24_attn_k_proj_repeat_kv_reshape_4, model_layers_24_attn_v_proj_repeat_kv_reshape_4, attention_mask = model_attn_mask_reformat_tile);  com_microsoft__model_layers_24_attn_q_rotary_rotary_embedding = model_layers_24_attn_k_proj_repeat_kv_reshape_4 = model_layers_24_attn_v_proj_repeat_kv_reshape_4 = None
        getitem_191 = com_microsoft__model_layers_24_attn_multi_head_attention[0];  com_microsoft__model_layers_24_attn_multi_head_attention = None
        initializers_onnx_initializer_893 = self.initializers.onnx_initializer_893
        model_layers_24_attn_o_proj_mat_mul = self.model_layers_24_attn_o_proj_MatMul(getitem_191, initializers_onnx_initializer_893);  getitem_191 = initializers_onnx_initializer_893 = None
        getitem_192 = com_microsoft__model_layers_24_input_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_24_input_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_894 = self.initializers.onnx_initializer_894
        com_microsoft__model_layers_24_post_attention_layernorm_skip_layer_norm = self.com_microsoft__model_layers_24_post_attention_layernorm_SkipLayerNorm(getitem_192, model_layers_24_attn_o_proj_mat_mul, initializers_onnx_initializer_894);  getitem_192 = model_layers_24_attn_o_proj_mat_mul = initializers_onnx_initializer_894 = None
        getitem_193 = com_microsoft__model_layers_24_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_895 = self.initializers.onnx_initializer_895
        model_layers_24_mlp_gate_proj_mat_mul = self.model_layers_24_mlp_gate_proj_MatMul(getitem_193, initializers_onnx_initializer_895);  getitem_193 = initializers_onnx_initializer_895 = None
        getitem_194 = com_microsoft__model_layers_24_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_896 = self.initializers.onnx_initializer_896
        model_layers_24_mlp_up_proj_mat_mul = self.model_layers_24_mlp_up_proj_MatMul(getitem_194, initializers_onnx_initializer_896);  getitem_194 = initializers_onnx_initializer_896 = None
        model_layers_24_mlp_act_fn_sigmoid = self.model_layers_24_mlp_act_fn_Sigmoid(model_layers_24_mlp_gate_proj_mat_mul)
        model_layers_24_mlp_act_fn_mul = self.model_layers_24_mlp_act_fn_Mul(model_layers_24_mlp_gate_proj_mat_mul, model_layers_24_mlp_act_fn_sigmoid);  model_layers_24_mlp_gate_proj_mat_mul = model_layers_24_mlp_act_fn_sigmoid = None
        model_layers_24_mlp_mul = self.model_layers_24_mlp_Mul(model_layers_24_mlp_act_fn_mul, model_layers_24_mlp_up_proj_mat_mul);  model_layers_24_mlp_act_fn_mul = model_layers_24_mlp_up_proj_mat_mul = None
        initializers_onnx_initializer_897 = self.initializers.onnx_initializer_897
        model_layers_24_mlp_down_proj_mat_mul = self.model_layers_24_mlp_down_proj_MatMul(model_layers_24_mlp_mul, initializers_onnx_initializer_897);  model_layers_24_mlp_mul = initializers_onnx_initializer_897 = None
        getitem_195 = com_microsoft__model_layers_24_post_attention_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_24_post_attention_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_898 = self.initializers.onnx_initializer_898
        com_microsoft__model_layers_25_input_layernorm_skip_layer_norm = self.com_microsoft__model_layers_25_input_layernorm_SkipLayerNorm(getitem_195, model_layers_24_mlp_down_proj_mat_mul, initializers_onnx_initializer_898);  getitem_195 = model_layers_24_mlp_down_proj_mat_mul = initializers_onnx_initializer_898 = None
        getitem_196 = com_microsoft__model_layers_25_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_899 = self.initializers.onnx_initializer_899
        model_layers_25_attn_q_proj_mat_mul = self.model_layers_25_attn_q_proj_MatMul(getitem_196, initializers_onnx_initializer_899);  getitem_196 = initializers_onnx_initializer_899 = None
        getitem_197 = com_microsoft__model_layers_25_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_900 = self.initializers.onnx_initializer_900
        model_layers_25_attn_k_proj_mat_mul = self.model_layers_25_attn_k_proj_MatMul(getitem_197, initializers_onnx_initializer_900);  getitem_197 = initializers_onnx_initializer_900 = None
        getitem_198 = com_microsoft__model_layers_25_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_901 = self.initializers.onnx_initializer_901
        model_layers_25_attn_v_proj_mat_mul = self.model_layers_25_attn_v_proj_MatMul(getitem_198, initializers_onnx_initializer_901);  getitem_198 = initializers_onnx_initializer_901 = None
        initializers_onnx_initializer_902 = self.initializers.onnx_initializer_902
        initializers_onnx_initializer_903 = self.initializers.onnx_initializer_903
        com_microsoft__model_layers_25_attn_q_rotary_rotary_embedding = self.com_microsoft__model_layers_25_attn_q_rotary_RotaryEmbedding(model_layers_25_attn_q_proj_mat_mul, input_3, initializers_onnx_initializer_902, initializers_onnx_initializer_903);  model_layers_25_attn_q_proj_mat_mul = initializers_onnx_initializer_902 = initializers_onnx_initializer_903 = None
        initializers_onnx_initializer_904 = self.initializers.onnx_initializer_904
        initializers_onnx_initializer_905 = self.initializers.onnx_initializer_905
        com_microsoft__model_layers_25_attn_k_rotary_rotary_embedding = self.com_microsoft__model_layers_25_attn_k_rotary_RotaryEmbedding(model_layers_25_attn_k_proj_mat_mul, input_3, initializers_onnx_initializer_904, initializers_onnx_initializer_905);  model_layers_25_attn_k_proj_mat_mul = initializers_onnx_initializer_904 = initializers_onnx_initializer_905 = None
        initializers_onnx_initializer_906 = self.initializers.onnx_initializer_906
        model_layers_25_attn_v_proj_repeat_kv_reshape_1 = self.model_layers_25_attn_v_proj_repeat_kv_Reshape_1(model_layers_25_attn_v_proj_mat_mul, initializers_onnx_initializer_906);  model_layers_25_attn_v_proj_mat_mul = initializers_onnx_initializer_906 = None
        initializers_onnx_initializer_907 = self.initializers.onnx_initializer_907
        model_layers_25_attn_k_proj_repeat_kv_reshape_1 = self.model_layers_25_attn_k_proj_repeat_kv_Reshape_1(com_microsoft__model_layers_25_attn_k_rotary_rotary_embedding, initializers_onnx_initializer_907);  com_microsoft__model_layers_25_attn_k_rotary_rotary_embedding = initializers_onnx_initializer_907 = None
        model_layers_25_attn_v_proj_repeat_kv_transpose_1 = self.model_layers_25_attn_v_proj_repeat_kv_Transpose_1(model_layers_25_attn_v_proj_repeat_kv_reshape_1);  model_layers_25_attn_v_proj_repeat_kv_reshape_1 = None
        model_layers_25_attn_k_proj_repeat_kv_transpose_1 = self.model_layers_25_attn_k_proj_repeat_kv_Transpose_1(model_layers_25_attn_k_proj_repeat_kv_reshape_1);  model_layers_25_attn_k_proj_repeat_kv_reshape_1 = None
        model_layers_25_attn_v_proj_repeat_kv_concat_1 = self.model_layers_25_attn_v_proj_repeat_kv_Concat_1(input_55, model_layers_25_attn_v_proj_repeat_kv_transpose_1);  input_55 = model_layers_25_attn_v_proj_repeat_kv_transpose_1 = None
        model_layers_25_attn_k_proj_repeat_kv_concat_1 = self.model_layers_25_attn_k_proj_repeat_kv_Concat_1(input_54, model_layers_25_attn_k_proj_repeat_kv_transpose_1);  input_54 = model_layers_25_attn_k_proj_repeat_kv_transpose_1 = None
        model_layers_25_attn_v_proj_repeat_kv_shape_1 = self.model_layers_25_attn_v_proj_repeat_kv_Shape_1(model_layers_25_attn_v_proj_repeat_kv_concat_1)
        model_layers_25_attn_v_proj_repeat_kv_unsqueeze_5 = self.model_layers_25_attn_v_proj_repeat_kv_Unsqueeze_5(model_layers_25_attn_v_proj_repeat_kv_concat_1)
        model_layers_25_attn_k_proj_repeat_kv_shape_1 = self.model_layers_25_attn_k_proj_repeat_kv_Shape_1(model_layers_25_attn_k_proj_repeat_kv_concat_1)
        model_layers_25_attn_k_proj_repeat_kv_unsqueeze_5 = self.model_layers_25_attn_k_proj_repeat_kv_Unsqueeze_5(model_layers_25_attn_k_proj_repeat_kv_concat_1)
        initializers_onnx_initializer_908 = self.initializers.onnx_initializer_908
        model_layers_25_attn_v_proj_repeat_kv_gather_1 = self.model_layers_25_attn_v_proj_repeat_kv_Gather_1(model_layers_25_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_908);  initializers_onnx_initializer_908 = None
        initializers_onnx_initializer_909 = self.initializers.onnx_initializer_909
        model_layers_25_attn_v_proj_repeat_kv_gather_3 = self.model_layers_25_attn_v_proj_repeat_kv_Gather_3(model_layers_25_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_909);  model_layers_25_attn_v_proj_repeat_kv_shape_1 = initializers_onnx_initializer_909 = None
        initializers_onnx_initializer_910 = self.initializers.onnx_initializer_910
        model_layers_25_attn_k_proj_repeat_kv_gather_1 = self.model_layers_25_attn_k_proj_repeat_kv_Gather_1(model_layers_25_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_910);  initializers_onnx_initializer_910 = None
        initializers_onnx_initializer_911 = self.initializers.onnx_initializer_911
        model_layers_25_attn_k_proj_repeat_kv_gather_3 = self.model_layers_25_attn_k_proj_repeat_kv_Gather_3(model_layers_25_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_911);  model_layers_25_attn_k_proj_repeat_kv_shape_1 = initializers_onnx_initializer_911 = None
        model_layers_25_attn_v_proj_repeat_kv_unsqueeze_1 = self.model_layers_25_attn_v_proj_repeat_kv_Unsqueeze_1(model_layers_25_attn_v_proj_repeat_kv_gather_1);  model_layers_25_attn_v_proj_repeat_kv_gather_1 = None
        model_layers_25_attn_v_proj_repeat_kv_unsqueeze_3 = self.model_layers_25_attn_v_proj_repeat_kv_Unsqueeze_3(model_layers_25_attn_v_proj_repeat_kv_gather_3);  model_layers_25_attn_v_proj_repeat_kv_gather_3 = None
        model_layers_25_attn_k_proj_repeat_kv_unsqueeze_1 = self.model_layers_25_attn_k_proj_repeat_kv_Unsqueeze_1(model_layers_25_attn_k_proj_repeat_kv_gather_1);  model_layers_25_attn_k_proj_repeat_kv_gather_1 = None
        model_layers_25_attn_k_proj_repeat_kv_unsqueeze_3 = self.model_layers_25_attn_k_proj_repeat_kv_Unsqueeze_3(model_layers_25_attn_k_proj_repeat_kv_gather_3);  model_layers_25_attn_k_proj_repeat_kv_gather_3 = None
        initializers_onnx_initializer_912 = self.initializers.onnx_initializer_912
        initializers_onnx_initializer_913 = self.initializers.onnx_initializer_913
        initializers_onnx_initializer_914 = self.initializers.onnx_initializer_914
        model_layers_25_attn_v_proj_repeat_kv_concat_2 = self.model_layers_25_attn_v_proj_repeat_kv_Concat_2(model_layers_25_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_912, initializers_onnx_initializer_913, model_layers_25_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_914);  initializers_onnx_initializer_912 = initializers_onnx_initializer_913 = initializers_onnx_initializer_914 = None
        initializers_onnx_initializer_915 = self.initializers.onnx_initializer_915
        initializers_onnx_initializer_916 = self.initializers.onnx_initializer_916
        model_layers_25_attn_v_proj_repeat_kv_concat_3 = self.model_layers_25_attn_v_proj_repeat_kv_Concat_3(model_layers_25_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_915, model_layers_25_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_916);  model_layers_25_attn_v_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_915 = model_layers_25_attn_v_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_916 = None
        initializers_onnx_initializer_917 = self.initializers.onnx_initializer_917
        initializers_onnx_initializer_918 = self.initializers.onnx_initializer_918
        initializers_onnx_initializer_919 = self.initializers.onnx_initializer_919
        model_layers_25_attn_k_proj_repeat_kv_concat_2 = self.model_layers_25_attn_k_proj_repeat_kv_Concat_2(model_layers_25_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_917, initializers_onnx_initializer_918, model_layers_25_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_919);  initializers_onnx_initializer_917 = initializers_onnx_initializer_918 = initializers_onnx_initializer_919 = None
        initializers_onnx_initializer_920 = self.initializers.onnx_initializer_920
        initializers_onnx_initializer_921 = self.initializers.onnx_initializer_921
        model_layers_25_attn_k_proj_repeat_kv_concat_3 = self.model_layers_25_attn_k_proj_repeat_kv_Concat_3(model_layers_25_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_920, model_layers_25_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_921);  model_layers_25_attn_k_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_920 = model_layers_25_attn_k_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_921 = None
        initializers_onnx_initializer_922 = self.initializers.onnx_initializer_922
        model_layers_25_attn_v_proj_repeat_kv_equal = self.model_layers_25_attn_v_proj_repeat_kv_Equal(model_layers_25_attn_v_proj_repeat_kv_concat_2, initializers_onnx_initializer_922);  initializers_onnx_initializer_922 = None
        initializers_onnx_initializer_923 = self.initializers.onnx_initializer_923
        model_layers_25_attn_k_proj_repeat_kv_equal = self.model_layers_25_attn_k_proj_repeat_kv_Equal(model_layers_25_attn_k_proj_repeat_kv_concat_2, initializers_onnx_initializer_923);  initializers_onnx_initializer_923 = None
        initializers_onnx_initializer_924 = self.initializers.onnx_initializer_924
        model_layers_25_attn_v_proj_repeat_kv_where = self.model_layers_25_attn_v_proj_repeat_kv_Where(model_layers_25_attn_v_proj_repeat_kv_equal, initializers_onnx_initializer_924, model_layers_25_attn_v_proj_repeat_kv_concat_2);  model_layers_25_attn_v_proj_repeat_kv_equal = initializers_onnx_initializer_924 = model_layers_25_attn_v_proj_repeat_kv_concat_2 = None
        initializers_onnx_initializer_925 = self.initializers.onnx_initializer_925
        model_layers_25_attn_k_proj_repeat_kv_where = self.model_layers_25_attn_k_proj_repeat_kv_Where(model_layers_25_attn_k_proj_repeat_kv_equal, initializers_onnx_initializer_925, model_layers_25_attn_k_proj_repeat_kv_concat_2);  model_layers_25_attn_k_proj_repeat_kv_equal = initializers_onnx_initializer_925 = model_layers_25_attn_k_proj_repeat_kv_concat_2 = None
        model_layers_25_attn_v_proj_repeat_kv_expand = self.model_layers_25_attn_v_proj_repeat_kv_Expand(model_layers_25_attn_v_proj_repeat_kv_unsqueeze_5, model_layers_25_attn_v_proj_repeat_kv_where);  model_layers_25_attn_v_proj_repeat_kv_unsqueeze_5 = model_layers_25_attn_v_proj_repeat_kv_where = None
        model_layers_25_attn_k_proj_repeat_kv_expand = self.model_layers_25_attn_k_proj_repeat_kv_Expand(model_layers_25_attn_k_proj_repeat_kv_unsqueeze_5, model_layers_25_attn_k_proj_repeat_kv_where);  model_layers_25_attn_k_proj_repeat_kv_unsqueeze_5 = model_layers_25_attn_k_proj_repeat_kv_where = None
        model_layers_25_attn_v_proj_repeat_kv_reshape_3 = self.model_layers_25_attn_v_proj_repeat_kv_Reshape_3(model_layers_25_attn_v_proj_repeat_kv_expand, model_layers_25_attn_v_proj_repeat_kv_concat_3);  model_layers_25_attn_v_proj_repeat_kv_expand = model_layers_25_attn_v_proj_repeat_kv_concat_3 = None
        model_layers_25_attn_k_proj_repeat_kv_reshape_3 = self.model_layers_25_attn_k_proj_repeat_kv_Reshape_3(model_layers_25_attn_k_proj_repeat_kv_expand, model_layers_25_attn_k_proj_repeat_kv_concat_3);  model_layers_25_attn_k_proj_repeat_kv_expand = model_layers_25_attn_k_proj_repeat_kv_concat_3 = None
        model_layers_25_attn_v_proj_repeat_kv_transpose_2 = self.model_layers_25_attn_v_proj_repeat_kv_Transpose_2(model_layers_25_attn_v_proj_repeat_kv_reshape_3);  model_layers_25_attn_v_proj_repeat_kv_reshape_3 = None
        model_layers_25_attn_k_proj_repeat_kv_transpose_2 = self.model_layers_25_attn_k_proj_repeat_kv_Transpose_2(model_layers_25_attn_k_proj_repeat_kv_reshape_3);  model_layers_25_attn_k_proj_repeat_kv_reshape_3 = None
        initializers_onnx_initializer_926 = self.initializers.onnx_initializer_926
        model_layers_25_attn_v_proj_repeat_kv_reshape_4 = self.model_layers_25_attn_v_proj_repeat_kv_Reshape_4(model_layers_25_attn_v_proj_repeat_kv_transpose_2, initializers_onnx_initializer_926);  model_layers_25_attn_v_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_926 = None
        initializers_onnx_initializer_927 = self.initializers.onnx_initializer_927
        model_layers_25_attn_k_proj_repeat_kv_reshape_4 = self.model_layers_25_attn_k_proj_repeat_kv_Reshape_4(model_layers_25_attn_k_proj_repeat_kv_transpose_2, initializers_onnx_initializer_927);  model_layers_25_attn_k_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_927 = None
        com_microsoft__model_layers_25_attn_multi_head_attention = self.com_microsoft__model_layers_25_attn_MultiHeadAttention(com_microsoft__model_layers_25_attn_q_rotary_rotary_embedding, model_layers_25_attn_k_proj_repeat_kv_reshape_4, model_layers_25_attn_v_proj_repeat_kv_reshape_4, attention_mask = model_attn_mask_reformat_tile);  com_microsoft__model_layers_25_attn_q_rotary_rotary_embedding = model_layers_25_attn_k_proj_repeat_kv_reshape_4 = model_layers_25_attn_v_proj_repeat_kv_reshape_4 = None
        getitem_199 = com_microsoft__model_layers_25_attn_multi_head_attention[0];  com_microsoft__model_layers_25_attn_multi_head_attention = None
        initializers_onnx_initializer_928 = self.initializers.onnx_initializer_928
        model_layers_25_attn_o_proj_mat_mul = self.model_layers_25_attn_o_proj_MatMul(getitem_199, initializers_onnx_initializer_928);  getitem_199 = initializers_onnx_initializer_928 = None
        getitem_200 = com_microsoft__model_layers_25_input_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_25_input_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_929 = self.initializers.onnx_initializer_929
        com_microsoft__model_layers_25_post_attention_layernorm_skip_layer_norm = self.com_microsoft__model_layers_25_post_attention_layernorm_SkipLayerNorm(getitem_200, model_layers_25_attn_o_proj_mat_mul, initializers_onnx_initializer_929);  getitem_200 = model_layers_25_attn_o_proj_mat_mul = initializers_onnx_initializer_929 = None
        getitem_201 = com_microsoft__model_layers_25_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_930 = self.initializers.onnx_initializer_930
        model_layers_25_mlp_gate_proj_mat_mul = self.model_layers_25_mlp_gate_proj_MatMul(getitem_201, initializers_onnx_initializer_930);  getitem_201 = initializers_onnx_initializer_930 = None
        getitem_202 = com_microsoft__model_layers_25_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_931 = self.initializers.onnx_initializer_931
        model_layers_25_mlp_up_proj_mat_mul = self.model_layers_25_mlp_up_proj_MatMul(getitem_202, initializers_onnx_initializer_931);  getitem_202 = initializers_onnx_initializer_931 = None
        model_layers_25_mlp_act_fn_sigmoid = self.model_layers_25_mlp_act_fn_Sigmoid(model_layers_25_mlp_gate_proj_mat_mul)
        model_layers_25_mlp_act_fn_mul = self.model_layers_25_mlp_act_fn_Mul(model_layers_25_mlp_gate_proj_mat_mul, model_layers_25_mlp_act_fn_sigmoid);  model_layers_25_mlp_gate_proj_mat_mul = model_layers_25_mlp_act_fn_sigmoid = None
        model_layers_25_mlp_mul = self.model_layers_25_mlp_Mul(model_layers_25_mlp_act_fn_mul, model_layers_25_mlp_up_proj_mat_mul);  model_layers_25_mlp_act_fn_mul = model_layers_25_mlp_up_proj_mat_mul = None
        initializers_onnx_initializer_932 = self.initializers.onnx_initializer_932
        model_layers_25_mlp_down_proj_mat_mul = self.model_layers_25_mlp_down_proj_MatMul(model_layers_25_mlp_mul, initializers_onnx_initializer_932);  model_layers_25_mlp_mul = initializers_onnx_initializer_932 = None
        getitem_203 = com_microsoft__model_layers_25_post_attention_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_25_post_attention_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_933 = self.initializers.onnx_initializer_933
        com_microsoft__model_layers_26_input_layernorm_skip_layer_norm = self.com_microsoft__model_layers_26_input_layernorm_SkipLayerNorm(getitem_203, model_layers_25_mlp_down_proj_mat_mul, initializers_onnx_initializer_933);  getitem_203 = model_layers_25_mlp_down_proj_mat_mul = initializers_onnx_initializer_933 = None
        getitem_204 = com_microsoft__model_layers_26_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_934 = self.initializers.onnx_initializer_934
        model_layers_26_attn_q_proj_mat_mul = self.model_layers_26_attn_q_proj_MatMul(getitem_204, initializers_onnx_initializer_934);  getitem_204 = initializers_onnx_initializer_934 = None
        getitem_205 = com_microsoft__model_layers_26_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_935 = self.initializers.onnx_initializer_935
        model_layers_26_attn_k_proj_mat_mul = self.model_layers_26_attn_k_proj_MatMul(getitem_205, initializers_onnx_initializer_935);  getitem_205 = initializers_onnx_initializer_935 = None
        getitem_206 = com_microsoft__model_layers_26_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_936 = self.initializers.onnx_initializer_936
        model_layers_26_attn_v_proj_mat_mul = self.model_layers_26_attn_v_proj_MatMul(getitem_206, initializers_onnx_initializer_936);  getitem_206 = initializers_onnx_initializer_936 = None
        initializers_onnx_initializer_937 = self.initializers.onnx_initializer_937
        initializers_onnx_initializer_938 = self.initializers.onnx_initializer_938
        com_microsoft__model_layers_26_attn_q_rotary_rotary_embedding = self.com_microsoft__model_layers_26_attn_q_rotary_RotaryEmbedding(model_layers_26_attn_q_proj_mat_mul, input_3, initializers_onnx_initializer_937, initializers_onnx_initializer_938);  model_layers_26_attn_q_proj_mat_mul = initializers_onnx_initializer_937 = initializers_onnx_initializer_938 = None
        initializers_onnx_initializer_939 = self.initializers.onnx_initializer_939
        initializers_onnx_initializer_940 = self.initializers.onnx_initializer_940
        com_microsoft__model_layers_26_attn_k_rotary_rotary_embedding = self.com_microsoft__model_layers_26_attn_k_rotary_RotaryEmbedding(model_layers_26_attn_k_proj_mat_mul, input_3, initializers_onnx_initializer_939, initializers_onnx_initializer_940);  model_layers_26_attn_k_proj_mat_mul = initializers_onnx_initializer_939 = initializers_onnx_initializer_940 = None
        initializers_onnx_initializer_941 = self.initializers.onnx_initializer_941
        model_layers_26_attn_v_proj_repeat_kv_reshape_1 = self.model_layers_26_attn_v_proj_repeat_kv_Reshape_1(model_layers_26_attn_v_proj_mat_mul, initializers_onnx_initializer_941);  model_layers_26_attn_v_proj_mat_mul = initializers_onnx_initializer_941 = None
        initializers_onnx_initializer_942 = self.initializers.onnx_initializer_942
        model_layers_26_attn_k_proj_repeat_kv_reshape_1 = self.model_layers_26_attn_k_proj_repeat_kv_Reshape_1(com_microsoft__model_layers_26_attn_k_rotary_rotary_embedding, initializers_onnx_initializer_942);  com_microsoft__model_layers_26_attn_k_rotary_rotary_embedding = initializers_onnx_initializer_942 = None
        model_layers_26_attn_v_proj_repeat_kv_transpose_1 = self.model_layers_26_attn_v_proj_repeat_kv_Transpose_1(model_layers_26_attn_v_proj_repeat_kv_reshape_1);  model_layers_26_attn_v_proj_repeat_kv_reshape_1 = None
        model_layers_26_attn_k_proj_repeat_kv_transpose_1 = self.model_layers_26_attn_k_proj_repeat_kv_Transpose_1(model_layers_26_attn_k_proj_repeat_kv_reshape_1);  model_layers_26_attn_k_proj_repeat_kv_reshape_1 = None
        model_layers_26_attn_v_proj_repeat_kv_concat_1 = self.model_layers_26_attn_v_proj_repeat_kv_Concat_1(input_57, model_layers_26_attn_v_proj_repeat_kv_transpose_1);  input_57 = model_layers_26_attn_v_proj_repeat_kv_transpose_1 = None
        model_layers_26_attn_k_proj_repeat_kv_concat_1 = self.model_layers_26_attn_k_proj_repeat_kv_Concat_1(input_56, model_layers_26_attn_k_proj_repeat_kv_transpose_1);  input_56 = model_layers_26_attn_k_proj_repeat_kv_transpose_1 = None
        model_layers_26_attn_v_proj_repeat_kv_shape_1 = self.model_layers_26_attn_v_proj_repeat_kv_Shape_1(model_layers_26_attn_v_proj_repeat_kv_concat_1)
        model_layers_26_attn_v_proj_repeat_kv_unsqueeze_5 = self.model_layers_26_attn_v_proj_repeat_kv_Unsqueeze_5(model_layers_26_attn_v_proj_repeat_kv_concat_1)
        model_layers_26_attn_k_proj_repeat_kv_shape_1 = self.model_layers_26_attn_k_proj_repeat_kv_Shape_1(model_layers_26_attn_k_proj_repeat_kv_concat_1)
        model_layers_26_attn_k_proj_repeat_kv_unsqueeze_5 = self.model_layers_26_attn_k_proj_repeat_kv_Unsqueeze_5(model_layers_26_attn_k_proj_repeat_kv_concat_1)
        initializers_onnx_initializer_943 = self.initializers.onnx_initializer_943
        model_layers_26_attn_v_proj_repeat_kv_gather_1 = self.model_layers_26_attn_v_proj_repeat_kv_Gather_1(model_layers_26_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_943);  initializers_onnx_initializer_943 = None
        initializers_onnx_initializer_944 = self.initializers.onnx_initializer_944
        model_layers_26_attn_v_proj_repeat_kv_gather_3 = self.model_layers_26_attn_v_proj_repeat_kv_Gather_3(model_layers_26_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_944);  model_layers_26_attn_v_proj_repeat_kv_shape_1 = initializers_onnx_initializer_944 = None
        initializers_onnx_initializer_945 = self.initializers.onnx_initializer_945
        model_layers_26_attn_k_proj_repeat_kv_gather_1 = self.model_layers_26_attn_k_proj_repeat_kv_Gather_1(model_layers_26_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_945);  initializers_onnx_initializer_945 = None
        initializers_onnx_initializer_946 = self.initializers.onnx_initializer_946
        model_layers_26_attn_k_proj_repeat_kv_gather_3 = self.model_layers_26_attn_k_proj_repeat_kv_Gather_3(model_layers_26_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_946);  model_layers_26_attn_k_proj_repeat_kv_shape_1 = initializers_onnx_initializer_946 = None
        model_layers_26_attn_v_proj_repeat_kv_unsqueeze_1 = self.model_layers_26_attn_v_proj_repeat_kv_Unsqueeze_1(model_layers_26_attn_v_proj_repeat_kv_gather_1);  model_layers_26_attn_v_proj_repeat_kv_gather_1 = None
        model_layers_26_attn_v_proj_repeat_kv_unsqueeze_3 = self.model_layers_26_attn_v_proj_repeat_kv_Unsqueeze_3(model_layers_26_attn_v_proj_repeat_kv_gather_3);  model_layers_26_attn_v_proj_repeat_kv_gather_3 = None
        model_layers_26_attn_k_proj_repeat_kv_unsqueeze_1 = self.model_layers_26_attn_k_proj_repeat_kv_Unsqueeze_1(model_layers_26_attn_k_proj_repeat_kv_gather_1);  model_layers_26_attn_k_proj_repeat_kv_gather_1 = None
        model_layers_26_attn_k_proj_repeat_kv_unsqueeze_3 = self.model_layers_26_attn_k_proj_repeat_kv_Unsqueeze_3(model_layers_26_attn_k_proj_repeat_kv_gather_3);  model_layers_26_attn_k_proj_repeat_kv_gather_3 = None
        initializers_onnx_initializer_947 = self.initializers.onnx_initializer_947
        initializers_onnx_initializer_948 = self.initializers.onnx_initializer_948
        initializers_onnx_initializer_949 = self.initializers.onnx_initializer_949
        model_layers_26_attn_v_proj_repeat_kv_concat_2 = self.model_layers_26_attn_v_proj_repeat_kv_Concat_2(model_layers_26_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_947, initializers_onnx_initializer_948, model_layers_26_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_949);  initializers_onnx_initializer_947 = initializers_onnx_initializer_948 = initializers_onnx_initializer_949 = None
        initializers_onnx_initializer_950 = self.initializers.onnx_initializer_950
        initializers_onnx_initializer_951 = self.initializers.onnx_initializer_951
        model_layers_26_attn_v_proj_repeat_kv_concat_3 = self.model_layers_26_attn_v_proj_repeat_kv_Concat_3(model_layers_26_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_950, model_layers_26_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_951);  model_layers_26_attn_v_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_950 = model_layers_26_attn_v_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_951 = None
        initializers_onnx_initializer_952 = self.initializers.onnx_initializer_952
        initializers_onnx_initializer_953 = self.initializers.onnx_initializer_953
        initializers_onnx_initializer_954 = self.initializers.onnx_initializer_954
        model_layers_26_attn_k_proj_repeat_kv_concat_2 = self.model_layers_26_attn_k_proj_repeat_kv_Concat_2(model_layers_26_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_952, initializers_onnx_initializer_953, model_layers_26_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_954);  initializers_onnx_initializer_952 = initializers_onnx_initializer_953 = initializers_onnx_initializer_954 = None
        initializers_onnx_initializer_955 = self.initializers.onnx_initializer_955
        initializers_onnx_initializer_956 = self.initializers.onnx_initializer_956
        model_layers_26_attn_k_proj_repeat_kv_concat_3 = self.model_layers_26_attn_k_proj_repeat_kv_Concat_3(model_layers_26_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_955, model_layers_26_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_956);  model_layers_26_attn_k_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_955 = model_layers_26_attn_k_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_956 = None
        initializers_onnx_initializer_957 = self.initializers.onnx_initializer_957
        model_layers_26_attn_v_proj_repeat_kv_equal = self.model_layers_26_attn_v_proj_repeat_kv_Equal(model_layers_26_attn_v_proj_repeat_kv_concat_2, initializers_onnx_initializer_957);  initializers_onnx_initializer_957 = None
        initializers_onnx_initializer_958 = self.initializers.onnx_initializer_958
        model_layers_26_attn_k_proj_repeat_kv_equal = self.model_layers_26_attn_k_proj_repeat_kv_Equal(model_layers_26_attn_k_proj_repeat_kv_concat_2, initializers_onnx_initializer_958);  initializers_onnx_initializer_958 = None
        initializers_onnx_initializer_959 = self.initializers.onnx_initializer_959
        model_layers_26_attn_v_proj_repeat_kv_where = self.model_layers_26_attn_v_proj_repeat_kv_Where(model_layers_26_attn_v_proj_repeat_kv_equal, initializers_onnx_initializer_959, model_layers_26_attn_v_proj_repeat_kv_concat_2);  model_layers_26_attn_v_proj_repeat_kv_equal = initializers_onnx_initializer_959 = model_layers_26_attn_v_proj_repeat_kv_concat_2 = None
        initializers_onnx_initializer_960 = self.initializers.onnx_initializer_960
        model_layers_26_attn_k_proj_repeat_kv_where = self.model_layers_26_attn_k_proj_repeat_kv_Where(model_layers_26_attn_k_proj_repeat_kv_equal, initializers_onnx_initializer_960, model_layers_26_attn_k_proj_repeat_kv_concat_2);  model_layers_26_attn_k_proj_repeat_kv_equal = initializers_onnx_initializer_960 = model_layers_26_attn_k_proj_repeat_kv_concat_2 = None
        model_layers_26_attn_v_proj_repeat_kv_expand = self.model_layers_26_attn_v_proj_repeat_kv_Expand(model_layers_26_attn_v_proj_repeat_kv_unsqueeze_5, model_layers_26_attn_v_proj_repeat_kv_where);  model_layers_26_attn_v_proj_repeat_kv_unsqueeze_5 = model_layers_26_attn_v_proj_repeat_kv_where = None
        model_layers_26_attn_k_proj_repeat_kv_expand = self.model_layers_26_attn_k_proj_repeat_kv_Expand(model_layers_26_attn_k_proj_repeat_kv_unsqueeze_5, model_layers_26_attn_k_proj_repeat_kv_where);  model_layers_26_attn_k_proj_repeat_kv_unsqueeze_5 = model_layers_26_attn_k_proj_repeat_kv_where = None
        model_layers_26_attn_v_proj_repeat_kv_reshape_3 = self.model_layers_26_attn_v_proj_repeat_kv_Reshape_3(model_layers_26_attn_v_proj_repeat_kv_expand, model_layers_26_attn_v_proj_repeat_kv_concat_3);  model_layers_26_attn_v_proj_repeat_kv_expand = model_layers_26_attn_v_proj_repeat_kv_concat_3 = None
        model_layers_26_attn_k_proj_repeat_kv_reshape_3 = self.model_layers_26_attn_k_proj_repeat_kv_Reshape_3(model_layers_26_attn_k_proj_repeat_kv_expand, model_layers_26_attn_k_proj_repeat_kv_concat_3);  model_layers_26_attn_k_proj_repeat_kv_expand = model_layers_26_attn_k_proj_repeat_kv_concat_3 = None
        model_layers_26_attn_v_proj_repeat_kv_transpose_2 = self.model_layers_26_attn_v_proj_repeat_kv_Transpose_2(model_layers_26_attn_v_proj_repeat_kv_reshape_3);  model_layers_26_attn_v_proj_repeat_kv_reshape_3 = None
        model_layers_26_attn_k_proj_repeat_kv_transpose_2 = self.model_layers_26_attn_k_proj_repeat_kv_Transpose_2(model_layers_26_attn_k_proj_repeat_kv_reshape_3);  model_layers_26_attn_k_proj_repeat_kv_reshape_3 = None
        initializers_onnx_initializer_961 = self.initializers.onnx_initializer_961
        model_layers_26_attn_v_proj_repeat_kv_reshape_4 = self.model_layers_26_attn_v_proj_repeat_kv_Reshape_4(model_layers_26_attn_v_proj_repeat_kv_transpose_2, initializers_onnx_initializer_961);  model_layers_26_attn_v_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_961 = None
        initializers_onnx_initializer_962 = self.initializers.onnx_initializer_962
        model_layers_26_attn_k_proj_repeat_kv_reshape_4 = self.model_layers_26_attn_k_proj_repeat_kv_Reshape_4(model_layers_26_attn_k_proj_repeat_kv_transpose_2, initializers_onnx_initializer_962);  model_layers_26_attn_k_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_962 = None
        com_microsoft__model_layers_26_attn_multi_head_attention = self.com_microsoft__model_layers_26_attn_MultiHeadAttention(com_microsoft__model_layers_26_attn_q_rotary_rotary_embedding, model_layers_26_attn_k_proj_repeat_kv_reshape_4, model_layers_26_attn_v_proj_repeat_kv_reshape_4, attention_mask = model_attn_mask_reformat_tile);  com_microsoft__model_layers_26_attn_q_rotary_rotary_embedding = model_layers_26_attn_k_proj_repeat_kv_reshape_4 = model_layers_26_attn_v_proj_repeat_kv_reshape_4 = None
        getitem_207 = com_microsoft__model_layers_26_attn_multi_head_attention[0];  com_microsoft__model_layers_26_attn_multi_head_attention = None
        initializers_onnx_initializer_963 = self.initializers.onnx_initializer_963
        model_layers_26_attn_o_proj_mat_mul = self.model_layers_26_attn_o_proj_MatMul(getitem_207, initializers_onnx_initializer_963);  getitem_207 = initializers_onnx_initializer_963 = None
        getitem_208 = com_microsoft__model_layers_26_input_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_26_input_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_964 = self.initializers.onnx_initializer_964
        com_microsoft__model_layers_26_post_attention_layernorm_skip_layer_norm = self.com_microsoft__model_layers_26_post_attention_layernorm_SkipLayerNorm(getitem_208, model_layers_26_attn_o_proj_mat_mul, initializers_onnx_initializer_964);  getitem_208 = model_layers_26_attn_o_proj_mat_mul = initializers_onnx_initializer_964 = None
        getitem_209 = com_microsoft__model_layers_26_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_965 = self.initializers.onnx_initializer_965
        model_layers_26_mlp_gate_proj_mat_mul = self.model_layers_26_mlp_gate_proj_MatMul(getitem_209, initializers_onnx_initializer_965);  getitem_209 = initializers_onnx_initializer_965 = None
        getitem_210 = com_microsoft__model_layers_26_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_966 = self.initializers.onnx_initializer_966
        model_layers_26_mlp_up_proj_mat_mul = self.model_layers_26_mlp_up_proj_MatMul(getitem_210, initializers_onnx_initializer_966);  getitem_210 = initializers_onnx_initializer_966 = None
        model_layers_26_mlp_act_fn_sigmoid = self.model_layers_26_mlp_act_fn_Sigmoid(model_layers_26_mlp_gate_proj_mat_mul)
        model_layers_26_mlp_act_fn_mul = self.model_layers_26_mlp_act_fn_Mul(model_layers_26_mlp_gate_proj_mat_mul, model_layers_26_mlp_act_fn_sigmoid);  model_layers_26_mlp_gate_proj_mat_mul = model_layers_26_mlp_act_fn_sigmoid = None
        model_layers_26_mlp_mul = self.model_layers_26_mlp_Mul(model_layers_26_mlp_act_fn_mul, model_layers_26_mlp_up_proj_mat_mul);  model_layers_26_mlp_act_fn_mul = model_layers_26_mlp_up_proj_mat_mul = None
        initializers_onnx_initializer_967 = self.initializers.onnx_initializer_967
        model_layers_26_mlp_down_proj_mat_mul = self.model_layers_26_mlp_down_proj_MatMul(model_layers_26_mlp_mul, initializers_onnx_initializer_967);  model_layers_26_mlp_mul = initializers_onnx_initializer_967 = None
        getitem_211 = com_microsoft__model_layers_26_post_attention_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_26_post_attention_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_968 = self.initializers.onnx_initializer_968
        com_microsoft__model_layers_27_input_layernorm_skip_layer_norm = self.com_microsoft__model_layers_27_input_layernorm_SkipLayerNorm(getitem_211, model_layers_26_mlp_down_proj_mat_mul, initializers_onnx_initializer_968);  getitem_211 = model_layers_26_mlp_down_proj_mat_mul = initializers_onnx_initializer_968 = None
        getitem_212 = com_microsoft__model_layers_27_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_969 = self.initializers.onnx_initializer_969
        model_layers_27_attn_q_proj_mat_mul = self.model_layers_27_attn_q_proj_MatMul(getitem_212, initializers_onnx_initializer_969);  getitem_212 = initializers_onnx_initializer_969 = None
        getitem_213 = com_microsoft__model_layers_27_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_970 = self.initializers.onnx_initializer_970
        model_layers_27_attn_k_proj_mat_mul = self.model_layers_27_attn_k_proj_MatMul(getitem_213, initializers_onnx_initializer_970);  getitem_213 = initializers_onnx_initializer_970 = None
        getitem_214 = com_microsoft__model_layers_27_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_971 = self.initializers.onnx_initializer_971
        model_layers_27_attn_v_proj_mat_mul = self.model_layers_27_attn_v_proj_MatMul(getitem_214, initializers_onnx_initializer_971);  getitem_214 = initializers_onnx_initializer_971 = None
        initializers_onnx_initializer_972 = self.initializers.onnx_initializer_972
        initializers_onnx_initializer_973 = self.initializers.onnx_initializer_973
        com_microsoft__model_layers_27_attn_q_rotary_rotary_embedding = self.com_microsoft__model_layers_27_attn_q_rotary_RotaryEmbedding(model_layers_27_attn_q_proj_mat_mul, input_3, initializers_onnx_initializer_972, initializers_onnx_initializer_973);  model_layers_27_attn_q_proj_mat_mul = initializers_onnx_initializer_972 = initializers_onnx_initializer_973 = None
        initializers_onnx_initializer_974 = self.initializers.onnx_initializer_974
        initializers_onnx_initializer_975 = self.initializers.onnx_initializer_975
        com_microsoft__model_layers_27_attn_k_rotary_rotary_embedding = self.com_microsoft__model_layers_27_attn_k_rotary_RotaryEmbedding(model_layers_27_attn_k_proj_mat_mul, input_3, initializers_onnx_initializer_974, initializers_onnx_initializer_975);  model_layers_27_attn_k_proj_mat_mul = initializers_onnx_initializer_974 = initializers_onnx_initializer_975 = None
        initializers_onnx_initializer_976 = self.initializers.onnx_initializer_976
        model_layers_27_attn_v_proj_repeat_kv_reshape_1 = self.model_layers_27_attn_v_proj_repeat_kv_Reshape_1(model_layers_27_attn_v_proj_mat_mul, initializers_onnx_initializer_976);  model_layers_27_attn_v_proj_mat_mul = initializers_onnx_initializer_976 = None
        initializers_onnx_initializer_977 = self.initializers.onnx_initializer_977
        model_layers_27_attn_k_proj_repeat_kv_reshape_1 = self.model_layers_27_attn_k_proj_repeat_kv_Reshape_1(com_microsoft__model_layers_27_attn_k_rotary_rotary_embedding, initializers_onnx_initializer_977);  com_microsoft__model_layers_27_attn_k_rotary_rotary_embedding = initializers_onnx_initializer_977 = None
        model_layers_27_attn_v_proj_repeat_kv_transpose_1 = self.model_layers_27_attn_v_proj_repeat_kv_Transpose_1(model_layers_27_attn_v_proj_repeat_kv_reshape_1);  model_layers_27_attn_v_proj_repeat_kv_reshape_1 = None
        model_layers_27_attn_k_proj_repeat_kv_transpose_1 = self.model_layers_27_attn_k_proj_repeat_kv_Transpose_1(model_layers_27_attn_k_proj_repeat_kv_reshape_1);  model_layers_27_attn_k_proj_repeat_kv_reshape_1 = None
        model_layers_27_attn_v_proj_repeat_kv_concat_1 = self.model_layers_27_attn_v_proj_repeat_kv_Concat_1(input_59, model_layers_27_attn_v_proj_repeat_kv_transpose_1);  input_59 = model_layers_27_attn_v_proj_repeat_kv_transpose_1 = None
        model_layers_27_attn_k_proj_repeat_kv_concat_1 = self.model_layers_27_attn_k_proj_repeat_kv_Concat_1(input_58, model_layers_27_attn_k_proj_repeat_kv_transpose_1);  input_58 = model_layers_27_attn_k_proj_repeat_kv_transpose_1 = None
        model_layers_27_attn_v_proj_repeat_kv_shape_1 = self.model_layers_27_attn_v_proj_repeat_kv_Shape_1(model_layers_27_attn_v_proj_repeat_kv_concat_1)
        model_layers_27_attn_v_proj_repeat_kv_unsqueeze_5 = self.model_layers_27_attn_v_proj_repeat_kv_Unsqueeze_5(model_layers_27_attn_v_proj_repeat_kv_concat_1)
        model_layers_27_attn_k_proj_repeat_kv_shape_1 = self.model_layers_27_attn_k_proj_repeat_kv_Shape_1(model_layers_27_attn_k_proj_repeat_kv_concat_1)
        model_layers_27_attn_k_proj_repeat_kv_unsqueeze_5 = self.model_layers_27_attn_k_proj_repeat_kv_Unsqueeze_5(model_layers_27_attn_k_proj_repeat_kv_concat_1)
        initializers_onnx_initializer_978 = self.initializers.onnx_initializer_978
        model_layers_27_attn_v_proj_repeat_kv_gather_1 = self.model_layers_27_attn_v_proj_repeat_kv_Gather_1(model_layers_27_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_978);  initializers_onnx_initializer_978 = None
        initializers_onnx_initializer_979 = self.initializers.onnx_initializer_979
        model_layers_27_attn_v_proj_repeat_kv_gather_3 = self.model_layers_27_attn_v_proj_repeat_kv_Gather_3(model_layers_27_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_979);  model_layers_27_attn_v_proj_repeat_kv_shape_1 = initializers_onnx_initializer_979 = None
        initializers_onnx_initializer_980 = self.initializers.onnx_initializer_980
        model_layers_27_attn_k_proj_repeat_kv_gather_1 = self.model_layers_27_attn_k_proj_repeat_kv_Gather_1(model_layers_27_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_980);  initializers_onnx_initializer_980 = None
        initializers_onnx_initializer_981 = self.initializers.onnx_initializer_981
        model_layers_27_attn_k_proj_repeat_kv_gather_3 = self.model_layers_27_attn_k_proj_repeat_kv_Gather_3(model_layers_27_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_981);  model_layers_27_attn_k_proj_repeat_kv_shape_1 = initializers_onnx_initializer_981 = None
        model_layers_27_attn_v_proj_repeat_kv_unsqueeze_1 = self.model_layers_27_attn_v_proj_repeat_kv_Unsqueeze_1(model_layers_27_attn_v_proj_repeat_kv_gather_1);  model_layers_27_attn_v_proj_repeat_kv_gather_1 = None
        model_layers_27_attn_v_proj_repeat_kv_unsqueeze_3 = self.model_layers_27_attn_v_proj_repeat_kv_Unsqueeze_3(model_layers_27_attn_v_proj_repeat_kv_gather_3);  model_layers_27_attn_v_proj_repeat_kv_gather_3 = None
        model_layers_27_attn_k_proj_repeat_kv_unsqueeze_1 = self.model_layers_27_attn_k_proj_repeat_kv_Unsqueeze_1(model_layers_27_attn_k_proj_repeat_kv_gather_1);  model_layers_27_attn_k_proj_repeat_kv_gather_1 = None
        model_layers_27_attn_k_proj_repeat_kv_unsqueeze_3 = self.model_layers_27_attn_k_proj_repeat_kv_Unsqueeze_3(model_layers_27_attn_k_proj_repeat_kv_gather_3);  model_layers_27_attn_k_proj_repeat_kv_gather_3 = None
        initializers_onnx_initializer_982 = self.initializers.onnx_initializer_982
        initializers_onnx_initializer_983 = self.initializers.onnx_initializer_983
        initializers_onnx_initializer_984 = self.initializers.onnx_initializer_984
        model_layers_27_attn_v_proj_repeat_kv_concat_2 = self.model_layers_27_attn_v_proj_repeat_kv_Concat_2(model_layers_27_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_982, initializers_onnx_initializer_983, model_layers_27_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_984);  initializers_onnx_initializer_982 = initializers_onnx_initializer_983 = initializers_onnx_initializer_984 = None
        initializers_onnx_initializer_985 = self.initializers.onnx_initializer_985
        initializers_onnx_initializer_986 = self.initializers.onnx_initializer_986
        model_layers_27_attn_v_proj_repeat_kv_concat_3 = self.model_layers_27_attn_v_proj_repeat_kv_Concat_3(model_layers_27_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_985, model_layers_27_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_986);  model_layers_27_attn_v_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_985 = model_layers_27_attn_v_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_986 = None
        initializers_onnx_initializer_987 = self.initializers.onnx_initializer_987
        initializers_onnx_initializer_988 = self.initializers.onnx_initializer_988
        initializers_onnx_initializer_989 = self.initializers.onnx_initializer_989
        model_layers_27_attn_k_proj_repeat_kv_concat_2 = self.model_layers_27_attn_k_proj_repeat_kv_Concat_2(model_layers_27_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_987, initializers_onnx_initializer_988, model_layers_27_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_989);  initializers_onnx_initializer_987 = initializers_onnx_initializer_988 = initializers_onnx_initializer_989 = None
        initializers_onnx_initializer_990 = self.initializers.onnx_initializer_990
        initializers_onnx_initializer_991 = self.initializers.onnx_initializer_991
        model_layers_27_attn_k_proj_repeat_kv_concat_3 = self.model_layers_27_attn_k_proj_repeat_kv_Concat_3(model_layers_27_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_990, model_layers_27_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_991);  model_layers_27_attn_k_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_990 = model_layers_27_attn_k_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_991 = None
        initializers_onnx_initializer_992 = self.initializers.onnx_initializer_992
        model_layers_27_attn_v_proj_repeat_kv_equal = self.model_layers_27_attn_v_proj_repeat_kv_Equal(model_layers_27_attn_v_proj_repeat_kv_concat_2, initializers_onnx_initializer_992);  initializers_onnx_initializer_992 = None
        initializers_onnx_initializer_993 = self.initializers.onnx_initializer_993
        model_layers_27_attn_k_proj_repeat_kv_equal = self.model_layers_27_attn_k_proj_repeat_kv_Equal(model_layers_27_attn_k_proj_repeat_kv_concat_2, initializers_onnx_initializer_993);  initializers_onnx_initializer_993 = None
        initializers_onnx_initializer_994 = self.initializers.onnx_initializer_994
        model_layers_27_attn_v_proj_repeat_kv_where = self.model_layers_27_attn_v_proj_repeat_kv_Where(model_layers_27_attn_v_proj_repeat_kv_equal, initializers_onnx_initializer_994, model_layers_27_attn_v_proj_repeat_kv_concat_2);  model_layers_27_attn_v_proj_repeat_kv_equal = initializers_onnx_initializer_994 = model_layers_27_attn_v_proj_repeat_kv_concat_2 = None
        initializers_onnx_initializer_995 = self.initializers.onnx_initializer_995
        model_layers_27_attn_k_proj_repeat_kv_where = self.model_layers_27_attn_k_proj_repeat_kv_Where(model_layers_27_attn_k_proj_repeat_kv_equal, initializers_onnx_initializer_995, model_layers_27_attn_k_proj_repeat_kv_concat_2);  model_layers_27_attn_k_proj_repeat_kv_equal = initializers_onnx_initializer_995 = model_layers_27_attn_k_proj_repeat_kv_concat_2 = None
        model_layers_27_attn_v_proj_repeat_kv_expand = self.model_layers_27_attn_v_proj_repeat_kv_Expand(model_layers_27_attn_v_proj_repeat_kv_unsqueeze_5, model_layers_27_attn_v_proj_repeat_kv_where);  model_layers_27_attn_v_proj_repeat_kv_unsqueeze_5 = model_layers_27_attn_v_proj_repeat_kv_where = None
        model_layers_27_attn_k_proj_repeat_kv_expand = self.model_layers_27_attn_k_proj_repeat_kv_Expand(model_layers_27_attn_k_proj_repeat_kv_unsqueeze_5, model_layers_27_attn_k_proj_repeat_kv_where);  model_layers_27_attn_k_proj_repeat_kv_unsqueeze_5 = model_layers_27_attn_k_proj_repeat_kv_where = None
        model_layers_27_attn_v_proj_repeat_kv_reshape_3 = self.model_layers_27_attn_v_proj_repeat_kv_Reshape_3(model_layers_27_attn_v_proj_repeat_kv_expand, model_layers_27_attn_v_proj_repeat_kv_concat_3);  model_layers_27_attn_v_proj_repeat_kv_expand = model_layers_27_attn_v_proj_repeat_kv_concat_3 = None
        model_layers_27_attn_k_proj_repeat_kv_reshape_3 = self.model_layers_27_attn_k_proj_repeat_kv_Reshape_3(model_layers_27_attn_k_proj_repeat_kv_expand, model_layers_27_attn_k_proj_repeat_kv_concat_3);  model_layers_27_attn_k_proj_repeat_kv_expand = model_layers_27_attn_k_proj_repeat_kv_concat_3 = None
        model_layers_27_attn_v_proj_repeat_kv_transpose_2 = self.model_layers_27_attn_v_proj_repeat_kv_Transpose_2(model_layers_27_attn_v_proj_repeat_kv_reshape_3);  model_layers_27_attn_v_proj_repeat_kv_reshape_3 = None
        model_layers_27_attn_k_proj_repeat_kv_transpose_2 = self.model_layers_27_attn_k_proj_repeat_kv_Transpose_2(model_layers_27_attn_k_proj_repeat_kv_reshape_3);  model_layers_27_attn_k_proj_repeat_kv_reshape_3 = None
        initializers_onnx_initializer_996 = self.initializers.onnx_initializer_996
        model_layers_27_attn_v_proj_repeat_kv_reshape_4 = self.model_layers_27_attn_v_proj_repeat_kv_Reshape_4(model_layers_27_attn_v_proj_repeat_kv_transpose_2, initializers_onnx_initializer_996);  model_layers_27_attn_v_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_996 = None
        initializers_onnx_initializer_997 = self.initializers.onnx_initializer_997
        model_layers_27_attn_k_proj_repeat_kv_reshape_4 = self.model_layers_27_attn_k_proj_repeat_kv_Reshape_4(model_layers_27_attn_k_proj_repeat_kv_transpose_2, initializers_onnx_initializer_997);  model_layers_27_attn_k_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_997 = None
        com_microsoft__model_layers_27_attn_multi_head_attention = self.com_microsoft__model_layers_27_attn_MultiHeadAttention(com_microsoft__model_layers_27_attn_q_rotary_rotary_embedding, model_layers_27_attn_k_proj_repeat_kv_reshape_4, model_layers_27_attn_v_proj_repeat_kv_reshape_4, attention_mask = model_attn_mask_reformat_tile);  com_microsoft__model_layers_27_attn_q_rotary_rotary_embedding = model_layers_27_attn_k_proj_repeat_kv_reshape_4 = model_layers_27_attn_v_proj_repeat_kv_reshape_4 = None
        getitem_215 = com_microsoft__model_layers_27_attn_multi_head_attention[0];  com_microsoft__model_layers_27_attn_multi_head_attention = None
        initializers_onnx_initializer_998 = self.initializers.onnx_initializer_998
        model_layers_27_attn_o_proj_mat_mul = self.model_layers_27_attn_o_proj_MatMul(getitem_215, initializers_onnx_initializer_998);  getitem_215 = initializers_onnx_initializer_998 = None
        getitem_216 = com_microsoft__model_layers_27_input_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_27_input_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_999 = self.initializers.onnx_initializer_999
        com_microsoft__model_layers_27_post_attention_layernorm_skip_layer_norm = self.com_microsoft__model_layers_27_post_attention_layernorm_SkipLayerNorm(getitem_216, model_layers_27_attn_o_proj_mat_mul, initializers_onnx_initializer_999);  getitem_216 = model_layers_27_attn_o_proj_mat_mul = initializers_onnx_initializer_999 = None
        getitem_217 = com_microsoft__model_layers_27_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_1000 = self.initializers.onnx_initializer_1000
        model_layers_27_mlp_gate_proj_mat_mul = self.model_layers_27_mlp_gate_proj_MatMul(getitem_217, initializers_onnx_initializer_1000);  getitem_217 = initializers_onnx_initializer_1000 = None
        getitem_218 = com_microsoft__model_layers_27_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_1001 = self.initializers.onnx_initializer_1001
        model_layers_27_mlp_up_proj_mat_mul = self.model_layers_27_mlp_up_proj_MatMul(getitem_218, initializers_onnx_initializer_1001);  getitem_218 = initializers_onnx_initializer_1001 = None
        model_layers_27_mlp_act_fn_sigmoid = self.model_layers_27_mlp_act_fn_Sigmoid(model_layers_27_mlp_gate_proj_mat_mul)
        model_layers_27_mlp_act_fn_mul = self.model_layers_27_mlp_act_fn_Mul(model_layers_27_mlp_gate_proj_mat_mul, model_layers_27_mlp_act_fn_sigmoid);  model_layers_27_mlp_gate_proj_mat_mul = model_layers_27_mlp_act_fn_sigmoid = None
        model_layers_27_mlp_mul = self.model_layers_27_mlp_Mul(model_layers_27_mlp_act_fn_mul, model_layers_27_mlp_up_proj_mat_mul);  model_layers_27_mlp_act_fn_mul = model_layers_27_mlp_up_proj_mat_mul = None
        initializers_onnx_initializer_1002 = self.initializers.onnx_initializer_1002
        model_layers_27_mlp_down_proj_mat_mul = self.model_layers_27_mlp_down_proj_MatMul(model_layers_27_mlp_mul, initializers_onnx_initializer_1002);  model_layers_27_mlp_mul = initializers_onnx_initializer_1002 = None
        getitem_219 = com_microsoft__model_layers_27_post_attention_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_27_post_attention_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_1003 = self.initializers.onnx_initializer_1003
        com_microsoft__model_layers_28_input_layernorm_skip_layer_norm = self.com_microsoft__model_layers_28_input_layernorm_SkipLayerNorm(getitem_219, model_layers_27_mlp_down_proj_mat_mul, initializers_onnx_initializer_1003);  getitem_219 = model_layers_27_mlp_down_proj_mat_mul = initializers_onnx_initializer_1003 = None
        getitem_220 = com_microsoft__model_layers_28_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_1004 = self.initializers.onnx_initializer_1004
        model_layers_28_attn_q_proj_mat_mul = self.model_layers_28_attn_q_proj_MatMul(getitem_220, initializers_onnx_initializer_1004);  getitem_220 = initializers_onnx_initializer_1004 = None
        getitem_221 = com_microsoft__model_layers_28_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_1005 = self.initializers.onnx_initializer_1005
        model_layers_28_attn_k_proj_mat_mul = self.model_layers_28_attn_k_proj_MatMul(getitem_221, initializers_onnx_initializer_1005);  getitem_221 = initializers_onnx_initializer_1005 = None
        getitem_222 = com_microsoft__model_layers_28_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_1006 = self.initializers.onnx_initializer_1006
        model_layers_28_attn_v_proj_mat_mul = self.model_layers_28_attn_v_proj_MatMul(getitem_222, initializers_onnx_initializer_1006);  getitem_222 = initializers_onnx_initializer_1006 = None
        initializers_onnx_initializer_1007 = self.initializers.onnx_initializer_1007
        initializers_onnx_initializer_1008 = self.initializers.onnx_initializer_1008
        com_microsoft__model_layers_28_attn_q_rotary_rotary_embedding = self.com_microsoft__model_layers_28_attn_q_rotary_RotaryEmbedding(model_layers_28_attn_q_proj_mat_mul, input_3, initializers_onnx_initializer_1007, initializers_onnx_initializer_1008);  model_layers_28_attn_q_proj_mat_mul = initializers_onnx_initializer_1007 = initializers_onnx_initializer_1008 = None
        initializers_onnx_initializer_1009 = self.initializers.onnx_initializer_1009
        initializers_onnx_initializer_1010 = self.initializers.onnx_initializer_1010
        com_microsoft__model_layers_28_attn_k_rotary_rotary_embedding = self.com_microsoft__model_layers_28_attn_k_rotary_RotaryEmbedding(model_layers_28_attn_k_proj_mat_mul, input_3, initializers_onnx_initializer_1009, initializers_onnx_initializer_1010);  model_layers_28_attn_k_proj_mat_mul = initializers_onnx_initializer_1009 = initializers_onnx_initializer_1010 = None
        initializers_onnx_initializer_1011 = self.initializers.onnx_initializer_1011
        model_layers_28_attn_v_proj_repeat_kv_reshape_1 = self.model_layers_28_attn_v_proj_repeat_kv_Reshape_1(model_layers_28_attn_v_proj_mat_mul, initializers_onnx_initializer_1011);  model_layers_28_attn_v_proj_mat_mul = initializers_onnx_initializer_1011 = None
        initializers_onnx_initializer_1012 = self.initializers.onnx_initializer_1012
        model_layers_28_attn_k_proj_repeat_kv_reshape_1 = self.model_layers_28_attn_k_proj_repeat_kv_Reshape_1(com_microsoft__model_layers_28_attn_k_rotary_rotary_embedding, initializers_onnx_initializer_1012);  com_microsoft__model_layers_28_attn_k_rotary_rotary_embedding = initializers_onnx_initializer_1012 = None
        model_layers_28_attn_v_proj_repeat_kv_transpose_1 = self.model_layers_28_attn_v_proj_repeat_kv_Transpose_1(model_layers_28_attn_v_proj_repeat_kv_reshape_1);  model_layers_28_attn_v_proj_repeat_kv_reshape_1 = None
        model_layers_28_attn_k_proj_repeat_kv_transpose_1 = self.model_layers_28_attn_k_proj_repeat_kv_Transpose_1(model_layers_28_attn_k_proj_repeat_kv_reshape_1);  model_layers_28_attn_k_proj_repeat_kv_reshape_1 = None
        model_layers_28_attn_v_proj_repeat_kv_concat_1 = self.model_layers_28_attn_v_proj_repeat_kv_Concat_1(input_61, model_layers_28_attn_v_proj_repeat_kv_transpose_1);  input_61 = model_layers_28_attn_v_proj_repeat_kv_transpose_1 = None
        model_layers_28_attn_k_proj_repeat_kv_concat_1 = self.model_layers_28_attn_k_proj_repeat_kv_Concat_1(input_60, model_layers_28_attn_k_proj_repeat_kv_transpose_1);  input_60 = model_layers_28_attn_k_proj_repeat_kv_transpose_1 = None
        model_layers_28_attn_v_proj_repeat_kv_shape_1 = self.model_layers_28_attn_v_proj_repeat_kv_Shape_1(model_layers_28_attn_v_proj_repeat_kv_concat_1)
        model_layers_28_attn_v_proj_repeat_kv_unsqueeze_5 = self.model_layers_28_attn_v_proj_repeat_kv_Unsqueeze_5(model_layers_28_attn_v_proj_repeat_kv_concat_1)
        model_layers_28_attn_k_proj_repeat_kv_shape_1 = self.model_layers_28_attn_k_proj_repeat_kv_Shape_1(model_layers_28_attn_k_proj_repeat_kv_concat_1)
        model_layers_28_attn_k_proj_repeat_kv_unsqueeze_5 = self.model_layers_28_attn_k_proj_repeat_kv_Unsqueeze_5(model_layers_28_attn_k_proj_repeat_kv_concat_1)
        initializers_onnx_initializer_1013 = self.initializers.onnx_initializer_1013
        model_layers_28_attn_v_proj_repeat_kv_gather_1 = self.model_layers_28_attn_v_proj_repeat_kv_Gather_1(model_layers_28_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_1013);  initializers_onnx_initializer_1013 = None
        initializers_onnx_initializer_1014 = self.initializers.onnx_initializer_1014
        model_layers_28_attn_v_proj_repeat_kv_gather_3 = self.model_layers_28_attn_v_proj_repeat_kv_Gather_3(model_layers_28_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_1014);  model_layers_28_attn_v_proj_repeat_kv_shape_1 = initializers_onnx_initializer_1014 = None
        initializers_onnx_initializer_1015 = self.initializers.onnx_initializer_1015
        model_layers_28_attn_k_proj_repeat_kv_gather_1 = self.model_layers_28_attn_k_proj_repeat_kv_Gather_1(model_layers_28_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_1015);  initializers_onnx_initializer_1015 = None
        initializers_onnx_initializer_1016 = self.initializers.onnx_initializer_1016
        model_layers_28_attn_k_proj_repeat_kv_gather_3 = self.model_layers_28_attn_k_proj_repeat_kv_Gather_3(model_layers_28_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_1016);  model_layers_28_attn_k_proj_repeat_kv_shape_1 = initializers_onnx_initializer_1016 = None
        model_layers_28_attn_v_proj_repeat_kv_unsqueeze_1 = self.model_layers_28_attn_v_proj_repeat_kv_Unsqueeze_1(model_layers_28_attn_v_proj_repeat_kv_gather_1);  model_layers_28_attn_v_proj_repeat_kv_gather_1 = None
        model_layers_28_attn_v_proj_repeat_kv_unsqueeze_3 = self.model_layers_28_attn_v_proj_repeat_kv_Unsqueeze_3(model_layers_28_attn_v_proj_repeat_kv_gather_3);  model_layers_28_attn_v_proj_repeat_kv_gather_3 = None
        model_layers_28_attn_k_proj_repeat_kv_unsqueeze_1 = self.model_layers_28_attn_k_proj_repeat_kv_Unsqueeze_1(model_layers_28_attn_k_proj_repeat_kv_gather_1);  model_layers_28_attn_k_proj_repeat_kv_gather_1 = None
        model_layers_28_attn_k_proj_repeat_kv_unsqueeze_3 = self.model_layers_28_attn_k_proj_repeat_kv_Unsqueeze_3(model_layers_28_attn_k_proj_repeat_kv_gather_3);  model_layers_28_attn_k_proj_repeat_kv_gather_3 = None
        initializers_onnx_initializer_1017 = self.initializers.onnx_initializer_1017
        initializers_onnx_initializer_1018 = self.initializers.onnx_initializer_1018
        initializers_onnx_initializer_1019 = self.initializers.onnx_initializer_1019
        model_layers_28_attn_v_proj_repeat_kv_concat_2 = self.model_layers_28_attn_v_proj_repeat_kv_Concat_2(model_layers_28_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_1017, initializers_onnx_initializer_1018, model_layers_28_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_1019);  initializers_onnx_initializer_1017 = initializers_onnx_initializer_1018 = initializers_onnx_initializer_1019 = None
        initializers_onnx_initializer_1020 = self.initializers.onnx_initializer_1020
        initializers_onnx_initializer_1021 = self.initializers.onnx_initializer_1021
        model_layers_28_attn_v_proj_repeat_kv_concat_3 = self.model_layers_28_attn_v_proj_repeat_kv_Concat_3(model_layers_28_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_1020, model_layers_28_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_1021);  model_layers_28_attn_v_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_1020 = model_layers_28_attn_v_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_1021 = None
        initializers_onnx_initializer_1022 = self.initializers.onnx_initializer_1022
        initializers_onnx_initializer_1023 = self.initializers.onnx_initializer_1023
        initializers_onnx_initializer_1024 = self.initializers.onnx_initializer_1024
        model_layers_28_attn_k_proj_repeat_kv_concat_2 = self.model_layers_28_attn_k_proj_repeat_kv_Concat_2(model_layers_28_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_1022, initializers_onnx_initializer_1023, model_layers_28_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_1024);  initializers_onnx_initializer_1022 = initializers_onnx_initializer_1023 = initializers_onnx_initializer_1024 = None
        initializers_onnx_initializer_1025 = self.initializers.onnx_initializer_1025
        initializers_onnx_initializer_1026 = self.initializers.onnx_initializer_1026
        model_layers_28_attn_k_proj_repeat_kv_concat_3 = self.model_layers_28_attn_k_proj_repeat_kv_Concat_3(model_layers_28_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_1025, model_layers_28_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_1026);  model_layers_28_attn_k_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_1025 = model_layers_28_attn_k_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_1026 = None
        initializers_onnx_initializer_1027 = self.initializers.onnx_initializer_1027
        model_layers_28_attn_v_proj_repeat_kv_equal = self.model_layers_28_attn_v_proj_repeat_kv_Equal(model_layers_28_attn_v_proj_repeat_kv_concat_2, initializers_onnx_initializer_1027);  initializers_onnx_initializer_1027 = None
        initializers_onnx_initializer_1028 = self.initializers.onnx_initializer_1028
        model_layers_28_attn_k_proj_repeat_kv_equal = self.model_layers_28_attn_k_proj_repeat_kv_Equal(model_layers_28_attn_k_proj_repeat_kv_concat_2, initializers_onnx_initializer_1028);  initializers_onnx_initializer_1028 = None
        initializers_onnx_initializer_1029 = self.initializers.onnx_initializer_1029
        model_layers_28_attn_v_proj_repeat_kv_where = self.model_layers_28_attn_v_proj_repeat_kv_Where(model_layers_28_attn_v_proj_repeat_kv_equal, initializers_onnx_initializer_1029, model_layers_28_attn_v_proj_repeat_kv_concat_2);  model_layers_28_attn_v_proj_repeat_kv_equal = initializers_onnx_initializer_1029 = model_layers_28_attn_v_proj_repeat_kv_concat_2 = None
        initializers_onnx_initializer_1030 = self.initializers.onnx_initializer_1030
        model_layers_28_attn_k_proj_repeat_kv_where = self.model_layers_28_attn_k_proj_repeat_kv_Where(model_layers_28_attn_k_proj_repeat_kv_equal, initializers_onnx_initializer_1030, model_layers_28_attn_k_proj_repeat_kv_concat_2);  model_layers_28_attn_k_proj_repeat_kv_equal = initializers_onnx_initializer_1030 = model_layers_28_attn_k_proj_repeat_kv_concat_2 = None
        model_layers_28_attn_v_proj_repeat_kv_expand = self.model_layers_28_attn_v_proj_repeat_kv_Expand(model_layers_28_attn_v_proj_repeat_kv_unsqueeze_5, model_layers_28_attn_v_proj_repeat_kv_where);  model_layers_28_attn_v_proj_repeat_kv_unsqueeze_5 = model_layers_28_attn_v_proj_repeat_kv_where = None
        model_layers_28_attn_k_proj_repeat_kv_expand = self.model_layers_28_attn_k_proj_repeat_kv_Expand(model_layers_28_attn_k_proj_repeat_kv_unsqueeze_5, model_layers_28_attn_k_proj_repeat_kv_where);  model_layers_28_attn_k_proj_repeat_kv_unsqueeze_5 = model_layers_28_attn_k_proj_repeat_kv_where = None
        model_layers_28_attn_v_proj_repeat_kv_reshape_3 = self.model_layers_28_attn_v_proj_repeat_kv_Reshape_3(model_layers_28_attn_v_proj_repeat_kv_expand, model_layers_28_attn_v_proj_repeat_kv_concat_3);  model_layers_28_attn_v_proj_repeat_kv_expand = model_layers_28_attn_v_proj_repeat_kv_concat_3 = None
        model_layers_28_attn_k_proj_repeat_kv_reshape_3 = self.model_layers_28_attn_k_proj_repeat_kv_Reshape_3(model_layers_28_attn_k_proj_repeat_kv_expand, model_layers_28_attn_k_proj_repeat_kv_concat_3);  model_layers_28_attn_k_proj_repeat_kv_expand = model_layers_28_attn_k_proj_repeat_kv_concat_3 = None
        model_layers_28_attn_v_proj_repeat_kv_transpose_2 = self.model_layers_28_attn_v_proj_repeat_kv_Transpose_2(model_layers_28_attn_v_proj_repeat_kv_reshape_3);  model_layers_28_attn_v_proj_repeat_kv_reshape_3 = None
        model_layers_28_attn_k_proj_repeat_kv_transpose_2 = self.model_layers_28_attn_k_proj_repeat_kv_Transpose_2(model_layers_28_attn_k_proj_repeat_kv_reshape_3);  model_layers_28_attn_k_proj_repeat_kv_reshape_3 = None
        initializers_onnx_initializer_1031 = self.initializers.onnx_initializer_1031
        model_layers_28_attn_v_proj_repeat_kv_reshape_4 = self.model_layers_28_attn_v_proj_repeat_kv_Reshape_4(model_layers_28_attn_v_proj_repeat_kv_transpose_2, initializers_onnx_initializer_1031);  model_layers_28_attn_v_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_1031 = None
        initializers_onnx_initializer_1032 = self.initializers.onnx_initializer_1032
        model_layers_28_attn_k_proj_repeat_kv_reshape_4 = self.model_layers_28_attn_k_proj_repeat_kv_Reshape_4(model_layers_28_attn_k_proj_repeat_kv_transpose_2, initializers_onnx_initializer_1032);  model_layers_28_attn_k_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_1032 = None
        com_microsoft__model_layers_28_attn_multi_head_attention = self.com_microsoft__model_layers_28_attn_MultiHeadAttention(com_microsoft__model_layers_28_attn_q_rotary_rotary_embedding, model_layers_28_attn_k_proj_repeat_kv_reshape_4, model_layers_28_attn_v_proj_repeat_kv_reshape_4, attention_mask = model_attn_mask_reformat_tile);  com_microsoft__model_layers_28_attn_q_rotary_rotary_embedding = model_layers_28_attn_k_proj_repeat_kv_reshape_4 = model_layers_28_attn_v_proj_repeat_kv_reshape_4 = None
        getitem_223 = com_microsoft__model_layers_28_attn_multi_head_attention[0];  com_microsoft__model_layers_28_attn_multi_head_attention = None
        initializers_onnx_initializer_1033 = self.initializers.onnx_initializer_1033
        model_layers_28_attn_o_proj_mat_mul = self.model_layers_28_attn_o_proj_MatMul(getitem_223, initializers_onnx_initializer_1033);  getitem_223 = initializers_onnx_initializer_1033 = None
        getitem_224 = com_microsoft__model_layers_28_input_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_28_input_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_1034 = self.initializers.onnx_initializer_1034
        com_microsoft__model_layers_28_post_attention_layernorm_skip_layer_norm = self.com_microsoft__model_layers_28_post_attention_layernorm_SkipLayerNorm(getitem_224, model_layers_28_attn_o_proj_mat_mul, initializers_onnx_initializer_1034);  getitem_224 = model_layers_28_attn_o_proj_mat_mul = initializers_onnx_initializer_1034 = None
        getitem_225 = com_microsoft__model_layers_28_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_1035 = self.initializers.onnx_initializer_1035
        model_layers_28_mlp_gate_proj_mat_mul = self.model_layers_28_mlp_gate_proj_MatMul(getitem_225, initializers_onnx_initializer_1035);  getitem_225 = initializers_onnx_initializer_1035 = None
        getitem_226 = com_microsoft__model_layers_28_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_1036 = self.initializers.onnx_initializer_1036
        model_layers_28_mlp_up_proj_mat_mul = self.model_layers_28_mlp_up_proj_MatMul(getitem_226, initializers_onnx_initializer_1036);  getitem_226 = initializers_onnx_initializer_1036 = None
        model_layers_28_mlp_act_fn_sigmoid = self.model_layers_28_mlp_act_fn_Sigmoid(model_layers_28_mlp_gate_proj_mat_mul)
        model_layers_28_mlp_act_fn_mul = self.model_layers_28_mlp_act_fn_Mul(model_layers_28_mlp_gate_proj_mat_mul, model_layers_28_mlp_act_fn_sigmoid);  model_layers_28_mlp_gate_proj_mat_mul = model_layers_28_mlp_act_fn_sigmoid = None
        model_layers_28_mlp_mul = self.model_layers_28_mlp_Mul(model_layers_28_mlp_act_fn_mul, model_layers_28_mlp_up_proj_mat_mul);  model_layers_28_mlp_act_fn_mul = model_layers_28_mlp_up_proj_mat_mul = None
        initializers_onnx_initializer_1037 = self.initializers.onnx_initializer_1037
        model_layers_28_mlp_down_proj_mat_mul = self.model_layers_28_mlp_down_proj_MatMul(model_layers_28_mlp_mul, initializers_onnx_initializer_1037);  model_layers_28_mlp_mul = initializers_onnx_initializer_1037 = None
        getitem_227 = com_microsoft__model_layers_28_post_attention_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_28_post_attention_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_1038 = self.initializers.onnx_initializer_1038
        com_microsoft__model_layers_29_input_layernorm_skip_layer_norm = self.com_microsoft__model_layers_29_input_layernorm_SkipLayerNorm(getitem_227, model_layers_28_mlp_down_proj_mat_mul, initializers_onnx_initializer_1038);  getitem_227 = model_layers_28_mlp_down_proj_mat_mul = initializers_onnx_initializer_1038 = None
        getitem_228 = com_microsoft__model_layers_29_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_1039 = self.initializers.onnx_initializer_1039
        model_layers_29_attn_q_proj_mat_mul = self.model_layers_29_attn_q_proj_MatMul(getitem_228, initializers_onnx_initializer_1039);  getitem_228 = initializers_onnx_initializer_1039 = None
        getitem_229 = com_microsoft__model_layers_29_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_1040 = self.initializers.onnx_initializer_1040
        model_layers_29_attn_k_proj_mat_mul = self.model_layers_29_attn_k_proj_MatMul(getitem_229, initializers_onnx_initializer_1040);  getitem_229 = initializers_onnx_initializer_1040 = None
        getitem_230 = com_microsoft__model_layers_29_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_1041 = self.initializers.onnx_initializer_1041
        model_layers_29_attn_v_proj_mat_mul = self.model_layers_29_attn_v_proj_MatMul(getitem_230, initializers_onnx_initializer_1041);  getitem_230 = initializers_onnx_initializer_1041 = None
        initializers_onnx_initializer_1042 = self.initializers.onnx_initializer_1042
        initializers_onnx_initializer_1043 = self.initializers.onnx_initializer_1043
        com_microsoft__model_layers_29_attn_q_rotary_rotary_embedding = self.com_microsoft__model_layers_29_attn_q_rotary_RotaryEmbedding(model_layers_29_attn_q_proj_mat_mul, input_3, initializers_onnx_initializer_1042, initializers_onnx_initializer_1043);  model_layers_29_attn_q_proj_mat_mul = initializers_onnx_initializer_1042 = initializers_onnx_initializer_1043 = None
        initializers_onnx_initializer_1044 = self.initializers.onnx_initializer_1044
        initializers_onnx_initializer_1045 = self.initializers.onnx_initializer_1045
        com_microsoft__model_layers_29_attn_k_rotary_rotary_embedding = self.com_microsoft__model_layers_29_attn_k_rotary_RotaryEmbedding(model_layers_29_attn_k_proj_mat_mul, input_3, initializers_onnx_initializer_1044, initializers_onnx_initializer_1045);  model_layers_29_attn_k_proj_mat_mul = initializers_onnx_initializer_1044 = initializers_onnx_initializer_1045 = None
        initializers_onnx_initializer_1046 = self.initializers.onnx_initializer_1046
        model_layers_29_attn_v_proj_repeat_kv_reshape_1 = self.model_layers_29_attn_v_proj_repeat_kv_Reshape_1(model_layers_29_attn_v_proj_mat_mul, initializers_onnx_initializer_1046);  model_layers_29_attn_v_proj_mat_mul = initializers_onnx_initializer_1046 = None
        initializers_onnx_initializer_1047 = self.initializers.onnx_initializer_1047
        model_layers_29_attn_k_proj_repeat_kv_reshape_1 = self.model_layers_29_attn_k_proj_repeat_kv_Reshape_1(com_microsoft__model_layers_29_attn_k_rotary_rotary_embedding, initializers_onnx_initializer_1047);  com_microsoft__model_layers_29_attn_k_rotary_rotary_embedding = initializers_onnx_initializer_1047 = None
        model_layers_29_attn_v_proj_repeat_kv_transpose_1 = self.model_layers_29_attn_v_proj_repeat_kv_Transpose_1(model_layers_29_attn_v_proj_repeat_kv_reshape_1);  model_layers_29_attn_v_proj_repeat_kv_reshape_1 = None
        model_layers_29_attn_k_proj_repeat_kv_transpose_1 = self.model_layers_29_attn_k_proj_repeat_kv_Transpose_1(model_layers_29_attn_k_proj_repeat_kv_reshape_1);  model_layers_29_attn_k_proj_repeat_kv_reshape_1 = None
        model_layers_29_attn_v_proj_repeat_kv_concat_1 = self.model_layers_29_attn_v_proj_repeat_kv_Concat_1(input_63, model_layers_29_attn_v_proj_repeat_kv_transpose_1);  input_63 = model_layers_29_attn_v_proj_repeat_kv_transpose_1 = None
        model_layers_29_attn_k_proj_repeat_kv_concat_1 = self.model_layers_29_attn_k_proj_repeat_kv_Concat_1(input_62, model_layers_29_attn_k_proj_repeat_kv_transpose_1);  input_62 = model_layers_29_attn_k_proj_repeat_kv_transpose_1 = None
        model_layers_29_attn_v_proj_repeat_kv_shape_1 = self.model_layers_29_attn_v_proj_repeat_kv_Shape_1(model_layers_29_attn_v_proj_repeat_kv_concat_1)
        model_layers_29_attn_v_proj_repeat_kv_unsqueeze_5 = self.model_layers_29_attn_v_proj_repeat_kv_Unsqueeze_5(model_layers_29_attn_v_proj_repeat_kv_concat_1)
        model_layers_29_attn_k_proj_repeat_kv_shape_1 = self.model_layers_29_attn_k_proj_repeat_kv_Shape_1(model_layers_29_attn_k_proj_repeat_kv_concat_1)
        model_layers_29_attn_k_proj_repeat_kv_unsqueeze_5 = self.model_layers_29_attn_k_proj_repeat_kv_Unsqueeze_5(model_layers_29_attn_k_proj_repeat_kv_concat_1)
        initializers_onnx_initializer_1048 = self.initializers.onnx_initializer_1048
        model_layers_29_attn_v_proj_repeat_kv_gather_1 = self.model_layers_29_attn_v_proj_repeat_kv_Gather_1(model_layers_29_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_1048);  initializers_onnx_initializer_1048 = None
        initializers_onnx_initializer_1049 = self.initializers.onnx_initializer_1049
        model_layers_29_attn_v_proj_repeat_kv_gather_3 = self.model_layers_29_attn_v_proj_repeat_kv_Gather_3(model_layers_29_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_1049);  model_layers_29_attn_v_proj_repeat_kv_shape_1 = initializers_onnx_initializer_1049 = None
        initializers_onnx_initializer_1050 = self.initializers.onnx_initializer_1050
        model_layers_29_attn_k_proj_repeat_kv_gather_1 = self.model_layers_29_attn_k_proj_repeat_kv_Gather_1(model_layers_29_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_1050);  initializers_onnx_initializer_1050 = None
        initializers_onnx_initializer_1051 = self.initializers.onnx_initializer_1051
        model_layers_29_attn_k_proj_repeat_kv_gather_3 = self.model_layers_29_attn_k_proj_repeat_kv_Gather_3(model_layers_29_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_1051);  model_layers_29_attn_k_proj_repeat_kv_shape_1 = initializers_onnx_initializer_1051 = None
        model_layers_29_attn_v_proj_repeat_kv_unsqueeze_1 = self.model_layers_29_attn_v_proj_repeat_kv_Unsqueeze_1(model_layers_29_attn_v_proj_repeat_kv_gather_1);  model_layers_29_attn_v_proj_repeat_kv_gather_1 = None
        model_layers_29_attn_v_proj_repeat_kv_unsqueeze_3 = self.model_layers_29_attn_v_proj_repeat_kv_Unsqueeze_3(model_layers_29_attn_v_proj_repeat_kv_gather_3);  model_layers_29_attn_v_proj_repeat_kv_gather_3 = None
        model_layers_29_attn_k_proj_repeat_kv_unsqueeze_1 = self.model_layers_29_attn_k_proj_repeat_kv_Unsqueeze_1(model_layers_29_attn_k_proj_repeat_kv_gather_1);  model_layers_29_attn_k_proj_repeat_kv_gather_1 = None
        model_layers_29_attn_k_proj_repeat_kv_unsqueeze_3 = self.model_layers_29_attn_k_proj_repeat_kv_Unsqueeze_3(model_layers_29_attn_k_proj_repeat_kv_gather_3);  model_layers_29_attn_k_proj_repeat_kv_gather_3 = None
        initializers_onnx_initializer_1052 = self.initializers.onnx_initializer_1052
        initializers_onnx_initializer_1053 = self.initializers.onnx_initializer_1053
        initializers_onnx_initializer_1054 = self.initializers.onnx_initializer_1054
        model_layers_29_attn_v_proj_repeat_kv_concat_2 = self.model_layers_29_attn_v_proj_repeat_kv_Concat_2(model_layers_29_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_1052, initializers_onnx_initializer_1053, model_layers_29_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_1054);  initializers_onnx_initializer_1052 = initializers_onnx_initializer_1053 = initializers_onnx_initializer_1054 = None
        initializers_onnx_initializer_1055 = self.initializers.onnx_initializer_1055
        initializers_onnx_initializer_1056 = self.initializers.onnx_initializer_1056
        model_layers_29_attn_v_proj_repeat_kv_concat_3 = self.model_layers_29_attn_v_proj_repeat_kv_Concat_3(model_layers_29_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_1055, model_layers_29_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_1056);  model_layers_29_attn_v_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_1055 = model_layers_29_attn_v_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_1056 = None
        initializers_onnx_initializer_1057 = self.initializers.onnx_initializer_1057
        initializers_onnx_initializer_1058 = self.initializers.onnx_initializer_1058
        initializers_onnx_initializer_1059 = self.initializers.onnx_initializer_1059
        model_layers_29_attn_k_proj_repeat_kv_concat_2 = self.model_layers_29_attn_k_proj_repeat_kv_Concat_2(model_layers_29_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_1057, initializers_onnx_initializer_1058, model_layers_29_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_1059);  initializers_onnx_initializer_1057 = initializers_onnx_initializer_1058 = initializers_onnx_initializer_1059 = None
        initializers_onnx_initializer_1060 = self.initializers.onnx_initializer_1060
        initializers_onnx_initializer_1061 = self.initializers.onnx_initializer_1061
        model_layers_29_attn_k_proj_repeat_kv_concat_3 = self.model_layers_29_attn_k_proj_repeat_kv_Concat_3(model_layers_29_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_1060, model_layers_29_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_1061);  model_layers_29_attn_k_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_1060 = model_layers_29_attn_k_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_1061 = None
        initializers_onnx_initializer_1062 = self.initializers.onnx_initializer_1062
        model_layers_29_attn_v_proj_repeat_kv_equal = self.model_layers_29_attn_v_proj_repeat_kv_Equal(model_layers_29_attn_v_proj_repeat_kv_concat_2, initializers_onnx_initializer_1062);  initializers_onnx_initializer_1062 = None
        initializers_onnx_initializer_1063 = self.initializers.onnx_initializer_1063
        model_layers_29_attn_k_proj_repeat_kv_equal = self.model_layers_29_attn_k_proj_repeat_kv_Equal(model_layers_29_attn_k_proj_repeat_kv_concat_2, initializers_onnx_initializer_1063);  initializers_onnx_initializer_1063 = None
        initializers_onnx_initializer_1064 = self.initializers.onnx_initializer_1064
        model_layers_29_attn_v_proj_repeat_kv_where = self.model_layers_29_attn_v_proj_repeat_kv_Where(model_layers_29_attn_v_proj_repeat_kv_equal, initializers_onnx_initializer_1064, model_layers_29_attn_v_proj_repeat_kv_concat_2);  model_layers_29_attn_v_proj_repeat_kv_equal = initializers_onnx_initializer_1064 = model_layers_29_attn_v_proj_repeat_kv_concat_2 = None
        initializers_onnx_initializer_1065 = self.initializers.onnx_initializer_1065
        model_layers_29_attn_k_proj_repeat_kv_where = self.model_layers_29_attn_k_proj_repeat_kv_Where(model_layers_29_attn_k_proj_repeat_kv_equal, initializers_onnx_initializer_1065, model_layers_29_attn_k_proj_repeat_kv_concat_2);  model_layers_29_attn_k_proj_repeat_kv_equal = initializers_onnx_initializer_1065 = model_layers_29_attn_k_proj_repeat_kv_concat_2 = None
        model_layers_29_attn_v_proj_repeat_kv_expand = self.model_layers_29_attn_v_proj_repeat_kv_Expand(model_layers_29_attn_v_proj_repeat_kv_unsqueeze_5, model_layers_29_attn_v_proj_repeat_kv_where);  model_layers_29_attn_v_proj_repeat_kv_unsqueeze_5 = model_layers_29_attn_v_proj_repeat_kv_where = None
        model_layers_29_attn_k_proj_repeat_kv_expand = self.model_layers_29_attn_k_proj_repeat_kv_Expand(model_layers_29_attn_k_proj_repeat_kv_unsqueeze_5, model_layers_29_attn_k_proj_repeat_kv_where);  model_layers_29_attn_k_proj_repeat_kv_unsqueeze_5 = model_layers_29_attn_k_proj_repeat_kv_where = None
        model_layers_29_attn_v_proj_repeat_kv_reshape_3 = self.model_layers_29_attn_v_proj_repeat_kv_Reshape_3(model_layers_29_attn_v_proj_repeat_kv_expand, model_layers_29_attn_v_proj_repeat_kv_concat_3);  model_layers_29_attn_v_proj_repeat_kv_expand = model_layers_29_attn_v_proj_repeat_kv_concat_3 = None
        model_layers_29_attn_k_proj_repeat_kv_reshape_3 = self.model_layers_29_attn_k_proj_repeat_kv_Reshape_3(model_layers_29_attn_k_proj_repeat_kv_expand, model_layers_29_attn_k_proj_repeat_kv_concat_3);  model_layers_29_attn_k_proj_repeat_kv_expand = model_layers_29_attn_k_proj_repeat_kv_concat_3 = None
        model_layers_29_attn_v_proj_repeat_kv_transpose_2 = self.model_layers_29_attn_v_proj_repeat_kv_Transpose_2(model_layers_29_attn_v_proj_repeat_kv_reshape_3);  model_layers_29_attn_v_proj_repeat_kv_reshape_3 = None
        model_layers_29_attn_k_proj_repeat_kv_transpose_2 = self.model_layers_29_attn_k_proj_repeat_kv_Transpose_2(model_layers_29_attn_k_proj_repeat_kv_reshape_3);  model_layers_29_attn_k_proj_repeat_kv_reshape_3 = None
        initializers_onnx_initializer_1066 = self.initializers.onnx_initializer_1066
        model_layers_29_attn_v_proj_repeat_kv_reshape_4 = self.model_layers_29_attn_v_proj_repeat_kv_Reshape_4(model_layers_29_attn_v_proj_repeat_kv_transpose_2, initializers_onnx_initializer_1066);  model_layers_29_attn_v_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_1066 = None
        initializers_onnx_initializer_1067 = self.initializers.onnx_initializer_1067
        model_layers_29_attn_k_proj_repeat_kv_reshape_4 = self.model_layers_29_attn_k_proj_repeat_kv_Reshape_4(model_layers_29_attn_k_proj_repeat_kv_transpose_2, initializers_onnx_initializer_1067);  model_layers_29_attn_k_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_1067 = None
        com_microsoft__model_layers_29_attn_multi_head_attention = self.com_microsoft__model_layers_29_attn_MultiHeadAttention(com_microsoft__model_layers_29_attn_q_rotary_rotary_embedding, model_layers_29_attn_k_proj_repeat_kv_reshape_4, model_layers_29_attn_v_proj_repeat_kv_reshape_4, attention_mask = model_attn_mask_reformat_tile);  com_microsoft__model_layers_29_attn_q_rotary_rotary_embedding = model_layers_29_attn_k_proj_repeat_kv_reshape_4 = model_layers_29_attn_v_proj_repeat_kv_reshape_4 = None
        getitem_231 = com_microsoft__model_layers_29_attn_multi_head_attention[0];  com_microsoft__model_layers_29_attn_multi_head_attention = None
        initializers_onnx_initializer_1068 = self.initializers.onnx_initializer_1068
        model_layers_29_attn_o_proj_mat_mul = self.model_layers_29_attn_o_proj_MatMul(getitem_231, initializers_onnx_initializer_1068);  getitem_231 = initializers_onnx_initializer_1068 = None
        getitem_232 = com_microsoft__model_layers_29_input_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_29_input_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_1069 = self.initializers.onnx_initializer_1069
        com_microsoft__model_layers_29_post_attention_layernorm_skip_layer_norm = self.com_microsoft__model_layers_29_post_attention_layernorm_SkipLayerNorm(getitem_232, model_layers_29_attn_o_proj_mat_mul, initializers_onnx_initializer_1069);  getitem_232 = model_layers_29_attn_o_proj_mat_mul = initializers_onnx_initializer_1069 = None
        getitem_233 = com_microsoft__model_layers_29_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_1070 = self.initializers.onnx_initializer_1070
        model_layers_29_mlp_gate_proj_mat_mul = self.model_layers_29_mlp_gate_proj_MatMul(getitem_233, initializers_onnx_initializer_1070);  getitem_233 = initializers_onnx_initializer_1070 = None
        getitem_234 = com_microsoft__model_layers_29_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_1071 = self.initializers.onnx_initializer_1071
        model_layers_29_mlp_up_proj_mat_mul = self.model_layers_29_mlp_up_proj_MatMul(getitem_234, initializers_onnx_initializer_1071);  getitem_234 = initializers_onnx_initializer_1071 = None
        model_layers_29_mlp_act_fn_sigmoid = self.model_layers_29_mlp_act_fn_Sigmoid(model_layers_29_mlp_gate_proj_mat_mul)
        model_layers_29_mlp_act_fn_mul = self.model_layers_29_mlp_act_fn_Mul(model_layers_29_mlp_gate_proj_mat_mul, model_layers_29_mlp_act_fn_sigmoid);  model_layers_29_mlp_gate_proj_mat_mul = model_layers_29_mlp_act_fn_sigmoid = None
        model_layers_29_mlp_mul = self.model_layers_29_mlp_Mul(model_layers_29_mlp_act_fn_mul, model_layers_29_mlp_up_proj_mat_mul);  model_layers_29_mlp_act_fn_mul = model_layers_29_mlp_up_proj_mat_mul = None
        initializers_onnx_initializer_1072 = self.initializers.onnx_initializer_1072
        model_layers_29_mlp_down_proj_mat_mul = self.model_layers_29_mlp_down_proj_MatMul(model_layers_29_mlp_mul, initializers_onnx_initializer_1072);  model_layers_29_mlp_mul = initializers_onnx_initializer_1072 = None
        getitem_235 = com_microsoft__model_layers_29_post_attention_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_29_post_attention_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_1073 = self.initializers.onnx_initializer_1073
        com_microsoft__model_layers_30_input_layernorm_skip_layer_norm = self.com_microsoft__model_layers_30_input_layernorm_SkipLayerNorm(getitem_235, model_layers_29_mlp_down_proj_mat_mul, initializers_onnx_initializer_1073);  getitem_235 = model_layers_29_mlp_down_proj_mat_mul = initializers_onnx_initializer_1073 = None
        getitem_236 = com_microsoft__model_layers_30_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_1074 = self.initializers.onnx_initializer_1074
        model_layers_30_attn_q_proj_mat_mul = self.model_layers_30_attn_q_proj_MatMul(getitem_236, initializers_onnx_initializer_1074);  getitem_236 = initializers_onnx_initializer_1074 = None
        getitem_237 = com_microsoft__model_layers_30_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_1075 = self.initializers.onnx_initializer_1075
        model_layers_30_attn_k_proj_mat_mul = self.model_layers_30_attn_k_proj_MatMul(getitem_237, initializers_onnx_initializer_1075);  getitem_237 = initializers_onnx_initializer_1075 = None
        getitem_238 = com_microsoft__model_layers_30_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_1076 = self.initializers.onnx_initializer_1076
        model_layers_30_attn_v_proj_mat_mul = self.model_layers_30_attn_v_proj_MatMul(getitem_238, initializers_onnx_initializer_1076);  getitem_238 = initializers_onnx_initializer_1076 = None
        initializers_onnx_initializer_1077 = self.initializers.onnx_initializer_1077
        initializers_onnx_initializer_1078 = self.initializers.onnx_initializer_1078
        com_microsoft__model_layers_30_attn_q_rotary_rotary_embedding = self.com_microsoft__model_layers_30_attn_q_rotary_RotaryEmbedding(model_layers_30_attn_q_proj_mat_mul, input_3, initializers_onnx_initializer_1077, initializers_onnx_initializer_1078);  model_layers_30_attn_q_proj_mat_mul = initializers_onnx_initializer_1077 = initializers_onnx_initializer_1078 = None
        initializers_onnx_initializer_1079 = self.initializers.onnx_initializer_1079
        initializers_onnx_initializer_1080 = self.initializers.onnx_initializer_1080
        com_microsoft__model_layers_30_attn_k_rotary_rotary_embedding = self.com_microsoft__model_layers_30_attn_k_rotary_RotaryEmbedding(model_layers_30_attn_k_proj_mat_mul, input_3, initializers_onnx_initializer_1079, initializers_onnx_initializer_1080);  model_layers_30_attn_k_proj_mat_mul = initializers_onnx_initializer_1079 = initializers_onnx_initializer_1080 = None
        initializers_onnx_initializer_1081 = self.initializers.onnx_initializer_1081
        model_layers_30_attn_v_proj_repeat_kv_reshape_1 = self.model_layers_30_attn_v_proj_repeat_kv_Reshape_1(model_layers_30_attn_v_proj_mat_mul, initializers_onnx_initializer_1081);  model_layers_30_attn_v_proj_mat_mul = initializers_onnx_initializer_1081 = None
        initializers_onnx_initializer_1082 = self.initializers.onnx_initializer_1082
        model_layers_30_attn_k_proj_repeat_kv_reshape_1 = self.model_layers_30_attn_k_proj_repeat_kv_Reshape_1(com_microsoft__model_layers_30_attn_k_rotary_rotary_embedding, initializers_onnx_initializer_1082);  com_microsoft__model_layers_30_attn_k_rotary_rotary_embedding = initializers_onnx_initializer_1082 = None
        model_layers_30_attn_v_proj_repeat_kv_transpose_1 = self.model_layers_30_attn_v_proj_repeat_kv_Transpose_1(model_layers_30_attn_v_proj_repeat_kv_reshape_1);  model_layers_30_attn_v_proj_repeat_kv_reshape_1 = None
        model_layers_30_attn_k_proj_repeat_kv_transpose_1 = self.model_layers_30_attn_k_proj_repeat_kv_Transpose_1(model_layers_30_attn_k_proj_repeat_kv_reshape_1);  model_layers_30_attn_k_proj_repeat_kv_reshape_1 = None
        model_layers_30_attn_v_proj_repeat_kv_concat_1 = self.model_layers_30_attn_v_proj_repeat_kv_Concat_1(input_65, model_layers_30_attn_v_proj_repeat_kv_transpose_1);  input_65 = model_layers_30_attn_v_proj_repeat_kv_transpose_1 = None
        model_layers_30_attn_k_proj_repeat_kv_concat_1 = self.model_layers_30_attn_k_proj_repeat_kv_Concat_1(input_64, model_layers_30_attn_k_proj_repeat_kv_transpose_1);  input_64 = model_layers_30_attn_k_proj_repeat_kv_transpose_1 = None
        model_layers_30_attn_v_proj_repeat_kv_shape_1 = self.model_layers_30_attn_v_proj_repeat_kv_Shape_1(model_layers_30_attn_v_proj_repeat_kv_concat_1)
        model_layers_30_attn_v_proj_repeat_kv_unsqueeze_5 = self.model_layers_30_attn_v_proj_repeat_kv_Unsqueeze_5(model_layers_30_attn_v_proj_repeat_kv_concat_1)
        model_layers_30_attn_k_proj_repeat_kv_shape_1 = self.model_layers_30_attn_k_proj_repeat_kv_Shape_1(model_layers_30_attn_k_proj_repeat_kv_concat_1)
        model_layers_30_attn_k_proj_repeat_kv_unsqueeze_5 = self.model_layers_30_attn_k_proj_repeat_kv_Unsqueeze_5(model_layers_30_attn_k_proj_repeat_kv_concat_1)
        initializers_onnx_initializer_1083 = self.initializers.onnx_initializer_1083
        model_layers_30_attn_v_proj_repeat_kv_gather_1 = self.model_layers_30_attn_v_proj_repeat_kv_Gather_1(model_layers_30_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_1083);  initializers_onnx_initializer_1083 = None
        initializers_onnx_initializer_1084 = self.initializers.onnx_initializer_1084
        model_layers_30_attn_v_proj_repeat_kv_gather_3 = self.model_layers_30_attn_v_proj_repeat_kv_Gather_3(model_layers_30_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_1084);  model_layers_30_attn_v_proj_repeat_kv_shape_1 = initializers_onnx_initializer_1084 = None
        initializers_onnx_initializer_1085 = self.initializers.onnx_initializer_1085
        model_layers_30_attn_k_proj_repeat_kv_gather_1 = self.model_layers_30_attn_k_proj_repeat_kv_Gather_1(model_layers_30_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_1085);  initializers_onnx_initializer_1085 = None
        initializers_onnx_initializer_1086 = self.initializers.onnx_initializer_1086
        model_layers_30_attn_k_proj_repeat_kv_gather_3 = self.model_layers_30_attn_k_proj_repeat_kv_Gather_3(model_layers_30_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_1086);  model_layers_30_attn_k_proj_repeat_kv_shape_1 = initializers_onnx_initializer_1086 = None
        model_layers_30_attn_v_proj_repeat_kv_unsqueeze_1 = self.model_layers_30_attn_v_proj_repeat_kv_Unsqueeze_1(model_layers_30_attn_v_proj_repeat_kv_gather_1);  model_layers_30_attn_v_proj_repeat_kv_gather_1 = None
        model_layers_30_attn_v_proj_repeat_kv_unsqueeze_3 = self.model_layers_30_attn_v_proj_repeat_kv_Unsqueeze_3(model_layers_30_attn_v_proj_repeat_kv_gather_3);  model_layers_30_attn_v_proj_repeat_kv_gather_3 = None
        model_layers_30_attn_k_proj_repeat_kv_unsqueeze_1 = self.model_layers_30_attn_k_proj_repeat_kv_Unsqueeze_1(model_layers_30_attn_k_proj_repeat_kv_gather_1);  model_layers_30_attn_k_proj_repeat_kv_gather_1 = None
        model_layers_30_attn_k_proj_repeat_kv_unsqueeze_3 = self.model_layers_30_attn_k_proj_repeat_kv_Unsqueeze_3(model_layers_30_attn_k_proj_repeat_kv_gather_3);  model_layers_30_attn_k_proj_repeat_kv_gather_3 = None
        initializers_onnx_initializer_1087 = self.initializers.onnx_initializer_1087
        initializers_onnx_initializer_1088 = self.initializers.onnx_initializer_1088
        initializers_onnx_initializer_1089 = self.initializers.onnx_initializer_1089
        model_layers_30_attn_v_proj_repeat_kv_concat_2 = self.model_layers_30_attn_v_proj_repeat_kv_Concat_2(model_layers_30_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_1087, initializers_onnx_initializer_1088, model_layers_30_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_1089);  initializers_onnx_initializer_1087 = initializers_onnx_initializer_1088 = initializers_onnx_initializer_1089 = None
        initializers_onnx_initializer_1090 = self.initializers.onnx_initializer_1090
        initializers_onnx_initializer_1091 = self.initializers.onnx_initializer_1091
        model_layers_30_attn_v_proj_repeat_kv_concat_3 = self.model_layers_30_attn_v_proj_repeat_kv_Concat_3(model_layers_30_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_1090, model_layers_30_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_1091);  model_layers_30_attn_v_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_1090 = model_layers_30_attn_v_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_1091 = None
        initializers_onnx_initializer_1092 = self.initializers.onnx_initializer_1092
        initializers_onnx_initializer_1093 = self.initializers.onnx_initializer_1093
        initializers_onnx_initializer_1094 = self.initializers.onnx_initializer_1094
        model_layers_30_attn_k_proj_repeat_kv_concat_2 = self.model_layers_30_attn_k_proj_repeat_kv_Concat_2(model_layers_30_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_1092, initializers_onnx_initializer_1093, model_layers_30_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_1094);  initializers_onnx_initializer_1092 = initializers_onnx_initializer_1093 = initializers_onnx_initializer_1094 = None
        initializers_onnx_initializer_1095 = self.initializers.onnx_initializer_1095
        initializers_onnx_initializer_1096 = self.initializers.onnx_initializer_1096
        model_layers_30_attn_k_proj_repeat_kv_concat_3 = self.model_layers_30_attn_k_proj_repeat_kv_Concat_3(model_layers_30_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_1095, model_layers_30_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_1096);  model_layers_30_attn_k_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_1095 = model_layers_30_attn_k_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_1096 = None
        initializers_onnx_initializer_1097 = self.initializers.onnx_initializer_1097
        model_layers_30_attn_v_proj_repeat_kv_equal = self.model_layers_30_attn_v_proj_repeat_kv_Equal(model_layers_30_attn_v_proj_repeat_kv_concat_2, initializers_onnx_initializer_1097);  initializers_onnx_initializer_1097 = None
        initializers_onnx_initializer_1098 = self.initializers.onnx_initializer_1098
        model_layers_30_attn_k_proj_repeat_kv_equal = self.model_layers_30_attn_k_proj_repeat_kv_Equal(model_layers_30_attn_k_proj_repeat_kv_concat_2, initializers_onnx_initializer_1098);  initializers_onnx_initializer_1098 = None
        initializers_onnx_initializer_1099 = self.initializers.onnx_initializer_1099
        model_layers_30_attn_v_proj_repeat_kv_where = self.model_layers_30_attn_v_proj_repeat_kv_Where(model_layers_30_attn_v_proj_repeat_kv_equal, initializers_onnx_initializer_1099, model_layers_30_attn_v_proj_repeat_kv_concat_2);  model_layers_30_attn_v_proj_repeat_kv_equal = initializers_onnx_initializer_1099 = model_layers_30_attn_v_proj_repeat_kv_concat_2 = None
        initializers_onnx_initializer_1100 = self.initializers.onnx_initializer_1100
        model_layers_30_attn_k_proj_repeat_kv_where = self.model_layers_30_attn_k_proj_repeat_kv_Where(model_layers_30_attn_k_proj_repeat_kv_equal, initializers_onnx_initializer_1100, model_layers_30_attn_k_proj_repeat_kv_concat_2);  model_layers_30_attn_k_proj_repeat_kv_equal = initializers_onnx_initializer_1100 = model_layers_30_attn_k_proj_repeat_kv_concat_2 = None
        model_layers_30_attn_v_proj_repeat_kv_expand = self.model_layers_30_attn_v_proj_repeat_kv_Expand(model_layers_30_attn_v_proj_repeat_kv_unsqueeze_5, model_layers_30_attn_v_proj_repeat_kv_where);  model_layers_30_attn_v_proj_repeat_kv_unsqueeze_5 = model_layers_30_attn_v_proj_repeat_kv_where = None
        model_layers_30_attn_k_proj_repeat_kv_expand = self.model_layers_30_attn_k_proj_repeat_kv_Expand(model_layers_30_attn_k_proj_repeat_kv_unsqueeze_5, model_layers_30_attn_k_proj_repeat_kv_where);  model_layers_30_attn_k_proj_repeat_kv_unsqueeze_5 = model_layers_30_attn_k_proj_repeat_kv_where = None
        model_layers_30_attn_v_proj_repeat_kv_reshape_3 = self.model_layers_30_attn_v_proj_repeat_kv_Reshape_3(model_layers_30_attn_v_proj_repeat_kv_expand, model_layers_30_attn_v_proj_repeat_kv_concat_3);  model_layers_30_attn_v_proj_repeat_kv_expand = model_layers_30_attn_v_proj_repeat_kv_concat_3 = None
        model_layers_30_attn_k_proj_repeat_kv_reshape_3 = self.model_layers_30_attn_k_proj_repeat_kv_Reshape_3(model_layers_30_attn_k_proj_repeat_kv_expand, model_layers_30_attn_k_proj_repeat_kv_concat_3);  model_layers_30_attn_k_proj_repeat_kv_expand = model_layers_30_attn_k_proj_repeat_kv_concat_3 = None
        model_layers_30_attn_v_proj_repeat_kv_transpose_2 = self.model_layers_30_attn_v_proj_repeat_kv_Transpose_2(model_layers_30_attn_v_proj_repeat_kv_reshape_3);  model_layers_30_attn_v_proj_repeat_kv_reshape_3 = None
        model_layers_30_attn_k_proj_repeat_kv_transpose_2 = self.model_layers_30_attn_k_proj_repeat_kv_Transpose_2(model_layers_30_attn_k_proj_repeat_kv_reshape_3);  model_layers_30_attn_k_proj_repeat_kv_reshape_3 = None
        initializers_onnx_initializer_1101 = self.initializers.onnx_initializer_1101
        model_layers_30_attn_v_proj_repeat_kv_reshape_4 = self.model_layers_30_attn_v_proj_repeat_kv_Reshape_4(model_layers_30_attn_v_proj_repeat_kv_transpose_2, initializers_onnx_initializer_1101);  model_layers_30_attn_v_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_1101 = None
        initializers_onnx_initializer_1102 = self.initializers.onnx_initializer_1102
        model_layers_30_attn_k_proj_repeat_kv_reshape_4 = self.model_layers_30_attn_k_proj_repeat_kv_Reshape_4(model_layers_30_attn_k_proj_repeat_kv_transpose_2, initializers_onnx_initializer_1102);  model_layers_30_attn_k_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_1102 = None
        com_microsoft__model_layers_30_attn_multi_head_attention = self.com_microsoft__model_layers_30_attn_MultiHeadAttention(com_microsoft__model_layers_30_attn_q_rotary_rotary_embedding, model_layers_30_attn_k_proj_repeat_kv_reshape_4, model_layers_30_attn_v_proj_repeat_kv_reshape_4, attention_mask = model_attn_mask_reformat_tile);  com_microsoft__model_layers_30_attn_q_rotary_rotary_embedding = model_layers_30_attn_k_proj_repeat_kv_reshape_4 = model_layers_30_attn_v_proj_repeat_kv_reshape_4 = None
        getitem_239 = com_microsoft__model_layers_30_attn_multi_head_attention[0];  com_microsoft__model_layers_30_attn_multi_head_attention = None
        initializers_onnx_initializer_1103 = self.initializers.onnx_initializer_1103
        model_layers_30_attn_o_proj_mat_mul = self.model_layers_30_attn_o_proj_MatMul(getitem_239, initializers_onnx_initializer_1103);  getitem_239 = initializers_onnx_initializer_1103 = None
        getitem_240 = com_microsoft__model_layers_30_input_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_30_input_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_1104 = self.initializers.onnx_initializer_1104
        com_microsoft__model_layers_30_post_attention_layernorm_skip_layer_norm = self.com_microsoft__model_layers_30_post_attention_layernorm_SkipLayerNorm(getitem_240, model_layers_30_attn_o_proj_mat_mul, initializers_onnx_initializer_1104);  getitem_240 = model_layers_30_attn_o_proj_mat_mul = initializers_onnx_initializer_1104 = None
        getitem_241 = com_microsoft__model_layers_30_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_1105 = self.initializers.onnx_initializer_1105
        model_layers_30_mlp_gate_proj_mat_mul = self.model_layers_30_mlp_gate_proj_MatMul(getitem_241, initializers_onnx_initializer_1105);  getitem_241 = initializers_onnx_initializer_1105 = None
        getitem_242 = com_microsoft__model_layers_30_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_1106 = self.initializers.onnx_initializer_1106
        model_layers_30_mlp_up_proj_mat_mul = self.model_layers_30_mlp_up_proj_MatMul(getitem_242, initializers_onnx_initializer_1106);  getitem_242 = initializers_onnx_initializer_1106 = None
        model_layers_30_mlp_act_fn_sigmoid = self.model_layers_30_mlp_act_fn_Sigmoid(model_layers_30_mlp_gate_proj_mat_mul)
        model_layers_30_mlp_act_fn_mul = self.model_layers_30_mlp_act_fn_Mul(model_layers_30_mlp_gate_proj_mat_mul, model_layers_30_mlp_act_fn_sigmoid);  model_layers_30_mlp_gate_proj_mat_mul = model_layers_30_mlp_act_fn_sigmoid = None
        model_layers_30_mlp_mul = self.model_layers_30_mlp_Mul(model_layers_30_mlp_act_fn_mul, model_layers_30_mlp_up_proj_mat_mul);  model_layers_30_mlp_act_fn_mul = model_layers_30_mlp_up_proj_mat_mul = None
        initializers_onnx_initializer_1107 = self.initializers.onnx_initializer_1107
        model_layers_30_mlp_down_proj_mat_mul = self.model_layers_30_mlp_down_proj_MatMul(model_layers_30_mlp_mul, initializers_onnx_initializer_1107);  model_layers_30_mlp_mul = initializers_onnx_initializer_1107 = None
        getitem_243 = com_microsoft__model_layers_30_post_attention_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_30_post_attention_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_1108 = self.initializers.onnx_initializer_1108
        com_microsoft__model_layers_31_input_layernorm_skip_layer_norm = self.com_microsoft__model_layers_31_input_layernorm_SkipLayerNorm(getitem_243, model_layers_30_mlp_down_proj_mat_mul, initializers_onnx_initializer_1108);  getitem_243 = model_layers_30_mlp_down_proj_mat_mul = initializers_onnx_initializer_1108 = None
        getitem_244 = com_microsoft__model_layers_31_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_1109 = self.initializers.onnx_initializer_1109
        model_layers_31_attn_q_proj_mat_mul = self.model_layers_31_attn_q_proj_MatMul(getitem_244, initializers_onnx_initializer_1109);  getitem_244 = initializers_onnx_initializer_1109 = None
        getitem_245 = com_microsoft__model_layers_31_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_1110 = self.initializers.onnx_initializer_1110
        model_layers_31_attn_k_proj_mat_mul = self.model_layers_31_attn_k_proj_MatMul(getitem_245, initializers_onnx_initializer_1110);  getitem_245 = initializers_onnx_initializer_1110 = None
        getitem_246 = com_microsoft__model_layers_31_input_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_1111 = self.initializers.onnx_initializer_1111
        model_layers_31_attn_v_proj_mat_mul = self.model_layers_31_attn_v_proj_MatMul(getitem_246, initializers_onnx_initializer_1111);  getitem_246 = initializers_onnx_initializer_1111 = None
        initializers_onnx_initializer_1112 = self.initializers.onnx_initializer_1112
        initializers_onnx_initializer_1113 = self.initializers.onnx_initializer_1113
        com_microsoft__model_layers_31_attn_q_rotary_rotary_embedding = self.com_microsoft__model_layers_31_attn_q_rotary_RotaryEmbedding(model_layers_31_attn_q_proj_mat_mul, input_3, initializers_onnx_initializer_1112, initializers_onnx_initializer_1113);  model_layers_31_attn_q_proj_mat_mul = initializers_onnx_initializer_1112 = initializers_onnx_initializer_1113 = None
        initializers_onnx_initializer_1114 = self.initializers.onnx_initializer_1114
        initializers_onnx_initializer_1115 = self.initializers.onnx_initializer_1115
        com_microsoft__model_layers_31_attn_k_rotary_rotary_embedding = self.com_microsoft__model_layers_31_attn_k_rotary_RotaryEmbedding(model_layers_31_attn_k_proj_mat_mul, input_3, initializers_onnx_initializer_1114, initializers_onnx_initializer_1115);  model_layers_31_attn_k_proj_mat_mul = input_3 = initializers_onnx_initializer_1114 = initializers_onnx_initializer_1115 = None
        initializers_onnx_initializer_1116 = self.initializers.onnx_initializer_1116
        model_layers_31_attn_v_proj_repeat_kv_reshape_1 = self.model_layers_31_attn_v_proj_repeat_kv_Reshape_1(model_layers_31_attn_v_proj_mat_mul, initializers_onnx_initializer_1116);  model_layers_31_attn_v_proj_mat_mul = initializers_onnx_initializer_1116 = None
        initializers_onnx_initializer_1117 = self.initializers.onnx_initializer_1117
        model_layers_31_attn_k_proj_repeat_kv_reshape_1 = self.model_layers_31_attn_k_proj_repeat_kv_Reshape_1(com_microsoft__model_layers_31_attn_k_rotary_rotary_embedding, initializers_onnx_initializer_1117);  com_microsoft__model_layers_31_attn_k_rotary_rotary_embedding = initializers_onnx_initializer_1117 = None
        model_layers_31_attn_v_proj_repeat_kv_transpose_1 = self.model_layers_31_attn_v_proj_repeat_kv_Transpose_1(model_layers_31_attn_v_proj_repeat_kv_reshape_1);  model_layers_31_attn_v_proj_repeat_kv_reshape_1 = None
        model_layers_31_attn_k_proj_repeat_kv_transpose_1 = self.model_layers_31_attn_k_proj_repeat_kv_Transpose_1(model_layers_31_attn_k_proj_repeat_kv_reshape_1);  model_layers_31_attn_k_proj_repeat_kv_reshape_1 = None
        model_layers_31_attn_v_proj_repeat_kv_concat_1 = self.model_layers_31_attn_v_proj_repeat_kv_Concat_1(input_67, model_layers_31_attn_v_proj_repeat_kv_transpose_1);  input_67 = model_layers_31_attn_v_proj_repeat_kv_transpose_1 = None
        model_layers_31_attn_k_proj_repeat_kv_concat_1 = self.model_layers_31_attn_k_proj_repeat_kv_Concat_1(input_66, model_layers_31_attn_k_proj_repeat_kv_transpose_1);  input_66 = model_layers_31_attn_k_proj_repeat_kv_transpose_1 = None
        model_layers_31_attn_v_proj_repeat_kv_shape_1 = self.model_layers_31_attn_v_proj_repeat_kv_Shape_1(model_layers_31_attn_v_proj_repeat_kv_concat_1)
        model_layers_31_attn_v_proj_repeat_kv_unsqueeze_5 = self.model_layers_31_attn_v_proj_repeat_kv_Unsqueeze_5(model_layers_31_attn_v_proj_repeat_kv_concat_1)
        model_layers_31_attn_k_proj_repeat_kv_shape_1 = self.model_layers_31_attn_k_proj_repeat_kv_Shape_1(model_layers_31_attn_k_proj_repeat_kv_concat_1)
        model_layers_31_attn_k_proj_repeat_kv_unsqueeze_5 = self.model_layers_31_attn_k_proj_repeat_kv_Unsqueeze_5(model_layers_31_attn_k_proj_repeat_kv_concat_1)
        initializers_onnx_initializer_1118 = self.initializers.onnx_initializer_1118
        model_layers_31_attn_v_proj_repeat_kv_gather_1 = self.model_layers_31_attn_v_proj_repeat_kv_Gather_1(model_layers_31_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_1118);  initializers_onnx_initializer_1118 = None
        initializers_onnx_initializer_1119 = self.initializers.onnx_initializer_1119
        model_layers_31_attn_v_proj_repeat_kv_gather_3 = self.model_layers_31_attn_v_proj_repeat_kv_Gather_3(model_layers_31_attn_v_proj_repeat_kv_shape_1, initializers_onnx_initializer_1119);  model_layers_31_attn_v_proj_repeat_kv_shape_1 = initializers_onnx_initializer_1119 = None
        initializers_onnx_initializer_1120 = self.initializers.onnx_initializer_1120
        model_layers_31_attn_k_proj_repeat_kv_gather_1 = self.model_layers_31_attn_k_proj_repeat_kv_Gather_1(model_layers_31_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_1120);  initializers_onnx_initializer_1120 = None
        initializers_onnx_initializer_1121 = self.initializers.onnx_initializer_1121
        model_layers_31_attn_k_proj_repeat_kv_gather_3 = self.model_layers_31_attn_k_proj_repeat_kv_Gather_3(model_layers_31_attn_k_proj_repeat_kv_shape_1, initializers_onnx_initializer_1121);  model_layers_31_attn_k_proj_repeat_kv_shape_1 = initializers_onnx_initializer_1121 = None
        model_layers_31_attn_v_proj_repeat_kv_unsqueeze_1 = self.model_layers_31_attn_v_proj_repeat_kv_Unsqueeze_1(model_layers_31_attn_v_proj_repeat_kv_gather_1);  model_layers_31_attn_v_proj_repeat_kv_gather_1 = None
        model_layers_31_attn_v_proj_repeat_kv_unsqueeze_3 = self.model_layers_31_attn_v_proj_repeat_kv_Unsqueeze_3(model_layers_31_attn_v_proj_repeat_kv_gather_3);  model_layers_31_attn_v_proj_repeat_kv_gather_3 = None
        model_layers_31_attn_k_proj_repeat_kv_unsqueeze_1 = self.model_layers_31_attn_k_proj_repeat_kv_Unsqueeze_1(model_layers_31_attn_k_proj_repeat_kv_gather_1);  model_layers_31_attn_k_proj_repeat_kv_gather_1 = None
        model_layers_31_attn_k_proj_repeat_kv_unsqueeze_3 = self.model_layers_31_attn_k_proj_repeat_kv_Unsqueeze_3(model_layers_31_attn_k_proj_repeat_kv_gather_3);  model_layers_31_attn_k_proj_repeat_kv_gather_3 = None
        initializers_onnx_initializer_1122 = self.initializers.onnx_initializer_1122
        initializers_onnx_initializer_1123 = self.initializers.onnx_initializer_1123
        initializers_onnx_initializer_1124 = self.initializers.onnx_initializer_1124
        model_layers_31_attn_v_proj_repeat_kv_concat_2 = self.model_layers_31_attn_v_proj_repeat_kv_Concat_2(model_layers_31_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_1122, initializers_onnx_initializer_1123, model_layers_31_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_1124);  initializers_onnx_initializer_1122 = initializers_onnx_initializer_1123 = initializers_onnx_initializer_1124 = None
        initializers_onnx_initializer_1125 = self.initializers.onnx_initializer_1125
        initializers_onnx_initializer_1126 = self.initializers.onnx_initializer_1126
        model_layers_31_attn_v_proj_repeat_kv_concat_3 = self.model_layers_31_attn_v_proj_repeat_kv_Concat_3(model_layers_31_attn_v_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_1125, model_layers_31_attn_v_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_1126);  model_layers_31_attn_v_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_1125 = model_layers_31_attn_v_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_1126 = None
        initializers_onnx_initializer_1127 = self.initializers.onnx_initializer_1127
        initializers_onnx_initializer_1128 = self.initializers.onnx_initializer_1128
        initializers_onnx_initializer_1129 = self.initializers.onnx_initializer_1129
        model_layers_31_attn_k_proj_repeat_kv_concat_2 = self.model_layers_31_attn_k_proj_repeat_kv_Concat_2(model_layers_31_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_1127, initializers_onnx_initializer_1128, model_layers_31_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_1129);  initializers_onnx_initializer_1127 = initializers_onnx_initializer_1128 = initializers_onnx_initializer_1129 = None
        initializers_onnx_initializer_1130 = self.initializers.onnx_initializer_1130
        initializers_onnx_initializer_1131 = self.initializers.onnx_initializer_1131
        model_layers_31_attn_k_proj_repeat_kv_concat_3 = self.model_layers_31_attn_k_proj_repeat_kv_Concat_3(model_layers_31_attn_k_proj_repeat_kv_unsqueeze_1, initializers_onnx_initializer_1130, model_layers_31_attn_k_proj_repeat_kv_unsqueeze_3, initializers_onnx_initializer_1131);  model_layers_31_attn_k_proj_repeat_kv_unsqueeze_1 = initializers_onnx_initializer_1130 = model_layers_31_attn_k_proj_repeat_kv_unsqueeze_3 = initializers_onnx_initializer_1131 = None
        initializers_onnx_initializer_1132 = self.initializers.onnx_initializer_1132
        model_layers_31_attn_v_proj_repeat_kv_equal = self.model_layers_31_attn_v_proj_repeat_kv_Equal(model_layers_31_attn_v_proj_repeat_kv_concat_2, initializers_onnx_initializer_1132);  initializers_onnx_initializer_1132 = None
        initializers_onnx_initializer_1133 = self.initializers.onnx_initializer_1133
        model_layers_31_attn_k_proj_repeat_kv_equal = self.model_layers_31_attn_k_proj_repeat_kv_Equal(model_layers_31_attn_k_proj_repeat_kv_concat_2, initializers_onnx_initializer_1133);  initializers_onnx_initializer_1133 = None
        initializers_onnx_initializer_1134 = self.initializers.onnx_initializer_1134
        model_layers_31_attn_v_proj_repeat_kv_where = self.model_layers_31_attn_v_proj_repeat_kv_Where(model_layers_31_attn_v_proj_repeat_kv_equal, initializers_onnx_initializer_1134, model_layers_31_attn_v_proj_repeat_kv_concat_2);  model_layers_31_attn_v_proj_repeat_kv_equal = initializers_onnx_initializer_1134 = model_layers_31_attn_v_proj_repeat_kv_concat_2 = None
        initializers_onnx_initializer_1135 = self.initializers.onnx_initializer_1135
        model_layers_31_attn_k_proj_repeat_kv_where = self.model_layers_31_attn_k_proj_repeat_kv_Where(model_layers_31_attn_k_proj_repeat_kv_equal, initializers_onnx_initializer_1135, model_layers_31_attn_k_proj_repeat_kv_concat_2);  model_layers_31_attn_k_proj_repeat_kv_equal = initializers_onnx_initializer_1135 = model_layers_31_attn_k_proj_repeat_kv_concat_2 = None
        model_layers_31_attn_v_proj_repeat_kv_expand = self.model_layers_31_attn_v_proj_repeat_kv_Expand(model_layers_31_attn_v_proj_repeat_kv_unsqueeze_5, model_layers_31_attn_v_proj_repeat_kv_where);  model_layers_31_attn_v_proj_repeat_kv_unsqueeze_5 = model_layers_31_attn_v_proj_repeat_kv_where = None
        model_layers_31_attn_k_proj_repeat_kv_expand = self.model_layers_31_attn_k_proj_repeat_kv_Expand(model_layers_31_attn_k_proj_repeat_kv_unsqueeze_5, model_layers_31_attn_k_proj_repeat_kv_where);  model_layers_31_attn_k_proj_repeat_kv_unsqueeze_5 = model_layers_31_attn_k_proj_repeat_kv_where = None
        model_layers_31_attn_v_proj_repeat_kv_reshape_3 = self.model_layers_31_attn_v_proj_repeat_kv_Reshape_3(model_layers_31_attn_v_proj_repeat_kv_expand, model_layers_31_attn_v_proj_repeat_kv_concat_3);  model_layers_31_attn_v_proj_repeat_kv_expand = model_layers_31_attn_v_proj_repeat_kv_concat_3 = None
        model_layers_31_attn_k_proj_repeat_kv_reshape_3 = self.model_layers_31_attn_k_proj_repeat_kv_Reshape_3(model_layers_31_attn_k_proj_repeat_kv_expand, model_layers_31_attn_k_proj_repeat_kv_concat_3);  model_layers_31_attn_k_proj_repeat_kv_expand = model_layers_31_attn_k_proj_repeat_kv_concat_3 = None
        model_layers_31_attn_v_proj_repeat_kv_transpose_2 = self.model_layers_31_attn_v_proj_repeat_kv_Transpose_2(model_layers_31_attn_v_proj_repeat_kv_reshape_3);  model_layers_31_attn_v_proj_repeat_kv_reshape_3 = None
        model_layers_31_attn_k_proj_repeat_kv_transpose_2 = self.model_layers_31_attn_k_proj_repeat_kv_Transpose_2(model_layers_31_attn_k_proj_repeat_kv_reshape_3);  model_layers_31_attn_k_proj_repeat_kv_reshape_3 = None
        initializers_onnx_initializer_1136 = self.initializers.onnx_initializer_1136
        model_layers_31_attn_v_proj_repeat_kv_reshape_4 = self.model_layers_31_attn_v_proj_repeat_kv_Reshape_4(model_layers_31_attn_v_proj_repeat_kv_transpose_2, initializers_onnx_initializer_1136);  model_layers_31_attn_v_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_1136 = None
        initializers_onnx_initializer_1137 = self.initializers.onnx_initializer_1137
        model_layers_31_attn_k_proj_repeat_kv_reshape_4 = self.model_layers_31_attn_k_proj_repeat_kv_Reshape_4(model_layers_31_attn_k_proj_repeat_kv_transpose_2, initializers_onnx_initializer_1137);  model_layers_31_attn_k_proj_repeat_kv_transpose_2 = initializers_onnx_initializer_1137 = None
        com_microsoft__model_layers_31_attn_multi_head_attention = self.com_microsoft__model_layers_31_attn_MultiHeadAttention(com_microsoft__model_layers_31_attn_q_rotary_rotary_embedding, model_layers_31_attn_k_proj_repeat_kv_reshape_4, model_layers_31_attn_v_proj_repeat_kv_reshape_4, attention_mask = model_attn_mask_reformat_tile);  com_microsoft__model_layers_31_attn_q_rotary_rotary_embedding = model_layers_31_attn_k_proj_repeat_kv_reshape_4 = model_layers_31_attn_v_proj_repeat_kv_reshape_4 = model_attn_mask_reformat_tile = None
        getitem_247 = com_microsoft__model_layers_31_attn_multi_head_attention[0];  com_microsoft__model_layers_31_attn_multi_head_attention = None
        initializers_onnx_initializer_1138 = self.initializers.onnx_initializer_1138
        model_layers_31_attn_o_proj_mat_mul = self.model_layers_31_attn_o_proj_MatMul(getitem_247, initializers_onnx_initializer_1138);  getitem_247 = initializers_onnx_initializer_1138 = None
        getitem_248 = com_microsoft__model_layers_31_input_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_31_input_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_1139 = self.initializers.onnx_initializer_1139
        com_microsoft__model_layers_31_post_attention_layernorm_skip_layer_norm = self.com_microsoft__model_layers_31_post_attention_layernorm_SkipLayerNorm(getitem_248, model_layers_31_attn_o_proj_mat_mul, initializers_onnx_initializer_1139);  getitem_248 = model_layers_31_attn_o_proj_mat_mul = initializers_onnx_initializer_1139 = None
        getitem_249 = com_microsoft__model_layers_31_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_1140 = self.initializers.onnx_initializer_1140
        model_layers_31_mlp_gate_proj_mat_mul = self.model_layers_31_mlp_gate_proj_MatMul(getitem_249, initializers_onnx_initializer_1140);  getitem_249 = initializers_onnx_initializer_1140 = None
        getitem_250 = com_microsoft__model_layers_31_post_attention_layernorm_skip_layer_norm[0]
        initializers_onnx_initializer_1141 = self.initializers.onnx_initializer_1141
        model_layers_31_mlp_up_proj_mat_mul = self.model_layers_31_mlp_up_proj_MatMul(getitem_250, initializers_onnx_initializer_1141);  getitem_250 = initializers_onnx_initializer_1141 = None
        model_layers_31_mlp_act_fn_sigmoid = self.model_layers_31_mlp_act_fn_Sigmoid(model_layers_31_mlp_gate_proj_mat_mul)
        model_layers_31_mlp_act_fn_mul = self.model_layers_31_mlp_act_fn_Mul(model_layers_31_mlp_gate_proj_mat_mul, model_layers_31_mlp_act_fn_sigmoid);  model_layers_31_mlp_gate_proj_mat_mul = model_layers_31_mlp_act_fn_sigmoid = None
        model_layers_31_mlp_mul = self.model_layers_31_mlp_Mul(model_layers_31_mlp_act_fn_mul, model_layers_31_mlp_up_proj_mat_mul);  model_layers_31_mlp_act_fn_mul = model_layers_31_mlp_up_proj_mat_mul = None
        initializers_onnx_initializer_1142 = self.initializers.onnx_initializer_1142
        model_layers_31_mlp_down_proj_mat_mul = self.model_layers_31_mlp_down_proj_MatMul(model_layers_31_mlp_mul, initializers_onnx_initializer_1142);  model_layers_31_mlp_mul = initializers_onnx_initializer_1142 = None
        getitem_251 = com_microsoft__model_layers_31_post_attention_layernorm_skip_layer_norm[3];  com_microsoft__model_layers_31_post_attention_layernorm_skip_layer_norm = None
        initializers_onnx_initializer_1143 = self.initializers.onnx_initializer_1143
        com_microsoft__model_layers_32_final_norm_layernorm_skip_layer_norm = self.com_microsoft__model_layers_32_final_norm_layernorm_SkipLayerNorm(getitem_251, model_layers_31_mlp_down_proj_mat_mul, initializers_onnx_initializer_1143);  getitem_251 = model_layers_31_mlp_down_proj_mat_mul = initializers_onnx_initializer_1143 = None
        initializers_onnx_initializer_1144 = self.initializers.onnx_initializer_1144
        lm_head_mat_mul = self.lm_head_MatMul(com_microsoft__model_layers_32_final_norm_layernorm_skip_layer_norm, initializers_onnx_initializer_1144);  com_microsoft__model_layers_32_final_norm_layernorm_skip_layer_norm = initializers_onnx_initializer_1144 = None
        return [lm_head_mat_mul, model_layers_0_attn_k_proj_repeat_kv_concat_1, model_layers_0_attn_v_proj_repeat_kv_concat_1, model_layers_1_attn_k_proj_repeat_kv_concat_1, model_layers_1_attn_v_proj_repeat_kv_concat_1, model_layers_2_attn_k_proj_repeat_kv_concat_1, model_layers_2_attn_v_proj_repeat_kv_concat_1, model_layers_3_attn_k_proj_repeat_kv_concat_1, model_layers_3_attn_v_proj_repeat_kv_concat_1, model_layers_4_attn_k_proj_repeat_kv_concat_1, model_layers_4_attn_v_proj_repeat_kv_concat_1, model_layers_5_attn_k_proj_repeat_kv_concat_1, model_layers_5_attn_v_proj_repeat_kv_concat_1, model_layers_6_attn_k_proj_repeat_kv_concat_1, model_layers_6_attn_v_proj_repeat_kv_concat_1, model_layers_7_attn_k_proj_repeat_kv_concat_1, model_layers_7_attn_v_proj_repeat_kv_concat_1, model_layers_8_attn_k_proj_repeat_kv_concat_1, model_layers_8_attn_v_proj_repeat_kv_concat_1, model_layers_9_attn_k_proj_repeat_kv_concat_1, model_layers_9_attn_v_proj_repeat_kv_concat_1, model_layers_10_attn_k_proj_repeat_kv_concat_1, model_layers_10_attn_v_proj_repeat_kv_concat_1, model_layers_11_attn_k_proj_repeat_kv_concat_1, model_layers_11_attn_v_proj_repeat_kv_concat_1, model_layers_12_attn_k_proj_repeat_kv_concat_1, model_layers_12_attn_v_proj_repeat_kv_concat_1, model_layers_13_attn_k_proj_repeat_kv_concat_1, model_layers_13_attn_v_proj_repeat_kv_concat_1, model_layers_14_attn_k_proj_repeat_kv_concat_1, model_layers_14_attn_v_proj_repeat_kv_concat_1, model_layers_15_attn_k_proj_repeat_kv_concat_1, model_layers_15_attn_v_proj_repeat_kv_concat_1, model_layers_16_attn_k_proj_repeat_kv_concat_1, model_layers_16_attn_v_proj_repeat_kv_concat_1, model_layers_17_attn_k_proj_repeat_kv_concat_1, model_layers_17_attn_v_proj_repeat_kv_concat_1, model_layers_18_attn_k_proj_repeat_kv_concat_1, model_layers_18_attn_v_proj_repeat_kv_concat_1, model_layers_19_attn_k_proj_repeat_kv_concat_1, model_layers_19_attn_v_proj_repeat_kv_concat_1, model_layers_20_attn_k_proj_repeat_kv_concat_1, model_layers_20_attn_v_proj_repeat_kv_concat_1, model_layers_21_attn_k_proj_repeat_kv_concat_1, model_layers_21_attn_v_proj_repeat_kv_concat_1, model_layers_22_attn_k_proj_repeat_kv_concat_1, model_layers_22_attn_v_proj_repeat_kv_concat_1, model_layers_23_attn_k_proj_repeat_kv_concat_1, model_layers_23_attn_v_proj_repeat_kv_concat_1, model_layers_24_attn_k_proj_repeat_kv_concat_1, model_layers_24_attn_v_proj_repeat_kv_concat_1, model_layers_25_attn_k_proj_repeat_kv_concat_1, model_layers_25_attn_v_proj_repeat_kv_concat_1, model_layers_26_attn_k_proj_repeat_kv_concat_1, model_layers_26_attn_v_proj_repeat_kv_concat_1, model_layers_27_attn_k_proj_repeat_kv_concat_1, model_layers_27_attn_v_proj_repeat_kv_concat_1, model_layers_28_attn_k_proj_repeat_kv_concat_1, model_layers_28_attn_v_proj_repeat_kv_concat_1, model_layers_29_attn_k_proj_repeat_kv_concat_1, model_layers_29_attn_v_proj_repeat_kv_concat_1, model_layers_30_attn_k_proj_repeat_kv_concat_1, model_layers_30_attn_v_proj_repeat_kv_concat_1, model_layers_31_attn_k_proj_repeat_kv_concat_1, model_layers_31_attn_v_proj_repeat_kv_concat_1]
        
