from attention import MultiHeadSparseAttention
from transformers.models.bart.modeling_tf_bart import TFBartForConditionalGeneration
from transformers import BartConfig



class MyTransformer(TFBartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        if 'attention_deviation' not in dir(config):
            config.attention_deviation = 384
            config.sequence_length = 4096
            config.global_tokens = 4
        #print(self.layers)
        for i, layer in enumerate(self.layers[0].encoder.layers):
            layer.self_attn = MultiHeadSparseAttention(
                d_model=config.d_model, 
                layer_id=i,
                n_heads=config.num_attention_heads,
                graph_connections=config.attention_deviation,
                sequence_length=config.sequence_length,
                global_tokens=config.global_tokens
            )
        #for i, layer in enumerate(self.layers[0].encoder.layers):
            #print(str(layer.self_attn))


class MyTransformerConfig(BartConfig):
    def __init__(self, attention_deviation: int=384, **kwargs):
        super().__init__(**kwargs)
        self.attention_deviation = attention_deviation
