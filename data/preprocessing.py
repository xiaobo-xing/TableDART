import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from project_config.config import cfg
_question_embed_tokenizer = None
_question_embed_model = None

def get_question_embedding_for_gate(question_text, device):
    global _question_embed_tokenizer, _question_embed_model
    model_id = cfg['MODEL']['QUESTION_EMBED_MODEL_FOR_GATE_ID']
    if _question_embed_tokenizer is None or _question_embed_model is None or _question_embed_tokenizer.name_or_path != model_id:
        print(f'INFO: Loading question embedding model for gate: {model_id}')
        _question_embed_tokenizer = AutoTokenizer.from_pretrained(model_id)
        _question_embed_model = AutoModel.from_pretrained(model_id).to(device)
        _question_embed_model.eval()
    with torch.no_grad():
        inputs = _question_embed_tokenizer(question_text, return_tensors='pt', truncation=True, padding='max_length', max_length=128).to(device)
        outputs = _question_embed_model(**inputs)
        if 'pooler_output' in outputs:
            embedding = outputs.pooler_output.squeeze(0)
        else:
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-09)
            embedding = (sum_embeddings / sum_mask).squeeze(0)
    return embedding
 
