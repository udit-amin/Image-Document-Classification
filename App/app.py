import os
os.system('pip install pyyaml==5.1')
os.system('pip install -q pytesseract')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dataset import create_features
from modeling import DocFormerEncoder,ResNetFeatureExtractor,DocFormerEmbeddings,LanguageFeatureExtractor
from transformers import BertTokenizerFast
from utils import DocFormer

import torch

seed = 42
target_size = (500, 384)
max_len = 128


device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = {
  "coordinate_size": 96,
  "hidden_dropout_prob": 0.6,
  "hidden_size": 768,
  "image_feature_pool_shape": [7, 7, 256],
  "intermediate_ff_size_factor": 4,
  "max_2d_position_embeddings": 1024,
  "max_position_embeddings": 128,
  "max_relative_positions": 8,
  "num_attention_heads": 12,
  "num_hidden_layers": 3,
  "pad_token_id": 0,
  "shape_size": 96,
  "vocab_size": 30522,
  "layer_norm_eps": 1e-12,
}

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

url = r'C:\Users\udits\Desktop\DL proj\App\epoch=0-step=2253.ckpt'


docformer = DocFormer(config).load_from_checkpoint(url)


id2label = ['resume',
 'memo',
 'scientific_publication',
 'news_article',
 'specification',
 'letter',
 'form',
 'invoice',
 'handwritten',
 'file_folder',
 'email',
 'scientific_report',
 'budget',
 'presentation',
 'questionnaire',
 'advertisement']

import gradio as gr

image = gr.inputs.Image(type="pil")
label = gr.outputs.Label(num_top_classes=5)
examples = [['00093726.png'], ['00866042.png']]
title = "Image Document Classification"
description = "Demo for classifying document images with DocFormer model. To use it, \
simply upload an image or use the example images below and click 'submit' to let the model predict the 5 most probable Document classes. \
Results will show up in a few seconds."

def classify_image(image):

  image.save('sample_img.png')
  final_encoding = create_features(
            './sample_img.png',
            tokenizer,
            add_batch_dim=True,
            target_size=target_size,
            max_seq_length=max_len,
            path_to_save=None,
            save_to_disk=False,
            apply_mask_for_mlm=False,
            extras_for_debugging=False,
            use_ocr = True
    )

  keys_to_reshape = ['x_features', 'y_features', 'resized_and_aligned_bounding_boxes']
  for key in keys_to_reshape:
      final_encoding[key] = final_encoding[key][:, :max_len]

  from torchvision import transforms
  transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

  final_encoding['resized_scaled_img'] = transform(final_encoding['resized_scaled_img'])
  output = docformer.forward(final_encoding)
  output = output[0].softmax(axis = -1)
  
  final_pred = {}
  for i, score in enumerate(output):
      score = output[i]
      final_pred[id2label[i]] = score.detach().cpu().tolist()
      
  return final_pred

gr.Interface(fn=classify_image, inputs=image, outputs=label, title=title, description=description, examples=examples).launch(debug=True, enable_queue=True)

