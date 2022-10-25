import spacy
import json
import random
from spacy.training.example import Example
import torch
from spacy.util import minibatch
from tqdm.auto import tqdm
import numpy
from thinc.api import set_gpu_allocator, require_gpu

# Debug functions
## Load the dataset
def load_dataset(path):
  ''' Read json file and reorganize ot for spacy
  :param path: path to the jsonl file, str
  '''
  data = []
  for line in open(path, 'r'):
      line_dict = json.loads(line)
      data.append((line_dict['data'].replace('\n', ' '), line_dict['label']))
  return data

## Display entity info
def show_ents(doc): 
  spacy.displacy.render(doc, style="ent", jupyter=False) # if from notebook else displacy.serve(doc, style="ent") generally


# Training Functions
## Set the learning rate
def cyclic_triangular_rate(min_lr, max_lr, period):
    it = 1
    while True:
        # https://towardsdatascience.com/adaptive-and-cyclical-learning-rates-using-pytorch-2bf904d18dee
        cycle = numpy.floor(1 + it / (2 * period))
        x = numpy.abs(it / period - 2 * cycle + 1)
        relative = max(0, 1 - x)
        yield min_lr + (max_lr - min_lr) * relative
        it += 1

def train(data, model, batch_size, rate):
  ''''Main function to update the given model
  :param data: List of dictionaries
  :param model: A transformer model
  :param batch_size: Int
  :param rate: Learning ratio of the training
  :return: Updated model
  '''
  if "ner" not in model.pipe_names:
      ner = model.create_pipe("ner") # "architecture": "ensemble" simple_cnn ensemble, bow # https://spacy.io/api/annotation
      model.add_pipe(ner)
  else:
      ner = nlp.get_pipe("ner")

  # Update the label list
  for annotations in data:
      for ent in annotations[1]:
          ner.add_label(ent[2])
  # Set the learning rate
  learn_rates = cyclic_triangular_rate(
    rate / 3, rate * 3, 2 * len(train_data) // batch_size
    )

  with model.select_pipes(enable=['ner', 'transformer']):  # only train NER
      optimizer = model.resume_training()
      i = 0
      for itn in range(n_iter):
        
          random.shuffle(train_data)
          losses = {}
          batches = spacy.util.minibatch(train_data, size=batch_size)
          for batch in batches:
              for text, annotations in batch:
                  print(text)
                  print(annotations)
                  # create Example 
                  #cupy.get_default_memory_pool().free_all_blocks()              
                  doc = model.make_doc(text)
                  annotations = {'entities' : annotations}
                  example = Example.from_dict(doc, annotations)
                  # try to visualize the content of the example

                  loss = model.update([example], sgd=optimizer, drop=0.1, losses=losses) # Be sure that you are defining batch size
                  print(loss)


  return model

def split(data, train_percantage):
  '''Split the data
  :param data: List of words
  :param train_percantage: Train/test ratio between 0-1, float 
  '''
  train_lenght = int(len(data)*train_percantage)
  train_data = data[:train_lenght]
  test_data = data[train_lenght:]
  return train_data, test_data

def test(test_data, model):
  for text, _ in test_data:
    doc = nlp(text)
    print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
    show_ents(doc)

def save_model(model, output_dir):
  if output_dir is not None:
      nlp.to_disk(output_dir)
      print("Saved model to", output_dir)


## Set the path parameters
json_path = './data/train.jsonl'
model_name = 'en_core_web_trf'

output_dir = "./Model"
n_iter = 100
rate=2e-5
batch = 8

# Main
## Use the GPU, with memory allocations directed via PyTorch.
## This prevents out-of-memory errors that would otherwise occur from competing
## memory pools.#
set_gpu_allocator("pytorch")
require_gpu(0)

data = load_dataset(json_path)
nlp = spacy.load(model_name)

train_data, test_data = split(data, 1)
finetuned_model = train(train_data, nlp, batch, rate)

if output_dir is not None:
    finetuned_model.to_disk(output_dir)
    print("Saved model to", output_dir)