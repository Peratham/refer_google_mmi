"""
Preprocess a raw json dataset into hdf5 and json files for use in data_loader.lua

Input: REFER Loader

Output: json file that has
'refs'       -[{'ref_id', 'ann_id', 'split', 'image_id', 'category_id', 'sent_ids'}]
'images'     -[{'image_id', 'ref_ids', 'file_name', 'width', 'height', 'h5_id'}]
'anns'       -[{'ann_id', 'category_id', 'image_id', 'bbox'}]
'sentences'  -[{'sent_id', 'tokens', 'h5_id'}]
'ix_to_word' -{ix: word}
'word_to_ix' -{word: ix}

Output: hdf5 file that has
/images is (N, 3, 256, 256) uint8 array of raw image data in RGB format
/labels is (M, max_length) uint32 array of encoded labels, zeros padded
"""

import os
import sys
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
# add root path
import os.path as osp
ROOT_DIR = './'

def build_vocab(refer, params):
  """
  remove bad words, and return final sentences (sent_id --> final)
  """
  count_thr = params['word_count_threshold']
  sentToTokens = refer.sentToTokens

  # count up the number of words
  word2count = {}
  for sent_id, tokens in sentToTokens.items():
    for wd in tokens:
      word2count[wd] = word2count.get(wd, 0) + 1

  # print some stats
  total_words = sum(word2count.itervalues())
  print 'total words: %s' % total_words
  bad_words = [w for w, n in word2count.items() if n <= count_thr]
  vocab = [w for w, n in word2count.items() if n > count_thr]
  bad_count = sum([word2count[w] for w in bad_words])
  print 'number of good words: %d' % len(vocab)
  print 'number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(word2count), len(bad_words)*100.0/len(word2count))
  print 'number of UNKs in sentences: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words)

  # add UNK
  if bad_count > 0:
    vocab.append('UNK')

  # lets now produce final tokens
  sentToFinal = {}
  for sent_id, tokens in sentToTokens.items():
    final = [w if word2count[w] > count_thr else 'UNK' for w in tokens]
    sentToFinal[sent_id] = final

  return vocab, sentToFinal 


def check_sentLength(sentToFinal):
  sent_lengths = {}
  for sent_id, tokens in sentToFinal.items():
    nw = len(tokens)
    sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
  max_len = max(sent_lengths.keys())
  print 'max length of sentence in raw data is %d' % max_len
  print 'sentence length distribution (count, number of words):'
  sum_len = sum(sent_lengths.values())
  acc = 0  # accumulative distribution
  for i in range(max_len+1):
    acc += sent_lengths.get(i, 0)
    print '%2d: %10d %.3f%% %.3f%%' % (i, sent_lengths.get(i, 0), 
      sent_lengths.get(i, 0)*100.0/sum_len, acc*100.0/sum_len)


def prepare_json(refer, sentToFinal, params):
  # prepare refs = [{'ref_id', 'split', 'category_id', 'ann_id', 'sent_ids', 'bbox', 'image_id'}]
  refs = []
  ref_ids = refer.getRefIds()
  for ref_id in ref_ids:
    ref = refer.Refs[ref_id]
    bbox = refer.refToAnn[ref_id]['bbox']
    refs += [{'ref_id': ref_id, 'split': ref['split'], 'category_id': ref['category_id'], 'ann_id': ref['ann_id'],
    'sent_ids': ref['sent_ids'], 'bbox': bbox, 'image_id': ref['image_id']}]

  # prepare images = [{'image_id', 'width', 'height', 'file_name', 'ref_ids', 'ann_ids', h5_id'}]
  images = []
  h5_id = 0
  for image_id, img in refer.imgs.items():
    width = img['width']
    height = img['height']
    file_name = img['file_name']
    ref_ids = [ref['ref_id'] for ref in refer.imgToRefs[image_id]]
    ann_ids = [ann['id'] for ann in refer.imgToAnns[image_id]]
    h5_id = h5_id + 1  # lua 1-indexed
    images += [{'image_id': image_id, 'height': height, 'width': width, 'file_name': file_name, 'ref_ids': ref_ids, 
    'ann_ids': ann_ids, 'h5_id': h5_id}]
  print 'There are in all %d images to be written into hdf5 file.' % h5_id

  # prepare related anns = [{'ann_id', 'category_id', 'bbox', 'image_id'}]
  anns = []
  for image_id in refer.imgs.keys():
    ann_ids = [ann['id'] for ann in refer.imgToAnns[image_id]]
    for ann_id in ann_ids:
      ann = refer.anns[ann_id]
      anns += [{'ann_id': ann_id, 'category_id': ann['category_id'], 'bbox': ann['bbox'], 'image_id': image_id}]
  print 'There are in all %d anns within the %d images to be written into hdf5 file.' % (len(anns), len(images))

  # prepare sentences = [{'sent_id', 'tokens', 'h5_id'}]
  sentences = []
  h5_id = 0
  for sent_id, tokens in sentToFinal.items():
    h5_id = h5_id + 1  # lua 1-indexed
    sentences += [{'sent_id': sent_id, 'tokens': tokens, 'h5_id': h5_id}]
  print 'There are in all %d sentences to be written into hdf5 file.' % h5_id

  return refs, images, sentences, anns


def encode_captions(sentences, wtoi, params):
  """
  Input: sentences is a list of {sent_id, h5_id, tokens}
  """
  max_length = params['max_length']
  M = len(sentences)
  L = np.zeros((M, max_length), dtype='uint32')
  for sent in sentences:
    h5_id = sent['h5_id']  # lua 1-indexed
    for j, w in enumerate(sent['tokens']):
      if j < max_length:
        L[h5_id-1, j] = wtoi[w]  # python 0-indexed

  return L


def main(params):

  # dataset_splitBy
  dataset = params['dataset']
  splitBy = params['splitBy']

  # mkdir and write json file
  if not osp.isdir(osp.join('cache/data', dataset+'_'+splitBy)):
    os.mkdir(osp.join('cache/data', dataset+'_'+splitBy))
  if not osp.isdir(osp.join('model', dataset+'_'+splitBy)):
    os.mkdir(osp.join('model', dataset+'_'+splitBy))  # we also mkdir model/dataset_splitBy here!

  # load refer
  sys.path.insert(0, osp.join(ROOT_DIR, 'pyutils/datasets'))
  from refer import REFER
  refer = REFER(dataset, splitBy = splitBy)

  # create vocab
  vocab, sentToFinal = build_vocab(refer, params)
  itow = {i+1: w for i, w in enumerate(vocab)}  # lua 1-indexed
  wtoi = {w: i+1 for i, w in enumerate(vocab)}  # lua 1-indexed

  # check sentence length
  check_sentLength(sentToFinal)

  # prepare refs, images, sentences and anns
  refs, images, sentences, anns = prepare_json(refer, sentToFinal, params)

  json.dump({'refs': refs, 'images': images, 'sentences': sentences, 'anns': anns, 'ix_to_word': itow, 
    'word_to_ix': wtoi}, open(osp.join('cache/data', dataset+'_'+splitBy, params['output_json']), 'w'))

  # /sentences
  f = h5py.File(osp.join('cache/data', dataset+'_'+splitBy, params['output_h5']), 'w')
  L = encode_captions(sentences, wtoi, params)
  f.create_dataset("labels", dtype='uint32', data=L)

  # /images
  N = len(images)
  dset = f.create_dataset('images', (N, 3, 256, 256), dtype='uint8')

  for i in range(N):

    I = imread(osp.join(refer.IMAGE_DIR, images[i]['file_name']))  # uint8
    if len(I.shape) == 2:
      I = np.tile(I[:,:,np.newaxis], (1,1,3))
    Ir = imresize(I, (256, 256))
    Ir = Ir.transpose(2,0,1)

    h5_id = images[i]['h5_id']  # 1-indexed
    dset[h5_id-1] = Ir  # 0-indexed

    if i % 1000 == 0:
      print 'processing %d/%d (%.2f%% done)' % (i, N, i*100.0/N)

  f.close()
  print 'wrote %s' % params['output_h5']

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # input json
  parser.add_argument('--output_json', default='data.json', help='output json file')
  parser.add_argument('--output_h5', default='data.h5', help='output h5 file')
  
  # options
  parser.add_argument('--dataset', default='refcoco', type=str, help='refcoco/refcoco+')
  parser.add_argument('--splitBy', default='licheng', type=str, help='licheng/google')
  parser.add_argument('--max_length', default=10, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
  parser.add_argument('--images_root', default='', help='root location in which images are stored, to be prepended to file_path in input json')
  parser.add_argument('--word_count_threshold', default=5, type=int, help='only words that occur more than this number of times will be put in vocab')
  parser.add_argument('--scan', default=0, type=int, help='scan the referred object and its expressions')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print 'parsed input parameters:'
  print json.dumps(params, indent = 2)
  main(params)
