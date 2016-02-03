"""
This code is to collect info for generated sentence.
The input is some json file in cache/lang/dataset_splitBy/, which contains
{'overall', 
 'refToEval: {'ref_id': {'sent', 'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'ROUGE_L', 'METEOR', 'CIDEr'} 
 }
We output "sents_id0.json" (compared with "sents_gd.json")
[{'sent', 'tokens', 'ref_id', 'image_id', 'bbox', 'split', 'bleu'}]  (we still record Bleu@1 here in case we want to check later)
"""
import os.path as osp
import os
import sys
import json
import argparse

# input
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_splitBy', default='refcoco_licheng', help='name of dataset+splitBy')
parser.add_argument('--model_id', default='0', help='model_id to be loaded')
parser.add_argument('--split', default='testA', help='split name, val|test|train')
args = parser.parse_args()
params = vars(args)

dataset_splitBy = params['dataset_splitBy'] # in Lua, we simply use dataset denoting dataset_splitBy
i = dataset_splitBy.find('_')
dataset, splitBy = dataset_splitBy[:i], dataset_splitBy[i+1:]
model_id = params['model_id']
split = params['split']

# load predicted sents by model_id
lang_result_path = osp.join('cache/lang', dataset_splitBy, 'model_id' + model_id + '_' + split + '_out.json')
if not osp.isfile(lang_result_path):
	print('%s not found.' % lang_result_path)
	sys.exit()
lang_result = json.load(open(lang_result_path, 'r'))
refToEval = lang_result['refToEval']

# load refer
ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, osp.join(ROOT_DIR, 'lib', 'datasets'))
from refer import REFER
refer = REFER(dataset, splitBy = splitBy)

# add ground-truth sents
sents = []
for ref_id, res in refToEval.items():
	ref_id = int(ref_id)  # json's key is string, convert it to number
	ref = refer.Refs[ref_id]
	if ref['split'] != split:
		continue
	# get ref info
	image_id = ref['image_id']
	bbox = refer.refToAnn[ref_id]['bbox']
	# get sent info
	sent = res['sent']
	tokens = sent.split()
	bleu = res['Bleu_1']
	# add to sents
	sents.append({'sent': sent, 'tokens': tokens, 'ref_id': ref_id, 'image_id': image_id, 'bbox': bbox, 'split': split, 'bleu': bleu})
	
# save
file_folder = 'cache/langToBox/'+dataset_splitBy
if not osp.isdir(file_folder):
	os.mkdir(file_folder)
file_name = osp.join(file_folder, 'sents_id'+model_id+'_'+split+'.json')
json.dump(sents, open(file_name, 'w'))

