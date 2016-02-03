
collected:
- 'dataset': name of dataset
- 'images' : [{'filename', 'coco_object_id', 'coco_image_id', 'coco_category_iud', 'sentences'}]
             where each 'sentences' is a list of {'raw', 'accuracy'}

cleaned: (after cleaning collected.json)
- [ref]: each ref is {'ref_id', 'file_name', 'ann_id', 'image_id', 'category_id', 'sent_ids', 'sentences', 'split'}
         where each 'sentences' is a list of {'sent_id', 'sent', 'tokens', 'raw'}

imageIdSplits:
- 'train': [image_id]
- 'val'  : [image_id]
- 'test' : [image_id]

instances:
- 'info'       : {'description', 'url', 'version', 'year', 'contributor', 'data_created'}
- 'licenses'
- 'images'     : [{'id', file_name', 'height', 'width'}]
- 'annotations': [{'id', 'image_id', 'bbox', 'category_id', 'area', 'segmentation', 'iscrowd'}]

sentToParse:
- {'sent_id': {'parsetree', 'text', 'dependencies', 'words'}}

sentToAtts:
- {'sent_id': {'r1': [atts], 'r2': [atts], ...}}

