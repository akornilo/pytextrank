#!/usr/bin/env python
# encoding: utf-8

from pytextrank import json_iter, parse_doc, pretty_print
from pytextrank import normalize_key_phrases, render_ranks, text_rank
from pytextrank import rank_kernel, top_sentences

import os
import sys

import jsonlines

prefix = 'test_docs'

def rank_bill(bill):
    bill_id = bill['bill_id']
    with open(prefix + '/{}_stage1'.format(bill_id), 'w') as f:
        for graf in parse_doc([bill]):
            f.write(pretty_print(graf._asdict()))
            f.write('\n')
   
    path_stage1 = prefix + '/{}_stage1'.format(bill_id)

    graph, ranks = text_rank(path_stage1)
    render_ranks(graph, ranks)

    for rl in normalize_key_phrases(path_stage1, ranks):
        output = pretty_print(rl._asdict())
        with open(prefix + '/{}_stage2'.format(bill_id), 'w') as f:
            f.write(output)
    
    path_stage1 = prefix + '/{}_stage1'.format(bill_id)
    path_stage2 = prefix + '/{}_stage2'.format(bill_id)

    kernel = rank_kernel(path_stage2)
    with open(prefix + '/{}_stage3'.format(bill_id), 'w') as f:
        for s in top_sentences(kernel, path_stage1):
            f.write(pretty_print(s._asdict()))


if __name__ == '__main__':

    with jsonlines.open(os.path.expanduser('~/tldr_data/final/final_data_107_clean.jsonl')) as reader:

        for line in reader:
            print(line['bill_id'])
            rank_bill(line)