#!/usr/bin/env python3
"""This script is for generating training datasets for the WBMET and LMBMET
models. A subset of PubTator annotated documents and the MeSH definitions from
the MeSH datafiles (desc2019 and supp2019) are used to generate datasets.

For WBMET, the pre-processed document set will be saved in a text file.
For LMBMET, the Corpus class in LMBET model will be saved as a serialized
pytorch object."""

import logging
from pathlib import Path
import time
import re
import multiprocessing as mp
import random as rnd
import unidecode
from bounter import bounter
from collections import Counter
import pickle

from lxml import etree
from nltk.tokenize import word_tokenize, sent_tokenize

from data import Corpus


# ------------------------------------------------------------------------------
# Parsing PubTator
# ------------------------------------------------------------------------------

def process_pubtator(sample_ratio=0.03):
    """Sample Pubtator documents and batchify for multiprocessing"""
    logger.info('Processing PubTator documents...')

    global docs
    docs = []
    total_meshes = 0
    total_docs_read = 0
    bsz = 10000  # batch size
    word_freq = bounter(size_mb=1024*4)

    def cb_proc_pubtator(res):
        nonlocal total_meshes
        docs_, words_, num_meshes = res
        docs.extend(docs_)
        word_freq.update(words_)
        total_meshes += num_meshes
        print('total_meshes {}, num_meshes {}\r'
              ''.format(total_meshes, num_meshes), end='')

    p = mp.Pool()
    # Read the PubTator datafile
    with pubtator_file.open('rt') as f:
        batch = []
        aDoc = []
        flgSample = rnd.random() < sample_ratio
        while True:
            line = f.readline()
            if not line:  # End of file
                break
            if not flgSample:  # Do nothing and just read on
                if line == '\n':  # End of document
                    flgSample = rnd.random() < sample_ratio
                    total_docs_read += 1
                else:
                    continue
            else:
                if line == '\n':  # End of document
                    total_docs_read += 1
                    batch.append(aDoc)  # Add current document
                    flgSample = rnd.random() < sample_ratio
                    aDoc = []
                    if len(batch) == bsz:  # If batch is full, assign a job
                        p.apply_async(mp_proc_pubtator, (batch, ),
                                      callback=cb_proc_pubtator)
                        batch = []
                else:
                    aDoc.append(line.rstrip())
        if len(batch) > 0:
            p.apply_async(mp_proc_pubtator, (batch,), callback=cb_proc_pubtator)
    p.close()
    p.join()

    # move from bounter to Counter (bounter dosen't have most_common())
    global words
    words = Counter({k: v for k, v in word_freq.items()})
    logger.info('{}/{} documents processed, {} mesh terms found ({} meshes per'
                ' doc)'.format(len(docs), total_docs_read,  total_meshes,
                               total_meshes/len(docs)))


def mp_proc_pubtator(batch):
    """ Run text pre-processing over given batch of documents.

    :param batch: batch of documents in raw xml format
    """
    # Parse the PubTator documents
    ptn_title = re.compile(r'^((\d+)\|t\|)(.*)$')
    ptn_body = re.compile(r'^((\d+)\|a\|)(.*)$')
    ptn_annt = re.compile(r'\d+\t\d+\t\d+\t\S+\t(Chemical|Disease)\t.*')

    docs = []
    words = Counter()
    total_meshes = 0
    for doc in batch:
        text = []
        annotations = []
        for line in doc:
            m = ptn_title.match(line)
            if m:
                text.append(m.group(3))
            m = ptn_body.match(line)
            if m:
                text.append(m.group(3))
            m = ptn_annt.match(line)
            if m:
                annotations.append(m.group(0))
        if len(annotations) == 0:
            continue
        out, num_ent = interpolate_entities(' '.join(text), annotations)
        total_meshes += num_ent
        tokens = word_tokenize(out)
        data = clean_pubtator_texts(' '.join(tokens)).split()
        words.update(data)
        docs.append(data)
    return docs, words, total_meshes


def interpolate_entities(text, annotations, ent_indicator='ε'):
    """Read a document fields as in PubTator data and interpolate entities
    into the texts"""

    out = text
    entities = []
    for line in annotations:
        fields = line.split('\t')
        if len(fields) != 6:
            continue
        # Annotation format:
        # docid \t loc_s \t loc_e \t name \t ent_type \t source:id
        docid, loc_s, loc_e, name, ent_type, ent_id = fields
        entity_code = None
        if ent_type in ['Disease', 'Chemical']:
            try:
                if '|' in ent_id:  # Multiple entities, use first one
                    ent_id = ent_id[:ent_id.find('|')]
                if ':' in ent_id:
                    src, code = ent_id.split(':')
                else:
                    src, code = ('MESH', ent_id)
                assert src in ['MESH', 'OMIM', 'CHEBI']
            except AssertionError as err:
                logger.warning('Unknown src: {}'.format(fields))
                continue
            except Exception as err:
                logger.warning('Unknown format: {}'.format(fields))
                continue
            if src == 'MESH':  # Only the MeSH terms for now
                entity_code = '{}MESH_{}'.format(ent_indicator, code)
        if entity_code:
            entities.append((int(loc_s), int(loc_e), entity_code))
    entities.sort(key=lambda x: x[0])
    pointer = 0
    for e in entities:
        out += text[pointer:e[1]] + ' ' + e[2] + ' '
        pointer = e[1]
    out += text[pointer:]  # the rest of document
    return out, len(entities)


def clean_pubtator_texts(x):
    x = unidecode.unidecode(x)  # this changes the entity indicator ε to e too.
    x = re.sub(r'emesh_', 'εmesh_', x)  # revert e to ε for the entity indicator
    x = x.lower()
    x = re.sub(r'\b[\d,.]+\b', '#', x)
    x = re.sub(r'[^a-z0-9ε_#,.\-\'\s]+', ' ', x)
    x = re.sub(r'\s+', ' ', x)
    return x


# ------------------------------------------------------------------------------
# Parsing MeSH concepts
# ------------------------------------------------------------------------------

def process_mesh():
    """Read the MeSH definitions and ScopeNotes from the descriptor file.
    Prepend MeSH codes to their definitions."""
    logger.info("Processing MeSH Definitions...                            ")

    # Read MeSH descriptors
    global mesh_def
    mesh_def = dict()
    data = etree.parse(mesh_desc_file.open('rt'))
    cnt = [0, 0]
    for rec in data.getiterator("DescriptorRecord"):
        cnt[0] += 1
        mshid = 'εmesh_' + rec.find("DescriptorUI").text.lower()
        name = rec.find("DescriptorName/String").text
        elm = 'ConceptList/Concept[@PreferredConceptYN="Y"]/ScopeNote'
        scope_elm = rec.find(elm)
        note = scope_elm.text if scope_elm is not None else ''
        body = []
        for sent in sent_tokenize(name + ', ' + note):
            cnt[1] += 1
            body.extend(word_tokenize(clean_pubtator_texts(mshid + ' ' + sent)))
        mesh_def[mshid] = {
            'descriptor': True,
            'name': name.strip(),
            'note': note.strip(),
            'enc_body': body
        }
    logger.info('>> {} MeSH descriptors read, {} sentences'.format(*cnt))

    # Read MeSH Supplmentary concepts
    # --------------------------------------------------------------------------
    # data = etree.parse(mesh_supp_file.open('rt'))
    # for rec in data.getiterator("SupplementalRecord"):
    #     mshid = 'εmesh_' + rec.find("SupplementalRecordUI").text.lower()
    #     name = rec.find("SupplementalRecordName/String").text
    #     note_ = rec.find("Note")
    #     note = note_.text if note_ is not None else ''
    #     note = clean_pubtator_texts(note)
    #     body = []
    #     for sent in sent_tokenize(note):
    #         body.extend(word_tokenize(mshid + ' ' + sent))
    #     mesh_def[mshid] = {
    #         'descriptor': False,
    #         'name': name.strip(),
    #         'note': note.strip(),
    #         'enc_body': body
    #     }
    return mesh_def


# ------------------------------------------------------------------------------
# Save
# ------------------------------------------------------------------------------

def save_wbmet_ds():
    logger.info('Saving WBMET training data...')
    with w_out_file.open('w') as f:
        # PubTator docs
        cnt = 0
        for doc in docs:
            f.write(' '.join(doc) + ' <eos> ')
            cnt += 1
            if cnt % 1000 == 0:
                print('- {} docs saved...\r'.format(cnt), end='')

        # Mesh defs
        if mesh_def is not None:
            cnt = 0
            for k, data in mesh_def.items():
                if not data['descriptor']:
                    continue
                f.write(' ' + ' '.join(data['enc_body']))
                cnt += 1
                if cnt % 1000 == 0:
                    print('- {} mesh defs saved in wbmet data...\r'
                          ''.format(cnt), end='')


def save_lmbmet_ds():
    """Instantiate a Corpus object, add vocabulary and encoded data"""
    logger.info('Saving LMBMET training data...')

    corpus = Corpus()

    # Identify all the tokens except MeSH descriptors, which are included in
    # vocab via mesh_def
    words_wo_mesh_d = []
    for w, _ in words.most_common():
        if w in mesh_def and mesh_def[w]['descriptor']:
            continue
        words_wo_mesh_d.append(w)
    corpus.vocab.load_vocab(words_wo_mesh_d,
                            mesh_def=mesh_def, specials=['<eos>', '<unk>'])
    corpus.mesh_def = mesh_def

    corpus.load_data(docs)
    pickle.dump(corpus, lm_out_file.open("wb"), protocol=4)


if __name__ == '__main__':
    # Logger
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s-%(levelname)s] -- %(message)s')
    logger = logging.getLogger()

    # Paths
    run_id = time.strftime('%m%d_%H%M')
    data_dir = Path(__file__).resolve().parent / 'data'
    pubtator_file = data_dir / 'pubtator/bioconcepts2pubtator_offsets'
    mesh_desc_file = data_dir / 'mesh/desc2019'
    mesh_supp_file = data_dir / 'mesh/supp2019'
    w_out_file = data_dir / 'wbmet-training-{}.txt'.format(run_id)
    lm_out_file = data_dir / 'pubtator/pubtator-corpus-{}.pickle'.format(run_id)

    for f in [w_out_file, lm_out_file]:
        if f.is_file():
            f.unlink()

    # RUN
    # --------------------------------------------------------------------------
    docs = None
    words = None
    mesh_def = None

    process_pubtator(sample_ratio=0.02)
    process_mesh()

    save_wbmet_ds()
    save_lmbmet_ds()

    logger.info('DONE. Writing training datasets for BMET. Datasets are saved '
                'as:\n{}\n'.format(w_out_file, lm_out_file))
