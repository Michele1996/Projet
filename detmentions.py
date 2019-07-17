#! /usr/bin/env python3
r"""Detect mentions

Usage:
  detmentions [options] <model> <spans> [<output>]

Options:
  --device <d>  	The device to use for computations (defaults to `cuda:0` or `cpu`)
  --format <f>  	The output format, either "csv", "json" or "prettyjson" [default: csv]
  --gold-key <g>  	Use column with header <g> as the gold mention tag
  --mentions  	Output only mentions (gold and predicted)
  --mistakes  	Output only mislassifications
  -h, --help  	Show this screen.
"""
import contextlib
import csv
import json
import sys
import re


import typing as ty

import docopt
import torch
import tqdm

from boltons import iterutils as itu
from loguru import logger

import datatools
import utils
import Model

import spacy 


from models.defaults import Detector

__version__ = 'detmentions 0.0.2'


def load_spans(
    spans_file: str, span_digitizer: ty.Callable[[ty.Dict[str, str]], ty.Any]
) -> ty.Tuple[
    ty.Tuple[ty.Dict[str, ty.Union[str, ty.List[str]]], ...], ty.Tuple[ty.Any, ...]
]:
    '''Load and digitize a span file.

    Output: (raw_spans, digitized_spans)
    '''
    raw = tuple(datatools.read_spans_tsv(spans_file))
    digitized = tuple(
        span_digitizer(row)
        for row in tqdm.tqdm(
            raw,
            unit='spans',
            desc='Digitizing',
            unit_scale=True,
            unit_divisor=1024,
            dynamic_ncols=True,
            leave=False,
        )
    )

    return raw, digitized


def tag(
    model: torch.nn.Module, data: ty.Iterable, batch_size: int = 100
) -> ty.List[ty.Tuple[int, ty.List[float]]]:
    '''Tag a dataset

    Output: (tag, scores)
    '''
    device = next(model.parameters()).device
    model.eval()
    sys_out = []  # type: ty.List[ty.Tuple[int, ty.List[float]]]
    try:
        data_len = len(data)
    except TypeError:
        data_len = None
    data = map(datatools.collate_spans, itu.chunked_iter(data, batch_size))
    pbar = tqdm.tqdm(
        data,
        total=None if data_len is None else (data_len - 1) // batch_size + 1,
        unit='batch',
        desc='Tagging',
        mininterval=2,
        unit_scale=True,
        dynamic_ncols=True,
    )
    with torch.no_grad():
        for d in pbar:
            r = model(datatools.move(d, device=device))
            sys_tags = r.argmax(dim=-1).tolist()
            scores = r.exp().tolist()
            sys_out.extend(zip(sys_tags, scores))
    return sys_out


@contextlib.contextmanager
def smart_open(
    filename: str = None, mode: str = 'r', *args, **kwargs
) -> ty.Generator[ty.IO, None, None]:
    '''Open files and i/o streams transparently.'''
    fh: ty.IO
    if filename == '-':
        if 'r' in mode:
            stream = sys.stdin
        else:
            stream = sys.stdout
        if 'b' in mode:
            fh = stream.buffer
        else:
            fh = stream
        close = False
    else:
        fh = open(filename, mode, *args, **kwargs)
        close = True

    try:
        yield fh
    finally:
        if close:
            try:
                fh.close()
            except AttributeError:
                pass


def main_entry_point(argv=None):
    arguments = docopt.docopt(__doc__, version=__version__, argv=argv)
    logger.add(
        utils.TqdmCompatibleStream(),
        format=(
            "[caren] "
            "<green>{time:YYYY-MM-DD}T{time:HH:mm:ss}</green> {level}: "
            "<level>{message}</level>"
        ),
        colorize=True,
    )
    if arguments['<output>'] is None:
        arguments['<output>'] = '-'

    if arguments['--device'] is None:
        if torch.cuda.is_available():
            arguments['--device'] = 'cuda'
        else:
            arguments['--device'] = 'cpu'

    gold_key = arguments['--gold-key']
    device = torch.device(arguments['--device'])

    detector = Detector.load(arguments['<model>'])
    detector.model.to(device)

    raw, digitized = load_spans(arguments['<spans>'], detector.digitize_span)

    sys_out = tag(detector.model, digitized)

    results = []
    for row, (sys_t, scores) in zip(raw, sys_out):
        sys_tag = detector.span_types_lexicon.i2t[sys_t]
        if (
            arguments["--mistakes"]
            and gold_key is not None
            and sys_tag == row[gold_key]
        ):
            continue
        if (
            arguments["--mentions"]
            and sys_tag == "None"
            or gold_key is not None
            and row[gold_key] == "None"
        ):
            continue

        results.append(
            {
                **row,
                'scores': {
                    tag: score
                    for tag, score in zip(detector.span_types_lexicon.i2t, scores)
                },
                'sys_tag': sys_tag,
            }
        )

    if arguments['--format'] == 'csv':
        with smart_open(arguments['<output>'], 'w', newline='') as out_stream:
            i=0
            nlp = spacy.load("fr_core_news_md")
            model = Model.Model('/mnt/c/Users/miche/Desktop/french-sequoia-ud-2.0-170801.udpipe')
            case=None
            lemma=""
            file = open('/mnt/c/Users/miche/Desktop/monfichier2.txt',encoding='latin-1',errors='ignore')
            
         
            uma1=""
            tk=0
            fi=None
            fo=None
            tktext=0
            lines=str(file.read())
            list=lines.split(".")
            op=1
            no=None
            si=""
            o=0
            for row1 in results:
                save=" "
                uma2= str(row1['content']).replace('[','').replace(']','').replace('\'','').replace(',','').replace('<start>','').replace('<end>','').replace('\"','').replace("\"m\" ","m'").replace("\"qu\" il","qu'il").replace("\"l\" ","l'").replace("  "," ").replace("\"d\" ","d'").replace("\"n\" ","n'").replace("m ","m'").replace("d ","d'").replace("l ","l'").replace("qu  ","qu'").replace("-","")
                if uma2 in "":
                   uma2="-------------"
                if uma2=="m":
                   uma2="m'"
                if uma2 in "n":
                   uma2=="n'"
                if uma2 =="en":
                   uma2=" en "
                
                control="#"+str(tktext)
                
                
                
                if uma2 in lines:
                   
                   uma3=uma2+"#"+str(tktext)
                   lines=lines.replace(uma2,uma3,1)
                   op=lines.find(uma3)
                   si=si+lines[:op]
                   lines=lines[op:]
                   
                   tktext=tktext+1
                elif uma2 in lines.replace("#"+str(tktext-1),""):
                   ti =uma2.split(" ")
                   for ti2 in ti:
                       if ti2 in uma2:
                          o=o+1
                       if o==len(ti):
                          lines.find(ti2)
                          lines=lines.replace(ti2,ti2+control)
                          tktext=tktext+1
                
            for row in results:
                uma2= str(row['content']).replace('[','').replace(']','').replace('\'','').replace(',','').replace('<start>','').replace('<end>','').replace('\"','').replace("\"m\" ","m'").replace("\"qu\" il","qu'il").replace("\"l\" ","l'").replace("  "," ").replace("\"d\" ","d'").replace("\"n\" ","n'").replace("m ","m'").replace("d ","d'").replace("l ","l'").replace("qu  ","qu'").replace("-","")
                if uma2!="":
                   uma22=uma2+"#"+str(tk)
                   
                   tk=tk+1
       	        for line in list:
                    if uma2 in line:
                       uma1=line
                       list.remove(line)
                       fi=True
                    if fi:
                       break
                
                if uma2=="m":
                   uma2="m'"
                if uma2=="qu":
                   uma2="qu'"
                gp = "CASE=NO"
                sentences = model.tokenize(uma2)
                sentences1= model.tokenize(uma1)
                
                for s in sentences:
                   model.tag(s)
                   model.parse(s)
                for s1 in sentences1:
                   model.tag(s1)
                   model.parse(s1)
                conllu1=model.write(sentences1,"conllu")
                conllu=model.write(sentences,"matxin")
                i1=0
                cas1=0
                conllu1=str(conllu1).split("\n")
                for c1 in conllu1:
                   le=c1.split("\t")
                   
                   i1=i1+1
                i=0
                cas=0
                conllu = str(conllu).split("\n")
                
                en =""
                gn=""
                num=""
                
                for c in conllu:
                    ca=c.find("lem=")
                    lem=c[ca:].split(" ")
                    lemt=lem[0]
                    if '>' not in lemt:
                       
                       lemt1=lemt.split("=")
                       
                       if len(lemt1) > 1:
                          lemma=lemt1[1].replace("\"","")
                    if "si=\"root\"" in c:
                       cas=i
                      
                      
                       if 'Gender=Fem' in c:
                          gn="FEM"
                       if 'Gender=Masc' in c:
                          gn="MASC"
                       if 'Number=Sing' in c:
                          num="SING"
                       if 'Number=Plur' in c:
                          num="PLUR"
                    if "si=\"case\"" in c and i==(cas+1):
                       gp="CASE=YES"
                       case=True
                    i=i+1
                        
              
                

                doc1 = nlp(uma2)
                
                if uma2 == "":
                   uma2="aa"
                skip = None
                for token in doc1:
                    
                    if "PUNCT" in token.tag_ or ("VERB" in token.tag_ and "ROOT" in token.dep_):
                       skip = True
                  
                    
                       
                if skip:
                   row['left_context'] =""
                   row['content']=""
                   row['right_context']
                   
                row['mention'] = ' '.join(
                    [
                        *row.pop('left_context'),
                        '|',
                        *row.pop('content'),
                        '|',
                        *row.pop('right_context'),
                        '|',
                         en,
                        '|',
                         gn,
                         '|',
                         num,
                         '|',
                         gp,
                        
                    ]
                )
                if skip:
                   row['mention']=""
                   row['scores']=""
                   row['sys_tag']=""

                else:
                    row['scores'] = '|'.join(
                       f"{t}={round(s, 5)}" for t, s in row['scores'].items()
                    )
                
            start_fields = ['mention']
            end_fields = ['scores', 'sys_tag']
            fieldnames = start_fields[:]
            
            for k in results[0].keys():
                if k not in start_fields and k not in end_fields:
                    fieldnames.append(k)
            fieldnames.extend(end_fields)
            writer = csv.DictWriter(
                out_stream, fieldnames=fieldnames, delimiter='\t', quotechar='"'
            )
            writer.writeheader()
            for res in results:
                
                car = str(res).replace("\'","")
                
                if "scores: , sys_tag: , mention: }" in car:
                  
                else:
                   writer.writerow(res)
    elif arguments['--format'] in ('json', 'prettyjson'):
        with smart_open(arguments['<output>'], 'w') as out_stream:
            json.dump(
                results,
                out_stream,
                indent=4 if arguments['--format'] == "prettyjson" else None,
            )


if __name__ == '__main__':
    main_entry_point()
