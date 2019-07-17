#! /usr/bin/env python3

import contextlib
import csv
import json
import sys
import re

import pathlib as pa
import typing as ty

import docopt
import torch
import tqdm

from boltons import iterutils as itu
from loguru import logger

import datatools
from models import utils
import Model

import spacy 


from models.defaults import Detector

__version__ = 'detmentions 0.0.2'

def smart_open(
    filename: str, mode: str = 'r', *args, **kwargs
) -> ty.Generator[ty.IO, None, None]:
    '''Open files and i/o streams transparently.'''
    if filename == '-':
        if 'r' in mode:
            stream = sys.stdin
        else:
            stream = sys.stdout
        if 'b' in mode:
            fh = stream.buffer  # type: ty.IO
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
        with smart_open(arguments['<output>'], 'w', newline='') as out_stream:
            i=0
            nlp = spacy.load("fr_core_news_md")
            #On initialise le model de udpipe pour le francais
            model = Model.Model('/mnt/c/Users/miche/Desktop/french-sequoia-ud-2.0-170801.udpipe')
            case=None
            lemma=""
            #On ouvre le fichier contenant le texte brut
            file = open('/mnt/c/Users/miche/Desktop/monfichier2.txt',encoding='latin-1',errors='ignore')
            #On ouvre aussi le fichier arff et on lui inscrit l'intete
            b = open("/mnt/c/Users/miche/Desktop/dete.arff", 'w', encoding='latin-1')
            import arff
            obj = {
                          'description': u'mention',
                          'relation':"men",
                          'attributes': [
                          ('gn', ['MASC', 'FEM']),
                          ('num', ['SING','PLUR']),
                          ('en', ['PERS','LOC','ORG','FUNC']),
                          ('gp', ['CASE=YES', 'CASE=NO']),
                          ('id_form',['YES','NO']),
                          ('subform',['YES','NO']),
                          ('inc_rate','REAL'),
                          ('dis_men','REAL'),
                          ('dis_phrase','REAL'),
                          ('dis_char','REAL'),
                        ],
                       }                  
            arff.dump(obj,b)
            uma1=""
            tk=0
            fi=None
            fo=None
            tktext=0
            #On lit le fichier et on cree la liste des phrases separées par un point
            lines=str(file.read())
            mots=lines
            list=lines.split(".")
            con = 0
            no=None
            si=""
            o=0
            
            u=0
            # On prende les mentions avec leur contexte droit et gauche"""
            

            with open('data.json') as json_file:  
            results = json.load(json_file)
            for row1 in results:
                save=" "
            # uma2 sera la seule mention. row1 est la mention dans son context, donc on cherche a avoir row[content] qui est la mention en lui enlevant les caracteres superflues """
                uma2= str(row1['content']).replace('[','').replace(']','').replace('\'','').replace(',','').replace('<start>','').replace('<end>','').replace('\"','').replace("\"m\" ","m'").replace("\"qu\" il","qu'il").replace("\"l\" ","l'").replace("  "," ").replace("\"d\" ","d'").replace("\"n\" ","n'").replace("m ","m'").replace("d ","d'").replace("l ","l'").replace("qu  ","qu'").replace("-","")
                uma2= uma2.replace("*","\'")
                sen = []
                bool=False
                sen.append(uma2[0])       # put first letter in list. First letter doesn't need a space.
                for char in uma2[1::]:         # begin iteration after first letter
                   if char.islower():
                      sen.append(char) # if character is lower add to list
                   elif char.isupper():
                      if bool== True:
                         sen.append("\'") # then add the upper case character to list 
                         sen.append(char) # then add the upper case character to list  
                      else:
                         sen.append("-") # if character is upper add a space to list
                         sen.append(char) # then add the upper case character to list 
                   elif char==" ":
                      sen.append(" ")
                   elif char=="\'":
                      bool=True  
                   result = ''.join(sen)    # use () join to convert the list to a string
            #Result est donc la mention reconstruite"""
                uma2=result
                
                if uma2 in "":
                   uma2="-------------"
                if uma2=="m":
                   uma2="m'"
                if uma2 in "n":
                   uma2=="n'"
                if uma2 =="en":
                   uma2=" en "
                
                control="#"+str(tktext)
                
                # On etiquette les mentions trouvees avec un #plus un nombre"""
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
            
            si=si+lines
            # si sera donc notre texte etiquetté"""
            print(si)
            # On appel mXS qui va traiter le texte etiquetté"""
            import os
            os.system("cd /home/mike/mXS")
            os.system("echo \""+mots+"\" | ./bin/tagEtapeModelPLOP.sh > /mnt/c/Users/miche/Desktop/entity.txt")
            b=open('/mnt/c/Users/miche/Desktop/entity.txt')
            pine = str(b.read())
            u=0
            print(pine)
            stringa=""
           #On reprende les mentions et on refait le meme travaille pour la reconstruction
            for row in results:
                
                uma2= str(row['content']).replace('[','').replace(']','').replace('\'','').replace(',','').replace('<start>','').replace('<end>','').replace('\"','').replace("\"m\" ","m'").replace("\"qu\" il","qu'il").replace("\"l\" ","l'").replace("  "," ").replace("\"d\" ","d'").replace("\"n\" ","n'").replace("m ","m'").replace("d ","d'").replace("l ","l'").replace("qu  ","qu'").replace("-","")
                uma2= uma2.replace("*","\'")
                sen = []
                bool=False
                sen.append(uma2[0])       # put first letter in list. First letter doesn't need a space.
                for char in uma2[1::]:         # begin iteration after first letter
                   if char.islower():
                      sen.append(char) # if character is lower add to list
                   elif char.isupper():
                      if bool== True:
                         sen.append("\'") # then add the upper case character to list 
                         sen.append(char) # then add the upper case character to list  
                      else:
                         sen.append("-") # if character is upper add a space to list
                         sen.append(char) # then add the upper case character to list 
                   elif char==" ":
                      sen.append(" ")
                   elif char=="\'":
                      bool=True  
                   result = ''.join(sen)    # use () join to convert the list to a string
                uma2=result
                #On donc a nouveau une etiquette a les mentions et on les cherches dans le texte etiquetté
                #uma2 sera la mention, uma22 la mention etiquetté et uma1 la phrase qui contient la mention
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
               # On initialise gp a NO et on utilise udpipe pour tokenizer les mentions
                gp1 = "CASE=NO"
                sentences = model.tokenize(uma2)
                
                #on lance donc le parseur et le tagger de udpipe et on recuper les resultats dans un format matxin similaire a XML avec des balises.
                for s in sentences:
                   model.tag(s)
                   model.parse(s)
                
                conllu=model.write(sentences,"matxin")
                #On prends l'arborescence et on fait la tokenisation par que de la phrase avec la mention
               
                # Une fois trouvé la tete de la mention, on cherche dans la meme ligne , les elements: gendre nombre 
                #Pour le gp, on regarde les fils de l'element tete de la mention, si la ligne : "si=case" est trouvée alors la mention est dans un gp
                i1=0
                cas1=0
                
               
                i=0
                cas=0
                conllu = str(conllu).split("\n")
                kas=0
                en1 =""
                gn1=""
                num1=""
                el = si.find(uma22)
                
                gm=model.tokenize(si[el-4:el+len(uma22)])
                for n in gm:
                    model.tag(n)
                    model.parse(n)
                gm = model.write(gm,"matxin")
                gm=gm.split("\n")
                if case:
                  
                   for k in gm:
                         
                       if "si=\"root\"" in k:
                          kas=i1
                       
                      
                       if "si=\"case\"" in k and i1==(kas+1):
                          gp="CASE=YES"
                       
                       i1=i1+1
                
                pine=pine.replace("' ","'")
                #Pour l'entité nommée on reprends la mention etiquetté et on la recherche dans le texte etiquetté et passé a mXS : pine
                #On recherche la mention et on regarde a sa gauche pour trouver la balise <func>,<org> etc...
                Stringa_comparaison=uma22
                uma5=uma2
                print(uma5)
                if "'" in uma5[:2]:
                   uma5=uma2[2:]
                if " " in uma5[:3]:
                   uma5=uma2[3:]
                elif " " in uma5[:4]:
                   uma5=uma2[4:]
                print(uma5)
                entity = pine.find(uma5)
                print(entity)
                ent=pine[entity-7:entity]
                print(ent)
                if "func" in ent:
                    en="FUNC"
                if "pers" in ent:
                    en="PERS"
                if "loc" in ent:
                    en="LOC"
                if "org" in ent:
                    en="ORG"
                if case:
                   if uma2 in stringa and str(con-1) in stringa:
                      gp="CASE=YES"
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
                       stringa=uma22
                       case=True
                    i=i+1
                        
               
                
                     
                
     
                
           
                
                #Une fois fait on enleve des metions les mentions non correctes, c'est a dire les mentions ou la tete est un verb ou une ponctoition
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
                uma2=uma2.replace('é','e')
                #on ecris les resultats dans le fichier arff et dans le fichier json
                with smart_open("/mnt/c/Users/miche/Desktop/det.json", 'a', encoding='latin-1') as g :
                     data={"mention":[{"mention" : uma2 , "Entité Nommée": en1 ,"Gendre": gn1 ,"Nombre": num1 ,"GP": gp1}]}
                     
                     json.dump(data,g,ensure_ascii=False, indent=4)
                     
                    
                #On va faire la meme chose pour chaque couple de mentions. Donc mention1, mention2, mention1, mention3 ectt, double for
                for row2 in results:
                    uma2= str(row['content']).replace('[','').replace(']','').replace('\'','').replace(',','').replace('<start>','').replace('<end>','').replace('\"','').replace("\"m\" ","m'").replace("\"qu\" il","qu'il").replace("\"l\" ","l'").replace("  "," ").replace("\"d\" ","d'").replace("\"n\" ","n'").replace("m ","m'").replace("d ","d'").replace("l ","l'").replace("qu  ","qu'").replace("-","")
                    uma2= uma2.replace("*","\'")
                    sen = []
                    bool=False
                    sen.append(uma2[0])       # put first letter in list. First letter doesn't need a space.
                    for char in uma2[1::]:         # begin iteration after first letter
                        if char.islower():
                           sen.append(char) # if character is lower add to list
                        elif char.isupper():
                             if bool== True:
                                sen.append("\'") # then add the upper case character to list 
                                sen.append(char) # then add the upper case character to list  
                             else:
                                sen.append("-") # if character is upper add a space to list
                                sen.append(char) # then add the upper case character to list 
                        elif char==" ":
                             sen.append(" ")
                        elif char=="\'":
                             bool=True  
                        result = ''.join(sen)    # use () join to convert the list to a string
                    uma2=result
                
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
                
                
                    for s in sentences:
                        model.tag(s)
                        model.parse(s)
                
                    conllu=model.write(sentences,"matxin")
                    i1=0
                    cas1=0
                
               
                    i=0
                    cas=0
                    conllu = str(conllu).split("\n")
                    kas=0
                    en =""
                    gn=""
                    num=""
                    el = si.find(uma22)
                    Stringa_comparaison1=uma22
                    gm=model.tokenize(si[el-4:el+len(uma22)])
                    for n in gm:
                        model.tag(n)
                        model.parse(n)
                    gm = model.write(gm,"matxin")
                    gm=gm.split("\n")
                    if case:
                  
                       for k in gm:
                         
                           if "si=\"root\"" in k:
                              kas=i1
                       
                      
                           if "si=\"case\"" in k and i1==(kas+1):
                              gp="CASE=YES"
                       
                           i1=i1+1
                
                    pine=pine.replace("' ","'")
                    uma5=uma2
                    print(uma5)
                    if "'" in uma5[:2]:
                       uma5=uma2[2:]
                    if " " in uma5[:3]:
                       uma5=uma2[3:]
                    elif " " in uma5[:4]:
                       uma5=uma2[4:]
                    print(uma5)
                    entity = pine.find(uma5)
                    print(entity)
                    ent=pine[entity-7:entity]
                    print(ent)
                    if "func" in ent:
                       en="FUNC"
                    if "pers" in ent:
                       en="PERS"
                    if "loc" in ent:
                       en="LOC"
                    if "org" in ent:
                       en="ORG"
                    if case:
                       if uma2 in stringa and str(con-1) in stringa:
                          gp="CASE=YES"
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
                              stringa=uma22
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
                    uma2=uma2.replace('é','e')
                    id_form= False
                    if Stringa_comparaison == Stringa_comparaison1:
                       id_form='YES'
                    else:
                       id_form='NO'
                    ciao = Stringa_comparaison.split('#')
                    ciao1= Stringa_comparaison1.split('#')
                    dis_men= int(ciao1[1])-int(ciao[1])
                    kl=0
                    kl1=0
                    for line in list:
                        if Stringa_comparaison in line:
                           linea = kl
                        else:
                           kl=kl+1
                    for line1 in list:
                        if Stringa_comparaison in line1:
                           linea1 = kl1
                        else:
                           kl1=kl1+1
                    dis_phrase=linea1-inea
                    it = si.find(Stringa_comparaison1)
                    it1=si.find(Stringa_comparaison)
                    dis_char=it-it1
                    if Stringa_comparaison1 in Stringa_comparaison:
                       sub_form='YES'
                    else:
                       sub_form='NO'
                    from fuzzywuzzy import fuzz
                    inc_rate=fuzz.ratio(String_comparaison1, Stringa_comparaison)
                    with smart_open("/mnt/c/Users/miche/Desktop/det.json", 'a', encoding='latin-1') as g :
                         data={"mention":[{"mention" : uma2 , "Entité Nommée": en ,"Gendre": gn ,"Nombre": num ,"GP": gp}]}
                     
                         json.dump(data,g,ensure_ascii=False, indent=4)
                         a={[gn1,gn,en1,en,num1,num,gp1,gp,id_form, sub_form,inc_rate,dis_men,dis_phrase, dis_char]}
                         arff.dump(b, a, relation="men") 
                    
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
                con=con+1
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
                   print("")
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
