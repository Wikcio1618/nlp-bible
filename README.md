# NLP Bible
Bible related projects that employ NLP methodology to this ancient text. In each folder, different endeavor is conducted

## Structure
* *data* - data from [OGNT repo](https://github.com/eliranwong/OpenGNT/tree/master) and processed token table of New Testament is stored. OGNT stands for Open Greek New Testament project and provides Nestle-Aland 27 and Nestle Aland-28 Text
* *nt_author_wording_stats* - some NT stats are displayed in *.ipynb* notebook:
  * Word histogram over authors
  * Word histogram over books
  * Lemma histogram over authors
  * Lemma histogram over books
  * New unique lemmas occurance over each NT book (cf. 17th chapter of Gospel of John introduces very few new lemmas, conversely 6th and 8th chapters of the same book introduce many new lemmas at once) 
* *authorship-attr* - in this directory we attruibute authorship of Epistle of Hebrews based on traditional authorships of other books. N-gram similarity metric is used following [this paper](https://www.cnts.ua.ac.be/~walter/educational/material/Stamatatos_survey2009.pdf) page 547 (access: February 2025)


