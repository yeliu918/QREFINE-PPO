# -*- coding: utf-8 -*
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import pickle
import sys
import lucene
import shutil
import os
import itertools
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
# from org.apache.lucene.analysis.cjk.CJKAnalyzer import CJKAnalyzer
from org.apache.lucene.document import Document, Field, FieldType,TextField
from org.apache.lucene.index import FieldInfo, DirectoryReader, IndexWriter, IndexWriterConfig, IndexOptions
from org.apache.lucene.index import IndexReader
from org.apache.lucene.store import SimpleFSDirectory, NIOFSDirectory, MMapDirectory
from org.apache.lucene.util import Version
from org.apache.lucene.search import IndexSearcher, MatchAllDocsQuery, BooleanQuery
from org.apache.lucene.queryparser.classic import QueryParser
from fuzzywuzzy import process

from org.apache.lucene.store import RAMDirectory
from org.apache.lucene.index import DirectoryReader


def make_doc_query(query_name):
    process_file_name = 'Huawei_result/Huawei_language_retrival'
    process_doc = open("Huawei_result/document.txt","w")
    process_query = open("Huawei_result/query.txt","w")
    f = open(process_file_name, 'rb')
    data = pickle.load(f)
    gen_sen = data['gen_sen']
    answer_sen = data['answer_sen']
    num_data = len(gen_sen)
    print "the num of query:", num_data
    eval_rate = 0.9

    # if query_name =="noisy":
    #     search_query = noisy_sen

    if query_name =="gen":
        search_query = gen_sen

    # if query_name =="Seq2Seq":
    #     search_query = gen_Seq
    #
    # if query_name =="Whole":
    #     search_query = gen_Who


    search_answer = answer_sen

    for sen in search_answer:
        process_doc.write(sen+"\n")
    for query in search_query:
        process_query.write(query+"\n")

    process_doc.close()
    process_query.close()

def retrival_answer(MAX):
    lucene.initVM()
    directory = RAMDirectory()

    indexDir = SimpleFSDirectory(Paths.get('index'))
    writerConfig = IndexWriterConfig(StandardAnalyzer())
    writer = IndexWriter(directory, writerConfig)

    print "%d docs in index" % writer.numDocs()
    print "Reading lines from Document..."


    process_doc = open("Huawei_result/document.txt","r")
    doc_line = process_doc.readlines()
    for l in doc_line:
        doc = Document()
        doc.add(TextField("text", l, Field.Store.YES))
        writer.addDocument(doc)
    print "Indexed from %d docs in index" % (writer.numDocs())
    print "Closing index of %d docs..." % writer.numDocs()
    writer.close()

    accuracy=[]
    process_query = open("Huawei_result/query.txt","r")
    query_line = process_query.readlines()
    for n, one_query in enumerate(query_line):
        analyzer = StandardAnalyzer()
        # reader = IndexReader.open(SimpleFSDirectory(Paths.get('index')))
        searcher = IndexSearcher(DirectoryReader.open(directory))
        # searcher = IndexSearcher(reader)
        query = QueryParser("text", analyzer).parse(one_query)
        hits = searcher.search(query, MAX)
        # print "Found %d document(s) that matched query '%s':" % (hits.totalHits, query)
        # print "The groundtruth document is:", doc_line[n]
        candidate_doc=[]
        for hit in hits.scoreDocs:
            # print hit.score, hit.doc, hit.toString()
            doc = searcher.doc(hit.doc)
            # print doc.get("text").encode("utf-8")
            candidate_doc.append(doc.get("text"))

        choices = process.extract(unicode(doc_line[n]), candidate_doc)
        flag = 0
        for i in range(len(choices)):
            if choices[i][1] >= 89:
                flag=1
        if flag==1:
            accuracy.append(1)
        else:
            accuracy.append(0)

    final_accuracy=float(sum(accuracy))/float(len(accuracy))

    print "the final accuracy is:", final_accuracy

def main():
    # print "the number of MAX retrival:\n"
    # MAX=10
    # # print "the clean query:"
    # # make_doc_query("clean")
    # # retrival_answer(MAX)
    # # print "the noisy query:"
    # # make_doc_query("noisy")
    # # retrival_answer(MAX)
    # print "the Seq2Seq query:\n"
    # make_doc_query("Seq2Seq")
    # retrival_answer(MAX)
    # print "the Whole query:\n"
    # make_doc_query("Whole")
    # retrival_answer(MAX)
    #
    # print "the number of MAX retrival:\n"
    # MAX=50
    # # print "the clean query:"
    # # make_doc_query("clean")
    # # retrival_answer(MAX)
    # # print "the noisy query:"
    # # make_doc_query("noisy")
    # # retrival_answer(MAX)
    # print "the Seq2Seq query:\n"
    # make_doc_query("Seq2Seq")
    # retrival_answer(MAX)
    # print "the Whole query:\n"
    # make_doc_query("Whole")
    # retrival_answer(MAX)

    print "the number of MAX retrival:\n"
    MAX=100
    # print "the clean query:"
    # make_doc_query("clean")
    # retrival_answer(MAX)
    # print "the noisy query:"
    # make_doc_query("noisy")
    # retrival_answer(MAX)
    print "the Seq2Seq query:\n"
    make_doc_query("Seq2Seq")
    retrival_answer(MAX)
    print "the Whole query:\n"
    make_doc_query("Whole")
    retrival_answer(MAX)

    print "the number of MAX retrival:\n"
    MAX=1000
    # print "the clean query:"
    # make_doc_query("clean")
    # retrival_answer(MAX)
    # print "the noisy query:"
    # make_doc_query("noisy")
    # retrival_answer(MAX)
    print "the Seq2Seq query:\n"
    make_doc_query("Seq2Seq")
    retrival_answer(MAX)
    print "the Whole query:\n"
    make_doc_query("Whole")
    retrival_answer(MAX)

if __name__ == "__main__":
    main()







# if __name__ == "__main__":

