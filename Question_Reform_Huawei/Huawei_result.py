import pickle
import matplotlib.pyplot as plt
import numpy as np

file1="Huawei_result/Huawei_bleu1_0.723041901761"
file2="Huawei_result/Huawei_bleu1_0.57488804814"
f1 = open(file1, "r")
f2 = open(file2, "r")

fname_result ="Huawei_result/Generated_question_Huawei"
f3 = open(fname_result,"wb")

whole_line = f1.readlines()
Seq2Seq_line = f2.readlines()
answer = []
nque = []
cque = []
Seq_que = []
Who_que = []


def get_length(seq):
    seq_len = []
    for s in seq:
        s_word = s.split(" ")
        seq_len.append(len(s_word))
    return seq_len


for seqline in Seq2Seq_line:
    if "answer:" in seqline:
        answer.append(seqline.replace("answer: ", "").replace("\n",""))
    if "noisy question: " in seqline:
        nque.append(seqline.replace("noisy question: ", "").replace("\n",""))
    if "clean question: " in seqline:
        cque.append(seqline.replace("clean question: ", "").replace("\n",""))
    if "generated question: " in seqline:
        Seq_que.append(seqline.replace("generated question: ", "").replace("\n",""))

for wholine in whole_line:
    if "generated question: " in wholine:
        Who_que.append(wholine.replace("generated question: ", "").replace("\n",""))

for i in range(len(Who_que)):
    f3.write("answer: " + answer[i].replace("answer: ", "") + "\n")
    f3.write("noisy question: " + nque[i].replace("noisy question: ", "") + "\n")
    f3.write("clean question: " + cque[i].replace("clean question: ", "").replace("EOS", "") + "\n")
    f3.write("Seq2Seq question: " + Seq_que[i].replace("generated question: ", "") + "\n")
    f3.write("RL question: " + Who_que[i].replace("generated question: ", "") + "\n")
    f3.write("\n")

print len(answer), len(nque), len(cque), len(Seq_que), len(Who_que)

output_path = 'Huawei_result/Huawei_language_retrival'
result = {'answer_sen': answer, "noisy_sen": nque, "clean_sen": cque, "Seq_gen": Seq_que, "Who_gen": Who_que}
output = open(output_path, 'wb')
pickle.dump(result, output)
output.close()

f3.close()
# Seq_len = get_length(Seq_que)
# print float(sum(Seq_len)) / float(len(Seq_len))
# Who_len = get_length(Who_que)
# print float(sum(Who_len)) / float(len(Who_len))
#
# X = []
# Y = [Seq_len, Who_len]
# X.append("Seq2Seq")
# X.append("Qrefine")
#
# # print "the max and min of Seq2Seq:", max(Seq_len), min(Seq_len)
# # print "the max and min of Whole:", max(Who_len), min(Who_len)
# #figsize=(2, 30)
# fig, ax = plt.subplots()
#
# bp = ax.violinplot(Y, points=40, widths=1, showmeans=False,
#                    showextrema=True, showmedians=True, bw_method=0.5)
#
# for vio in bp['bodies']:
#     vio.set_facecolor('blue')
#     vio.set_edgecolor('black')
#
# fig.autofmt_xdate()
# ax.set_xticks(range(1, 3))
# ax.set_xticklabels(X, fontsize=14)
#
# plt.savefig("violin_test.pdf")
