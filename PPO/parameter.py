# import os
# import pickle
# value_add=[]
# value={}
# from tensorflow.python import pywrap_tensorflow
# checkpoint_path = "/home/shared/software/uncased_L-12_H-768_A-12/bert_model.ckpt"
# reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
# var_to_shape_map = reader.get_variable_to_shape_map()
# for key in var_to_shape_map:
#     # print("tensor_name: ", key)
#     # print reader.get_tensor(key).shape
#     ##"cls/predictions/output_bias"
#     if "word_embeddings" in key:
#         value_add.append(key)
#         word_embeddings = reader.get_tensor(key)
#     if "cls/predictions/output_bias" in key:
#         bias = reader.get_tensor(key)

# for v in value_add:
#     print v


    # input_path =path + dataset + str
    # output_path = path + dataset + str
    # f = h5py.File(input_path,"r")
    # sen = list(f["sen"])
    # f.close()
    # bc = BertClient()
    # my_sentences = [str(s) for s in sen]
    # # doing encoding in one-shot
    # vec = bc.encode(my_sentences)
    # print("saving")
    # numpy.save(output_path,vec)
    # print("saved")

# print word_embeddings.shape, bias.shape
# data_path = "/mnt/WDRed4T/ye/DataR/YAHOO/"
# bert_weight_path = data_path + "bert_w_b"
#
#
# result = {'bert_W': word_embeddings, 'bert_b': bias}
# output = open(bert_weight_path,"w")
# pickle.dump(result, output)
# output.close()
#
# f = open(bert_weight_path, "rb")
# data = pickle.load(f)
# output_weights = data["bert_W"]
# output_bias = data["bert_b"]