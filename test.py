from transformers import AutoTokenizer,AutoModel

tokenizer = AutoTokenizer.from_pretrained('glm_model',trust_remote_code=True)
model = AutoModel.from_pretrained('glm_model',trust_remote_code=True).half().cuda()
model  = model.eval()
qusetion = '"ljq和lrx今晚要在云南大学电影院看电影"请对这句话进行知识抽取，抽取其中的人物，地点，机构，其他，如果这句话里面不包含人物，地点，机构，其他，则将对应值设置为‘无’'
response,history = model.chat(tokenizer,qusetion,history=[])
print('抽取结果如下：\n'+response)
for line in response.split('\n'):
    item_name = line.split(',')[0]
    result = line.split(',')[1]
    print(str(item_name)+':'+str(result))
