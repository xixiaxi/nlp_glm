from transformers import AutoModel,AutoTokenizer
from prompt import prompt_test

class VisualGGLM:

    def __init__(self):
        self.model_path = './glm_model'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path,trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_path,trust_remote_code=True).half().cuda()
        self.model = self.model.eval()

    def create_prompt(self,text):
        prompt = prompt_test.format(text)
        prompt = str(prompt)

        return prompt
    def responser(self,question):
        response,history = self.model.chat(self.tokenizer,question,history=[])

        return response

if __name__ == '__main__':
    visual_glm = VisualGGLM()
    question_input = input('请输入您的问题：')
    question = visual_glm.create_prompt(question_input)
    response = visual_glm.responser(question)
    print('抽取结果是：\n',response)