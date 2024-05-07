from flask import Flask, render_template, request, jsonify
from chat_glm import VisualGGLM
from llms.applications.knowledge_extraction import ScoringAgent
import argparse
from llms.applications.tmp_utils import set_logger
from llms.remote import RemoteLLMs
from llms.remote.ChatGPT import ChatGPTLLM
import json

chatglm = VisualGGLM()

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default="./llms/remote/configs/wsx_gpt35.json")
args = parser.parse_args()

# 定义一个Logger
logger = set_logger("tmp.log")

# 定义一个Agent
chat_gpt = ChatGPTLLM(args.config_path)

# 定义参数
task_name = "Knowledge Excavation"
language = "Chinese"

more_guidance = ['Each score should have two digits.']

in_context_examples = [
    {
        "Input": {
            "text" : "比尔·盖茨和保罗·艾伦于1975年在美国的新墨西哥州阿尔伯克基创办了微软公司。"
        },
        "Output": {
            "person": "比尔·盖茨，保罗·艾伦",
            "location": "新墨西哥州",
            "organization": "微软公司",
            "other": "None"
        }
    }
]

result_pattern = {
"person": "str",
"location": "str",
"organization": "str",
"other": "str"
}
score_agent = ScoringAgent(logger, chat_gpt, task_name, result_pattern,
                           language=language, more_guidance=more_guidance, in_context_examples=in_context_examples)
app = Flask(__name__,static_url_path='/static')


def text_replay(text):
    while True:
        response = chatglm.responser(text)

        return response

@app.route('/')
def index():
    return render_template('kenan.html')
@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']
    model_type = request.form['model_type']
    if model_type == 'Local-Glm':
        question = chatglm.create_prompt(str(user_input))
        reply = text_replay(question)
        reply = str('抽取结果为'+reply)
    else:
        data = {
            "{{TEXT}}": str(user_input)
        }
        prompt1, result1 = score_agent.judge_a_case(data)
        result_dict = result1['content']
        result_dict = json.loads(result_dict)
        reply = '抽取结果如下：\n'
        for key, item in result_dict.items():
            reply = reply + str(key) + '包含' + str(item) + '\n'
    return jsonify({'response': reply})
#
#
if __name__ == '__main__':
    with app.app_context():
        app.run(debug=True)