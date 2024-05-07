import argparse
import json

from llms.applications.prompts import TEXT_KNOWLEDGE_EXTRACTION_PROMPT_PATTERN, TEXT_EVAL_METRICS
from llms.applications.tmp_utils import set_logger
from llms.remote import RemoteLLMs
from llms.remote.ChatGPT import ChatGPTLLM


class ScoringAgent:
    def __init__(self, logger, llm_model: RemoteLLMs, task_name: str, metrics: dict,
                 language,  in_context_examples=[],
                 more_guidance=[], more_task_definition=[]):

        self.logger = logger
        self.llm_model = llm_model
        self.prompt_pattern = TEXT_KNOWLEDGE_EXTRACTION_PROMPT_PATTERN
        self.result_pattern = metrics

        # 处理额外的输入
        more_task_definition = '\n'.join(more_task_definition)

        # 评价的指标
        metric_dict = dict()
        data_type = None

        for metric, score_format in metrics.items():
            metric_dict[metric] = "[Your Result]"
            items = score_format.split('_')

            if data_type is None:
                data_type = items[0]
            else:
                assert data_type == items[0]


        output_pattern = json.dumps(metric_dict, ensure_ascii=False, indent=4)

        # In-Context Examples 的设置
        if len(in_context_examples) > 0:
            more_guidance.append('To help your judgment, some examples are provided in [Examples].')
            in_context_prompt = ["[Examples]", "'''"]
            in_context_prompt.append(json.dumps(in_context_examples, ensure_ascii=False, indent=4))
            in_context_prompt.append("'''")
            in_context_prompt = '\n'.join(in_context_prompt)
        else:
            in_context_prompt = ""

        # 是否有更多需要补充的指南
        tmp = []
        for idx, guidance in enumerate(more_guidance):
            tmp.append('%s. %s' % (idx + 3, guidance))
        more_guidance = '\n'.join(tmp)

        # 根据评价指标设置评价的标准
        input_format = {}
        criteria = []

        criteria = '\n'.join(criteria)

        # 给定输入的模板格式
        input_format = json.dumps(input_format, ensure_ascii=False, indent=4)

        self.meta_dict = {
            "{{Language}}": language,
            "{{Output}}": output_pattern,
            "{{Input}}": input_format,
            "{{Criteria}}": criteria,
            "{{TASK_NAME}}": task_name,
            "{{MORE_GUIDANCE}}": more_guidance,
            "{{MORE_TASK_DEFINITION}}": more_task_definition,
            "{{In-Context Examples}}": in_context_prompt
        }

    def judge_a_case(self, case_data):
        llm_model = self.llm_model
        repeat_times = -1

        while True:
            repeat_times += 1
            if repeat_times >= llm_model.max_retries:
                break
            # 首先构造prompt
            prompt = llm_model.fit_case(pattern=self.prompt_pattern, data=case_data, meta_dict=self.meta_dict)
            contexts = llm_model.create_prompt(prompt)
            results = llm_model.request_llm(contexts, repeat_times=repeat_times)

            return prompt,results[-1]


if __name__ == '__main__':
    # https://platform.openai.com/docs/api-reference

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="../remote/configs/wsx_gpt35.json")
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
    # while True:
    #     text = input('您的文本是：')
    #     data = {"{{TEXT}}": str(text)}
    #     prompt, result = score_agent.judge_a_case(data)
    #     print(result)
    data1 = {
        "{{TEXT}}": "比尔·盖茨和保罗·艾伦于1975年在美国的新墨西哥州阿尔伯克基创办了微软公司"
    }
    prompt1,result1 = score_agent.judge_a_case(data1)
    result_dict = result1['content']
    result_dict = json.loads(result_dict)
    reply = '抽取结果如下：\n'
    for key,item in result_dict.items():
        reply = reply + str(key) +'包含'+str(item)+' '
    print(reply)



