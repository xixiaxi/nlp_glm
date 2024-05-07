import json
import logging
import argparse
import openai
from openai import OpenAI
import time
import socket
import os

from llms.remote import RemoteLLMs
from nltk import sent_tokenize, word_tokenize, pos_tag
from itertools import product


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/wsx_gpt35.json")
    args = parser.parse_args()
    return args


class ChatGPTLLM(RemoteLLMs):
    def init_local_client(self):
        try:
            self.model = self.args['model']
            client = OpenAI(api_key=self.args['api_key'], base_url=self.args['base_url'])
            return client
        except:
            return None

    def create_prompt(self, current_query, context=None):
        if context is None:
            context = []
        context.append(
            {
                "role": "user",
                "content": current_query,
            }
        )
        # # 提取常识知识三元组
        # knowledge_triplets = self.extract_knowledge_triplets(current_query)
        # for triplet in knowledge_triplets:
        #     context.append({
        #         "role": "system",
        #         "content": triplet,
        #     })

        return context

    def extract_knowledge_triplets(self, text):
        """
        从文本中提取常识知识三元组。

        Args:
            text (str): 要提取知识的文本。

        Returns:
            list: 常识知识三元组的列表，每个三元组为（头实体，关系，尾实体）。
        """
        # 将文本分解为句子
        sentences = sent_tokenize(text)
        triplets = []

        for sentence in sentences:
            # 将句子分解为单词并标注词性
            words = word_tokenize(sentence)
            tagged_words = pos_tag(words)

            # 寻找名词对
            nouns = [word for word, pos in tagged_words if pos.startswith('NN')]
            noun_pairs = list(product(nouns, repeat=2))

            # 构建三元组
            for pair in noun_pairs:
                triplet = (pair[0], "is_related_to", pair[1])  # 这里可以根据实际情况定义关系
                triplets.append(triplet)

        return triplets

    def request_llm(self, context, seed=1234, sleep_time=1, repeat_times=0):
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=context,
                    stream=False,
                    seed=seed+repeat_times
                )
                context.append(
                    {
                        'role': response.choices[0].message.role,
                        'content': response.choices[0].message.content
                    }
                )
                return context
            except openai.RateLimitError as e:
                logging.error(str(e))
                raise e
            except (openai.APIError, openai.InternalServerError, socket.timeout) as e:
                logging.error(str(e))
                raise e
            except Exception as e:
                # 捕捉未预料的异常，考虑是否终止循环或做其他处理
                logging.error(f"An unexpected error occurred: {str(e)}")
                raise e
            time.sleep(sleep_time)

if __name__ == '__main__':
    # https://platform.openai.com/docs/api-reference
    args = read_args()
    chat_gpt = ChatGPTLLM(args.config_path)
    chat_gpt.interactive_dialogue()