


TEXT_EVAL_METRICS = {
    "Fluency": """ Fluency: The fluency of the '{{TGT}}'. """,
    "Relevance": """ Relevance: The semantic consistency and relevance between '{{SRC}}' and '{{TGT}}'. """,
    "Informativeness": """ Informativeness: Does '{{TGT}}' contain sufficient and rational information.""",
}

TEXT_EVAL_GENERAL_PROMPT_PATTERN = """
[Task Description]
Here is a point-wise {{TASK_NAME}} task. All [Input] are in {{Language}}.
{{MORE_TASK_DEFINITION}}
Your are required to acted as a professional native-speaker human annotator to judge the given {{TGT}} in [Input].
Your evaluation should follow the [Criteria] and  [Guidance].
The output format should follow the [Output Format].

[Guidance]
You should strictly follow my guidance:
1. Each score is between {{MIN_SCORE}} (lowest) and {{MAX_SCORE}} (highest).
2. Each score should be {{DATATYPE}} score.
3. You should strictly follow the given output format and can't output other information.
{{MORE_GUIDANCE}}
If you break my guidance, you will be penalized.

[Criteria]
{{Criteria}}

{{In-Context Examples}}

[Output Format]
Your output should strictly follow this format and can be directly decoded by Python:
'''
{{Output}}
'''

[Input]
'''
{
    "{{SRC}}": {{SRC_VALUE}},
    "{{TGT}}": {{TGT_VALUE}}
}
'''

"""

TEXT_KNOWLEDGE_EXTRACTION_PROMPT_PATTERN =  """
[Task Description] 
Here is a point-wise named entity extraction task. All [Input] are in Chinese.
{{MORE_TASK_DEFINITION}}
You are required to act as a professional native-speaker human annotator to extract named entities from the given text and categorize them into different types (e.g., person, location, organization).
Your extraction should follow the [Criteria] and [Guidance].
The output format should follow the [Output Format].

[Guidance]
You should strictly follow my guidance:
1. Each named entity should be categorized into one of the specified types.
2. You should strictly adhere to the given output format and cannot output additional information.
If you deviate from my guidance, you will be penalized.

[Criteria]
{{Criteria}} 

{{In-Context Examples}}

[Output Format]
Your output should strictly follow this format and can be directly decoded by Python:
'''
{{Output}}
'''

[Input]
'''
{
    "text": "{{TEXT}}"
}
'''
"""
