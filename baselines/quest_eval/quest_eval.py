import os
import re
import json
import copy
import jieba
import logging
import numpy as np
from openai import OpenAI
from collections import Counter
from abc import ABC, abstractmethod

jieba.setLogLevel(logging.ERROR)

json_response = """
{
    "key_info": ["Newly installed grid-connected photovoltaic capacity of 10.6 million kilowatts", "One-quarter", "National newly added photovoltaic power station capacity of 8.55 million kilowatts", "Distributed photovoltaic capacity of 2.05 million kilowatts", "China's photovoltaic power generation in 2014 reached 25 billion kilowatt-hours", "Year-on-year growth exceeded 200%"],

    "question": ["What was China's newly installed grid-connected photovoltaic capacity in 2014?", "What fraction of the global newly installed capacity did China's newly installed grid-connected photovoltaic capacity account for in 2014?", "What was the capacity of newly added photovoltaic power stations nationwide?", "What was the capacity of distributed photovoltaics?", "What was China's photovoltaic power generation in 2014?", "By how much did China's photovoltaic power generation grow compared to the previous year in 2014?"]
}
"""


class BaseLLM(ABC):
    def __init__(
            self, 
            model_name: str = "gpt-3.5-turbo", 
            temperature: float = 1.0,
            max_new_tokens: int = 1024, 
            top_p: float = 0.9,
            top_k: int = 5,
            **more_params
        ):
        self.params = {
            'model_name': model_name if model_name else self.__class__.__name__,
            'temperature': temperature,
            'max_new_tokens': max_new_tokens,
            'top_p': top_p,
            'top_k': top_k,
            **more_params
        }

    def update_params(self, inplace: bool = True, **params):
        if inplace:
            self.params.update(params)
            return self
        else:
            new_obj = copy.deepcopy(self)
            new_obj.params.update(params)
            return new_obj

    @abstractmethod
    def request(self, query:str) -> str:
        return ''

    def safe_request(self, query: str) -> str:
        """Safely make a request to the language model, handling exceptions."""
        response = self.request(query)
        return response

class GPT(BaseLLM):
    def __init__(self, model_name='gpt-3.5-turbo', temperature=1.0, max_new_tokens=1024, report=False):
        super().__init__(model_name, temperature, max_new_tokens)
        self.report = report
        self.client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ["OPENAI_API_BASE"],
        )

    def request(self, query: str) -> str:
        res = self.client.chat.completions.create(
            model = self.params['model_name'],
            messages = [{"role": "user","content": query}],
            temperature = self.params['temperature'],
            max_tokens = self.params['max_new_tokens'],
            top_p = self.params['top_p'],
        )
        real_res = res.choices[0].message.content
        return real_res

class QuestEval(GPT):
    def __init__(self, model_name='gpt-3.5-turbo', temperature=1.0, max_new_tokens=1024, report=False):
        super().__init__(model_name, temperature, max_new_tokens)
        self.report = report
        self.quest_gt_save = {}

    def parse_json(self, text: str) -> dict:
        try:
            start = text.find('{')
            end = text.rfind('}')
            assert start >= 0 or end >= 0, "Model response must contain JSON"
            json_content = text[start:end+1].replace('\\','\\\\')
            return json.loads(json_content)
        except Exception as e:
            raise e
    
    def question_generation(self, text4gen: str):
        prompt = self._read_prompt_template("quest_eval_gen.txt")
        query = prompt.format(json_response=json_response, news=text4gen)
        extracted_content = self.safe_request(query)
        question4eval = self.parse_json(extracted_content)
        return question4eval

    def question_answer(self, context, question):
        template = self._read_prompt_template('quest_eval_answer.txt')
        query = template.format(
            context=context,
            questions=question
        )
        answers = self.safe_request(query)
        
        pattern = r'<response>\n(.*?)\n</response>'
        real_answers = re.findall(pattern, answers, re.DOTALL)
        return real_answers
    
    def _read_prompt_template(self, filename: str):
        path = os.path.join(os.path.dirname(__file__), 'prompts/', filename)
        if os.path.exists(path):
            with open(path) as f:
                return f.read()
        else:
            return ''
    
    def get_QA_pair(self, data_point: dict):
        ground_truth_text = data_point["ground_truth"]
        generated_text = data_point["answer"]
        
        # if data_point["id"] in self.quest_gt_save.keys():
        #    questions_gt = self.quest_gt_save[data_point["id"]]["question"]
        #    answers_gt4gt = self.quest_gt_save[data_point["id"]]["answers"]
        #else:
        keyinfo_and_questions = self.question_generation(ground_truth_text)
        questions_gt = keyinfo_and_questions["question"]           
        answers_gt4gt = self.question_answer(ground_truth_text, questions_gt) 
        
        keyinfo_and_questions["answers"] = answers_gt4gt
            # self.quest_gt_save[data_point["id"]] = keyinfo_and_questions
    
        answers_gm4gt = self.question_answer(generated_text, questions_gt) 

        return questions_gt, answers_gt4gt, answers_gm4gt

    def quest_eval(self, data_point: dict):
        questions_gt, answers_gt4gt, answers_gm4gt = self.get_QA_pair(data_point)

        quest_eval_save = {}
        quest_eval_save["questions_gt"] = questions_gt
        quest_eval_save["answers_gt4gt"] = answers_gt4gt
        quest_eval_save["answers_gm4gt"] = answers_gm4gt

        indices = [i for i, x in enumerate(answers_gt4gt) if x != "Unanswerable"]
        answers_gm4gt = [answers_gm4gt[i] for i in indices]
        answers_gt4gt = [answers_gt4gt[i] for i in indices]

        if len(answers_gm4gt) == 0:
            return 0, 0, quest_eval_save

        undetermined_ratio = answers_gm4gt.count("Unanswerable") / len(answers_gm4gt)
        quest_recall = 1 - undetermined_ratio

        indices = [i for i, x in enumerate(answers_gm4gt) if x != "Unanswerable"]
        answers_gm4gt = [answers_gm4gt[i] for i in indices]
        answers_gt4gt = [answers_gt4gt[i] for i in indices]
        
        if answers_gm4gt == []:
            return 0, 0, quest_eval_save

        quest_avg_f1 = word_based_f1_score(answers_gt4gt, answers_gm4gt)
        
        return quest_avg_f1, quest_recall, quest_eval_save

def compute_f1(a_gold, a_pred):
    gold_toks = list(jieba.cut(a_gold)) 
    pred_toks = list(jieba.cut(a_pred)) 
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def word_based_f1_score(a_gold_list, a_pred_list):
    f1_list=[]
    for a_gold,a_pred in zip(a_gold_list, a_pred_list):
        f1_list.append(compute_f1(a_gold,a_pred))
    return np.mean(f1_list)
