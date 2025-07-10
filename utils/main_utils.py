'''Utils for the main.py file'''
import os
import json
from utils.api_utils import str2image
from utils.call_agent import ask_agent
import re
from typing import List, Tuple
from statistics import mean
from PIL import Image
import base64

class MainUtils:
    def __init__(self, path2prompts, path2save, agent, obj, tools, system, experiment):
        self.path2prompts = path2prompts
        self.path2save = path2save
        self.agent = agent
        self.obj = obj
        self.tools = tools
        self.system = system
        self.experiment = experiment
        if self.tools.exemplars:
            data_acts = [act for (act, image) in self.tools.dataset_exemplars(self.system)]
            self.avg_acts = mean(data_acts)

    def return_prompt(self, setting='bias_discovery', prev_bias=None):
        with open(f'{self.path2prompts}/api_{setting}.txt', 'r') as file:
            sysPrompt = file.read()
        with open(f'{self.path2prompts}/user_{setting}.txt', 'r') as file:
            user_prompt = file.read()
        if self.obj is not None:
            user_prompt = user_prompt.replace("{0}", self.obj)
        if prev_bias is not None:
            user_prompt = user_prompt.replace("{BIAS}", prev_bias)
        return sysPrompt, user_prompt
    
    def return_user_prompt(self, setting='bias_discovery', prev_bias=None):
        with open(f'{self.path2prompts}/user_{setting}.txt', 'r') as file:
            user_prompt = file.read()
        if self.obj is not None:
            user_prompt = user_prompt.replace("{0}", self.obj)
        if prev_bias is not None:
            user_prompt = user_prompt.replace("{BIAS}", prev_bias)
        return user_prompt

    def _get_field(self, history, field_name):
        text2save = None
        # print(history[-1])
        if history[-1]['role'] == 'assistant':
            for entry in history[-1]['content']:
                if field_name in entry["text"]:
                    text2save = entry["text"].split(field_name)[-1].split('.')[0].strip()
        return text2save

    # def save_history(self, history, round=0):
    #     with open(f'{self.path2save}/history_{round}.json', 'w') as file:
    #         json.dump(history, file)
    #     file.close()

    # def save_dialogue(self, history, round=0):
    #     self.save_history(history, round)
        # save_field(history, path2save+'/description.txt', '[DESCRIPTION]: ')
        # save_field(history, path2save+'/label.txt', '[LABEL]: ', end=True)
        # self.save_field(history, f'{self.path2save}/bias_{round}.txt', '[BIAS]:')

    def _get_prompts(self, prev_bias):
        def parse_resp(response: str):
            """
            Parse the response to extract positive and negative prompts.
            
            Args:
                response: The response text containing tagged prompts
            
            Returns:
                Lists of positive and negative prompts
            """
            # Extract content between tags using regex
            positive_pattern = r"<POSITIVE_PROMPTS>(.*?)</POSITIVE_PROMPTS>"
            negative_pattern = r"<NEGATIVE_PROMPTS>(.*?)</NEGATIVE_PROMPTS>"
            
            # Find all matches (using re.DOTALL to match across lines)
            positive_match = re.search(positive_pattern, response, re.DOTALL)
            negative_match = re.search(negative_pattern, response, re.DOTALL)
            
            def clean_prompts(text: str) -> List[str]:
                """Helper function to clean and extract numbered prompts."""
                if not text:
                    return []
                
                # Split into lines and clean
                lines = [line.strip() for line in text.strip().split('\n')]
                
                # Remove empty lines and extract actual prompts
                prompts = []
                for line in lines:
                    # Skip empty lines
                    if not line:
                        continue
                    # Remove numbering (e.g., "1. ", "2. ", etc.)
                    cleaned = re.sub(r'^\d+\.\s*', '', line)
                    if cleaned:
                        prompts.append(cleaned)
                        
                return prompts
            
            positive_prompts = clean_prompts(positive_match.group(1) if positive_match else "")
            negative_prompts = clean_prompts(negative_match.group(1) if negative_match else "")
            
            return positive_prompts, negative_prompts
        
        print("Started scoring images...")
        log = []
        user_prompt = self.return_user_prompt(setting="scorer", prev_bias=prev_bias)
        log.append({'role': 'user', 'content': [{"type":'text', "text": user_prompt}]})
        resp = ask_agent(self.agent, log)
        pos_prompts, neg_prompts = parse_resp(resp)
        if not pos_prompts or not neg_prompts:
            return [], [], [], []
        
        print("Positive scoring image prompts:")
        print(pos_prompts)
        print("Negative scoring image prompts:")
        print(neg_prompts)

        return pos_prompts, neg_prompts

    def save_results(self, round_num=0):
        """
        Save results of the experiment round to the history.json file at the end of the round.
        
        The file is a list of dictionaries, where each dictionary contains:
        - round_num: the round number
        - bias_conclusion: the bias conclusion from the experiment log
        - pos_acts: the positive confidence scores from the scoring
        - neg_acts: the negative confidence scores from the scoring
        - pos_images_names: the names of the positive images
        - neg_images_names: the names of the negative images
        
        :param round_num: The round number of the experiment
        """
        if os.path.exists(os.path.join(self.path2save, f"history_{self.experiment}.json")):
            with open(os.path.join(self.path2save, f"history_{self.experiment}.json"), 'r') as file:
                history = json.load(file)
            file.close()
        else:
            history = []
        
        for tag in ["[BIAS LABEL]:", "[BIAS LABEL]"]:
            bias_conclusion = self._get_field(self.tools.experiment_log, tag)
            if bias_conclusion:
                break
        if not bias_conclusion:
            return None, None, None

        pos_prompts, neg_prompts = self._get_prompts(bias_conclusion)
        if not pos_prompts or not neg_prompts:
            return None, None, None

        pos_path = os.path.join(self.path2save, f"positive_scoring_images_{self.experiment}", f"round_{round_num}")
        os.makedirs(pos_path, exist_ok=True)
        neg_path = os.path.join(self.path2save, f"negative_scoring_images_{self.experiment}", f"round_{round_num}")
        os.makedirs(neg_path, exist_ok=True)

        pos_images = self.tools.text2image(pos_prompts)
        pos_acts, pos_image_list = self.system.call_classifier(pos_images)
        pos_acts = [round(a, 2) for a in pos_acts]
        print("Positive scoring image confidence scores:")
        print(pos_acts)
        neg_images = self.tools.text2image(neg_prompts)
        neg_acts, neg_image_list = self.system.call_classifier(neg_images)
        neg_acts = [round(a, 2) for a in neg_acts]
        print("Negative scoring image confidence scores:")
        print(neg_acts)

        pos_image_names = []
        neg_image_names = []
        for i in range(len(pos_prompts)):
            buffer = base64.b64decode(pos_image_list[i])
            image_name = os.path.join(pos_path, f"{i}.png")
            with open(image_name, 'wb') as file:
                file.write(buffer)
            pos_image_names.append(image_name)

        for i in range(len(neg_prompts)):
            buffer = base64.b64decode(neg_image_list[i])
            image_name = os.path.join(neg_path, f"{i}.png")
            with open(image_name, 'wb') as file:
                file.write(buffer)
            neg_image_names.append(image_name)

        history.append({'round': round_num, 'bias_conclusion': bias_conclusion, 'pos_images_paths': pos_image_names, 'neg_images_paths': neg_image_names, 'pos_prompts': pos_prompts, 'neg_prompts': neg_prompts, 'pos_acts': pos_acts, 'neg_acts': neg_acts})

        return mean(pos_acts), mean(neg_acts), history
    
    def load_prev_context(self, round_num=0, setting="context"):
        if os.path.exists(os.path.join(self.path2save, f"history_{self.experiment}.json")):
            with open(os.path.join(self.path2save, f"history_{self.experiment}.json"), 'r') as file:
                history = json.load(file)
            file.close()
        else:
            history = []

        def get_act_level(score: int):
            if score < 0.3:
                return "LOW"
            elif score < 0.8:
                return "MODERATE"
            else:
                return "HIGH"
        
        pos_avg = None
        neg_avg = None
        for entry in history:
            if entry['round'] == round_num:
                prev_bias = entry['bias_conclusion']
                context_prompt = self.return_user_prompt(setting=setting, prev_bias=prev_bias)
                self.tools.update_experiment_log(role='user', type="text", type_content=context_prompt)

                pos_acts = entry['pos_acts']
                neg_acts = entry['neg_acts']
                pos_prompts = entry['pos_prompts']
                neg_prompts = entry['neg_prompts']
                pos_images_paths = entry['pos_images_paths']
                neg_images_paths = entry['neg_images_paths']
                
                self.tools.display("[DATASET EXEMPLARS]")
                exemplar_data = self.tools.dataset_exemplars(self.system)
                # Display the confidence score values along with the exemplar images.
                for score, image in exemplar_data:
                    self.tools.display(image, f"Confidence Score Value: {score}\nConfidence Score Level: {get_act_level(score/self.avg_acts)}")
                
                self.tools.display("[POSITIVE EXAMPLES]")
                for i, (image_path, prompt, score) in enumerate(zip(pos_images_paths, pos_prompts, pos_acts)):
                    image = Image.open(image_path)
                    self.tools.display(image, f"Prompt: {prompt}\nConfidence Score Value: {score}\nConfidence Score Level: {get_act_level(score/self.avg_acts)}")
                
                self.tools.display("[NEGATIVE EXAMPLES]")
                for i, (image_path, prompt, score) in enumerate(zip(neg_images_paths, neg_prompts, neg_acts)):
                    image = Image.open(image_path)
                    self.tools.display(image, f"Prompt: {prompt}\nConfidence Score Value: {score}\nConfidence Score Level: {get_act_level(score/self.avg_acts)}")

                self.tools.display(f"Average dataset exemplar confidence score: {round(self.avg_acts, 2)}")
                pos_avg = round(mean(pos_acts), 2)
                neg_avg = round(mean(neg_acts), 2)
                self.tools.display(f"Average positive prompt confidence score: {pos_avg}")
                self.tools.display(f"Average negative prompt confidence score: {neg_avg}")

                # _, user_prompt = self.return_prompt(setting="adversarial_generate_list", prev_bias=prev_bias)
                # self.tools.update_experiment_log(role='user', type="text", type_content=user_prompt)
                break
        
        return pos_avg, neg_avg

    def plot_results_notebook(self, experiment_log):
        for entry in experiment_log:
            if (entry['role'] == 'assistant'):
                print('\n\n*** MAIA: ***\n\n')  
            else: 
                print('\n\n*** Experiment Execution: ***\n\n')
            for item in entry['content']:
                if item['type'] == 'text':
                    print(item['text'])
                elif item['type'] == 'image_url':
                    display(str2image(item['image_url']['url'].split(',')[1]))

    def save_gt_eval(self, bias_conclusion, round_num=0):
        """
        Save results of the experiment round to the history.json file at the end of the round.
        
        The file is a list of dictionaries, where each dictionary contains:
        - round_num: the round number
        - bias_conclusion: the bias conclusion from the experiment log
        - pos_acts: the positive confidence scores from the scoring
        - neg_acts: the negative confidence scores from the scoring
        - pos_images_names: the names of the positive images
        - neg_images_names: the names of the negative images
        
        :param round_num: The round number of the experiment
        """
        history = []
        pos_prompts, neg_prompts = self._get_prompts(bias_conclusion)
        if not pos_prompts or not neg_prompts:
            return None, None, None

        pos_path = os.path.join(self.path2save, "positive_scoring_images", f"round_{round_num}")
        os.makedirs(pos_path, exist_ok=True)
        neg_path = os.path.join(self.path2save, "negative_scoring_images", f"round_{round_num}")
        os.makedirs(neg_path, exist_ok=True)

        pos_images = self.tools.text2image(pos_prompts)
        pos_acts, pos_image_list = self.system.call_classifier(pos_images)
        pos_acts = [round(a, 2) for a in pos_acts]
        print("Positive scoring image confidence scores:")
        print(pos_acts)
        neg_images = self.tools.text2image(neg_prompts)
        neg_acts, neg_image_list = self.system.call_classifier(neg_images)
        neg_acts = [round(a, 2) for a in neg_acts]
        print("Negative scoring image confidence scores:")
        print(neg_acts)

        pos_image_names = []
        neg_image_names = []
        for i in range(len(pos_prompts)):
            buffer = base64.b64decode(pos_image_list[i])
            image_name = os.path.join(pos_path, f"{i}.png")
            with open(image_name, 'wb') as file:
                file.write(buffer)
            pos_image_names.append(image_name)

        for i in range(len(neg_prompts)):
            buffer = base64.b64decode(neg_image_list[i])
            image_name = os.path.join(neg_path, f"{i}.png")
            with open(image_name, 'wb') as file:
                file.write(buffer)
            neg_image_names.append(image_name)

        history.append({'round': round_num, 'bias_conclusion': bias_conclusion, 'pos_images_paths': pos_image_names, 'neg_images_paths': neg_image_names, 'pos_prompts': pos_prompts, 'neg_prompts': neg_prompts, 'pos_acts': pos_acts, 'neg_acts': neg_acts})
        with open(os.path.join(self.path2save, f"history_{self.experiment}.json"), 'w') as file:
            json.dump(history, file)
        file.close()