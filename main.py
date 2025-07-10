import argparse
import os

import torch
import gc
import openai
import anthropic
import traceback
from dotenv import load_dotenv

from utils.call_agent import ask_agent
from agent_api import Synthetic_System, Tools
from utils.ExperimentEnvironment import ExperimentEnvironment
from utils.SyntheticExemplars import SyntheticExemplars
from utils.main_utils import MainUtils
from utils.api_utils import str2image
import json
import time
import logging
import subprocess

def call_argparse():
    parser = argparse.ArgumentParser(description='Process Arguments')	
    parser.add_argument('--agent', type=str, default='claude', choices=['gpt-4-turbo', 'gpt-4o', 'claude'], help='agent agent name')	
    parser.add_argument('--path2save', type=str, default='./results', help='a path to save the experiment outputs')	
    parser.add_argument('--path2prompts', type=str, default='./prompts', help='path to prompt to use')	
    parser.add_argument('--path2exemplars', type=str, default='./exemplars', help='path to net disect top 15 exemplars images')	
    parser.add_argument('--device', type=int, default=0, help='gpu device to use (e.g. 1)')	
    parser.add_argument('--text2image', type=str, default='flux', choices=['sd','dalle', 'flux'], help='name of text2image model')	
    parser.add_argument('--p2p_model', type=str, default='instdiff', choices=['ip2p', 'instdiff'], help='name of p2p model')
    parser.add_argument('--bias_mode', type=str, choices=['age', 'gender', 'color', 'state', 'setting', 'material', 'clip'], help='bias mode')
    parser.add_argument('--bias', type=str, help='bias')		
    parser.add_argument('--bias_discount', type=float, default=0.9, help='amount of discount applied for bias')	
    parser.add_argument('--n_experiment', type=int, default=0, help='experiment number')	
    parser.add_argument('--classifiers', nargs='+', type=int, default=None, help='synthetic classifier number')
    args = parser.parse_args()
    return args

def main(args):
    path2save = os.path.join(
        args.path2save, args.bias_mode, args.bias, f"{args.bias_discount}_discount"
    )
    path2exemplars = os.path.join(
        args.path2exemplars, args.bias_mode, args.bias, f"{args.bias_discount}_discount"
    )
    os.makedirs(path2save, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(path2save, "run.log")),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Load exemplars & prompts
    net_dissect = SyntheticExemplars(path2exemplars, path2save, args.bias_mode)
    try:
        with open(os.path.join(path2exemplars, 'data.json'), 'r') as f:
            classifier_data = json.load(f)
    except FileNotFoundError:
        raise RuntimeError(f"Could not find data.json in {path2exemplars}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in data.json: {e}")

    try:
        with open(os.path.join(args.path2prompts, 'user_adversarial_experiment.txt'), 'r') as f:
            experiment_prompt = f.read()
    except FileNotFoundError:
        raise RuntimeError(f"Could not find user_adversarial_experiment.txt in {args.path2prompts}")
    except IOError as e:
        raise RuntimeError(f"Error reading prompt file: {e}")

    tools = Tools(
        args.device,
        net_dissect,
        text2image_model_name=args.text2image,
        p2p_model_name=args.p2p_model
    )

    system = Synthetic_System(
                0,
                classifier_data[0]["label"].rsplit('_')[1:],  # initialize with label of first classifier
                args.bias_mode,
                args.device,
                bias=args.bias,
                bias_discount=args.bias_discount
            )

    # Per-classifier loop for clean exit
    try:
        for classifier_number, item in enumerate(classifier_data):
            if args.classifiers and classifier_number not in args.classifiers:
                continue

            gt_label = item["label"].rsplit('_')[1:]
            obj = gt_label[0]
            n_path2save = os.path.join(path2save, item["label"])
            os.makedirs(n_path2save, exist_ok=True)

            # Determine starting round
            history_file = os.path.join(n_path2save, f"history_{args.n_experiment}.json")
            if os.path.exists(history_file):
                try:
                    with open(history_file, 'r') as hf:
                        start_round = len(json.load(hf))
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"Invalid JSON in {history_file}: {e}")
            else:
                start_round = 0

            # Instead of reinitializing the system, just update the classifier labels
            system.classifier_num = classifier_number
            system.classifier.labels = gt_label
            experiment_env = ExperimentEnvironment(system, tools, globals())
            main_utils = MainUtils(
                args.path2prompts,
                n_path2save,
                args.agent,
                obj,
                tools,
                system,
                args.n_experiment
            )

            logger.info(f"Starting experiments for {args.bias}-{obj}")
            
            agent_api, user_query = main_utils.return_prompt(setting='bias_discovery')

            current_r = start_round
            while current_r < 10:
                badrequest_count = 0
                try:
                    round_start_time = time.time()
                    logger.info(f"Classifier {classifier_number} ('{item['label']}'): Starting Round {current_r}")
                    
                    tools.experiment_log = [] # Reset log for each round attempt
                    tools.update_experiment_log(role='system', type="text", type_content=agent_api)

                    ask_agent_failed_non_badrequest = False

                    if current_r == 0:
                        # Start fresh experiment
                        tools.update_experiment_log(role='user', type="text", type_content=user_query)
                    else:
                        # Load context from previous round
                        prev_pos_avg, prev_neg_avg = main_utils.load_prev_context(round_num=current_r-1)
                        first_resp = ask_agent(args.agent, tools.experiment_log)
                        if first_resp is None:
                            logger.error(f"Classifier {classifier_number} ('{item['label']}'): ask_agent returned None for initial response in Round {current_r}. Aborting rounds for this classifier.")
                            ask_agent_failed_non_badrequest = True
                            break
                        tools.update_experiment_log(role='agent', type="text", type_content=str(first_resp))
                        tools.update_experiment_log(role='user', type="text", type_content=experiment_prompt)
                    
                    if ask_agent_failed_non_badrequest: break

                    # Main interaction loop for the current round
                    for i in range(20): # Max 20 interactions
                        resp = ask_agent(args.agent, tools.experiment_log)
                        if resp is None:
                            logger.error(f"Classifier {classifier_number} ('{item['label']}'): ask_agent returned None during interaction {i} in Round {current_r}. Aborting rounds for this classifier.")
                            ask_agent_failed_non_badrequest = True
                            break
                        
                        tools.update_experiment_log(role='agent', type="text", type_content=str(resp))
                        tools.generate_html(n_path2save, name=f'experiment_{args.n_experiment}_r{current_r}') # Original call
                        
                        if "[BIAS LABEL]" in resp:
                            logger.info(f"Classifier {classifier_number} ('{item['label']}'): Bias found in Round {current_r}, interaction {i}. Ending round early.")
                            break
                        
                        try:
                            output = experiment_env.execute_experiment(resp)
                            if output:
                                tools.update_experiment_log(role='user', type="text", type_content=output)
                        except Exception as exec_e:
                            logger.error(f"Classifier {classifier_number} ('{item['label']}'): Experiment execution failed in Round {current_r}, interaction {i}: {exec_e}", exc_info=True)
                            tools.update_experiment_log(role='user', type="text", type_content=f"Error during experiment execution: {str(exec_e)}")
                    
                    if ask_agent_failed_non_badrequest: break

                    pos_avg, neg_avg, history = None, None, None
                    save_successful = False
                    for attempt in range(1, 6):
                        try:
                            pos_avg, neg_avg, history = main_utils.save_results(round_num=current_r)
                            if pos_avg is not None and neg_avg is not None:
                                save_successful = True
                                break
                            logger.warning(f"Classifier {classifier_number} ('{item['label']}'): Save attempt {attempt} for Round {current_r} returned None values, retrying...")
                        except Exception as save_e:
                            logger.error(f"Classifier {classifier_number} ('{item['label']}'): Save results attempt {attempt} for Round {current_r} failed: {save_e}", exc_info=True)
                        time.sleep(2)

                    if not save_successful:
                        logger.error(f"Classifier {classifier_number} ('{item['label']}'): All save attempts failed for Round {current_r}. Aborting rounds for this classifier.")
                        break

                    # Persist history
                    with open(history_file, 'w') as hf:
                        json.dump(history, hf)

                    logger.info(f"Classifier {classifier_number} ('{item['label']}'): Round {current_r} completed. Stats: pos_avg={pos_avg}, neg_avg={neg_avg}, exemplar_avg={main_utils.avg_acts}")
                    logger.info(f"Classifier {classifier_number} ('{item['label']}'): Round {current_r} duration: {time.time() - round_start_time:.2f}s")

                    current_r += 1

                except anthropic.BadRequestError as bad_req_e:
                    # Wait 10 seconds before retrying the same round r
                    logger.error(f"Classifier {classifier_number} ('{item['label']}'): Anthropic BadRequestError in Round {current_r}: {bad_req_e}. Retrying this round after a delay.")
                    badrequest_count += 1
                    if badrequest_count > 3:
                        logger.error(f"Classifier {classifier_number} ('{item['label']}'): Anthropic BadRequestError in Round {current_r}: {bad_req_e}. Aborting rounds for this classifier.")
                        break
                    time.sleep(10)

                except Exception as general_e:
                    # Abort rounds for THIS classifier and move to the next one.
                    logger.error(f"Classifier {classifier_number} ('{item['label']}'): Unexpected error during Round {current_r} processing: {general_e}", exc_info=True)
                    break

    except KeyboardInterrupt:
        logger.warning("Interrupted by user, saving progress and exiting")
    except Exception as e:
        logger.error("Unexpected error in main loop", exc_info=True)
        raise

if __name__ == '__main__':
    args = call_argparse()
    main(args)