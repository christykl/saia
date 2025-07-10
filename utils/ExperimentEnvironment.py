from io import StringIO
import sys
import traceback
import re
from typing import List
from IPython import embed
import statistics

from agent_api import Synthetic_System, Tools

class ExperimentEnvironment:
    '''Executes and stores the variables of the agent's experiments'''

    def __init__(self, system: Synthetic_System, tools: Tools, global_vars: dict):
        self.system = system
        self.tools = tools
        self.experiment_vars = global_vars

        self.experiment_vars["system"] = system
        self.experiment_vars["tools"] = tools

    # Parse the agent's code
    def get_code(self, agent_experiment: str)->List[str]:
        '''Parses code from the agent's experiment. There may be multiple code blocks.
        A block is defined by "```python ```"."
        
        '''
        # Extract the code blocks
        pattern = r"```python(.*?)```"
        unstripped_agent_code = re.findall(pattern, agent_experiment, re.DOTALL)
        # Remove leading and trailing whitespaces
        agent_code = [code.strip() for code in unstripped_agent_code]

        if len(agent_code) == 0:
            raise ValueError("No code blocks found in the experiment.")
        
        return agent_code

    # Run the code on python
    def execute_experiment(self, agent_experiment: str)->str:
        code_blocks = self.get_code(agent_experiment)

        out = StringIO()
        err = StringIO()
        # Store original stdout and stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        for code in code_blocks:
            try:
                # Redirect stdout and stderr
                sys.stdout = out
                sys.stderr = err
                
                # Execute the code with the system and tools objects, as well as any
                # variables defined in previous experiments
                exec(compile(code, 'code', 'exec'), globals(), self.experiment_vars)
            except Exception as e:
                # Capture traceback for exceptions 
                traceback.print_exc(file=err)
                # Output error
                err.write(str(e))
                # Stop execution
                break

        # Restore original stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        # Get the captured output
        output = ""
        # if out.getvalue() != "":
        #     output += f"Standard Output:\n{out.getvalue()}"
        print(f"Standard Output:\n{out.getvalue()}")
        if err.getvalue() != "":
            output += f"\n\nStandard Error:\n{err.getvalue()}"

        return output