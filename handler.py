import argparse, os, importlib, sys, inspect
from art import *

# import classes
class MULTITOOL:
    def __init__(self):
        sys.path.append(f'{os.sep}classes')

        actions = {}

        for module in [x for x in os.listdir('classes') if (x.find('.py') != -1 and x != "__init__.py"  and x != "utils.py")]:
            module = importlib.import_module(f'classes.{module.split(".")[0]}')
            class_name = [x for x in inspect.getmembers(module, inspect.isclass) if x[0]==str.upper(module.__name__.split('.')[1])][0][0]
            actions[str.lower(class_name)] = getattr(module, class_name)
        del module
        
        tprint('hello world', font="tarty1")

        self.actions = actions
        self.parser = argparse.ArgumentParser(description='OCR')
        # self.parser.add_argument('action', choices=list(self.actions.keys()) + [str(i) for i in range(1, len(self.actions))], help='Choose an action by name or number')
        self.parser.add_argument('action', choices=list(self.actions.keys()), help='Choose an action by name or number')
        self.parser.add_argument('--function', type=str, help='Choose a function within the action class')
        self.parser.add_argument('--param1', type=str, help='Parameter 1')
        self.parser.add_argument('--param2', type=str, help='Parameter 2')

    def run(self):
        args = self.parser.parse_args()

        if args.action.isdigit():
            # User provided a number
            action_num = int(args.action)
            if 1 <= action_num <= len(self.actions) - 1:
                action_name = f'action{action_num}'
        else:
            action_name = args.action

        if action_name in self.actions:
            # User provided action name
            action_instance = self.actions[action_name]()
            if args.function:
                self._execute_function(action_instance, args.function, args.param1, args.param2)
            else:
                self._list_functions(action_instance)
        else:
            print("Invalid action. Please choose a valid action name or number.")

    def _list_functions(self, action_instance):
        functions = [func for func in dir(action_instance) if callable(getattr(action_instance, func)) and not func.startswith("_")]
        print(f"Functions available in {action_instance.__class__.__name__}: {', '.join(functions)}")

    def _execute_function(self, action_instance, function_name, param1, param2):
        if hasattr(action_instance, function_name) and callable(func := getattr(action_instance, function_name)):
            # Check if the action class has the specified function
            # func(param1, param2)
            func()
        else:
            print(f"Invalid function '{function_name}' for the chosen action.")

if __name__ == "__main__":
    multiool = MULTITOOL()
    multiool.run()