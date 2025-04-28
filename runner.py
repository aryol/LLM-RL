import yaml
import subprocess
import argparse

# Function to load the YAML file
def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# Function to execute a command
def execute_command(config_file, name, model, command, id, mode='vllm'):
    try:
        # Run the command in the shell
        command = command.strip() + " trainer.experiment_name=" + name.strip() + f" log_dir=$BOLT_ARTIFACT_DIR trainer.n_gpus_per_node=8"
        with open(config_file, 'r') as f:
            content = f.read()
            content = content.replace('command_template', command)
            content = content.replace('Progressive-RL', f'Progressive-RL - {id}')
        with open(f"tmp/config_for_{name.replace('/', '-')}.yaml", 'w') as f:
            f.write(content)
        result = subprocess.run(f"zsh -i -c 'conda activate dev; bolt task submit --tar=. --config=tmp/config_for_{name.replace('/', '-')}.yaml'", shell=True, check=True, capture_output=True, text=True)
        # print(result.stdout)
        # print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing command: {e}")

# Function to get the settings based on the user input
def get_entities(entities, user_input):
    selected_entities = []
    for entity in entities:
        if str(entity['id']) == str(user_input):
            selected_entities.append(entity)
    return selected_entities

# Main function
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Runs selected tasks based on their IDs.")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--inputs', metavar='N', type=str, nargs='+', help='List of IDs to execute', required=True)
    parser.add_argument('--mode', type=str, help='How to run the code, mostly for backward compatibilty', required=False, default='vllm')
    args = parser.parse_args()

    # Path to your YAML file
    yaml_file = 'experiments_verl.yaml'
    
    # Load the entities from the YAML file
    entities = load_yaml(yaml_file)
    
    # Find and execute commands for matching entities
    for input_item in args.inputs:
        matching_entities = get_entities(entities, input_item)
        
        if len(matching_entities) > 0:
            for entity in matching_entities:
                print(f"Executing command for {entity['name']} (ID: {entity['id']})")
                execute_command(args.config, entity['name'], entity['model'], entity['command'], entity['id'], args.mode)
        else:
            print(f"No matching entity found for input: {input_item}")

if __name__ == '__main__':
    main()
