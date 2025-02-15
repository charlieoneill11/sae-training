from huggingface_hub import HfApi
from datasets import load_dataset, Dataset, concatenate_datasets
import pandas as pd

def format_medqa(example):
    options = [x for x in list(example['data']['Options'].values())]
    options_text = "\nThe options are:\n" + "\n".join(options)
    question = example['data']['Question']
    correct_option = example['data']['Options'][example['data']['Correct Option']]
    
    return {
        'text': f"{question}{options_text}\nCorrect option: {correct_option}",
        'source_dataset': 'medqa'
    }

def format_medmcqa(example):
    option_keys = ['opa', 'opb', 'opc', 'opd']
    
    question = example['question'].rstrip(':') + '?'
    options = "\nThe options are:\n" + "\n".join(example[key] for key in option_keys)
    correct_idx = int(example['cop'])
    correct_option = example[option_keys[correct_idx]]
    
    explanation = f"\nExplanation: {example['exp']}" if 'exp' in example else ""
    explanation = explanation.split('Ref')[0]
    
    return {
        'text': f"{question}{options}\nCorrect option: {correct_option}{explanation}",
        'source_dataset': 'medmcqa'
    }

def format_pubmed_qa(example):
    context = "Context:\n" + "\n".join(example['data']['Context'])
    question = f"\nQuestion: {example['data']['Question']}"
    options = "\nThe options are:\n" + "\n".join(f"{v}" for v in example['data']['Options'].values())
    correct_option = f"\nCorrect option: {example['data']['Correct Answer']}"
    long_answer = f"\nDetailed answer: {example['data']['Long Answer']}"
    
    return {
        'text': f"{context}{question}{options}{correct_option}{long_answer}",
        'source_dataset': 'pubmed_qa'
    }

def format_mmlu(example, dataset_name):
    question = example['data']['Question']
    options = "\nThe options are:\n" + "\n".join(f"{v}" for v in example['data']['Options'].values())
    correct_option = example['data']['Options'][example['data']['Correct Option']]
    
    return {
        'text': f"{question}{options}\nCorrect option: {correct_option}",
        'source_dataset': dataset_name
    }

# Dataset loading and processing
dataset_configs = {
    'medqa': {
        'path': 'openlifescienceai/medqa',
        'split': 'train',
        'format_fn': format_medqa
    },
    'medmcqa': {
        'path': 'openlifescienceai/medmcqa',
        'split': 'train',
        'format_fn': format_medmcqa
    },
    'pubmed_qa': {
        'path': 'openlifescienceai/pubmedqa',
        'split': 'train',
        'format_fn': format_pubmed_qa
    },
    'mmlu_college_medicine': {
        'path': 'openlifescienceai/mmlu_college_medicine',
        'split': 'test',
        'format_fn': lambda x: format_mmlu(x, 'mmlu_college_medicine')
    },
    'mmlu_clinical_knowledge': {
        'path': 'openlifescienceai/mmlu_clinical_knowledge',
        'split': 'test',
        'format_fn': lambda x: format_mmlu(x, 'mmlu_clinical_knowledge')
    },
    'mmlu_professional_medicine': {
        'path': 'openlifescienceai/mmlu_professional_medicine',
        'split': 'test',
        'format_fn': lambda x: format_mmlu(x, 'mmlu_professional_medicine')
    }
}

# Process each dataset
formatted_datasets = []

for dataset_name, config in dataset_configs.items():
    print(f"Processing {dataset_name}...")
    dataset = load_dataset(config['path'], split=config['split'])
    formatted_dataset = dataset.map(
        config['format_fn'],
        remove_columns=dataset.column_names
    )
    formatted_datasets.append(formatted_dataset)

# Combine all datasets
combined_dataset = concatenate_datasets(formatted_datasets)

# Push to hub
dataset_name = "medical-qa-combined"
combined_dataset.push_to_hub(f"charlieoneill/{dataset_name}", private=False)

print(f"Dataset uploaded successfully to charlieoneill/{dataset_name}")
print(f"Total examples: {len(combined_dataset)}")
print("\nSample distribution:")
print(combined_dataset.to_pandas()['source_dataset'].value_counts())