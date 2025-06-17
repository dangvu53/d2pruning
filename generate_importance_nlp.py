import os
import torch
import numpy as np
import argparse
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--base-dir', type=str, default='./data-model/nlp', help='Base directory for storing data and model')
    parser.add_argument('--task-name', type=str, default='all-data', help='Task name')
    parser.add_argument('--feature', action='store_true', help='Generate and save sample embeddings')
    parser.add_argument('--dataset', type=str, default='imdb', 
                       choices=['imdb', 'cola', 'sst2', 'mrpc', 'qnli', 'qqp', 'rte', 'wnli'], 
                       help='Dataset name')
    parser.add_argument('--model-name', type=str, default='roberta-base', help='RoBERTa model name')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--num-samples', type=int, default=1000, help='Number of samples for feature extraction')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--td-metrics', action='store_true', help='Use training dynamics metrics')
    parser.add_argument('--td-epochs', type=int, default=5, help='Number of epochs for training dynamics')
    parser.add_argument('--metric', type=str, default='el2n', 
                       choices=['el2n', 'correctness', 'forgetting', 'accumulated_margin', 'variance'],
                       help='Which training dynamics metric to use for importance')
    parser.add_argument('--max-length', type=int, default=512, help='Maximum sequence length')
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_dataset_config(dataset_name):
    """Get dataset configuration for different NLP tasks."""
    configs = {
        "imdb": {"path": "imdb", "text_column": "text", "num_labels": 2, "subset": None},
        "cola": {"path": "glue", "text_column": "sentence", "num_labels": 2, "subset": "cola"},
        "sst2": {"path": "glue", "text_column": "sentence", "num_labels": 2, "subset": "sst2"},
        "mrpc": {"path": "glue", "text_column": ["sentence1", "sentence2"], "num_labels": 2, "subset": "mrpc"},
        "qnli": {"path": "glue", "text_column": ["question", "sentence"], "num_labels": 2, "subset": "qnli"},
        "qqp": {"path": "glue", "text_column": ["question1", "question2"], "num_labels": 2, "subset": "qqp"},
        "rte": {"path": "glue", "text_column": ["sentence1", "sentence2"], "num_labels": 2, "subset": "rte"},
        "wnli": {"path": "glue", "text_column": ["sentence1", "sentence2"], "num_labels": 2, "subset": "wnli"},
    }

    return configs[dataset_name]

def prepare_dataset(dataset_name, tokenizer, batch_size, max_length=512):
    """Prepare NLP dataset for RoBERTa model."""
    config = get_dataset_config(dataset_name)
    
    # Load dataset
    if config['subset']:
        dataset = load_dataset(config['path'], config['subset'])
    else:
        dataset = load_dataset(config['path'])
    
    # Tokenization function
    def preprocess_function(examples):
        text_column = config['text_column']
        
        if isinstance(text_column, list):
            # For paired sentences
            return tokenizer(
                examples[text_column[0]], 
                examples[text_column[1]],
                truncation=True, 
                padding='max_length', 
                max_length=max_length
            )
        else:
            # For single sentences
            return tokenizer(
                examples[text_column], 
                truncation=True, 
                padding='max_length', 
                max_length=max_length
            )
    
    # Apply tokenization
    encoded_dataset = dataset.map(preprocess_function, batched=True)
    
    # Standardize label column name
    if 'label' in encoded_dataset['train'].column_names:
        if 'label' != 'labels':
            encoded_dataset = encoded_dataset.rename_column('label', 'labels')
    
    # Set format for pytorch
    encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    # Create dataloaders
    train_dataloader = DataLoader(encoded_dataset['train'], batch_size=batch_size, shuffle=True)
    
    # Validation/test set
    if 'validation' in encoded_dataset:
        eval_set = 'validation'
    else:
        eval_set = 'test'
    
    eval_dataloader = DataLoader(encoded_dataset[eval_set], batch_size=batch_size)
    
    return train_dataloader, eval_dataloader, encoded_dataset, config['num_labels'], config['text_column']

def get_token_importance(model, dataloader, device):
    """Compute importance scores for tokens using attention weights."""
    importance_scores = {}
    
    model.eval()
    for batch in tqdm(dataloader, desc="Computing token importance"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        
        # Get attention weights from all layers
        attentions = outputs.attentions
        
        # Average attention across all heads and layers
        all_attentions = torch.cat([layer.detach().mean(dim=1) for layer in attentions], dim=0)
        avg_attention = all_attentions.mean(dim=0)
        
        # For each example, calculate token importance
        for i, (ids, att, mask) in enumerate(zip(input_ids, avg_attention, attention_mask)):
            for token_idx, token_id in enumerate(ids):
                if mask[token_idx] == 0:  # Skip padding tokens
                    continue
                    
                token_id_item = token_id.item()
                if token_id_item in importance_scores:
                    importance_scores[token_id_item].append(att[token_idx].mean().item())
                else:
                    importance_scores[token_id_item] = [att[token_idx].mean().item()]
    
    # Average the importance scores for each token ID
    final_scores = {}
    for token_id in importance_scores:
        final_scores[token_id] = np.mean(importance_scores[token_id])
    
    return final_scores

def extract_features(model, dataset, text_column, tokenizer, device, num_samples=1000, seed=42, max_length=512):
    """Extract embeddings from samples."""
    set_seed(seed)
    
    # Select random samples
    train_samples = dataset['train'].shuffle(seed=seed).select(range(min(num_samples, len(dataset['train']))))
    
    features = []
    labels = []
    
    model.eval()
    for i in tqdm(range(len(train_samples)), desc="Extracting features"):
        sample = train_samples[i]
        
        # Handle different dataset structures
        if isinstance(text_column, list):
            # For paired sentences
            text1 = sample.get(text_column[0], "")
            text2 = sample.get(text_column[1], "")
            inputs = tokenizer(text1, text2, return_tensors="pt", 
                              truncation=True, padding='max_length', max_length=max_length)
        elif text_column in sample:
            # Use text column if available
            inputs = tokenizer(sample[text_column], return_tensors="pt", 
                              truncation=True, padding='max_length', max_length=max_length)
        else:
            # Use pre-tokenized inputs
            inputs = {
                'input_ids': sample['input_ids'].unsqueeze(0),
                'attention_mask': sample['attention_mask'].unsqueeze(0)
            }
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model.roberta(**inputs, output_hidden_states=True)
            # Use <s> token (position 0) from last layer as feature vector
            embedding = outputs.hidden_states[-1][:, 0].cpu().numpy()
        
        features.append(embedding[0])
        labels.append(sample['labels'].item())
    
    features = np.array(features)
    labels = np.array(labels)
    
    return features, labels

def collect_training_dynamics(model, tokenizer, dataset, device, num_epochs=5, batch_size=16, max_length=512):
    """Collect training dynamics data for NLP models."""
    train_dataset = dataset['train']
    
    # Initialize model for training
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # Store training dynamics
    td_log = []
    
    # Track training dynamics over epochs
    for epoch in range(num_epochs):
        model.train()
        
        # Process in batches
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Get original sample indices
            indices = torch.arange(
                batch_idx * batch_size, 
                min((batch_idx + 1) * batch_size, len(train_dataset))
            )
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Record training dynamics
            logits = outputs.logits
            log_probs = F.log_softmax(logits, dim=1)
            
            td_log.append({
                'epoch': epoch,
                'idx': indices,
                'output': log_probs.detach().cpu(),
            })
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return td_log

def EL2N(td_log, dataset, data_importance, num_labels, max_epoch=10):
    """Calculate Error L2-Norm (EL2N) metrics for NLP data."""
    targets = []
    data_size = len(dataset)

    for i in range(data_size):
        example = dataset[i]
        targets.append(example["labels"])
    targets = torch.tensor(targets)
    data_importance['targets'] = targets.type(torch.int32)
    data_importance['el2n'] = torch.zeros(data_size).type(torch.float32)
    l2_loss = torch.nn.MSELoss(reduction='none')

    def record_training_dynamics(td_log):
        output = torch.exp(td_log['output'].type(torch.float))
        index = td_log['idx'].type(torch.long)
        label = targets[index]
        label_onehot = torch.nn.functional.one_hot(label, num_classes=num_labels)
        el2n_score = torch.sqrt(l2_loss(label_onehot, output).sum(dim=1))
        data_importance['el2n'][index] += el2n_score

    for i, item in enumerate(td_log):
        if i % 1000 == 0:
            print(f"Processing batch {i}")
        if item['epoch'] == max_epoch:
            return
        record_training_dynamics(item)

def training_dynamics_metrics(td_log, dataset, data_importance):
    """Calculate various training dynamics metrics for NLP data."""
    targets = []
    data_size = len(dataset)

    for i in range(data_size):
        example = dataset[i]
        targets.append(example["labels"])
    targets = torch.tensor(targets)
    data_importance['targets'] = targets.type(torch.int32)

    data_importance['correctness'] = torch.zeros(data_size).type(torch.int32)
    data_importance['forgetting'] = torch.zeros(data_size).type(torch.int32)
    data_importance['last_correctness'] = torch.zeros(data_size).type(torch.int32)
    data_importance['accumulated_margin'] = torch.zeros(data_size).type(torch.float32)
    data_importance['variance'] = []
    for i in range(data_size):
        data_importance['variance'].append([])

    def record_training_dynamics(td_log):
        output = torch.exp(td_log['output'].type(torch.float))
        predicted = output.argmax(dim=1)
        index = td_log['idx'].type(torch.long)
        label = targets[index]

        correctness = (predicted == label).type(torch.int)
        data_importance['forgetting'][index] += torch.logical_and(data_importance['last_correctness'][index] == 1, correctness == 0)
        data_importance['last_correctness'][index] = correctness
        data_importance['correctness'][index] += data_importance['last_correctness'][index]

        batch_idx = range(output.shape[0])
        target_prob = output[batch_idx, label]
        for i, idx in enumerate(index):
            data_importance['variance'][idx].append(target_prob[i].item())
        output[batch_idx, label] = 0
        other_highest_prob = torch.max(output, dim=1)[0]
        margin = target_prob - other_highest_prob
        data_importance['accumulated_margin'][index] += margin

    for i, item in enumerate(td_log):
        if i % 1000 == 0:
            print(f"Processing batch {i}")
        record_training_dynamics(item)

    # compute variance
    sizes = [len(data_importance['variance'][i]) for i in range(data_size)]
    for i, s in enumerate(sizes):
        if s != sizes[0]:
            for j in range(sizes[0] - s):
                data_importance['variance'][i].append(1.0)
    data_importance['variance'] = torch.tensor(np.std(np.array(data_importance['variance']), axis=-1))

def compute_token_importance_from_sample_metrics(tokenizer, dataset, data_importance, metric_name='el2n', text_column=None):
    """Convert sample-level metrics to token-level importance scores."""
    token_importance = {}
    
    # Normalize metric values
    metric_values = data_importance[metric_name]
    if metric_name in ['el2n', 'forgetting', 'variance']:
        normalized_values = 1.0 - (metric_values - metric_values.min()) / (metric_values.max() - metric_values.min() + 1e-8)
    else:
        normalized_values = (metric_values - metric_values.min()) / (metric_values.max() - metric_values.min() + 1e-8)
    
    # Assign importance to tokens
    for i, example in enumerate(dataset):
        sample_importance = normalized_values[i].item()
        
        # Get tokens for this sample
        if 'input_ids' in example:
            tokens = example['input_ids'].tolist()
        else:
            continue
        
        # Distribute sample importance to its tokens
        for token_id in tokens:
            if token_id in token_importance:
                token_importance[token_id].append(sample_importance)
            else:
                token_importance[token_id] = [sample_importance]
    
    # Average importance scores for each token
    final_token_importance = {}
    for token_id, importances in token_importance.items():
        final_token_importance[token_id] = np.mean(importances)
    
    return final_token_importance

def main():
    args = parse_args()
    
    # Set device
    device = torch.device(f"cuda:{args.gpuid}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set seed
    set_seed(args.seed)
    
    # Create directories
    os.makedirs(args.base_dir, exist_ok=True)
    task_dir = os.path.join(args.base_dir, args.task_name)
    os.makedirs(task_dir, exist_ok=True)
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    
    # Prepare dataset
    print(f"Loading {args.dataset} dataset")
    train_dataloader, eval_dataloader, encoded_dataset, num_labels, text_column = prepare_dataset(
        args.dataset, tokenizer, args.batch_size, args.max_length
    )
    
    # Load model
    print(f"Loading model: {args.model_name}")
    model = RobertaForSequenceClassification.from_pretrained(
        args.model_name, 
        num_labels=num_labels, 
        output_attentions=True, 
        output_hidden_states=True
    )
    model.to(device)
    
    # Compute importance scores based on selected method
    if args.td_metrics:
        print(f"Collecting training dynamics data over {args.td_epochs} epochs...")
        td_log = collect_training_dynamics(
            model, tokenizer, encoded_dataset, device, 
            num_epochs=args.td_epochs, batch_size=args.batch_size, max_length=args.max_length
        )
        
        # Initialize data importance dict
        data_importance = {}
        
        # Calculate metrics
        print(f"Computing {args.metric} importance scores...")
        if args.metric == 'el2n':
            EL2N(td_log, encoded_dataset['train'], data_importance, num_labels, max_epoch=args.td_epochs)
        else:
            training_dynamics_metrics(td_log, encoded_dataset['train'], data_importance)
        
        # Convert sample metrics to token importance
        importance_scores = compute_token_importance_from_sample_metrics(
            tokenizer, encoded_dataset['train'], data_importance, args.metric, text_column
        )
        
        # Save the full data importance dictionary
        data_importance_file = os.path.join(task_dir, f"{args.dataset}_data_importance.pkl")
        with open(data_importance_file, 'wb') as f:
            pickle.dump(data_importance, f)
        print(f"Data importance metrics saved to {data_importance_file}")
    else:
        print("Computing token importance scores using attention weights...")
        importance_scores = get_token_importance(model, train_dataloader, device)
    
    # Save importance scores
    importance_file = os.path.join(task_dir, f"{args.dataset}_importance.pkl")
    with open(importance_file, 'wb') as f:
        pickle.dump(importance_scores, f)
    print(f"Importance scores saved to {importance_file}")
    
    # Also save as numpy array for compatibility
    max_token_id = max(importance_scores.keys())
    importance_array = np.zeros(max_token_id + 1)
    for token_id, score in importance_scores.items():
        importance_array[token_id] = score
    
    np_importance_file = os.path.join(task_dir, f"{args.dataset}_importance.npy")
    np.save(np_importance_file, importance_array)
    print(f"Importance scores (numpy) saved to {np_importance_file}")
    
    # Extract and save features if requested
    if args.feature:
        print(f"Extracting features from {args.num_samples} samples...")
        features, labels = extract_features(
            model, encoded_dataset, text_column, tokenizer, device, 
            args.num_samples, args.seed, args.max_length
        )
        
        # Save features and labels
        feature_file = os.path.join(task_dir, f"{args.dataset}_features.npy")
        np.save(feature_file, features)
        
        label_file = os.path.join(task_dir, f"{args.dataset}_labels.npy")
        np.save(label_file, labels)
        
        print(f"Features saved to {feature_file}")
        print(f"Labels saved to {label_file}")
    
    print("Done!")

if __name__ == "__main__":
    main()
