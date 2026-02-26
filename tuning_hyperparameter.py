#%%
import gc
import wandb
import torch
from src.utils import set_seed
import torch.nn.functional as F
from train_and_eval import get_dataset, train, test, negative_sampling, negative_sampling_filtered

from src.hetero_rgcn import HeterogeneousRGCN as rgcn
from src.hetero_rgat import HeterogeneousRGAT as rgat
from src.hetero_compgcn import HeterogeneousCompGCN as compgcn

# WandB configuration
PROJECT_NAME = "PathogenKG"  # Replace with your WandB project name
# ENTITY = "gidek"  # Replace with your WandB entity
ENTITY = "giovannimaria-defilippis-university-of-naples-federico-ii"  # Replace with your WandB entity
#%%
# Hyperparameter search space
SWEEP_CONFIG = {
	'method': 'bayes',
	'metric': {
		'name': 'val_mixed_metric',
		'goal': 'maximize'
	},
	'parameters': {
		'learning_rate': {
			'distribution': 'log_uniform_values',
			'min': 1e-4,
			'max': 1e-2
		},
		'regularization': {
			'distribution': 'log_uniform_values',
			'min': 1e-6,
			'max': 1e-2
		},
		'grad_norm': {
			'distribution': 'uniform',
			'min': 0.5,
			'max': 5.0
		},
		'dropout': {
			'distribution': 'uniform',
			'min': 0.0,
			'max': 0.7
		},
		'conv_layer_num': {
			'values': [1, 2, 3]
		},
		'mlp_out_layer': {
			'values': [8, 16, 32, 64]
		},
		'layer_0': {
			'values': [8, 16, 32, 64]
		},
		'layer_1': {
			'values': [8, 16, 32, 64]
		},
		'layer_2': {
			'values': [8, 16, 32, 64]
		},
		'num_bases': {
			'values': [10, 20, 30, 40, 50]
		},
		'opn': {  # For CompGCN only
			'values': ['mult', 'sub', 'corr']
		},

	}
}

# AVAILABLE_MODELS = ['rgcn', 'rgat', 'compgcn']
# AVAILABLE_MODELS = ['rgat']
AVAILABLE_MODELS = ['rgcn', 'compgcn']
AVAILABLE_MODELS = ['compgcn']
BASE_SEED = 42
negative_sampling = negative_sampling_filtered  # Use filtered negative sampling for better evaluation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cleanup_cuda():
	"""Clean up CUDA memory"""
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
		torch.cuda.synchronize()
	gc.collect()

def create_model_from_config(model_name, config, in_channels_dict, num_nodes_per_type, num_entities, num_relations):
	"""Create model with hyperparameters from WandB config"""    
	# Build conv_hidden_channels dict
	conv_hidden_channels = {}
	for i in range(config.conv_layer_num):
		layer_key = f'layer_{i}'
		if layer_key in config:
			conv_hidden_channels[layer_key] = getattr(config, layer_key)
		else:
			# Default fallback
			conv_hidden_channels[layer_key] = 128
	
	if model_name == 'rgcn':
		model = rgcn(
			in_channels_dict,
			None,
			config.mlp_out_layer,
			conv_hidden_channels,
			num_nodes_per_type,
			num_entities,
			num_relations + 1,
			config.conv_layer_num,
			config.num_bases,
			activation_function=F.relu,
			device=device
		)
	elif model_name == 'rgat':
		model = rgat(
			in_channels_dict,
			None,
			config.mlp_out_layer,
			conv_hidden_channels,
			num_nodes_per_type,
			num_entities,
			num_relations + 1,
			conv_num_layers=config.conv_layer_num,
			num_bases=config.num_bases,
			activation_function=F.relu,
			device=device
		)
	elif model_name == 'compgcn':
		model = compgcn(
			in_channels_dict,
			mlp_out_emb_size=config.mlp_out_layer,
			conv_hidden_channels=conv_hidden_channels,
			num_nodes_per_type=num_nodes_per_type,
			num_entities=num_entities,
			num_relations=num_relations,
			dropout=config.dropout,
			conv_num_layers=config.conv_layer_num,
			opn=config.opn,
		)
	
	return model

def train_model():
	"""Training function called by WandB sweep"""
	# Initialize WandB run
	wandb.init()
	config = wandb.config
	
	# Get model name from config (set by sweep)
	model_name = config.model_name
	
	# Fixed training parameters
	tsv_path = 'dataset/PathogenKG_n34_core.tsv.zip'  
	task = "TARGET"  #'Compound-ExtGene'
	validation_size = 0.1
	test_size = 0.2
	epochs = 200  # Reduced for hyperopt
	patience = 50
	evaluate_every = 5
	negative_rate = 1
	alone = False

	# added new {2025-12-15}
	oversample_rate =  5
	undersample_rate = 0.5
	alpha = 0.25
	gamma = 3.0
	alpha_adv = 2.0
	
	# Set seed for reproducibility
	seed = BASE_SEED
	set_seed(seed)
	
	try:
		# Load dataset
		# error missing oversample_rate, undersample_rate in get_dataset call
		(in_channels_dict, num_nodes_per_type, num_entities, num_relations,
		 train_triplets, train_index, flattened_features_per_type, val_triplets,
		 train_val_triplets, test_triplets, train_val_test_triplets,
		 edge_index, ent2id, relation2id) = get_dataset(
			tsv_path, task, validation_size, test_size, True, seed,
			# added new {2025-12-15}
			oversample_rate=oversample_rate,
			undersample_rate=undersample_rate
		)
		
		# Create model with hyperparameters from config
		model = create_model_from_config(
			model_name, config, in_channels_dict, 
			num_nodes_per_type, num_entities, num_relations
		)
		
		# Move to device
		model = model.to(device)
		train_index = train_index.to(device)
		flattened_features_per_type = {
			node_type: (features.to(device) if features is not None else None) 
			for node_type, features in flattened_features_per_type.items()
		}
		
		# Handle change points for RGAT
		change_points = None
		if model_name == 'rgat':
			sorted_indices = torch.argsort(train_index[:, 1])
			train_index = train_index[sorted_indices]
			
			rel_ids_sorted = train_index[:, 1]
			change_points = torch.cat([
				torch.tensor([0], device=device),
				(rel_ids_sorted[1:] != rel_ids_sorted[:-1]).nonzero(as_tuple=False).view(-1) + 1,
				torch.tensor([rel_ids_sorted.size(0)], device=device)
			])
		
		# Optimizer
		optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
		
		# Training loop
		best_mixed_metric = 0
		patience_trigger = 0
		
		for epoch in range(1, epochs + 1):
			# Training
			train_triplets_np = train_triplets.cpu().numpy() if torch.is_tensor(train_triplets) else train_triplets
			training_triplets, train_labels = negative_sampling(train_triplets_np, int(negative_rate))
			training_triplets, train_labels = training_triplets.to(device), train_labels.to(device)
			
			train_metrics = train(
				model, optimizer, config.grad_norm, config.regularization,
				flattened_features_per_type, train_index,
				training_triplets, train_labels,
				alpha, gamma, alpha_adv, change_points
			)
			
			# Validation
			if epoch % evaluate_every == 0:
				val_triplets_np = val_triplets.cpu().numpy() if torch.is_tensor(val_triplets) else val_triplets
				validation_triplets, val_labels = negative_sampling(val_triplets_np, int(negative_rate))
				validation_triplets, val_labels = validation_triplets.to(device), val_labels.to(device)
				
				val_metrics = test(
					model, config.regularization,
					flattened_features_per_type, train_index,
					validation_triplets, val_labels,
					train_val_triplets,
					alpha, gamma, alpha_adv, change_points
				)
				
				# Calculate mixed metric
				mixed_metric = 0.2 * val_metrics["Auroc"] + 0.4 * val_metrics["Auprc"] + 0.4 * val_metrics["MRR"]
				
				# Log metrics to WandB
				wandb.log({
					'epoch': epoch,
					'train_loss': train_metrics["Loss"],
					'train_auroc': train_metrics["Auroc"],
					'train_auprc': train_metrics["Auprc"],
					'val_loss': val_metrics["Loss"],
					'val_auroc': val_metrics["Auroc"],
					'val_auprc': val_metrics["Auprc"],
					'val_mrr': val_metrics["MRR"],
					'val_mixed_metric': mixed_metric
				})
				
				# Early stopping
				if mixed_metric > best_mixed_metric:
					best_mixed_metric = mixed_metric
					patience_trigger = 0
				else:
					patience_trigger += 1
				
				if patience_trigger > patience:
					break
		
		# Final test evaluation
		test_triplets_np = test_triplets.cpu().numpy() if torch.is_tensor(test_triplets) else test_triplets
		testing_triplets, test_labels = negative_sampling(test_triplets_np, int(negative_rate))
		testing_triplets, test_labels = testing_triplets.to(device), test_labels.to(device)
		
		test_metrics = test(
			model, config.regularization,
			flattened_features_per_type, train_index,
			testing_triplets, test_labels,
			train_val_test_triplets,
			alpha, gamma, alpha_adv, change_points
		)
		
		# Log final test metrics
		wandb.log({
			'test_auroc': test_metrics["Auroc"],
			'test_auprc': test_metrics["Auprc"],
			'test_mrr': test_metrics["MRR"],
			'test_hits@1': test_metrics["Hits@"][1],
			'test_hits@3': test_metrics["Hits@"][3],
			'test_hits@10': test_metrics["Hits@"][10],
			'final_mixed_metric': 0.2 * test_metrics["Auroc"] + 0.4 * test_metrics["Auprc"] + 0.4 * test_metrics["MRR"]
		})
		"""
		final_mixed_metric
		AUROC (20%): misura separazione classi, robusta ma meno sensibile a imbalance.
		AUPRC (40%): cruciale per link prediction (pochi edge positivi), enfatizza precision/recall su positivi.
		MRR (40%): prioritizza top ranking (essenziale per raccomandazioni link).
		Pesi enfatizzano metriche ranking-specifiche vs AUROC generica.
		"""
		
		cleanup_cuda()
		
	except torch.cuda.OutOfMemoryError as e:
		print(f"🚨 CUDA OOM Error: {e}")
		print("💡 Trying to recover by cleaning up memory...")
		cleanup_cuda()
		
		# Log the error but don't fail the run
		wandb.log({
			'error': 'CUDA_OOM',
			'error_details': str(e),
			'status': 'skipped'
		})
		
		# Skip this configuration
		return
		
	except Exception as e:
		print(f"❌ Error in training: {e}")
		wandb.log({
			'error': 'training_error',
			'error_details': str(e),
			'status': 'failed'
		})
		cleanup_cuda()
		raise e
	
	finally:
		cleanup_cuda()
		wandb.finish()

def run_hyperparameter_optimization():
	"""Run hyperparameter optimization for all models"""
	
	for model_name in AVAILABLE_MODELS:
		print(f"Starting hyperparameter optimization for {model_name}")
		
		# Create model-specific sweep config
		# sweep_config = SWEEP_CONFIG.copy()
		import copy
		sweep_config = copy.deepcopy(SWEEP_CONFIG)

		sweep_config['name'] = f'{model_name}-hyperopt'
		
		# Add model_name as a fixed parameter
		sweep_config['parameters']['model_name'] = {'value': model_name}
		
		# Filter parameters based on model
		if model_name != 'compgcn':
			# Remove CompGCN-specific parameters
			if 'opn' in sweep_config['parameters']:
				del sweep_config['parameters']['opn']
			if 'dropout' in sweep_config['parameters']:
				del sweep_config['parameters']['dropout']
		
		# Create sweep with model-specific project name
		model_project = f"{PROJECT_NAME}-{model_name}"
		sweep_id = wandb.sweep(
			sweep_config, 
			project=model_project,
			entity=ENTITY
		)
		
		print(f"Created sweep {sweep_id} for {model_name}")
		print(f"Project: {model_project}")
		print(f"Run: wandb agent {ENTITY}/{model_project}/{sweep_id}")
		
		# Run the sweep (200 runs per model)
		wandb.agent(
			sweep_id, 
			train_model, 
			count=200,
			project=model_project,
			entity=ENTITY
		)
		
		print(f"Completed hyperparameter optimization for {model_name}")

if __name__ == "__main__":
	
	print("🔬 Starting hyperparameter optimization with WandB")
	print(f"🖥️ Device: {device}")
	print(f"📊 Total planned runs: {len(AVAILABLE_MODELS) * 200}")
	print("-" * 50)
	
	run_hyperparameter_optimization()
