config = {
	"defaults": {
		"task": "topic",
		"profile": "default",
		"seed": 42,
	},
	"common": {
		"model": {
			"name": "vinai/phobert-base",
			"max_length": 128,
		},
		"data": {
			"use_context_prev": True,
			"use_context_next": True,
		},
		"training": {
			"epochs": 5,
			"train_batch_size": 16,
			"eval_batch_size": 32,
			"learning_rate": 2.0e-5,
			"weight_decay": 0.01,
			'warmup_steps': 0.1,
			"use_class_weights": True,
			"weight_method": "sqrt_inverse",
			"gradient_accumulation_steps": 1,
			"max_grad_norm": 1.0,
			"label_smoothing_factor": 0.0,
			"fp16": "auto",
			"metric_for_best_model": "macro_f1",
		},
		"early_stopping": {
			"enabled": True,
			"patience": 2,
			"threshold": 0.0,
		},
		"neuro_symbolic": {
			"enabled": True,
			"constraint_lambda": 0.3,
			"exactly_one_weight": 1.0,
			"implication_weight": 0.5,
			"negation_weight": 0.5,
			"inference_alpha": 0.3,
			"min_confidence": 0.3,
		},
		"experiment": {
			"alpha_grid": [0.3, 0.5, 0.7, 0.9],
		},
	},
	"tasks": {
		"topic": {
			"labels": ["E", "S_labor", "S_community", "S_product", "G", "Non_ESG"],
			"paths": {
				"train_data": "data/labels/topic/train.parquet",
				"val_data": "data/labels/topic/val.parquet",
				"test_data": "data/labels/topic/test.parquet",
				"output_dir": "output/models/topic_classifier",
			},
			"profiles": {
				"default": {},
			},
		},
		"action": {
			"labels": ["Implemented", "Planning", "Indeterminate"],
			"paths": {
				"train_data": "data/labels/action/train.parquet",
				"val_data": "data/labels/action/val.parquet",
				"test_data": "data/labels/action/test.parquet",
				"output_dir": "output/models/action_classifier",
			},
			"profiles": {
				"default": {},
			},
		},
	},
}