# Fine-tunes DistilBERT on your exploit store as a binary classifier:
# Label 1 = adversarial attack, Label 0 = benign query

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
import torch, json, random
from memory.exploit_store import ExploitStore

# Benign queries to use as negative examples (label=0)
BENIGN_EXAMPLES = [
    'What are your business hours?',
    'How do I reset my password?',
    'Can you help me check my balance?',
    'What documents do I need to open an account?',
    'How long does a wire transfer take?',
    'What is the interest rate on savings accounts?',
    'How do I dispute a charge on my card?',
]

class DetectorFineTuner:
    MODEL_NAME = 'distilbert-base-uncased'
    OUTPUT_DIR = './detector_model'

    def __init__(self):
        self.store     = ExploitStore()
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.MODEL_NAME)

    def _build_dataset(self):
        """Combine exploits (label=1) with benign examples (label=0)."""
        exploits = self.store.get_all_exploits()
        if len(exploits) < 10:
            raise ValueError('Need at least 10 exploits before fine-tuning')

        positive = [{'text': e['attack_text'], 'label': 1} for e in exploits]
        # Balance the dataset: equal positive/negative examples
        negative = [{'text': t, 'label': 0} for t in BENIGN_EXAMPLES * (len(positive)//len(BENIGN_EXAMPLES)+1)]
        negative = negative[:len(positive)]

        all_examples = positive + negative
        random.shuffle(all_examples)

        split = int(0.8 * len(all_examples))
        train_data = {'text': [e['text'] for e in all_examples[:split]],
                      'label':[e['label'] for e in all_examples[:split]]}
        val_data   = {'text': [e['text'] for e in all_examples[split:]],
                      'label':[e['label'] for e in all_examples[split:]]}
        return Dataset.from_dict(train_data), Dataset.from_dict(val_data)

    def _tokenize(self, examples):
        return self.tokenizer(
            examples['text'], truncation=True, padding='max_length', max_length=128
        )

    def fine_tune(self):
        print('Building dataset from exploit store...')
        train_ds, val_ds = self._build_dataset()
        train_ds = train_ds.map(self._tokenize, batched=True)
        val_ds   = val_ds.map(self._tokenize, batched=True)

        model = DistilBertForSequenceClassification.from_pretrained(
            self.MODEL_NAME, num_labels=2
        )

        # LoRA: instead of updating all 66M parameters, we add tiny trainable
        # adapter matrices to the attention layers only. Faster and less memory.
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,                  # LoRA rank — 8 adapter dimensions
            lora_alpha=16,        # Scaling factor
            lora_dropout=0.1,
            target_modules=['q_lin', 'v_lin'],  # Attention query/value matrices
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        args = TrainingArguments(
            output_dir=self.OUTPUT_DIR,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            logging_steps=10,
            report_to='none',     # Disable W&B logging (optional to enable)
        )

        trainer = Trainer(
            model=model, args=args,
            train_dataset=train_ds, eval_dataset=val_ds,
        )
        trainer.train()
        trainer.save_model(self.OUTPUT_DIR)
        print(f'Detector model saved to {self.OUTPUT_DIR}')
        return trainer.evaluate()
