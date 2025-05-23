import json
import random
import re
from typing import List
import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoTokenizer,
    AutoModelForCausalLM,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

from torch.utils.data import DataLoader
import math
from datasets import load_dataset

class DataAugmentation:
    """
    Class for augmenting handicap mission text data.
    
    This class provides various data augmentation techniques to expand the corpus
    for NLG model training. It implements multiple augmentation strategies including
    synonym replacement, sentence shuffling, template-based generation, and paraphrasing
    to create synthetic training data from the original handicap mission descriptions.
    
    Attributes:
        corpus: The complete text corpus loaded from JSON files
        sentences: List of extracted sentences from the corpus
    """
    
    def __init__(self, json_files: List[str]):
        """
        Initialize the data augmentation class.
        
        Args:
            json_files: List of JSON file paths containing handicap mission data
        """
        self.corpus = self._load_corpus(json_files)
        self.sentences = self._extract_sentences()
        
    def _load_corpus(self, json_files: List[str]) -> str:
        """
        Load and concatenate text corpus from multiple JSON files.
        
        This method reads each JSON file, extracts the 'description' fields,
        and combines them into a single text corpus for augmentation.
        
        Args:
            json_files: List of paths to JSON files
            
        Returns:
            str: Concatenated text corpus from all descriptions
        """
        corpus = ""
        for filename in json_files:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Extract description from each item in the JSON
                for item in data:
                    if 'description' in item:
                        corpus += item['description'] + " "
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        return corpus
    
    def _extract_sentences(self) -> List[str]:
        """
        Extract clean sentences from the raw corpus.
        
        Splits the corpus by sentence delimiters and filters out
        sentences that are too short to be meaningful.
        
        Returns:
            List[str]: List of cleaned sentences
        """
        # Split by common sentence endings in French
        sentences = re.split(r'[.!?]+', self.corpus)
        # Keep only sentences with more than 10 characters
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        return sentences
    
    def synonym_replacement(self, text: str, n: int = 2) -> str:
        """
        Replace words with their synonyms to create variations.
        
        This method uses a predefined dictionary of synonyms specific to
        the handicap/disability support domain to replace words while
        maintaining semantic meaning.
        
        Args:
            text: Input text to augment
            n: Number of words to replace with synonyms
            
        Returns:
            str: Text with synonym replacements
        """
        # Domain-specific synonym dictionary for handicap mission context
        synonyms = {
            'handicap': ['situation de handicap', 'déficience', 'incapacité'],
            'étudiant': ['apprenant', 'élève', 'personne en formation'],
            'aide': ['assistance', 'soutien', 'accompagnement'],
            'aménagement': ['adaptation', 'ajustement', 'modification'],
            'accompagnement': ['soutien', 'aide', 'assistance'],
            'service': ['dispositif', 'structure', 'unité'],
            'mission': ['service', 'cellule', 'pôle'],
            'accessible': ['adapté', 'aménagé', 'praticable']
        }
        
        words = text.split()
        modified_count = 0
        
        # Replace up to n words with their synonyms
        for i, word in enumerate(words):
            if modified_count >= n:
                break
            word_lower = word.lower()
            if word_lower in synonyms:
                words[i] = random.choice(synonyms[word_lower])
                modified_count += 1
        
        return ' '.join(words)
    
    def sentence_shuffling(self, n: int = 3) -> str:
        """
        Create new text by randomly shuffling sentences.
        
        This technique creates new training examples by combining
        existing sentences in different orders, maintaining coherence
        while providing variety.
        
        Args:
            n: Number of sentences to select and shuffle
            
        Returns:
            str: New text created from shuffled sentences
        """
        # Ensure we don't select more sentences than available
        if len(self.sentences) < n:
            n = len(self.sentences)
        # Randomly select and shuffle sentences
        selected = random.sample(self.sentences, n)
        random.shuffle(selected)
        return ' '.join(selected)
    
    def template_generation(self) -> List[str]:
        """
        Generate new sentences using predefined templates.
        
        This method uses template-based generation to create synthetic
        sentences that follow common patterns found in handicap mission
        communications, ensuring domain-appropriate content.
        
        Returns:
            List[str]: List of generated sentences from templates
        """
        # Template patterns commonly found in handicap mission texts
        templates = [
            "La mission handicap propose {service} pour {beneficiaire}.",
            "Les étudiants en situation de handicap peuvent bénéficier de {aide}.",
            "L'université met en place {dispositif} pour {objectif}.",
            "Pour {action}, contactez la mission handicap.",
            "Les aménagements incluent {liste}.",
            "Un accompagnement {type} est disponible pour les étudiants."
        ]
        
        # Vocabulary slots for template filling
        vocab = {
            'service': ['accompagnement personnalisé', 'aménagements d\'examens', 'soutien pédagogique'],
            'beneficiaire': ['étudiants en situation de handicap', 'personnes à mobilité réduite'],
            'aide': ['temps supplémentaire', 'secrétariat', 'documents adaptés'],
            'dispositif': ['aménagements spécifiques', 'mesures d\'accompagnement'],
            'objectif': ['garantir l\'égalité des chances', 'faciliter l\'accès aux études'],
            'action': ['demander un aménagement', 'obtenir un soutien'],
            'liste': ['temps majoré, secrétariat, matériel adapté'],
            'type': ['individuel', 'collectif', 'spécialisé']
        }
        
        generated = []
        # Generate 30 new sentences from templates
        for _ in range(30):
            template = random.choice(templates)
            # Fill template slots with random vocabulary
            for key in vocab:
                if '{' + key + '}' in template:
                    template = template.replace('{' + key + '}', random.choice(vocab[key]))
            generated.append(template)
        
        return generated
    
    def paraphrase(self, sentence: str) -> str:
        """
        Apply simple rule-based paraphrasing to create variations.
        
        This method uses predefined substitution patterns to create
        paraphrased versions of sentences while maintaining their
        original meaning.
        
        Args:
            sentence: Input sentence to paraphrase
            
        Returns:
            str: Paraphrased version of the sentence
        """
        # Paraphrasing rules for common expressions
        replacements = [
            ("peut bénéficier de", "a droit à"),
            ("Les étudiants", "Les personnes"),
            ("en situation de handicap", "ayant un handicap"),
            ("contactez", "prenez contact avec"),
            ("propose", "offre"),
            ("accompagne", "soutient")
        ]
        
        result = sentence
        # Apply first matching replacement to avoid over-modification
        for old, new in replacements:
            if old in result:
                result = result.replace(old, new)
                break
        
        return result
    
    def augment(self) -> str:
        """
        Main augmentation method that combines all techniques.
        
        This method orchestrates all augmentation strategies to create
        a significantly expanded corpus from the original data. It applies
        multiple augmentation techniques and removes duplicates to ensure
        quality.
        
        Returns:
            str: Complete augmented corpus ready for model training
        """
        print("Starting data augmentation...")
        augmented = []
        
        # Add original sentences as the foundation
        augmented.extend(self.sentences)
        print(f"Original sentences: {len(self.sentences)}")
        
        # Apply synonym replacement to subset of sentences
        for sentence in self.sentences[:30]:
            augmented.append(self.synonym_replacement(sentence))
        
        # Generate shuffled sentence combinations
        for _ in range(20):
            augmented.append(self.sentence_shuffling())
        
        # Add template-generated sentences
        augmented.extend(self.template_generation())
        
        # Apply paraphrasing to subset of sentences
        for sentence in self.sentences[:20]:
            augmented.append(self.paraphrase(sentence))
        
        # Remove duplicates and shuffle for training variety
        augmented = list(set(augmented))
        print(f"Total augmented sentences: {len(augmented)}")
        
        return ' '.join(augmented)


class HandicapNLGFineTuner:
    """
    Fine-tuner for handicap mission NLG using pre-trained French models.
    
    This class implements a simple fine-tuning approach using a pre-trained
    French language model, which is more suitable for limited data scenarios
    than training from scratch.
    
    Attributes:
        corpus: The augmented text corpus
        model_name: Name of the pre-trained model
        device: Computing device (cuda/cpu)
    """
    
    def __init__(self, augmented_corpus: str):
        """
        Initialize the fine-tuner with augmented corpus.
        
        Args:
            augmented_corpus: The augmented text data for fine-tuning
        """
        self.corpus = augmented_corpus
        # Use a small French GPT model for efficiency
        self.model_name = "asi/gpt-fr-cased-small"
        self.output_dir = "./handicap_finetuned_model"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = None
        self.model = None
        
    def prepare_data(self):
        """
        Prepare training and validation datasets.
        
        Splits the corpus and saves it in the format required for
        TextDataset used by the Trainer.
        """
        # Save complete augmented corpus
        with open('augmented_corpus.txt', 'w', encoding='utf-8') as f:
            f.write(self.corpus)
        
        # Split into sentences
        sentences = self.corpus.split('.')
        split_idx = int(0.8 * len(sentences))
        
        # Create train/validation split
        train_text = '. '.join(sentences[:split_idx])
        val_text = '. '.join(sentences[split_idx:])
        
        # Save splits
        with open('train.txt', 'w', encoding='utf-8') as f:
            f.write(train_text)
        
        with open('val.txt', 'w', encoding='utf-8') as f:
            f.write(val_text)
        
        print(f"Data prepared: {len(sentences)} sentences")
        print(f"Train: {split_idx} sentences")
        print(f"Validation: {len(sentences) - split_idx} sentences")
    
    def setup_model(self):
        """
        Load the pre-trained French model and tokenizer.
        
        This method loads asi/gpt-fr-cased-small, a French GPT model
        that's already trained on French text, making it ideal for
        fine-tuning on our specific domain.
        """
        print(f"\nLoading French model: {self.model_name}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Move model to device
        self.model.to(self.device)
        
        print(f"Model loaded with {self.model.num_parameters():,} parameters")
        print(f"Using device: {self.device}")
    
    def fine_tune(self):
        """
        Execute the fine-tuning process.
        
        This method performs efficient fine-tuning with settings optimized
        for small datasets and limited computational resources.
        """
        print("\nStarting fine-tuning...")
        
        # Training arguments optimized for fine-tuning
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            
            # Reduced epochs for fine-tuning
            num_train_epochs=2,
            
            # Small batch sizes for limited memory
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            
            # Learning rate for fine-tuning (smaller than training from scratch)
            learning_rate=2e-5,
            warmup_steps=100,
            
            # Evaluation and saving
            evaluation_strategy="steps",
            eval_steps=200,
            save_steps=200,
            save_total_limit=2,
            load_best_model_at_end=True,
            
            # Logging
            logging_dir='./logs',
            logging_steps=50,
            
            # Memory optimization
            gradient_checkpointing=True if self.device.type == "cuda" else False,
            fp16=True if self.device.type == "cuda" else False,
            
            # Early stopping
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        
        # Load datasets
        train_dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path="train.txt",
            block_size=128
        )
        
        val_dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path="val.txt",
            block_size=128
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Initialize trainer with early stopping
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Fine-tune the model
        trainer.train()
        
        # Save the fine-tuned model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"\nFine-tuning completed!")
        print(f"Model saved to {self.output_dir}")
    
    def generate_samples(self, num_samples: int = 5):
        """
        Generate sample texts using the fine-tuned model.
        
        Args:
            num_samples: Number of samples to generate
        """
        print("\n" + "="*60)
        print("GENERATED SAMPLES FROM FINE-TUNED MODEL")
        print("="*60)
        
        # Test prompts
        prompts = [
            "La mission handicap",
            "Les étudiants en situation de handicap",
            "Pour bénéficier d'un aménagement",
            "L'université propose",
            "Les services d'accompagnement"
        ]
        
        self.model.eval()
        
        for i, prompt in enumerate(prompts[:num_samples]):
            inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=60,
                    num_return_sequences=1,
                    temperature=0.8,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=True,
                    top_p=0.9
                )
            
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\n{i+1}. Prompt: '{prompt}'")
            print(f"   Generated: {generated}")

    def evaluate_perplexity(self):
        """
        Evaluate model using perplexity on validation set.
        Perplexity = exp(loss)
        """
        
        print("\n" + "="*60)
        print("EVALUATING MODEL WITH PERPLEXITY")
        print("="*60)

  # Load dataset using Hugging Face Datasets library
        dataset = load_dataset("text", data_files={"validation": "val.txt"})
        tokenized = dataset.map(lambda x: self.tokenizer(x["text"], truncation=True, padding="max_length", max_length=128), batched=True)
        val_dataset = tokenized["validation"].with_format("torch")

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        val_loader = DataLoader(val_dataset, batch_size=2, collate_fn=data_collator)

        self.model.eval()
        total_loss = 0.0
        total_batches = 0

        for batch in val_loader:
            inputs = {k: v.to(self.device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            print(f"Batch Loss: {loss.item():.4f}")  # Print loss for each batch
            total_loss += loss.item()
            total_batches += 1

        if total_batches == 0:
            print("No batches found. Evaluation skipped.")
            return

        avg_loss = total_loss / total_batches
        perplexity = math.exp(avg_loss)
        print(f"\nValidation Loss: {avg_loss:.4f}")
        print(f"Perplexity: {perplexity:.2f}")





def main():
    """
    Main execution function for the complete NLG pipeline.
    
    This function orchestrates:
    1. Data loading and augmentation
    2. Fine-tuning a pre-trained French model
    3. Generating sample outputs
    """
    # Input files
    json_files = ['inalco.json', 'nanterre.json', 'sorbonne-nouvelle.json']
    
    # Step 1: Data Augmentation
    print("="*60)
    print("STEP 1: DATA AUGMENTATION")
    print("="*60)
    
    augmenter = DataAugmentation(json_files)
    augmented_corpus = augmenter.augment()
    
    # Save augmented corpus
    with open('augmented_corpus_full.txt', 'w', encoding='utf-8') as f:
        f.write(augmented_corpus)
    
    print(f"\nAugmented corpus saved")
    print(f"Corpus size: {len(augmented_corpus.split())} words")
    
    # Step 2: Fine-tuning
    print("\n" + "="*60)
    print("STEP 2: FINE-TUNING PRE-TRAINED FRENCH MODEL")
    print("="*60)
    
    # Initialize fine-tuner
    fine_tuner = HandicapNLGFineTuner(augmented_corpus)
    fine_tuner.prepare_data()
    fine_tuner.setup_model()
    
    # Execute fine-tuning
    try:
        fine_tuner.evaluate_perplexity()
        fine_tuner.fine_tune()
        fine_tuner.generate_samples()
    except Exception as e:
        print(f"\nNote: Fine-tuning requires significant memory.")

    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()