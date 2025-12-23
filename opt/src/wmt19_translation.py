"""
WMT19 Translation Dataset for Parallel DistZO2 + DP-AggZO
支持英中翻译任务
"""

from .tasks import Dataset, Sample
from datasets import load_dataset
from typing import Optional


class WMT19TranslationDataset(Dataset):
    """
    WMT19 English-Chinese Translation Dataset
    用于机器翻译任务
    """
    metric_name = "bleu"  # 翻译任务使用 BLEU 分数
    generation = True  # 生成式任务
    
    def __init__(self, source_lang="en", target_lang="zh", max_samples=None, **kwargs):
        """
        Initialize WMT19 Translation Dataset
        
        Args:
            source_lang: Source language code (e.g., 'en')
            target_lang: Target language code (e.g., 'zh')
            max_samples: Maximum number of training samples to use (for scaling law experiments)
                         If None, use all available samples
        """
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_samples = max_samples
        self.load_dataset()
    
    def load_dataset(self):
        """Load WMT19 dataset from HuggingFace"""
        # Try direct language pair first
        dataset_config = f"{self.source_lang}-{self.target_lang}"
        swapped = False
        dataset = None
        
        try:
            dataset = load_dataset("wmt19", dataset_config)
            print(f"✓ Successfully loaded WMT19 dataset '{dataset_config}'")
        except Exception as e1:
            # If direct pair fails, try reversed pair (e.g., use zh-en for en-zh)
            reversed_config = f"{self.target_lang}-{self.source_lang}"
            print(f"Warning: Language pair '{dataset_config}' not found. Trying reversed pair '{reversed_config}'...")
            print(f"  Error: {str(e1)[:200]}")
            
            try:
                dataset = load_dataset("wmt19", reversed_config)
                swapped = True  # Mark that we need to swap source and target
                print(f"✓ Successfully loaded reversed dataset '{reversed_config}'")
            except Exception as e2:
                print(f"✗ Failed to load WMT19 dataset '{dataset_config}' or '{reversed_config}'")
                print(f"  First error: {str(e1)[:200]}")
                print(f"  Second error: {str(e2)[:200]}")
                print("Creating dummy dataset for testing...")
                # 创建虚拟数据集用于测试
                self.samples = {
                    "train": [
                        Sample(
                            id=0,
                            data={"source": "Hello world", "target": "你好世界"},
                            candidates=None,
                            correct_candidate="你好世界"
                        )
                    ] * 100,
                    "valid": [
                        Sample(
                            id=0,
                            data={"source": "Hello", "target": "你好"},
                            candidates=None,
                            correct_candidate="你好"
                        )
                    ] * 10
                }
                return
        
        # Load dataset successfully - process splits
        try:
            train_examples = dataset.get("train", [])
            valid_examples = dataset.get("validation", [])
            
            original_train_size = len(train_examples) if hasattr(train_examples, '__len__') else 'unknown'
            print(f"  Train examples: {original_train_size}")
            print(f"  Validation examples: {len(valid_examples) if hasattr(valid_examples, '__len__') else 'streaming'}")
        except Exception as e:
            print(f"✗ Error accessing dataset splits: {e}")
            print("Creating dummy dataset for testing...")
            self.samples = {
                "train": [
                    Sample(
                        id=0,
                        data={"source": "Hello world", "target": "你好世界"},
                        candidates=None,
                        correct_candidate="你好世界"
                    )
                ] * 100,
                "valid": [
                    Sample(
                        id=0,
                        data={"source": "Hello", "target": "你好"},
                        candidates=None,
                        correct_candidate="你好"
                    )
                ] * 10
            }
            return
        
        # [Critical Fix] Downsample at HF Dataset level BEFORE loading to memory
        # This prevents OOM when dealing with 25M+ samples but only need 10k-50k
        if self.max_samples is not None:
            if hasattr(train_examples, '__len__') and len(train_examples) > self.max_samples:
                print(f"  ⚠ Downsampling training set from {len(train_examples)} to {self.max_samples} samples (Lazy Loading)...")
                # Use HF's shuffle and select - this is fast and memory-efficient
                train_examples = train_examples.shuffle(seed=42).select(range(self.max_samples))
        
        # Build samples (now train_examples is already downsampled if needed)
        print("  Building samples...")
        train_samples = []
        for idx, example in enumerate(train_examples):
            sample = self.build_sample(example, idx, swapped=False)
            train_samples.append(sample)
        
        # Validation set is usually small, can process directly
        valid_samples = []
        for idx, example in enumerate(valid_examples):
            sample = self.build_sample(example, idx, swapped=False)
            valid_samples.append(sample)
        
        self.samples = {"train": train_samples, "valid": valid_samples}
        
        print(f"  ✓ Created {len(train_samples)} training samples and {len(valid_samples)} validation samples")
        
        if swapped:
            print(f"  Note: Using reversed dataset '{self.target_lang}-{self.source_lang}' for '{self.source_lang}-{self.target_lang}' translation")
    
    def build_sample(self, example, idx, swapped=False):
        """
        Build a translation sample from WMT19 dataset
        
        WMT19 数据集格式:
        - 'translation': dict with fixed language keys (e.g., {'zh': '中文...', 'en': 'English...'})
        
        Important: The dictionary keys are fixed language codes, not dependent on which config
        (zh-en or en-zh) was loaded. We directly use self.source_lang and self.target_lang to
        extract the correct text, regardless of the swapped flag.
        
        Args:
            swapped: This parameter is kept for backward compatibility but is no longer used
                     to swap dictionary keys, since keys are fixed language codes.
        """
        if isinstance(example, dict) and 'translation' in example:
            translation = example['translation']
            # Directly use source_lang and target_lang to get values
            # translation['en'] is always English, translation['zh'] is always Chinese
            # No need to swap keys because keys are fixed language codes
            source_text = translation.get(self.source_lang, "")
            target_text = translation.get(self.target_lang, "")
        
        # Handle potential alternative data formats (fallback)
        elif isinstance(example, dict) and self.source_lang in example:
            source_text = example.get(self.source_lang, "")
            target_text = example.get(self.target_lang, "")
        else:
            # Very rare case: data has 'source'/'target' fields instead of language keys
            # Only in this case might swapped be useful, but WMT19 standard format doesn't need it
            # Keep this fallback for safety
            if swapped:
                source_text = example.get('target', str(example))
                target_text = example.get('source', "")
            else:
                source_text = example.get('source', str(example))
                target_text = example.get('target', "")
        
        return Sample(
            id=idx,
            data={
                "source": source_text,
                "target": target_text
            },
            candidates=None,
            correct_candidate=target_text
        )
    
    def get_template(self, template_version=0):
        """Translation tasks don't need templates in the same way"""
        # Return a simple template that just returns the source text
        class TranslationTemplate:
            def encode(self, sample):
                return sample.data["source"]
            
            def verbalize(self, sample, candidate):
                return candidate or sample.data["target"]
        
        return TranslationTemplate()

