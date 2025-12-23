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
    
    def __init__(self, source_lang="en", target_lang="zh", **kwargs):
        self.source_lang = source_lang
        self.target_lang = target_lang
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
            
            print(f"  Train examples: {len(train_examples) if hasattr(train_examples, '__len__') else 'streaming'}")
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
        
        # Build samples, swapping source/target if needed
        print("  Building samples...")
        train_samples = []
        for idx, example in enumerate(train_examples):
            sample = self.build_sample(example, idx, swapped=swapped)
            train_samples.append(sample)
        
        valid_samples = []
        for idx, example in enumerate(valid_examples):
            sample = self.build_sample(example, idx, swapped=swapped)
            valid_samples.append(sample)
        
        self.samples = {"train": train_samples, "valid": valid_samples}
        
        print(f"  ✓ Created {len(train_samples)} training samples and {len(valid_samples)} validation samples")
        
        if swapped:
            print(f"  Note: Using reversed dataset '{self.target_lang}-{self.source_lang}' for '{self.source_lang}-{self.target_lang}' translation")
    
    def build_sample(self, example, idx, swapped=False):
        """
        Build a translation sample from WMT19 dataset
        
        WMT19 数据集格式:
        - 'translation': dict with source_lang and target_lang keys
        
        Args:
            swapped: If True, swap source and target (used when loading reversed language pair)
        """
        if isinstance(example, dict) and 'translation' in example:
            translation = example['translation']
            if swapped:
                # If we loaded reversed pair, swap the languages
                source_text = translation.get(self.target_lang, "")
                target_text = translation.get(self.source_lang, "")
            else:
                source_text = translation.get(self.source_lang, "")
                target_text = translation.get(self.target_lang, "")
        elif isinstance(example, dict) and self.source_lang in example:
            if swapped:
                source_text = example.get(self.target_lang, "")
                target_text = example.get(self.source_lang, "")
            else:
                source_text = example.get(self.source_lang, "")
                target_text = example.get(self.target_lang, "")
        else:
            # Fallback: assume example is a dict with 'source' and 'target'
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

