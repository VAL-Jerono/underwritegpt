"""
LLM Engine for UnderwriteGPT - FORCE TEMPLATE MODE
This version forces template mode for reliable deployment
"""

import os
import requests
from typing import Dict, List, Optional
import json
from pathlib import Path

# FORCE TEMPLATE MODE for deployment
LLM_BACKEND = os.getenv('LLM_BACKEND', 'template')  # Changed from 'ollama' to 'template'


class UnderwriteLLM:
    """Unified interface for different LLM backends"""
    
    def __init__(self, backend: str = LLM_BACKEND, cache_dir: str = "models/llm_cache"):
        self.backend = backend
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Force template mode for deployment
        if self.backend == 'ollama':
            print("⚠️ Ollama not available in cloud deployment, using template mode")
            self.backend = 'template'
        
        self.model = self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the selected LLM backend"""
        
        if self.backend == 'template':
            print("✅ Using template mode (no LLM required)")
            return None
        elif self.backend == 'ollama':
            return self._init_ollama()
        elif self.backend == 'huggingface':
            return self._init_huggingface()
        elif self.backend == 'gpt4all':
            return self._init_gpt4all()
        else:
            print(f"⚠️ Unknown backend: {self.backend}, falling back to template mode")
            self.backend = 'template'
            return None
            
    def _init_ollama(self):
        """Initialize Ollama"""
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=2)
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])
            
                if not models:
                    print("❌ No Ollama models found")
                    return None
            
                print(f"✅ Ollama is running")
                
                preferred_models = [
                    'zephyr', 'phi3:mini', 'phi3', 'gemma:2b', 
                    'tinyllama', 'mistral', 'llama2'
                ]
            
                for preferred in preferred_models:
                    for model in models:
                        model_name = model.get('name', '')
                        if preferred in model_name.lower():
                            print(f"✅ Using model: {model_name}")
                            return model_name
            
                first_model = models[0]['name']
                print(f"⚠️ Using first available model: {first_model}")
                return first_model
            else:
                raise Exception("Ollama not responding")
        except Exception as e:
            print(f"❌ Ollama initialization failed: {e}")
            return None
    
    def _init_huggingface(self):
        """Initialize HuggingFace local model"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            import torch
            
            model_name = "HuggingFaceH4/zephyr-7b-beta"
            print("✅ Loading HuggingFace model...")
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="cpu",
                local_files_only=True
            )
            
            print("✅ HuggingFace model loaded")
            return (model, tokenizer)
        except Exception as e:
            print(f"❌ HuggingFace initialization failed: {e}")
            return None
    
    def _init_gpt4all(self):
        """Initialize GPT4All"""
        try:
            from gpt4all import GPT4All
            print("✅ Loading GPT4All...")
            model = GPT4All("mistral-7b-openorca.Q4_0.gguf")
            print("✅ GPT4All initialized!")
            return model
        except Exception as e:
            print(f"❌ GPT4All initialization failed: {e}")
            return None
    
    def generate_underwriting_response(
        self,
        decision: Dict,
        features: Dict,
        evidence: Dict,
        risk_analysis: Dict,
        mode: str = 'underwriter'
    ) -> str:
        """Generate a concise paragraph explaining the underwriting decision"""
        
        # ALWAYS use fallback templates for deployment
        print(f"✅ Using template mode for {mode}")
        return self._fallback_response(decision, features, evidence, risk_analysis, mode)
    
    def _fallback_response(self, decision, features, evidence, risk_analysis, mode='underwriter') -> str:
        """
        Dynamic template that uses ACTUAL client details
        Honest about assumptions when features aren't specified
        """
        
        tier = decision['tier']
        age = features['customer_age']
        v_age = features['vehicle_age']
        sub = features['subscription_length']
        claims = evidence['claims']
        total = evidence['total']
        claim_rate = (claims / total * 100) if total > 0 else 0
        risk_pct = risk_analysis['overall'] * 100
        
        # Check what was actually provided
        age_provided = features.get('customer_age_provided', True)  # Default True to show values
        v_age_provided = features.get('vehicle_age_provided', True)
        sub_provided = features.get('subscription_provided', True)
        airbags_provided = features.get('airbags_provided', True)
        esc_provided = features.get('esc_provided', True)
        region_provided = features.get('region_provided', True)
        
        # Build descriptions based on what was provided
        driver_desc = f"{age}-year-old driver" if age_provided else "driver"
        
        # Vehicle description
        vehicle_parts = []
        if v_age_provided:
            vehicle_parts.append(f"{v_age:.0f}-year-old vehicle" if v_age >= 1 else "new vehicle")
        else:
            vehicle_parts.append("vehicle")
        
        # Safety features (only mention if provided)
        safety_parts = []
        if airbags_provided:
            airbags = features['airbags']
            safety_parts.append(f"{airbags} airbag{'s' if airbags != 1 else ''}")
        if esc_provided:
            has_esc = features['has_esc']
            safety_parts.append("with ESC" if has_esc else "without ESC")
        
        if safety_parts:
            vehicle_desc = f"{vehicle_parts[0]} ({', '.join(safety_parts)})"
        else:
            vehicle_desc = vehicle_parts[0]
        
        # Subscription description
        if sub_provided:
            sub_desc = f"{sub}-month subscription"
        else:
            sub_desc = "policy"
        
        # Location description
        if region_provided:
            is_urban = features['is_urban']
            location_desc = "urban area" if is_urban else "rural area"
            location_phrase = f" in a {location_desc}"
        else:
            location_phrase = ""
        
        # Calculate multiplier vs baseline
        baseline_rate = 6.4
        multiplier = claim_rate / baseline_rate if baseline_rate > 0 else 1
        
        # Build "missing info" note if critical features are unspecified
        missing_features = []
        if not sub_provided:
            missing_features.append("subscription length")
        if not airbags_provided:
            missing_features.append("airbag count")
        if not esc_provided:
            missing_features.append("ESC status")
        
        missing_note = ""
        if missing_features and tier in ['MONITOR', 'CONDITIONAL', 'REJECT']:
            missing_note = f" Note: Our analysis assumes standard features for {', '.join(missing_features)} as these weren't specified—providing complete details could improve your assessment."
        
        if mode == 'mycar':
            # CONSUMER MODE
            templates = {
                'APPROVE': (
                    f"Great news! Your {vehicle_desc} qualifies for standard insurance rates. "
                    f"{'At ' + str(age) + ' years old' if age_provided else 'Based on your profile'} "
                    f"{'with a ' + sub_desc if sub_provided else ''}{location_phrase}, "
                    f"you're in a strong position—your profile shows {risk_pct:.1f}% risk, "
                    f"and similar drivers filed claims just {claim_rate:.1f}% of the time "
                    f"({'well below' if claim_rate < baseline_rate else 'near'} the {baseline_rate}% average). "
                    f"You can expect approval within 24-48 hours with no extra conditions or fees.{missing_note}"
                ),
                
                'MONITOR': (
                    f"We can insure your {vehicle_desc}, though we'll need to add a 15-20% premium adjustment. "
                    f"{'As a ' + driver_desc if age_provided else 'Your profile'} "
                    f"{'with a ' + sub_desc if sub_provided else ''}{location_phrase}, "
                    f"your risk assessment shows {risk_pct:.1f}%. When we analyzed {total} similar profiles, "
                    f"{claims} filed claims ({claim_rate:.1f}% rate). "
                    f"The good news? After you build a claims-free history over the next 12 months, "
                    f"we'll review your rates and likely reduce that premium.{missing_note}"
                ),
                
                'CONDITIONAL': (
                    f"Your {vehicle_desc} can be insured, but we need to be upfront about the conditions. "
                    f"{'As a ' + driver_desc if age_provided else 'Based on your profile'} "
                    f"{'with a ' + sub_desc if sub_provided else ''}{location_phrase}, "
                    f"your risk assessment shows {risk_pct:.1f}%. Similar drivers in our database filed claims "
                    f"{claim_rate:.1f}% of the time, which is {multiplier:.1f}x higher than typical. "
                    f"We can offer coverage with a 30-40% premium increase and a higher deductible. "
                    f"Consider this a bridge policy: drive safely for 12 months, and we'll reassess for better terms.{missing_note}"
                ),
                
                'REJECT': (
                    f"Unfortunately, we can't insure your {vehicle_desc} under our standard policies right now. "
                    f"{'As a ' + driver_desc if age_provided else 'Based on your profile'} "
                    f"{'with a ' + sub_desc if sub_provided else ''}{location_phrase}, "
                    f"your risk assessment shows {risk_pct:.1f}%. When we analyzed {total} similar cases, "
                    f"{claims} resulted in claims—that's {claim_rate:.1f}%, or {multiplier:.1f}x higher than "
                    f"what our pricing can support. We recommend trying a specialized high-risk insurer.{missing_note}"
                )
            }
        else:
            # UNDERWRITER MODE:
            templates = {
                'APPROVE': (
                    f"After analyzing this application against our database, we recommend standard approval. "
                    f"The profile presents a {driver_desc} with a {vehicle_desc} "
                    f"{'on a ' + sub_desc if sub_provided else ''}{location_phrase}, "
                    f"with a calculated risk score of {risk_pct:.1f}%. "
                    f"Our analysis of {total} similar cases revealed only {claims} claims ({claim_rate:.1f}% rate), "
                    f"{'significantly below' if claim_rate < baseline_rate else 'aligned with'} our {baseline_rate}% baseline. "
                    f"This application qualifies for standard rates with no additional conditions.{missing_note}"
                ),
                
                'MONITOR': (
                    f"We can offer coverage with some adjustments to ensure mutual protection. "
                    f"The profile presents a {driver_desc} with a {vehicle_desc} "
                    f"{'on a ' + sub_desc if sub_provided else ''}{location_phrase}, "
                    f"showing moderate risk at {risk_pct:.1f}%. "
                    f"Comparing against {total} similar policies, we found {claims} claims ({claim_rate:.1f}% rate), "
                    f"{'slightly above' if multiplier > 1 else 'near'} our {baseline_rate}% baseline. "
                    f"We recommend approval with a 15-20% premium adjustment and quarterly monitoring during the first year; "
                    f"as a claims-free history builds, we can revisit these terms for improved rates.{missing_note}"
                ),
                
                'CONDITIONAL': (
                    f"We want to find a path forward, though this profile requires certain conditions for approval. "
                    f"The application presents a {driver_desc} with a {vehicle_desc} "
                    f"{'on a ' + sub_desc if sub_provided else ''}{location_phrase}, "
                    f"with a risk profile scoring at {risk_pct:.1f}%, placing it in our higher-risk category. "
                    f"Our analysis of {total} similar cases revealed {claims} claims—a {claim_rate:.1f}% rate "
                    f"{'notably higher' if multiplier > 1.5 else 'elevated'} compared to our {baseline_rate}% average. "
                    f"We propose approval with a 30-40% premium loading and enhanced risk management terms; "
                    f"after 12 months of claims-free driving, we'll review for improved rates.{missing_note}"
                ),
                
                'REJECT': (
                    f"After thorough analysis, our underwriting team must decline this application at this time. "
                    f"The profile of a {driver_desc} with a {vehicle_desc} "
                    f"{'on a ' + sub_desc if sub_provided else ''}{location_phrase} "
                    f"results in a {risk_pct:.1f}% risk score. "
                    f"Our analysis of {total} similar policies found {claims} claims—a {claim_rate:.1f}% rate "
                    f"that's {multiplier:.1f}x our industry standard—exceeding acceptable exposure levels. "
                    f"We encourage building history with a specialized carrier or revisiting after establishing a longer driving record.{missing_note}"
                )
            }
        
        return templates.get(tier, templates['CONDITIONAL'])
    
    def _get_cache_key(self, prompt: str) -> str:
        """Generate cache key from prompt"""
        import hashlib
        return hashlib.md5(prompt.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Retrieve cached response if exists"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return None
    
    def _cache_response(self, cache_key: str, response: str):
        """Cache response"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(response, f)
        except Exception as e:
            print(f"⚠️ Cache write failed: {e}")


# Singleton instance
_llm_instance = None

def get_llm_engine():
    """Get or create LLM engine singleton"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = UnderwriteLLM()
    return _llm_instance