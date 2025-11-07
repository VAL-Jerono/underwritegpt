"""
LLM Engine for UnderwriteGPT
Supports multiple FREE/Open-Source LLM options
"""

import os
import requests
from typing import Dict, List, Optional
import json
from pathlib import Path

# LLM Options (choose one based on what you install)
LLM_BACKEND = os.getenv('LLM_BACKEND', 'ollama')  # Changed default to ollama


class UnderwriteLLM:
    """Unified interface for different LLM backends"""
    
    def __init__(self, backend: str = LLM_BACKEND, cache_dir: str = "models/llm_cache"):
        self.backend = backend
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the selected LLM backend"""
        
        if self.backend == 'ollama':
            return self._init_ollama()
        elif self.backend == 'huggingface':
            return self._init_huggingface()
        elif self.backend == 'gpt4all':
            return self._init_gpt4all()
        elif self.backend == 'template':
            print("✅ Using template mode (no LLM required)")
            return None
        else:
            print(f"⚠️ Unknown backend: {self.backend}, falling back to template mode")
            self.backend = 'template'
            return None
    
    def _init_ollama(self):
        """Initialize Ollama (RECOMMENDED - Best balance of quality and speed)"""
        try:
            # Test if Ollama is running
            response = requests.get('http://localhost:11434/api/tags', timeout=2)
            if response.status_code == 200:
                print("✅ Loading Ollama (Mistral-7B)...")
                print("✅ Ollama initialized successfully!")
                return 'mistral'  # Return model name
            else:
                raise Exception("Ollama not responding")
        except Exception as e:
            print(f"❌ Ollama initialization failed: {e}")
            print("💡 Install Ollama from https://ollama.ai then run: ollama pull mistral")
            return None
    
    def _init_huggingface(self):
        """Initialize HuggingFace local model (Requires setup)"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            import torch
            
            model_name = "HuggingFaceH4/zephyr-7b-beta"
            print("✅ Loading HuggingFace model (Zephyr-7B-beta, 4-bit CPU mode)...")

            # 4-bit quantization for Intel Mac CPU performance
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=True
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="cpu",
                local_files_only=True
            )

            print("✅ HuggingFace model loaded in efficient CPU mode")
            return (model, tokenizer)

        except Exception as e:
            print(f"❌ HuggingFace initialization failed: {e}")
            print("💡 Ensure model is downloaded into ~/.cache/huggingface/hub/")
            return None
    
    def _init_gpt4all(self):
        """Initialize GPT4All (Completely offline, no setup needed)"""
        try:
            from gpt4all import GPT4All
            print("✅ Loading GPT4All (Mistral-7B-OpenOrca)...")
            model = GPT4All("mistral-7b-openorca.Q4_0.gguf")
            print("✅ GPT4All initialized successfully!")
            return model
        except Exception as e:
            print(f"❌ GPT4All initialization failed: {e}")
            print("💡 Install: pip install gpt4all")
            return None
    
    def generate_underwriting_response(
        self,
        decision: Dict,
        features: Dict,
        evidence: Dict,
        risk_analysis: Dict
    ) -> str:
        """
        Generate a SINGLE concise paragraph explaining the underwriting decision
        Returns a single string (not a list)
        """
        
        # Build context-rich prompt
        prompt = self._build_prompt(decision, features, evidence, risk_analysis)
        
        # Check cache first
        cache_key = self._get_cache_key(prompt)
        cached = self._get_cached_response(cache_key)
        if cached:
            return cached
        
        # Generate response
        if self.model is None:
            return self._fallback_response(decision, features, evidence, risk_analysis)
        
        try:
            response = self._call_llm(prompt)
            # Clean up the response to ensure it's a single paragraph
            cleaned_response = self._clean_response(response)
            
            # Cache the response
            self._cache_response(cache_key, cleaned_response)
            
            return cleaned_response
        except Exception as e:
            print(f"⚠️ LLM generation failed: {e}")
            return self._fallback_response(decision, features, evidence, risk_analysis)
    
    def _build_prompt(self, decision, features, evidence, risk_analysis):
        """Build a detailed prompt for the LLM - requesting single paragraph"""
        
        claim_rate = evidence['claims'] / evidence['total'] * 100
        risk_pct = risk_analysis['overall'] * 100
        
        prompt = f"""You are an expert insurance underwriter. Write ONE concise paragraph (4-6 sentences) explaining this underwriting decision.

DECISION: {decision['tier']} ({decision['action']})
CONFIDENCE: {decision['confidence']:.0f}%

APPLICANT PROFILE:
- {features['customer_age']}-year-old driver
- {features['vehicle_age']}-year-old vehicle
- {features['subscription_length']}-month subscription
- Location: {'Urban' if features['is_urban'] else 'Rural'}
- Safety features: {features['airbags']} airbags, ESC: {'Yes' if features['has_esc'] else 'No'}

RISK ANALYSIS:
- Overall risk score: {risk_pct:.1f}%
- Top risk factor: Subscription ({risk_analysis['components']['subscription']*100:.1f}% risk)

EVIDENCE:
- Analyzed {evidence['total']} similar cases
- {evidence['claims']} filed claims ({claim_rate:.1f}% rate vs 6.4% baseline)

Write ONE flowing paragraph that:
1. States the decision clearly
2. Explains the applicant's key characteristics
3. References the evidence from similar cases
4. Indicates next steps or premium adjustment

Be professional, empathetic, and direct. No bullet points. No line breaks. Just one natural paragraph.

Response:"""

        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """Call the appropriate LLM backend"""
        if self.backend == 'ollama':
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.4,
                        "num_predict": 256,  # Reduced since we want single paragraph
                        "top_p": 0.9
                    }
                },
                timeout=30
            )
            data = response.json()
            return data.get("response", "")
        
        elif self.backend == 'huggingface':
            model, tokenizer = self.model
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            outputs = model.generate(
                input_ids,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.4,
                top_p=0.9
            )
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return full_text.replace(prompt, '').strip()

        elif self.backend == 'gpt4all':
            return self.model.generate(prompt, max_tokens=256, temp=0.4)
        
        return ""
    
    def _clean_response(self, response: str) -> str:
        """Clean up LLM response to ensure it's a single paragraph"""
        # Remove any markdown formatting
        response = response.replace('**', '')
        
        # Remove multiple newlines and replace with space
        response = ' '.join(response.split('\n'))
        
        # Remove multiple spaces
        response = ' '.join(response.split())
        
        # Trim to reasonable length (max ~300 words)
        words = response.split()
        if len(words) > 300:
            response = ' '.join(words[:300]) + '...'
        
        return response.strip()
    
    def _fallback_response(self, decision, features, evidence, risk_analysis) -> str:
        """High-quality single-paragraph template fallback"""
        
        tier = decision['tier']
        age = features['customer_age']
        v_age = features['vehicle_age']
        sub = features['subscription_length']
        claims = evidence['claims']
        total = evidence['total']
        claim_rate = (claims / total * 100) if total > 0 else 0
        risk_pct = risk_analysis['overall'] * 100
        
        templates = {
            'APPROVE': f"After analyzing your application against our database of 58,000+ policies, I'm pleased to recommend standard approval for your coverage. As a {age}-year-old driver with a {v_age}-year-old vehicle on a {sub}-month subscription, your profile demonstrates strong low-risk characteristics with a calculated risk score of {risk_pct:.1f}%. Our analysis of {total} similar cases revealed only {claims} claims ({claim_rate:.1f}% claim rate), which is significantly below our industry baseline of 6.4%. You qualify for standard rates with no additional conditions, and your policy can be processed within 24-48 hours.",
            
            'MONITOR': f"Thank you for your application—I can offer you coverage with some adjustments to ensure mutual protection. Your profile as a {age}-year-old driver with a {v_age}-year-old vehicle on a {sub}-month plan shows moderate risk characteristics, reflected in your {risk_pct:.1f}% risk score. When comparing against {total} similar policies in our database, we found {claims} resulted in claims ({claim_rate:.1f}% rate), which is slightly above our 6.4% baseline. Based on this analysis, I recommend approval with a 15-20% premium adjustment and quarterly monitoring during your first year; as you build a claims-free history, we can revisit these terms for better rates.",
            
            'CONDITIONAL': f"I appreciate your application and want to work with you, though your profile requires some conditions for approval. As a {age}-year-old with a {v_age}-year-old vehicle on a {sub}-month subscription, your risk profile scores at {risk_pct:.1f}%, placing you in our higher-risk category based on actuarial data. Our analysis of {total} similar cases revealed {claims} claims—a {claim_rate:.1f}% rate notably higher than our 6.4% industry average—which necessitates additional safeguards. I propose approval with a 30-40% premium loading, higher deductible ($1,500), and enhanced documentation requirements; after 12 months of claims-free driving, we'll review your policy for improved rates.",
            
            'REJECT': f"Thank you for considering us for your insurance needs, though after thorough analysis, I must recommend we decline this application at this time. Your profile as a {age}-year-old driver with a {v_age}-year-old vehicle on a {sub}-month plan triggers multiple high-risk indicators in our underwriting system, resulting in a {risk_pct:.1f}% risk score. Our analysis of {total} similar policies found {claims} filed claims within their policy period—a {claim_rate:.1f}% rate that's {claim_rate/6.4:.1f}x our industry standard—which exceeds the exposure levels our risk models can responsibly accommodate. I encourage you to consider building insurance history with a specialized carrier, exploring vehicles with advanced safety features, or revisiting your application with us after establishing a longer driving record."
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
        """Cache response for faster future lookups"""
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