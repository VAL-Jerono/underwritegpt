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
                data = response.json()
                models = data.get('models', [])
            
                if not models:
                    print("❌ No Ollama models found. Please run: ollama pull mistral")
                    return None
            
                print(f"✅ Ollama is running")
                print(f"📋 Available models: {[m.get('name', 'unknown') for m in models]}")
            
                # Priority order: mistral > zephyr > llama2 > first available
                preferred_models = [
                    'phi3:mini',      # Fast & good quality (recommended)
                    'phi3',           # Also good
                    'gemma:2b',       # Very fast
                    'tinyllama',      # Fastest but lower quality
                    'mistral',        # Slower but good quality
                    'zephyr',         # Current model (slow on CPU)
                    'llama2'          # Fallback
                    ]

            
                for preferred in preferred_models:
                    for model in models:
                        model_name = model.get('name', '')
                        if preferred in model_name.lower():
                            print(f"✅ Using model: {model_name}")
                            return model_name  # Return exact name like "zephyr:latest"
            
                # Fallback to first available model
                first_model = models[0]['name']
                print(f"⚠️ Using first available model: {first_model}")
                return first_model
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
        risk_analysis: Dict,
        mode: str = 'underwriter'
    ) -> str:
        """
        Generate a SINGLE concise paragraph explaining the underwriting decision
        Returns a single string (not a list)
        """
        
        # Build context-rich prompt
        prompt = self._build_prompt(decision, features, evidence, risk_analysis, mode)
        
        # Check cache first
        cache_key = self._get_cache_key(prompt)
        cached = self._get_cached_response(cache_key)
        if cached:
            print(f"✅ Using cached response")
            return cached
        
        # Generate response
        if self.model is None:
            print(f"⚠️ No model loaded, using fallback for mode: {mode}")
            return self._fallback_response(decision, features, evidence, risk_analysis, mode)
        
        try:
            print(f"🤖 Calling LLM backend: {self.backend}")
            response = self._call_llm(prompt)
            
            if not response or len(response.strip()) == 0:
                print(f"⚠️ Empty response from LLM, using fallback")
                return self._fallback_response(decision, features, evidence, risk_analysis, mode)
            
            # Clean up the response to ensure it's a single paragraph
            cleaned_response = self._clean_response(response)
            print(f"✅ Generated response: {len(cleaned_response)} chars")
            
            # Cache the response
            self._cache_response(cache_key, cleaned_response)
            
            return cleaned_response
        except Exception as e:
            print(f"⚠️ LLM generation failed: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_response(decision, features, evidence, risk_analysis, mode)
    
    def _build_prompt(self, decision, features, evidence, risk_analysis, mode='underwriter'):
        """Build prompt based on mode"""
        
        if mode == 'mycar':
            voice = "You are a helpful insurance advisor speaking directly to a car owner. Use 'your car' and 'you' frequently. Be warm and personal."
        else:
            voice = "You are an insurance underwriter explaining a decision to your colleagues. Use 'the applicant' and 'we recommend'."
        
        claim_rate = evidence['claims'] / evidence['total'] * 100
        risk_pct = risk_analysis['overall'] * 100
        
        prompt = f"""{voice}

You are an expert insurance underwriter. Write ONE concise paragraph (4-6 sentences) explaining this underwriting decision.

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
            try:
                print(f"📡 Sending request to Ollama with model: {self.model}")
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": True, #False,
                        "options": {
                            "temperature": 0.4,
                            "num_predict": 128, # Down from 256,
                            "top_p": 0.9
                        }
                    },
                    stream=True, #Enable streaming on request side
                    timeout=None #INCREASED TIMEOUT
                )
                
                # Check if request was successful
                if response.status_code != 200:
                    print(f"⚠️ Ollama returned status {response.status_code}")
                    return ""
                
                # Collect the streamed response
                full_response = ""
                chunk_count = 0
                
                print("📥 Receiving streaming response...")
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        try:
                            json_response = json.loads(line)
                            chunk_count += 1

                            # Print progress every 10 chunks
                            if chunk_count % 10 == 0:
                                print(f"  ... received {chunk_count} chunks")
                                
                            if "response" in json_response:
                                full_response += json_response["response"]
                                
                            if json_response.get("done", False):
                                print(f"✅ Streaming complete! Received {chunk_count} chunks, {len(full_response)} characters")
                                break
                        except json.JSONDecodeError as e:
                            print(f"⚠️ Failed to parse JSON: {line[:100]}")
                            continue
                    
                if not full_response:
                    print("⚠️ No response text received from Ollama")
            
                return full_response.strip()
            
            except requests.exceptions.Timeout:
                print(f"⚠️ Ollama API call timed out after 60 seconds")
                return ""
            except Exception as e:
                print(f"⚠️ Ollama API call failed: {e}")
                return ""
                
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
    
    def _fallback_response(self, decision, features, evidence, risk_analysis, mode='underwriter') -> str:
        """High-quality single-paragraph template fallback"""
        
        tier = decision['tier']
        age = features['customer_age']
        v_age = features['vehicle_age']
        sub = features['subscription_length']
        claims = evidence['claims']
        total = evidence['total']
        claim_rate = (claims / total * 100) if total > 0 else 0
        risk_pct = risk_analysis['overall'] * 100
        
        if mode == 'mycar':
            # PERSONALIZED TEMPLATES FOR MY CAR CHECK
            templates = {
                'APPROVE': f"Great news about your car! Your {v_age}-year-old vehicle with its {features['airbags']} airbags and {'excellent' if features['has_esc'] else 'basic'} safety features qualifies for standard insurance rates. At {age} years old with a {sub}-month policy, you're in a strong position—your profile shows only {risk_pct:.1f}% risk, and similar drivers filed claims just {claim_rate:.1f}% of the time (well below the 6.4% average). You can expect approval within 24-48 hours with no extra conditions or fees.",
            
                'MONITOR': f"We can definitely insure your {v_age}-year-old car, though we'll need to add a 15-20% premium adjustment for the first year. Your profile shows moderate risk at {risk_pct:.1f}%—when we looked at {total} drivers similar to you, {claims} filed claims ({claim_rate:.1f}% rate). The good news? After you build a claims-free history over the next 12 months, we'll review your rates and likely reduce that premium. Your {'strong' if features['has_esc'] else 'adequate'} safety features help, and we're confident this will work out well.",
            
                'CONDITIONAL': f"Your {v_age}-year-old vehicle can be insured, but we need to be upfront about the conditions. Your profile shows elevated risk at {risk_pct:.1f}%—similar drivers in our database filed claims {claim_rate:.1f}% of the time, which is notably higher than typical. We can offer coverage with a 30-40% premium increase and a $1,500 deductible. Consider this a bridge policy: drive safely for 12 months, and we'll reassess for better terms. You might also explore adding more safety features to your car to improve your profile.",
            
                'REJECT': f"Unfortunately, we can't insure your {v_age}-year-old vehicle under our standard policies right now. Your profile shows {risk_pct:.1f}% risk, and when we analyzed {total} similar cases, {claims} resulted in claims—that's {claim_rate/6.4:.1f}x higher than what our pricing can support. This isn't personal—it's about matching risk with appropriate coverage. We recommend: (1) Consider a newer vehicle with better safety features, (2) Try a specialized high-risk insurer, or (3) Build insurance history elsewhere and reapply with us in 12-18 months."
            }
        else:
            # UNDERWRITER MODE TEMPLATES - COMPANY VOICE
            templates = {
                'APPROVE': f"After analyzing this application against our database of 58,000+ policies, we're pleased to recommend standard approval for coverage. The profile shows a {age}-year-old driver with a {v_age}-year-old vehicle on a {sub}-month subscription, demonstrating strong low-risk characteristics with a calculated risk score of {risk_pct:.1f}%. Our analysis of {total} similar cases revealed only {claims} claims ({claim_rate:.1f}% claim rate), significantly below our industry baseline of 6.4%. This application qualifies for standard rates with no additional conditions, and the policy can be processed within 24-48 hours.",
        
                'MONITOR': f"We appreciate this application and can offer coverage with some adjustments to ensure mutual protection. The profile presents a {age}-year-old driver with a {v_age}-year-old vehicle on a {sub}-month plan, showing moderate risk characteristics reflected in the {risk_pct:.1f}% risk score. Comparing against {total} similar policies in our database, we found {claims} resulted in claims ({claim_rate:.1f}% rate), slightly above our 6.4% baseline. Based on this analysis, we recommend approval with a 15-20% premium adjustment and quarterly monitoring during the first year; as a claims-free history builds, we can revisit these terms for improved rates.",
        
                'CONDITIONAL': f"We value this application and want to find a path forward, though the profile requires certain conditions for approval. As a {age}-year-old with a {v_age}-year-old vehicle on a {sub}-month subscription, the risk profile scores at {risk_pct:.1f}%, placing it in our higher-risk category based on actuarial data. Our analysis of {total} similar cases revealed {claims} claims—a {claim_rate:.1f}% rate notably higher than our 6.4% industry average—necessitating additional safeguards. We propose approval with a 30-40% premium loading, higher deductible ($1,500), and enhanced documentation requirements; after 12 months of claims-free driving, we'll review the policy for improved rates.",
        
                'REJECT': f"We appreciate the interest in our coverage, though after thorough analysis, our underwriting team must decline this application at this time. The profile of a {age}-year-old driver with a {v_age}-year-old vehicle on a {sub}-month plan triggers multiple high-risk indicators in our system, resulting in a {risk_pct:.1f}% risk score. Our analysis of {total} similar policies found {claims} filed claims within their policy period—a {claim_rate:.1f}% rate that's {claim_rate/6.4:.1f}x our industry standard—exceeding the exposure levels our risk models can responsibly accommodate. We encourage considering building insurance history with a specialized carrier, exploring vehicles with advanced safety features, or revisiting the application after establishing a longer driving record."
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
    
    def _init_huggingface_api(self):
        """Use HuggingFace Inference API (free tier available)"""
        try:
            import os
            api_key = os.getenv('HF_API_KEY')  # Set in Streamlit secrets
            if api_key:
                return {'api_key': api_key, 'model': 'mistralai/Mistral-7B-Instruct-v0.2'}
        except:
            pass
        return None


# Singleton instance
_llm_instance = None

def get_llm_engine():
    """Get or create LLM engine singleton"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = UnderwriteLLM()
    return _llm_instance