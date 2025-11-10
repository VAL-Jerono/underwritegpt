"""
LLM Engine for UnderwriteGPT - FIXED VERSION
Now generates personalized responses matching actual client details
"""

import os
import requests
from typing import Dict, List, Optional
import json
from pathlib import Path

LLM_BACKEND = os.getenv('LLM_BACKEND', 'ollama')


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
            response = requests.get('http://localhost:11434/api/tags', timeout=2)
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])
            
                if not models:
                    print("❌ No Ollama models found. Please run: ollama pull mistral")
                    return None
            
                print(f"✅ Ollama is running")
                print(f"📋 Available models: {[m.get('name', 'unknown') for m in models]}")
            
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
            print("💡 Falling back to template mode")
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
            print("✅ GPT4All initialized successfully!")
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
        """Generate a SINGLE concise paragraph explaining the underwriting decision"""
        
        prompt = self._build_prompt(decision, features, evidence, risk_analysis, mode)
        
        # Check cache first
        cache_key = self._get_cache_key(prompt)
        cached = self._get_cached_response(cache_key)
        if cached:
            print(f"✅ Using cached response")
            return cached
        
        # Generate response
        if self.model is None:
            print(f"⚠️ No model loaded, using dynamic fallback for mode: {mode}")
            return self._fallback_response(decision, features, evidence, risk_analysis, mode)
        
        try:
            print(f"🤖 Calling LLM backend: {self.backend}")
            response = self._call_llm(prompt)
            
            if not response or len(response.strip()) == 0:
                print(f"⚠️ Empty response from LLM, using fallback")
                return self._fallback_response(decision, features, evidence, risk_analysis, mode)
            
            cleaned_response = self._clean_response(response)
            print(f"✅ Generated response: {len(cleaned_response)} chars")
            
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
                        "stream": True,
                        "options": {
                            "temperature": 0.4,
                            "num_predict": 200,
                            "top_p": 0.9,
                            "num_ctx": 2048,
                            "stop": ["\n\n", "---"]
                        }
                    },
                    stream=True,
                    timeout=(5, None)
                )
                
                if response.status_code != 200:
                    print(f"⚠️ Ollama returned status {response.status_code}")
                    return ""
                
                full_response = ""
                chunk_count = 0
                
                print("📥 Receiving streaming response...")
                
                try:
                    for line in response.iter_lines(decode_unicode=True):
                        if line:
                            try:
                                json_response = json.loads(line)
                                chunk_count += 1
                                
                                if chunk_count % 10 == 0:
                                    print(f"  ... received {chunk_count} chunks")
                                
                                if "response" in json_response:
                                    full_response += json_response["response"]
                                
                                if json_response.get("done", False):
                                    print(f"✅ Streaming complete! {chunk_count} chunks, {len(full_response)} chars")
                                    break
                            except json.JSONDecodeError:
                                continue
                
                except Exception as stream_error:
                    print(f"⚠️ Streaming error: {stream_error}")
                    if full_response:
                        return full_response.strip()
                    return ""
                    
                if not full_response:
                    print("⚠️ No response text received from Ollama")
            
                return full_response.strip()
            
            except requests.exceptions.Timeout:
                print(f"⚠️ Ollama connection timeout")
                return ""
            except Exception as e:
                print(f"⚠️ Unexpected error in Ollama call: {e}")
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
        response = response.replace('**', '')
        response = ' '.join(response.split('\n'))
        response = ' '.join(response.split())
        
        words = response.split()
        if len(words) > 300:
            response = ' '.join(words[:300]) + '...'
        
        return response.strip()
    
    def _fallback_response(self, decision, features, evidence, risk_analysis, mode='underwriter') -> str:
        """
        FIXED: Dynamic template fallback that uses ACTUAL client details
        No more hardcoded values!
        """
        
        tier = decision['tier']
        age = features['customer_age']
        v_age = features['vehicle_age']
        sub = features['subscription_length']
        claims = evidence['claims']
        total = evidence['total']
        claim_rate = (claims / total * 100) if total > 0 else 0
        risk_pct = risk_analysis['overall'] * 100
        airbags = features['airbags']
        has_esc = features['has_esc']
        is_urban = features['is_urban']
        
        # Build safety description
        safety_desc = f"{airbags} airbag{'s' if airbags != 1 else ''}"
        if has_esc:
            safety_desc += " with ESC"
        else:
            safety_desc += " without ESC"
        
        location_desc = "urban area" if is_urban else "rural area"
        
        # Calculate multiplier vs baseline
        baseline_rate = 6.4
        multiplier = claim_rate / baseline_rate if baseline_rate > 0 else 1
        
        if mode == 'mycar':
            # PERSONALIZED TEMPLATES FOR MY CAR CHECK
            templates = {
                'APPROVE': (
                    f"Great news about your car! Your {v_age:.1f}-year-old vehicle with {safety_desc} "
                    f"qualifies for standard insurance rates. At {age} years old with a {sub}-month policy "
                    f"in a {location_desc}, you're in a strong position—your profile shows {risk_pct:.1f}% risk, "
                    f"and similar drivers filed claims just {claim_rate:.1f}% of the time "
                    f"({'well below' if claim_rate < baseline_rate else 'near'} the {baseline_rate}% average). "
                    f"You can expect approval within 24-48 hours with no extra conditions or fees."
                ),
                
                'MONITOR': (
                    f"We can definitely insure your {v_age:.1f}-year-old car with {safety_desc}, "
                    f"though we'll need to add a 15-20% premium adjustment for the first year. "
                    f"Your profile shows moderate risk at {risk_pct:.1f}%—when we looked at {total} drivers "
                    f"similar to you (age {age}, {sub}-month subscription, {location_desc}), {claims} filed claims "
                    f"({claim_rate:.1f}% rate). The good news? After you build a claims-free history over the next "
                    f"12 months, we'll review your rates and likely reduce that premium. "
                    f"Your {'strong' if has_esc else 'adequate'} safety features help, and we're confident this will work out well."
                ),
                
                'CONDITIONAL': (
                    f"Your {v_age:.1f}-year-old vehicle with {safety_desc} can be insured, "
                    f"but we need to be upfront about the conditions. As a {age}-year-old driver with a "
                    f"{sub}-month subscription in a {location_desc}, your profile shows elevated risk at {risk_pct:.1f}%—"
                    f"similar drivers in our database filed claims {claim_rate:.1f}% of the time, which is "
                    f"{multiplier:.1f}x higher than typical. We can offer coverage with a 30-40% premium increase "
                    f"and a $1,500 deductible. Consider this a bridge policy: drive safely for 12 months, "
                    f"and we'll reassess for better terms. "
                    f"{'Adding ESC or other safety features would significantly improve your profile.' if not has_esc else ''}"
                ),
                
                'REJECT': (
                    f"Unfortunately, we can't insure your {v_age:.1f}-year-old vehicle with {safety_desc} "
                    f"under our standard policies right now. As a {age}-year-old driver with a {sub}-month subscription "
                    f"in a {location_desc}, your profile shows {risk_pct:.1f}% risk. When we analyzed {total} similar cases, "
                    f"{claims} resulted in claims—that's {claim_rate:.1f}%, or {multiplier:.1f}x higher than what our "
                    f"pricing can support. This isn't personal—it's about matching risk with appropriate coverage. "
                    f"We recommend: (1) Consider {'a newer vehicle with better safety features' if v_age > 5 else 'adding ESC and other safety features'}, "
                    f"(2) Try a specialized high-risk insurer, or (3) Build insurance history elsewhere and reapply in 12-18 months."
                )
            }
        else:
            # UNDERWRITER MODE TEMPLATES - COMPANY VOICE
            templates = {
                'APPROVE': (
                    f"After analyzing this application against our database of 58,000+ policies, we're pleased to "
                    f"recommend standard approval for coverage. The profile shows a {age}-year-old driver with a "
                    f"{v_age:.1f}-year-old vehicle ({safety_desc}) on a {sub}-month subscription in a {location_desc}, "
                    f"demonstrating strong low-risk characteristics with a calculated risk score of {risk_pct:.1f}%. "
                    f"Our analysis of {total} similar cases revealed only {claims} claims ({claim_rate:.1f}% claim rate), "
                    f"{'significantly below' if claim_rate < baseline_rate else 'aligned with'} our industry baseline of {baseline_rate}%. "
                    f"This application qualifies for standard rates with no additional conditions, and the policy can be "
                    f"processed within 24-48 hours."
                ),
                
                'MONITOR': (
                    f"We appreciate this application and can offer coverage with some adjustments to ensure mutual protection. "
                    f"The profile presents a {age}-year-old driver with a {v_age:.1f}-year-old vehicle ({safety_desc}) "
                    f"on a {sub}-month plan in a {location_desc}, showing moderate risk characteristics reflected in the "
                    f"{risk_pct:.1f}% risk score. Comparing against {total} similar policies in our database, we found "
                    f"{claims} resulted in claims ({claim_rate:.1f}% rate), {'slightly above' if multiplier > 1 else 'near'} "
                    f"our {baseline_rate}% baseline. Based on this analysis, we recommend approval with a 15-20% premium "
                    f"adjustment and quarterly monitoring during the first year; as a claims-free history builds, we can "
                    f"revisit these terms for improved rates."
                ),
                
                'CONDITIONAL': (
                    f"We value this application and want to find a path forward, though the profile requires certain "
                    f"conditions for approval. As a {age}-year-old driver with a {v_age:.1f}-year-old vehicle "
                    f"({safety_desc}) on a {sub}-month subscription in a {location_desc}, the risk profile scores at "
                    f"{risk_pct:.1f}%, placing it in our higher-risk category based on actuarial data. Our analysis of "
                    f"{total} similar cases revealed {claims} claims—a {claim_rate:.1f}% rate notably "
                    f"{'higher' if multiplier > 1.5 else 'elevated'} compared to our {baseline_rate}% industry average—"
                    f"necessitating additional safeguards. We propose approval with a 30-40% premium loading, higher "
                    f"deductible ($1,500), and enhanced documentation requirements; after 12 months of claims-free driving, "
                    f"we'll review the policy for improved rates."
                ),
                
                'REJECT': (
                    f"We appreciate the interest in our coverage, though after thorough analysis, our underwriting team "
                    f"must decline this application at this time. The profile of a {age}-year-old driver with a "
                    f"{v_age:.1f}-year-old vehicle ({safety_desc}) on a {sub}-month plan in a {location_desc} triggers "
                    f"multiple high-risk indicators in our system, resulting in a {risk_pct:.1f}% risk score. Our analysis "
                    f"of {total} similar policies found {claims} filed claims within their policy period—a {claim_rate:.1f}% "
                    f"rate that's {multiplier:.1f}x our industry standard—exceeding the exposure levels our risk models can "
                    f"responsibly accommodate. We encourage considering building insurance history with a specialized carrier, "
                    f"exploring vehicles with advanced safety features{'(particularly ESC)' if not has_esc else ''}, or "
                    f"revisiting the application after establishing a longer driving record."
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