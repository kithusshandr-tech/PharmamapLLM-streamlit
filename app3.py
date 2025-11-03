import streamlit as st
import os
import json
import uuid
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass, asdict
from openai import OpenAI
import requests
from dotenv import load_dotenv
import logging
import time
from pathlib import Path

load_dotenv()
logging.basicConfig(level=logging.INFO)


@dataclass
class AgentResponse:
    agent_name: str
    response: str
    confidence: float
    reasoning: str
    timestamp: str
    model_used: str
    processing_time: float


@dataclass
class ConversationLog:
    session_id: str
    timestamp: str
    user_input: str
    drug1: str
    drug2: str
    smiles1: str
    smiles2: str
    agent_responses: List[AgentResponse]
    final_answer: str
    total_processing_time: float
    user_settings: Dict

class BaseAgent:
    def __init__(self, name: str, model_type: str):
        self.name = name
        self.model_type = model_type
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)
        return logger
    
    def process(self, input_data: Dict) -> AgentResponse:
        raise NotImplementedError

class HuggingFaceAgent(BaseAgent):
    def __init__(self, name: str, endpoint_url: str | None = None):
        super().__init__(name, "hf_endpoint")
        self.endpoint_url = endpoint_url or os.getenv("HUGGINGFACE_ENDPOINT_URL")
        if not self.endpoint_url:
            raise ValueError("HUGGINGFACE_ENDPOINT_URL is not set.")
        self.hf_token = os.getenv("HUGGINGFACE_API_KEY")  # optional if endpoint is public

    def _call_huggingface_api(self, prompt: str, max_retries: int = 3) -> str:
        """Call your HF Inference Endpoint with retries."""
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 200,
                "temperature": 0.7,
                "top_p": 0.9,
                "return_full_text": False,
                "do_sample": True,
            }
        }

        for attempt in range(max_retries):
            try:
                resp = requests.post(self.endpoint_url, headers=headers, json=payload, timeout=60)
                # Endpoint can return 503 while spinning up
                if resp.status_code == 503:
                    wait_s = min(20 * (attempt + 1), 60)
                    st.info(f"⏳ Endpoint warming up… retrying in {wait_s}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_s)
                    continue

                resp.raise_for_status()
                data = resp.json()

                # Endpoint may return [{"generated_text": "..."}] or {"generated_text": "..."}
                if isinstance(data, list) and data and isinstance(data[0], dict):
                    return data[0].get("generated_text", "No response generated")
                if isinstance(data, dict) and "generated_text" in data:
                    return data["generated_text"]
                if isinstance(data, dict) and "error" in data:
                    return f"API Error: {data['error']}"
                return str(data)

            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    return f"API Error after {max_retries} attempts: {e}"
                time.sleep(5 * (attempt + 1))

        return "Failed to get response from HF Endpoint"


# class HuggingFaceAgent(BaseAgent):
#     def __init__(self, name: str, model_name: str):
#         super().__init__(name, "huggingface_api")
#         self.model_name = model_name
#         self.hf_token = os.getenv('HUGGINGFACE_API_KEY')
#         self.hf_api_url = f"https://api-inference.huggingface.co/models/{model_name}"

#     def _call_huggingface_api(self, prompt: str, max_retries: int = 3) -> str:
#         """Call HuggingFace Inference API with retries"""
#         headers = {"Authorization": f"Bearer {self.hf_token}"}
#         payload = {
#             "inputs": prompt,
#             "parameters": {
#                 "max_new_tokens": 200,
#                 "temperature": 0.7,
#                 "top_p": 0.9,
#                 "return_full_text": False,
#                 "do_sample": True
#             }
#         }
        
#         for attempt in range(max_retries):
#             try:
#                 response = requests.post(
#                     self.hf_api_url, 
#                     headers=headers, 
#                     json=payload,
#                     timeout=30
#                 )
                
#                 if response.status_code == 503:
#                     # Model is loading, wait and retry
#                     wait_time = min(20 * (attempt + 1), 60)
#                     st.info(f"⏳ Model is loading, waiting {wait_time}s... (attempt {attempt + 1})")
#                     time.sleep(wait_time)
#                     continue
                
#                 response.raise_for_status()
#                 result = response.json()
                
#                 if isinstance(result, list) and len(result) > 0:
#                     return result[0].get('generated_text', 'No response generated')
#                 elif isinstance(result, dict) and 'error' in result:
#                     return f"API Error: {result['error']}"
#                 else:
#                     return str(result)
                    
#             except requests.exceptions.RequestException as e:
#                 if attempt == max_retries - 1:
#                     return f"API Error after {max_retries} attempts: {str(e)}"
#                 time.sleep(5 * (attempt + 1))
                
#         return "Failed to get response from API"

class PredictorAgent(HuggingFaceAgent):
    def __init__(self):
        super().__init__("🔮 Predictor Agent")

    def process(self, input_data: Dict) -> AgentResponse:
        start_time = time.time()

        system_prompt = "You are an expert in drug-drug interaction prediction. Analyze the given drug pair and respond with either 'interaction' or 'no interaction'."

        user_prompt = f"""Drug1: {input_data['drug1']}
SMILES for drug1: {input_data.get('smiles1', 'Not provided')}
Drug2: {input_data['drug2']}
SMILES for drug2: {input_data.get('smiles2', 'Not provided')}

INSTRUCTIONS FOR ANALYSIS:
1. Analyze the molecular structures using the provided SMILES strings
2. Consider the target genes and their pathways for potential interactions
3. Evaluate pharmacokinetic factors (absorption, distribution, metabolism, excretion)
4. Assess pharmacodynamic interactions (synergistic, antagonistic, additive effects)
5. Consider shared metabolic pathways (CYP enzymes, transporters)

Based on your analysis, does this drug pair have a drug-drug interaction?

CLASSIFICATION:"""

        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        response = self._call_huggingface_api(full_prompt)

        # Extract classification from response
        classification = "interaction" if "interaction" in response.lower() and "no interaction" not in response.lower() else "no interaction"
        confidence = 0.8  

        processing_time = time.time() - start_time

        return AgentResponse(
            agent_name=self.name,
            response=classification,
            confidence=confidence,
            reasoning=response,
            timestamp=datetime.now().isoformat(),
            model_used="Pharmamapllm/MedLLaMA3-DDI-QLoRA-V1 (API)",
            processing_time=processing_time
        )

class ValidatorAgent1(HuggingFaceAgent):
    def __init__(self):
        super().__init__("✅ Validator Agent 1")

    def process(self, input_data: Dict) -> AgentResponse:
        start_time = time.time()
        predictor_result = input_data['predictor_result']

        system_prompt = "You are a medical validator specializing in drug-drug interactions. Review the DDI prediction and provide validation."
        
        user_prompt = f"""Please validate this drug interaction prediction:

Drug1: {input_data['drug1']}
Drug2: {input_data['drug2']}
Initial Prediction: {predictor_result.response}
Initial Reasoning: {predictor_result.reasoning}

Please provide your validation assessment. Consider:
1. Medical plausibility of the interaction
2. Known pharmacological mechanisms
3. Clinical significance
4. Potential for adverse effects

Respond with 'VALIDATED' if you agree with the prediction or 'NEEDS_REVIEW' if you have concerns, followed by your reasoning."""
        
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        response = self._call_huggingface_api(full_prompt)
        
        validation = "VALIDATED" if "validated" in response.lower() else "NEEDS_REVIEW"
        confidence = 0.75
        
        processing_time = time.time() - start_time
        
        return AgentResponse(
            agent_name=self.name,
            response=validation,
            confidence=confidence,
            reasoning=response,
            timestamp=datetime.now().isoformat(),
            model_used="Pharmamapllm/MedLLaMA3-DDI-QLoRA-V1 (API)",
            processing_time=processing_time
        )

class OpenAIAgent(BaseAgent):
    def __init__(self, name: str, model_name: str):
        super().__init__(name, "openai")
        self.model_name = model_name
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def _call_openai(self, prompt: str) -> str:
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"

class RiskAgent(OpenAIAgent):
    def __init__(self):
        super().__init__("⚠️ Risk Assessment Agent", "gpt-4o")
    
    def process(self, input_data: Dict) -> AgentResponse:
        start_time = time.time()
        predictor_result = input_data['predictor_result']
        
        prompt = f"""As a clinical pharmacologist, assess the risk level of this drug interaction:

Drug1: {input_data['drug1']}
Drug2: {input_data['drug2']}
Predicted Interaction: {predictor_result.response}

Please provide:
1. Risk level (LOW/MODERATE/HIGH/SEVERE)
2. Potential adverse effects
3. Clinical management recommendations
4. Monitoring requirements

Format your response clearly with each section."""
        
        response = self._call_openai(prompt)
        
        # Extract risk level from response
        risk_level = "MODERATE"  
        if "severe" in response.lower():
            risk_level = "SEVERE"
        elif "high" in response.lower():
            risk_level = "HIGH"
        elif "low" in response.lower():
            risk_level = "LOW"
        
        confidence = 0.7
        processing_time = time.time() - start_time
        
        return AgentResponse(
            agent_name=self.name,
            response=risk_level,
            confidence=confidence,
            reasoning=response,
            timestamp=datetime.now().isoformat(),
            model_used="gpt-4o",
            processing_time=processing_time
        )

class ValidatorAgent2(OpenAIAgent):
    def __init__(self):
        super().__init__("🔍 Validator Agent 2", "gpt-4o")
    
    def process(self, input_data: Dict) -> AgentResponse:
        start_time = time.time()
        predictor_result = input_data['predictor_result']
        risk_result = input_data['risk_result']
        validator1_result = input_data['validator1_result']
        
        prompt = f"""As an independent medical reviewer, please validate this drug interaction analysis:

Drug Pair: {input_data['drug1']} + {input_data['drug2']}

Initial Prediction: {predictor_result.response}
Risk Assessment: {risk_result.response}
First Validation: {validator1_result.response}

Please provide your independent validation considering:
1. Consistency across assessments
2. Medical literature support
3. Clinical evidence
4. Overall confidence in the prediction

Respond with: FINAL_APPROVED, NEEDS_REVISION, or CONFLICTING_EVIDENCE
Include your detailed reasoning."""
        
        response = self._call_openai(prompt)
        
        if "final_approved" in response.lower():
            validation = "FINAL_APPROVED"
        elif "conflicting" in response.lower():
            validation = "CONFLICTING_EVIDENCE"
        else:
            validation = "NEEDS_REVISION"
        
        confidence = 0.8
        processing_time = time.time() - start_time
        
        return AgentResponse(
            agent_name=self.name,
            response=validation,
            confidence=confidence,
            reasoning=response,
            timestamp=datetime.now().isoformat(),
            model_used="gpt-4o",
            processing_time=processing_time
        )

class OrchestratorAgent(OpenAIAgent):
    def __init__(self):
        super().__init__("🎯 Orchestrator Agent", "gpt-4o")
    
    def process(self, input_data: Dict) -> AgentResponse:
        start_time = time.time()
        all_results = input_data['all_results']
        
        prompt = f"""As the chief medical AI orchestrator, synthesize these drug interaction analyses into a final decision:

Drug Pair: {input_data['drug1']} + {input_data['drug2']}

AGENT ANALYSES:
"""
        
        for result in all_results:
            prompt += f"\n{result.agent_name}: {result.response}"
            prompt += f"\nConfidence: {result.confidence}"
            prompt += f"\nReasoning: {result.reasoning[:200]}...\n"
        
        prompt += """
Please provide:
1. FINAL DECISION: interaction/no interaction
2. CONFIDENCE LEVEL: 0.0-1.0
3. CLINICAL RECOMMENDATION
4. RATIONALE: Explain how you weighed different agent inputs

Be decisive but acknowledge any uncertainty."""
        
        response = self._call_openai(prompt)
        
        final_decision = "interaction" if "interaction" in response.lower() and "no interaction" not in response.lower() else "no interaction"
        
        confidence = 0.9 if "high confidence" in response.lower() else 0.7
        
        processing_time = time.time() - start_time
        
        return AgentResponse(
            agent_name=self.name,
            response=final_decision,
            confidence=confidence,
            reasoning=response,
            timestamp=datetime.now().isoformat(),
            model_used="gpt-4o",
            processing_time=processing_time
        )

def save_conversation_log(log: ConversationLog):
    """Save conversation log to JSON file"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    filename = log_dir / f"ddi_session_{log.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    log_dict = asdict(log)
    
    with open(filename, 'w') as f:
        json.dump(log_dict, f, indent=2, default=str)
    
    return filename

def display_agent_response(response: AgentResponse, idx: int):
    """Display agent response in Streamlit (clean, spacious)"""
    with st.container():
        st.markdown(
            f"""
            <div style="
                padding:16px;
                margin-bottom:14px;
                border:1px solid rgba(0,0,0,0.08);
                border-radius:12px;
                background: #fff;
                box-shadow: 0 1px 3px rgba(0,0,0,0.06);
            ">
              <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;">
                <div style="font-weight:600;font-size:1.05rem;">
                    {response.agent_name} <span style="opacity:.7;">— {response.response}</span>
                </div>
                <div style="display:flex;gap:12px;opacity:.8;">
                    <div>Confidence: <b>{response.confidence:.2f}</b></div>
                    <div>Time: <b>{response.processing_time:.2f}s</b></div>
                    <div>Model: <code>{response.model_used}</code></div>
                </div>
              </div>
              <div style="margin-top:10px;">
                <div style="font-size:.9rem;opacity:.75;">Reasoning</div>
                <pre style="
                    margin-top:6px;
                    padding:12px;
                    background:#0f172a;
                    color:#e5e7eb;
                    border-radius:8px;
                    max-height:320px;
                    overflow:auto;
                    white-space:pre-wrap;
                    line-height:1.4;
                    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;
                    font-size:.92rem;
                ">{response.reasoning}</pre>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )


def check_api_status():
    """Check if APIs are accessible"""
    status = {}
    
    # Check OpenAI
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        client.models.list()
        status['openai'] = "✅ Connected"
    except:
        status['openai'] = "❌ Error"
    
    # Check HuggingFace
    try:
        headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}
        response = requests.get("https://cl1yyn4n4oicz0d9.us-east-1.aws.endpoints.huggingface.cloud", 
                              headers=headers, timeout=10)
        if response.status_code == 200:
            status['huggingface'] = "✅ Connected"
        else:
            status['huggingface'] = f"⚠️ Status {response.status_code}"
    except:
        status['huggingface'] = "❌ Error"
    
    return status

def main():
    st.set_page_config(
        page_title="DDI Multi-Agent",
        page_icon="💊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # --- Minimal CSS polish: wider content, cleaner fonts, nicer headers
    st.markdown("""
    <style>
      .block-container {max-width: 1200px; padding-top: 1rem; padding-bottom: 2rem;}
      h1, h2, h3 {letter-spacing: .2px;}
      /* Hide Streamlit default menu/footer */
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

    st.title("💊 Multi-Agent Drug-Drug Interaction (DDI) Analysis")

    # ----- Sidebar: kept only what’s useful
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.caption("Set keys via environment variables.")

        # API quick checks (optional)
        if st.button("🔄 Check API Status"):
            with st.spinner("Checking APIs..."):
                api_status = check_api_status()
                st.text(f"OpenAI: {api_status['openai']}")
                st.text(f"HuggingFace: {api_status['huggingface']}")

        st.divider()
        st.subheader("🔑 API Keys")
        openai_key = "✅" if os.getenv('OPENAI_API_KEY') else "❌"
        hf_key     = "✅" if os.getenv('HUGGINGFACE_API_KEY') else "❌"
        st.text(f"OpenAI: {openai_key}")
        st.text(f"Hugging Face: {hf_key}")

        st.divider()
        st.subheader("📋 Recent Analyses")
        log_dir = Path("logs")
        if log_dir.exists():
            log_files = list(log_dir.glob("*.json"))
            log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            for log_file in log_files[:5]:
                try:
                    with open(log_file) as f:
                        content = f.read().strip()
                        if not content:
                            continue
                        log_data = json.loads(content)
                    st.markdown(
                        f"- **{log_data['drug1']} + {log_data['drug2']}** — "
                        f"{log_data['final_answer']} · {log_data['total_processing_time']:.2f}s"
                    )
                except Exception:
                    pass

    # ----- Main area
    col1, col2 = st.columns([2, 1], vertical_alignment="top")

    with col1:
        st.subheader("🔬 New Analysis")
        with st.form("ddi_form", clear_on_submit=False):
            c1, c2 = st.columns(2)
            with c1:
                drug1 = st.text_input("Drug 1", placeholder="e.g., Warfarin")
                smiles1 = st.text_area("SMILES 1 (optional)", height=80)
            with c2:
                drug2 = st.text_input("Drug 2", placeholder="e.g., Aspirin")
                smiles2 = st.text_area("SMILES 2 (optional)", height=80)

            submitted = st.form_submit_button("🚀 Analyze", type="primary", use_container_width=True)

        if submitted and drug1 and drug2:
            session_id = str(uuid.uuid4())
            start_time = time.time()

            try:
                # Initialize lightweight agents (API-based)
                predictor_agent   = PredictorAgent()
                validator_agent1  = ValidatorAgent1()
                risk_agent        = RiskAgent()
                validator_agent2  = ValidatorAgent2()
                orchestrator_agent= OrchestratorAgent()

                progress = st.progress(0)
                status   = st.empty()

                input_data = {
                    'drug1': drug1, 'drug2': drug2,
                    'smiles1': smiles1, 'smiles2': smiles2
                }
                agent_responses = []

                status.text("🔮 Predictor Agent…")
                progress.progress(20)
                predictor_result = predictor_agent.process(input_data)
                agent_responses.append(predictor_result)
                input_data['predictor_result'] = predictor_result

                status.text("✅ Validator Agent 1…")
                progress.progress(40)
                validator1_result = validator_agent1.process(input_data)
                agent_responses.append(validator1_result)
                input_data['validator1_result'] = validator1_result

                status.text("⚠️ Risk Assessment…")
                progress.progress(60)
                risk_result = risk_agent.process(input_data)
                agent_responses.append(risk_result)
                input_data['risk_result'] = risk_result

                status.text("🔍 Validator Agent 2…")
                progress.progress(80)
                validator2_result = validator_agent2.process(input_data)
                agent_responses.append(validator2_result)
                input_data['validator2_result'] = validator2_result

                status.text("🎯 Orchestrator…")
                progress.progress(95)
                input_data['all_results'] = agent_responses
                final_result = orchestrator_agent.process(input_data)
                agent_responses.append(final_result)

                total_time = time.time() - start_time
                progress.progress(100)
                status.text("✅ Complete")

                # Final decision card
                st.markdown("---")
                decision_color = "🔴" if final_result.response == "interaction" else "🟢"
                st.markdown(
                    f"""
                    <div style="
                        padding: 18px; border-radius: 12px;
                        background: {'#ffefef' if final_result.response == 'interaction' else '#eef8ef'};
                        border: 1px solid {'#ffd3d3' if final_result.response == 'interaction' else '#cfe9cf'};
                    ">
                      <div style="font-size:1.15rem; font-weight:700; margin-bottom:6px;">
                        {decision_color} Final Decision: {final_result.response.upper()}
                      </div>
                      <div>Confidence: <b>{final_result.confidence:.2f}</b></div>
                      <div>Total time: <b>{total_time:.2f}s</b></div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Agent tabs
                st.subheader("🤖 Agent Analyses")
                tabs = st.tabs([r.agent_name for r in agent_responses])
                for tab, resp in zip(tabs, agent_responses):
                    with tab:
                        display_agent_response(resp, idx=hash(resp.agent_name) % 10_000)

                # Save log
                conversation_log = ConversationLog(
                    session_id=session_id,
                    timestamp=datetime.now().isoformat(),
                    user_input=f"DDI Query: {drug1} + {drug2}",
                    drug1=drug1, drug2=drug2,
                    smiles1=smiles1, smiles2=smiles2,
                    agent_responses=agent_responses,
                    final_answer=final_result.response,
                    total_processing_time=total_time,
                    user_settings={
                        'predictor_agent':  {'type': 'huggingface_api', 'model': 'Pharmamapllm/MedLLaMA3-DDI-QLoRA-V1'},
                        'validator1_agent': {'type': 'huggingface_api', 'model': 'Pharmamapllm/MedLLaMA3-DDI-QLoRA-V1'},
                        'risk_agent':       {'type': 'openai', 'model': 'gpt-4o'},
                        'validator2_agent': {'type': 'openai', 'model': 'gpt-4o'},
                        'orchestrator_agent': {'type': 'openai', 'model': 'gpt-4o'}
                    }
                )
                log_file = save_conversation_log(conversation_log)
                with open(log_file, "rb") as f:
                    st.download_button(
                        "💾 Download session log",
                        f,
                        file_name=os.path.basename(log_file),
                        mime="application/json",
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"❌ Error during analysis: {e}")
                st.error("Please check your API keys and Internet connection.")

    with col2:
        st.subheader("📈 System Status")
        st.metric("Model Loading", "Not required")
        st.metric("Mode", "API-only")

        st.caption("Tip: Keep logs for auditability and reproducibility.")


if __name__ == "__main__":
    main()
