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
    def __init__(self, name: str):
        super().__init__(name, "hf_endpoint")

    def _call_huggingface_api(self, prompt: str) -> str:
        """Call the finetuned HuggingFace endpoint exactly as in zz.py."""
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        url = "https://cl1yyn4n4oicz0d9.us-east-1.aws.endpoints.huggingface.cloud"
        payload = {
            "inputs": prompt,
            "parameters": {}
        }
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            # The endpoint returns a list of dicts or a dict
            if isinstance(data, list) and data and isinstance(data[0], dict):
                return data[0].get("generated_text", str(data[0]))
            if isinstance(data, dict) and "generated_text" in data:
                return data["generated_text"]
            return str(data)
        except Exception as e:
            return f"API Error: {e}"


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
        super().__init__("🔮 Predictor Agent", "Pharmamapllm/MedLLaMA3-DDI-QLoRA-V1")

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
        super().__init__("✅ Validator Agent 1", "Pharmamapllm/MedLLaMA3-DDI-QLoRA-V1")

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
    """Display agent response in Streamlit"""
    with st.expander(f"{response.agent_name} - {response.response}", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Confidence", f"{response.confidence:.2f}")
            st.metric("Processing Time", f"{response.processing_time:.2f}s")
        
        with col2:
            st.text(f"Model: {response.model_used}")
            st.text(f"Timestamp: {response.timestamp}")
        
        # st.text_area("Reasoning", response.reasoning, height=100, disabled=True)
        st.text_area(
            "Reasoning",
            response.reasoning,
            height=100,
            disabled=True,
            key=f"reasoning_{idx}"
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
        page_title="Multi-Agent DDI Prediction System",
        page_icon="💊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("💊 Multi-Agent Drug-Drug Interaction Prediction System")
    st.markdown("**🚀 API-Based Version - No Local Model Loading Required**")
    st.markdown("---")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("Model Information")
        st.info("🤖 **Predictor & Validator 1**: MedLLaMA3-DDI via HF API")
        st.info("🧠 **Risk, Validator 2 & Orchestrator**: GPT-4o")
        
        st.subheader("📊 API Status")
        if st.button("🔄 Check API Status"):
            with st.spinner("Checking APIs..."):
                api_status = check_api_status()
                st.text(f"OpenAI: {api_status['openai']}")
                st.text(f"HuggingFace: {api_status['huggingface']}")
        
        st.subheader("💡 Benefits")
        st.markdown("- 🚀 **Fast startup** (no model loading)")
        st.markdown("- 💾 **Low memory usage**")
        st.markdown("- 🔄 **Always up-to-date models**")
        st.markdown("- ⚡ **Scalable processing**")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔬 Drug Interaction Analysis")
        
        with st.form("ddi_form"):
            col_drug1, col_drug2 = st.columns(2)
            
            with col_drug1:
                drug1 = st.text_input("Drug 1 Name", placeholder="e.g., Warfarin")
                smiles1 = st.text_area("SMILES 1 (optional)", placeholder="Chemical structure", height=80)
            
            with col_drug2:
                drug2 = st.text_input("Drug 2 Name", placeholder="e.g., Aspirin")
                smiles2 = st.text_area("SMILES 2 (optional)", placeholder="Chemical structure", height=80)
            
            submitted = st.form_submit_button("🚀 Analyze Interaction", type="primary")
        
        if submitted and drug1 and drug2:
            session_id = str(uuid.uuid4())
            start_time = time.time()

            try:
                st.info("🚀 **API-Based Processing**: Using cloud APIs for analysis...")
                
                # Initialize agents (lightweight, no model loading)
                predictor_agent = PredictorAgent()
                validator_agent1 = ValidatorAgent1()
                risk_agent = RiskAgent()
                validator_agent2 = ValidatorAgent2()
                orchestrator_agent = OrchestratorAgent()

                st.success("✅ All agents initialized!")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                input_data = {
                    'drug1': drug1,
                    'drug2': drug2,
                    'smiles1': smiles1,
                    'smiles2': smiles2
                }
                
                agent_responses = []
                
                # Predictor Agent
                status_text.text("🔮 Running Predictor Agent (MedLLaMA3 API)...")
                progress_bar.progress(20)
                predictor_result = predictor_agent.process(input_data)
                agent_responses.append(predictor_result)
                input_data['predictor_result'] = predictor_result
                
                # Validator Agent 1
                status_text.text("✅ Running Validator Agent 1 (MedLLaMA3 API)...")
                progress_bar.progress(40)
                validator1_result = validator_agent1.process(input_data)
                agent_responses.append(validator1_result)
                input_data['validator1_result'] = validator1_result
                
                # Risk Agent
                status_text.text("⚠️ Running Risk Assessment Agent (GPT-4o)...")
                progress_bar.progress(60)
                risk_result = risk_agent.process(input_data)
                agent_responses.append(risk_result)
                input_data['risk_result'] = risk_result
                
                # Validator Agent 2
                status_text.text("🔍 Running Validator Agent 2 (GPT-4o)...")
                progress_bar.progress(80)
                validator2_result = validator_agent2.process(input_data)
                agent_responses.append(validator2_result)
                input_data['validator2_result'] = validator2_result
                
                # Orchestrator Agent
                status_text.text("🎯 Running Orchestrator Agent (GPT-4o)...")
                progress_bar.progress(90)
                input_data['all_results'] = agent_responses
                final_result = orchestrator_agent.process(input_data)
                agent_responses.append(final_result)
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis Complete!")
                
                total_time = time.time() - start_time
                
                st.markdown("---")
                st.header("📊 Analysis Results")
                
                decision_color = "🔴" if final_result.response == "interaction" else "🟢"
                st.markdown(f"""
                    <div style="
                        padding: 20px;
                        border-radius: 10px;
                        background-color: {'#ffebee' if final_result.response == 'interaction' else '#e8f5e8'};
                        border-left: 5px solid {'#f44336' if final_result.response == 'interaction' else '#4caf50'};
                        color: black;">
                        <h2>{decision_color} Final Decision: {final_result.response.upper()}</h2>
                        <p><strong>Confidence:</strong> {final_result.confidence:.2f}</p>
                        <p><strong>Processing Time:</strong> {total_time:.2f} seconds</p>
                    </div>
                    """, unsafe_allow_html=True)

                
                st.subheader("🤖 Agent Analysis Details")
                for idx, response in enumerate(agent_responses):
                    display_agent_response(response, idx)

                
                conversation_log = ConversationLog(
                    session_id=session_id,
                    timestamp=datetime.now().isoformat(),
                    user_input=f"DDI Query: {drug1} + {drug2}",
                    drug1=drug1,
                    drug2=drug2,
                    smiles1=smiles1,
                    smiles2=smiles2,
                    agent_responses=agent_responses,
                    final_answer=final_result.response,
                    total_processing_time=total_time,
                    user_settings={
                        'predictor_agent': {'type': 'huggingface_api', 'model': 'Pharmamapllm/MedLLaMA3-DDI-QLoRA-V1'},
                        'validator1_agent': {'type': 'huggingface_api', 'model': 'Pharmamapllm/MedLLaMA3-DDI-QLoRA-V1'},
                        'risk_agent': {'type': 'openai', 'model': 'gpt-4o'},
                        'validator2_agent': {'type': 'openai', 'model': 'gpt-4o'},
                        'orchestrator_agent': {'type': 'openai', 'model': 'gpt-4o'}
                    }
                )
                
                log_file = save_conversation_log(conversation_log)
                st.success(f"💾 Analysis saved to: {log_file}")
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {e}")
                st.error("Please check your API keys and internet connection.")
    
    with col2:
        st.header("📈 System Status")
        
        st.metric("Memory Usage", "Low ✅")
        st.metric("Model Loading", "None Required ✅")
        
        st.subheader("🔑 API Keys Status")
        openai_key = "✅" if os.getenv('OPENAI_API_KEY') else "❌"
        hf_key = "✅" if os.getenv('HUGGINGFACE_API_KEY') else "❌"
        
        st.text(f"OpenAI: {openai_key}")
        st.text(f"Hugging Face: {hf_key}")

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

                    with st.expander(f"{log_data['drug1']} + {log_data['drug2']}", expanded=False):
                        st.text(f"Result: {log_data['final_answer']}")
                        st.text(f"Time: {log_data['total_processing_time']:.2f}s")
                        st.text(f"Session: {log_data['session_id'][:8]}")

                except Exception as e:
                    st.warning(f"⚠️ Skipping log file {log_file.name}: {e}")

        st.subheader("ℹ️ About This Version")
        st.info("""
        This version uses **HuggingFace Inference API** instead of loading models locally:
        
        ✅ **No model downloads**  
        ✅ **Minimal memory usage**  
        ✅ **Fast startup**  
        ✅ **Always latest models**  
        
        Perfect for laptops and development!
        """)

if __name__ == "__main__":
    main()