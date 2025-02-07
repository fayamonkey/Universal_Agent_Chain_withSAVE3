import streamlit as st
import openai
from typing import Dict, Any, List
import json
from datetime import datetime
import time
import os

# Corporate Identity
AUTHOR = "Dirk Wonhoefer"
COMPANY = "AI Engineering"
EMAIL = "dirk.wonhoefer@ai-engineering.ai"
WEBSITE = "ai-engineering.ai"

class BaseAgent:
    def __init__(self, model_name: str = "gpt-4"):
        self.model_name = model_name
        
    def execute(self, prompt: str, system_prompt: str, previous_results: Dict[str, str] = None) -> str:
        try:
            client = openai.OpenAI(api_key=st.session_state.api_key)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            return completion.choices[0].message.content
        except Exception as e:
            st.error(f"Error in agent execution: {str(e)}")
            return ""

class AgentChain:
    def __init__(self, system_prompts: List[str], model_name: str = "gpt-4"):
        self.num_agents = len(system_prompts)
        self.agents = [BaseAgent(model_name) for _ in range(self.num_agents)]
        self.system_prompts = system_prompts
        
    def process_input(self, initial_input: str) -> Dict[str, str]:
        results = {"initial_input": initial_input}
        
        for i in range(self.num_agents):
            agent_num = i + 1
            st.write(f"🔄 Agent {agent_num}: Running...")
            
            # Prepare input based on previous results
            if i == 0:
                # First agent gets initial input
                agent_input = initial_input
            else:
                # Other agents get formatted previous results
                agent_input = "Previous work:\n\n"
                for j in range(i):
                    agent_input += f"AGENT {j+1} OUTPUT:\n{results[f'agent{j+1}_output']}\n\n"
            
            # Execute agent
            results[f"agent{agent_num}_output"] = self.agents[i].execute(
                agent_input,
                self.system_prompts[i]
            )
            st.write(f"✅ Agent {agent_num}: Complete")
        
        return results

def create_markdown(results: Dict[str, str]) -> str:
    markdown = f"""# Agent Chain Results
Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Initial Input
{results['initial_input']}
"""
    
    # Add each agent's output
    for i in range(1, len(results)-1):  # -1 for initial_input
        markdown += f"\n## Agent {i} Output\n{results[f'agent{i}_output']}\n"
    
    return markdown

def save_configuration(num_agents: int, system_prompts: List[str], file_name: str) -> tuple[bool, str, str]:
    """Save the current configuration as a downloadable file."""
    try:
        # Prepare configuration data
        config = {
            "num_agents": num_agents,
            "system_prompts": system_prompts,
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Convert to JSON string
        json_str = json.dumps(config, indent=2, ensure_ascii=False)
        
        # Ensure the filename ends with .json
        if not file_name.endswith('.json'):
            file_name += '.json'
            
        # Return the JSON string for download
        return True, json_str, file_name
        
    except Exception as e:
        return False, str(e), ""

def load_configuration(file_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading configuration: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="Universal Agent Chain Builder",
        page_icon="🤖",
        layout="wide"
    )
    
    # Header with corporate identity
    st.title("🤖 Universal Agent Chain Builder")
    st.markdown(f"""
    Build your own chain of AI agents! Each agent will process the output of previous agents.
    First, choose how many agents you want in your chain, then configure each agent's behavior.
    
    ---
    **Created by [{AUTHOR}]({WEBSITE}) | [{COMPANY}]({WEBSITE})**  
    Contact: [{EMAIL}](mailto:{EMAIL})
    """)
    
    # API Key input in sidebar
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("OpenAI API Key", type="password")
        if api_key:
            st.session_state.api_key = api_key
        
        # Number of agents selector
        num_agents = st.number_input(
            "Number of Agents in Chain",
            min_value=2,
            max_value=99,
            value=5,
            help="Choose how many agents you want in your chain (2-99)"
        )
        
        # Save/Load Configuration
        st.markdown("---")
        st.header("Save/Load Configuration")
        
        # Save current configuration
        save_col1, save_col2 = st.columns([3, 1])
        with save_col1:
            save_name = st.text_input("Configuration Name", 
                value="my_config",
                help="Enter a name for your configuration")
        
        with save_col2:
            if st.button("💾 Save Configuration"):
                if 'system_prompts' in st.session_state:
                    success, content, filename = save_configuration(num_agents, st.session_state.system_prompts, save_name)
                    if success:
                        st.download_button(
                            label="📥 Download Configuration",
                            data=content,
                            file_name=filename,
                            mime="application/json",
                            help="Click to download your configuration file"
                        )
                    else:
                        st.error(f"Error saving configuration: {content}")
        
        # Load configuration
        st.markdown("---")
        uploaded_file = st.file_uploader("Load Configuration", type=['json'],
            help="Upload a previously saved configuration file")
        
        if uploaded_file is not None:
            try:
                config = json.load(uploaded_file)
                if config:
                    num_agents = config['num_agents']
                    st.session_state.system_prompts = config['system_prompts']
                    st.success(f"Configuration loaded! (Saved on: {config['saved_at']})")
            except Exception as e:
                st.error(f"Error loading configuration: {str(e)}")
        
        st.markdown("---")
        st.markdown("""
        ### How it works
        1. Enter your OpenAI API key
        2. Choose number of agents
        3. Define system prompts for each agent
        4. Enter your initial input
        5. Click 'Run Agent Chain'
        6. Watch the progress
        7. Download results as Markdown
        
        You can save your configuration at any time and load it later!
        """)
    
    # Main content
    if 'api_key' not in st.session_state:
        st.warning("Please enter your OpenAI API key in the sidebar to continue.")
        return
    
    # System prompts input
    st.header("Agent System Prompts")
    
    # Create columns for prompts (2 columns)
    system_prompts = []
    for i in range(0, num_agents, 2):
        col1, col2 = st.columns(2)
        
        with col1:
            if i < num_agents:
                default_value = st.session_state.system_prompts[i] if 'system_prompts' in st.session_state and i < len(st.session_state.system_prompts) else ""
                prompt = st.text_area(
                    f"Agent {i+1} System Prompt", 
                    height=100,
                    value=default_value,
                    help=f"Agent {i+1} " + ("receives the initial input" if i == 0 else f"receives outputs from Agents 1-{i}")
                )
                system_prompts.append(prompt)
        
        with col2:
            if i+1 < num_agents:
                default_value = st.session_state.system_prompts[i+1] if 'system_prompts' in st.session_state and i+1 < len(st.session_state.system_prompts) else ""
                prompt = st.text_area(
                    f"Agent {i+2} System Prompt", 
                    height=100,
                    value=default_value,
                    help=f"Agent {i+2} receives outputs from Agents 1-{i+1}"
                )
                system_prompts.append(prompt)
    
    # Store current prompts in session state
    st.session_state.system_prompts = system_prompts
    
    # Initial input
    st.header("Initial Input")
    initial_input = st.text_area("Enter your input for the first agent", height=150)
    
    # Process button
    if st.button("Run Agent Chain", type="primary"):
        if not all(system_prompts) or not initial_input:
            st.error("Please fill in all system prompts and the initial input.")
            return
        
        try:
            # Create progress container
            progress_container = st.empty()
            with progress_container.container():
                st.write("🚀 Starting agent chain process...")
                
                # Initialize and run agent chain
                chain = AgentChain(system_prompts)
                results = chain.process_input(initial_input)
                
                # Create markdown output
                markdown_output = create_markdown(results)
                
                st.write("✨ Process complete!")
                
                # Display results in expandable sections
                st.header("Results")
                for i in range(1, len(system_prompts) + 1):
                    with st.expander(f"Agent {i} Output", expanded=(i==1)):
                        st.markdown(results[f"agent{i}_output"])
                
                # Download button
                st.download_button(
                    label="Download Results as Markdown",
                    data=markdown_output,
                    file_name=f"agent_chain_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 