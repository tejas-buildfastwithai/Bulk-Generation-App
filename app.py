import streamlit as st
import json
import os
import tempfile
import csv
import io
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Import necessary libraries
from langchain_openai import ChatOpenAI
from educhain import Educhain, LLMConfig

# Set page config
st.set_page_config(
    page_title="Educhain - Bulk Question Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 42px !important;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 20px;
    }
    .sub-title {
        font-size: 26px !important;
        font-weight: 600;
        color: #333;
        margin-bottom: 15px;
    }
    .card {
        border-radius: 5px;
        padding: 20px;
        margin-bottom: 20px;
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<p class="main-title">📚 Educhain - Bulk Question Generator</p>', unsafe_allow_html=True)
st.markdown("""
Bulk Generation Questions using Educhain is a powerful feature that allows educators to quickly create large sets of 
high-quality questions for exams, quizzes, and practice sessions. With automated generation based on subject, topic, 
and difficulty level, it helps streamline content creation, saving time and ensuring comprehensive coverage of learning objectives.
""")

# Sidebar for API Keys and model selection
st.sidebar.markdown('<p class="sub-title">⚙️ Configuration</p>', unsafe_allow_html=True)

# API Key inputs
with st.sidebar.expander("API Keys", expanded=False):
    openai_api_key = st.text_input("OpenAI API Key", type="password", key="openai_key")
    anthropic_api_key = st.text_input("Anthropic API Key (Optional)", type="password", key="anthropic_key")
    google_api_key = st.text_input("Google API Key (Optional)", type="password", key="google_key")
    groq_api_key = st.text_input("Groq API Key (Optional)", type="password", key="groq_key")

# Set environment variables from provided keys if available
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
if anthropic_api_key:
    os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key 
if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key
if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key

# Model selection
model_option = st.sidebar.selectbox(
    "Select LLM Model",
    [
        "Default OpenAI",
        "GPT-4o",
        "o3-mini",
        "Claude 3.5 Sonnet",
        "Gemini Pro",
        "Gemini Flash",
        "Llama-3 (via Groq)"
    ],
    index=0,
    key="model_selection"
)

# Initialize client based on selected model
@st.cache_resource(show_spinner=False)
def get_educhain_client(model_selected: str, _openai_key: str = None, _anthropic_key: str = None,
                      _google_key: str = None, _groq_key: str = None) -> Educhain:
    """Initialize and return an Educhain client with the specified model."""
    # Check if any key is available
    if not any([_openai_key, _anthropic_key, _google_key, _groq_key]):
        st.sidebar.error("Please provide at least one API key.")
        return None

    try:
        if model_selected == "Default OpenAI":
            return Educhain()
        elif model_selected == "GPT-4o":
            llm = ChatOpenAI(model="gpt-4o")
            config = LLMConfig(custom_model=llm)
            return Educhain(config)
        elif model_selected == "o3-mini":
            llm = ChatOpenAI(model="o3-mini")
            config = LLMConfig(custom_model=llm)
            return Educhain(config)
        elif model_selected == "Claude 3.5 Sonnet":
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
            config = LLMConfig(custom_model=llm)
            return Educhain(config)
        elif model_selected == "Gemini Pro":
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
            config = LLMConfig(custom_model=llm)
            return Educhain(config)
        elif model_selected == "Gemini Flash":
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
            config = LLMConfig(custom_model=llm)
            return Educhain(config)
        elif model_selected == "Llama-3 (via Groq)":
            llm = ChatOpenAI(
                model="deepseek-r1-distill-llama-70b", 
                openai_api_base="https://api.groq.com/openai/v1",
            )
            config = LLMConfig(custom_model=llm)
            return Educhain(config)
        else:
            return Educhain()  # Default
    except Exception as e:
        st.sidebar.error(f"Error initializing model: {str(e)}")
        return None

# Initialize the client if API keys are provided
client = get_educhain_client(
    model_option,
    _openai_key=openai_api_key,
    _anthropic_key=anthropic_api_key,
    _google_key=google_api_key,
    _groq_key=groq_api_key
)

# Main content area
# Create tabs for different input methods
tab1, tab2 = st.tabs(["Form Input", "JSON Upload"])

topics_data = None

# Form input tab
with tab1:
    st.markdown('<p class="sub-title">📝 Define Topics and Learning Objectives</p>', unsafe_allow_html=True)
    
    with st.form(key="topic_form"):
        # Basic structure for topics
        main_topic = st.text_input("Main Topic (e.g., Mathematics, Science, History)", key="main_topic")
        
        # Create expandable sections for subtopics
        st.write("### Subtopics")
        col1, col2 = st.columns(2)
        
        # First subtopic (always available)
        with col1:
            subtopic1 = st.text_input("Subtopic 1 Name", key="subtopic1")
            learning_obj1 = st.text_area("Learning Objectives (One per line)", 
                                        height=150,
                                        placeholder="e.g.\nUnderstand the concept of fractions\nAdd and subtract fractions with like denominators",
                                        key="learning_obj1")
        
        with col2:
            subtopic2 = st.text_input("Subtopic 2 Name (Optional)", key="subtopic2")
            learning_obj2 = st.text_area("Learning Objectives (One per line)", 
                                        height=150,
                                        placeholder="e.g.\nUnderstand decimal place values\nConvert between fractions and decimals",
                                        key="learning_obj2")
        
        # Option for additional subtopics
        show_more = st.checkbox("Add more subtopics", key="show_more")
        
        if show_more:
            col3, col4 = st.columns(2)
            with col3:
                subtopic3 = st.text_input("Subtopic 3 Name (Optional)", key="subtopic3")
                learning_obj3 = st.text_area("Learning Objectives (One per line)", height=150, key="learning_obj3")
            
            with col4:
                subtopic4 = st.text_input("Subtopic 4 Name (Optional)", key="subtopic4")
                learning_obj4 = st.text_area("Learning Objectives (One per line)", height=150, key="learning_obj4")
        else:
            subtopic3, learning_obj3, subtopic4, learning_obj4 = "", "", "", ""
        
        submit_form = st.form_submit_button("Use This Topic Structure")
    
    if submit_form:
        # Convert form data to topics structure
        topics_data = [{"topic": main_topic, "subtopics": []}]
        
        # Process subtopics
        for subtopic, objectives in [
            (subtopic1, learning_obj1),
            (subtopic2, learning_obj2),
            (subtopic3, learning_obj3),
            (subtopic4, learning_obj4)
        ]:
            if subtopic and objectives:
                # Split objectives by newline and filter empty lines
                objective_list = [obj.strip() for obj in objectives.split("\n") if obj.strip()]
                if objective_list:
                    topics_data[0]["subtopics"].append({
                        "name": subtopic,
                        "learning_objectives": objective_list
                    })
        
        st.success("Topic structure created successfully!")
        st.json(topics_data)

# JSON Upload tab
with tab2:
    st.markdown('<p class="sub-title">📁 Upload JSON File</p>', unsafe_allow_html=True)
    st.write("Upload a JSON file with your topics structure.")
    
    # Example JSON structure
    with st.expander("View example JSON structure"):
        example_json = """
        [
          {
            "topic": "Mathematics",
            "subtopics": [
              {
                "name": "Fractions",
                "learning_objectives": [
                  "Convert proper fractions to improper fractions and mixed numbers",
                  "Add and subtract fractions with like denominators",
                  "Find equivalent fractions using multiplication and division"
                ]
              },
              {
                "name": "Decimals",
                "learning_objectives": [
                  "Understand decimal place values",
                  "Convert between fractions and decimals",
                  "Add and subtract decimals"
                ]
              }
            ]
          }
        ]
        """
        st.code(example_json, language="json")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a JSON file", type="json", key="json_uploader")
    
    if uploaded_file is not None:
        try:
            # Parse the uploaded JSON file
            topics_data = json.load(uploaded_file)
            st.success("JSON file loaded successfully!")
            st.json(topics_data)
        except json.JSONDecodeError:
            st.error("Invalid JSON file. Please check the format.")
            topics_data = None

# Generation settings
st.markdown('<p class="sub-title">⚙️ Generation Settings</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    question_type = st.selectbox(
        "Question Type",
        ["Multiple Choice", "True/False", "Fill in the Blank", "Short Answer"],
        index=0,
        key="question_type"
    )

with col2:
    generation_method = st.radio(
        "Generation Method",
        ["Questions Per Objective", "Total Questions"],
        horizontal=True,
        key="generation_method"
    )
    
    if generation_method == "Questions Per Objective":
        question_count = st.number_input("Questions per objective", 
                                        min_value=1, 
                                        max_value=50, 
                                        value=5,
                                        key="questions_per_objective")
    else:
        question_count = st.number_input("Total questions", 
                                        min_value=1, 
                                        max_value=100, 
                                        value=20,
                                        key="total_questions")

with col3:
    difficulty = st.select_slider(
        "Difficulty Level",
        options=["easy", "medium", "hard"],
        value="medium",
        key="difficulty"
    )
    
    output_format = st.selectbox(
        "Output Format",
        ["pdf", "csv", "json"],
        key="output_format"
    )

# Custom instructions field
custom_instructions = st.text_area(
    "Custom Instructions (Optional)",
    placeholder="Additional instructions for the question generation (e.g., 'Generate questions appropriate for 8th grade students' or 'Focus on conceptual understanding rather than calculations')",
    height=100,
    key="custom_instructions"
)

# Generate button
generate_button = st.button("Generate Questions", type="primary", use_container_width=True, disabled=(topics_data is None or client is None), key="generate_button")

# Handle generation
if generate_button and topics_data and client:
    with st.spinner("Generating questions... This may take a few minutes."):
        try:
            # Save topics data to a temporary JSON file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                json.dump(topics_data, temp_file)
                topics_file_path = temp_file.name
            
            # Set generation parameters
            gen_params = {
                "topic": topics_file_path,
                "max_workers": 3,  # Default value
                "output_format": output_format,
                "max_retries": 2,  # Default value
                "difficulty": difficulty,
                "batch_size": 5    # Default value
            }
            
            # Add custom instructions if provided
            if custom_instructions:
                gen_params["custom_instructions"] = custom_instructions
            
            # Set question type
            if question_type != "Multiple Choice":
                if question_type == "True/False":
                    gen_params["question_type"] = "True/False"
                elif question_type == "Fill in the Blank":
                    gen_params["question_type"] = "Fill in the Blank"
                elif question_type == "Short Answer":
                    gen_params["question_type"] = "Short Answer"
            
            # Set generation method
            if generation_method == "Questions Per Objective":
                gen_params["questions_per_objective"] = question_count
            else:
                gen_params["total_questions"] = question_count
            
            # Start generation
            result, output_file, total_generated, failed_batches = client.qna_engine.bulk_generate_questions(**gen_params)
            
            # Clean up the temporary file
            if os.path.exists(topics_file_path):
                os.unlink(topics_file_path)
            
            # Display results
            st.success(f"Generation completed! {total_generated} questions were generated.")
            
            # Show output file path and download button if applicable
            if output_file and os.path.exists(output_file):
                file_ext = Path(output_file).suffix[1:]  # Get file extension without dot
                output_filename = Path(output_file).name
                
                with open(output_file, "rb") as file:
                    file_bytes = file.read()
                    st.download_button(
                        label=f"Download {file_ext.upper()} file",
                        data=file_bytes,
                        file_name=output_filename,
                        mime={
                            "pdf": "application/pdf",
                            "csv": "text/csv",
                            "json": "application/json"
                        }.get(file_ext, "application/octet-stream"),
                        key="download_button"
                    )
            
            # Show any failures
            if failed_batches:
                st.warning(f"There were {len(failed_batches)} failed generation batches.")
        
        except Exception as e:
            st.error(f"An error occurred during question generation: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

# Documentation and Help
with st.expander("Help & Documentation"):
    st.markdown("""
    ## How to Use This App
    
    1. **Configure API Keys**: Enter your API keys in the sidebar (at least OpenAI is required)
    2. **Select a Model**: Choose an LLM model for generation
    3. **Define Topics**: Either use the form or upload a JSON file
    4. **Set Generation Parameters**: Configure question type, count, difficulty, etc.
    5. **Generate Questions**: Click the Generate button to start the process
    
    ## Tips for Best Results
    
    - Be specific in your learning objectives
    - Start with a small number of questions to test
    - For complex topics, consider using more advanced models like GPT-4o or Claude
    - Custom instructions can help tailor questions to your specific needs
    
    ## Output Formats
    
    - **PDF**: Best for printing or distributing to students
    - **CSV**: Useful for importing into spreadsheets or LMS systems
    - **JSON**: Perfect for further processing or integration with other systems
    
    ## Question Types
    
    - **Multiple Choice**: Questions with 4 options and one correct answer
    - **True/False**: Simple true or false statements
    - **Fill in the Blank**: Sentences with missing words or phrases
    - **Short Answer**: Questions requiring brief text responses
    """)

# Footer
st.markdown("---")
st.markdown("Made using Educhain and Streamlit")