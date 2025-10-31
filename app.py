import os
import streamlit as st
from openai import AzureOpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
from datetime import datetime
import urllib.parse
from constant import JOB_PLATFORMS, Prompts

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="Career AI", 
    layout="centered"
)
st.title("CareerBuilder AI Assistant")

# Constants
PINECONE_DIMENSION = 1536

# Initialize Azure OpenAI client
try:
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv(
            "AZURE_OPENAI_API_VERSION", 
            "2024-02-15-preview"
        ),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
except Exception as e:
    st.error(f"Azure OpenAI client failed: {e}")

# Initialize Pinecone client
try:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
except Exception as e:
    st.error(f"Pinecone client failed: {e}")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_skills" not in st.session_state:
    st.session_state.user_skills = ""

if "user_interests" not in st.session_state:
    st.session_state.user_interests = ""

if "user_profile_saved" not in st.session_state:
    st.session_state.user_profile_saved = False

if "show_recommendations" not in st.session_state:
    st.session_state.show_recommendations = False

if "current_jobs" not in st.session_state:
    st.session_state.current_jobs = ""

if "current_roadmap" not in st.session_state:
    st.session_state.current_roadmap = ""

if "chat_expanded" not in st.session_state:
    st.session_state.chat_expanded = True

def get_career_response(user_input, system_prompt=None):
    try:
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt}
            ]
        else:
            messages = [
                {
                    "role": "system", 
                    "content": "You are a career advisor. Provide helpful career guidance."
                }
            ]
        
        messages.append({"role": "user", "content": user_input})
        
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=messages,
            temperature=0.7,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"I apologize, but I'm having trouble responding right now. Error: {str(e)}"

def get_direct_response(user_input, system_prompt):
    """Get response without chat history for job recommendations and roadmap"""
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

def get_embeddings(text):
    """Get embeddings and ensure 1536 dimensions for Pinecone"""
    try:
        deployment = os.getenv(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", 
            "text-embedding-ada-002"
        )
        response = client.embeddings.create(
            model=deployment,
            input=text
        )
        embedding = response.data[0].embedding
        
        if len(embedding) != PINECONE_DIMENSION:
            if len(embedding) > PINECONE_DIMENSION:
                embedding = embedding[:PINECONE_DIMENSION]
            else:
                embedding = embedding + [0.0] * (
                    PINECONE_DIMENSION - len(embedding)
                )
        
        return embedding
        
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return [0.1] * PINECONE_DIMENSION

def save_user_profile(skills, interests):
    """Save user profile to Pinecone"""
    profile_text = f"Skills: {skills}. Interests: {interests}"
    try:
        vector = get_embeddings(profile_text)
        
        if len(vector) != PINECONE_DIMENSION:
            if len(vector) > PINECONE_DIMENSION:
                vector = vector[:PINECONE_DIMENSION]
            else:
                vector = vector + [0.0] * (
                    PINECONE_DIMENSION - len(vector)
                )
        
        import hashlib
        unique_id = hashlib.md5(
            f"{skills}_{interests}".encode()
        ).hexdigest()[:10]
        
        index.upsert(vectors=[{
            "id": f"user_{unique_id}",
            "values": vector,
            "metadata": {
                "skills": skills, 
                "interests": interests, 
                "type": "user_profile",
                "dimension": len(vector),
                "timestamp": datetime.now().isoformat()
            }
        }])
        st.session_state.user_profile_saved = True
        return True
    except Exception as e:
        st.error(f"Error saving profile: {e}")
        return False

def create_job_search_url(job_title, company, platform):
    """Create search URL for different job platforms"""
    search_query = f"{job_title} {company}".strip()
    encoded_query = urllib.parse.quote_plus(search_query)
    
    if platform in JOB_PLATFORMS:
        return f"{JOB_PLATFORMS[platform]}{encoded_query}"
    else:
        return f"{JOB_PLATFORMS['LinkedIn']}{encoded_query}"

def display_job_recommendations(jobs_text):
    """Display job recommendations with clickable links"""
    st.subheader("AI Job Recommendations")
    st.info("💡 Click on any job title to search for similar positions on job platforms")
    
    jobs_lines = jobs_text.split('\n')
    
    for line in jobs_lines:
        line = line.strip()
        if line.startswith('•') or line.startswith('-'):
            job_text = line[1:].strip()
            
            if '|' in job_text:
                parts = job_text.split('|')
                job_title_company = parts[0].strip()
                
                if ' at ' in job_title_company:
                    job_title, company = job_title_company.split(' at ', 1)
                    job_title = job_title.strip()
                    company = company.strip()
                else:
                    job_title = job_title_company
                    company = "Various Companies"
                
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                
                with col1:
                    st.write(f"**{job_title}** at {company}")
                
                with col2:
                    linkedin_url = create_job_search_url(
                        job_title, company, "LinkedIn"
                    )
                    st.markdown(
                        f"[LinkedIn]({linkedin_url})", 
                        unsafe_allow_html=True
                    )
                
                with col3:
                    indeed_url = create_job_search_url(
                        job_title, company, "Indeed"
                    )
                    st.markdown(
                        f"[Indeed]({indeed_url})", 
                        unsafe_allow_html=True
                    )
                
                with col4:
                    glassdoor_url = create_job_search_url(
                        job_title, company, "Glassdoor"
                    )
                    st.markdown(
                        f"[Glassdoor]({glassdoor_url})", 
                        unsafe_allow_html=True
                    )
                
                if len(parts) > 1:
                    details = ' | '.join(parts[1:])
                    st.caption(f"Details: {details}")
                
                st.markdown("---")

def get_ai_job_recommendations(skills, interests):
    """Get AI-generated job recommendations based on skills and interests"""
    user_prompt = f"""
    User Skills: {skills}
    Desired Roles: {interests}
    
    Recommend 5-8 current job opportunities that match these skills and interests.
    Include positions from various companies and different seniority levels.
    Focus on realistic companies that actually hire for these roles.
    """
    
    try:
        with st.spinner("Searching for current job opportunities..."):
            response = get_direct_response(
                user_prompt, 
                Prompts.recommendation_system_prompt
            )
        return response
    except Exception as e:
        return f"Error generating job recommendations: {str(e)}"

def get_personalized_roadmap(skills, interests):
    """Get personalized learning roadmap"""
    user_prompt = f"""
    Create a personalized learning roadmap for:
    Current Skills: {skills}
    Target Role: {interests}
    
    Make it a 6-month actionable plan with specific technologies and learning resources.
    """
    
    try:
        with st.spinner("Creating your personalized roadmap..."):
            response = get_direct_response(
                user_prompt, 
                Prompts.personalized_roadmap_prompt
            )
        return response
    except Exception as e:
        return f"Error generating roadmap: {str(e)}"

# Create main layout with tabs
tab1, tab2, tab3 = st.tabs([
    "Career Chat", 
    "Profile & Jobs", 
    "Job Search"
])

# Tab 1: Career Chat
with tab1:
    st.header("Career Guidance Chat")
    
    # Chat interface in expander
    with st.expander(
        "Ask Career Questions", 
        expanded=st.session_state.chat_expanded
    ):
        user_input = st.text_input(
            "Enter your career question:", 
            key="chat_input"
        )
        
        if user_input:
            with st.spinner("Thinking..."):
                response = get_career_response(user_input)
                st.session_state.messages.append({
                    "role": "user", 
                    "content": user_input
                })
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response
                })
            
            # Auto-collapse after response
            st.session_state.chat_expanded = False
    
    # Display conversation history
    if st.session_state.messages:
        st.subheader("Conversation History")
        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(
                "user" if msg['role']=='user' else "assistant"
            ):
                st.write(msg['content'])
    
    if st.button("Clear Chat History", key="clear_chat"):
        st.session_state.messages = []
        st.session_state.chat_expanded = True
        st.rerun()

# Tab 2: Profile & Jobs
with tab2:
    st.header("Profile Setup & Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        skills = st.text_input(
            "Your Skills:", 
            value=st.session_state.user_skills, 
            placeholder="e.g., Python, SQL, Machine Learning",
            key="skills_input"
        )
    
    with col2:
        interests = st.text_input(
            "Interested Roles:", 
            value=st.session_state.user_interests,
            placeholder="e.g., Data Scientist, ML Engineer",
            key="interests_input"
        )
    
    if st.button(
        "Save Profile & Get Recommendations", 
        key="save_profile"
    ):
        if skills and interests:
            st.session_state.user_skills = skills
            st.session_state.user_interests = interests
            
            with st.spinner("Saving profile and generating recommendations..."):
                if save_user_profile(skills, interests):
                    st.success("Profile saved successfully!")
                    
                    # Generate recommendations
                    st.session_state.current_jobs = get_ai_job_recommendations(
                        skills, 
                        interests
                    )
                    st.session_state.current_roadmap = get_personalized_roadmap(
                        skills, 
                        interests
                    )
                    st.session_state.show_recommendations = True
                    st.rerun()
                else:
                    st.error("Failed to save profile to database")
        else:
            st.error("Please enter both skills and interests")
    
    # Display recommendations
    if st.session_state.show_recommendations:
        if st.session_state.current_jobs:
            display_job_recommendations(st.session_state.current_jobs)
        
        if st.session_state.current_roadmap:
            st.subheader("Personalized Career Roadmap")
            st.write(st.session_state.current_roadmap)
        
        # Refresh buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Refresh Job Matches", key="refresh_jobs"):
                if st.session_state.user_skills:
                    st.session_state.current_jobs = get_ai_job_recommendations(
                        st.session_state.user_skills, 
                        st.session_state.user_interests
                    )
                    st.rerun()
        
        with col2:
            if st.button("Refresh Roadmap", key="refresh_roadmap"):
                if st.session_state.user_skills:
                    st.session_state.current_roadmap = get_personalized_roadmap(
                        st.session_state.user_skills, 
                        st.session_state.user_interests
                    )
                    st.rerun()

# Tab 3: Direct Job Search
with tab3:
    st.header("Direct Job Search")
    
    search_query = st.text_input(
        "Search for specific job titles:", 
        placeholder="e.g., Python Developer, Data Scientist",
        key="job_search_input"
    )
    
    if search_query:
        st.subheader("Search on Job Platforms:")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            linkedin_url = create_job_search_url(
                search_query, "", "LinkedIn"
            )
            st.markdown(
                f"[LinkedIn]({linkedin_url})", 
                unsafe_allow_html=True
            )
        
        with col2:
            indeed_url = create_job_search_url(
                search_query, "", "Indeed"
            )
            st.markdown(
                f"[Indeed]({indeed_url})", 
                unsafe_allow_html=True
            )
        
        with col3:
            glassdoor_url = create_job_search_url(
                search_query, "", "Glassdoor"
            )
            st.markdown(
                f"[Glassdoor]({glassdoor_url})", 
                unsafe_allow_html=True
            )
        
        with col4:
            monster_url = create_job_search_url(
                search_query, "", "Monster"
            )
            st.markdown(
                f"[Monster]({monster_url})", 
                unsafe_allow_html=True
            )

# Weekly News Section
if st.session_state.user_profile_saved:
    st.header("Weekly Industry News")
    
    if st.button("Get This Week's Industry Update", key="get_news"):
        def get_weekly_industry_news(interests):
            user_prompt = f"Generate a weekly industry news digest for: {interests}"
            try:
                with st.spinner("Gathering latest industry news..."):
                    response = get_direct_response(
                        user_prompt, 
                        Prompts.weekly_update_system_prompt
                    )
                return response
            except Exception as e:
                return f"Error generating news digest: {str(e)}"
        
        news = get_weekly_industry_news(st.session_state.user_interests)
        st.session_state.last_news_update = news
        st.session_state.news_timestamp = datetime.now()
        st.write(news)

if hasattr(st.session_state, 'last_news_update'):
    st.subheader("Last News Update")
    st.write(st.session_state.last_news_update)
    
    if hasattr(st.session_state, 'news_timestamp'):
        st.caption(
            f"Last updated: {st.session_state.news_timestamp.strftime('%Y-%m-%d %H:%M')}"
        )