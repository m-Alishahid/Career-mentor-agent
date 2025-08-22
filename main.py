# main.py
import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
from roadmap_tools import get_career_roadmap

# Load environment variables
load_dotenv()

# Gemini API client setup
client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Model setup
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)

# Run configuration
config = RunConfig(model=model, tracing_disabled=True)

# Agents
career_agent = Agent(
    name="CareerAgent",
    instructions="Ask about interests and suggest a career field.",
      handoffs=["skill_agent"] 
)

skill_agent = Agent(
    name="SkillAgent",
    instructions="Share the roadmap using the get_career_roadmap tool.",
    model=model,
   handoffs=["job_agent"]  
)

job_agent = Agent(
    name="JobAgent",
    instructions="Suggest job titles in the chosen career.",
    model=model
     handoffs=[]  
)

def main():
    print("\U0001F393 Career Mentor Agent\n")
    interest = input("💬 What are your interests? → ")

    # Step 1: Career suggestion
    result1 = Runner.run_sync(career_agent, interest, run_config=config)
    field = result1.final_output.strip()
    print("\n📌 Suggested Career:", field)

    # Step 2: Skills roadmap
    result2 = Runner.run_sync(skill_agent, field, run_config=config)
    print("\n📚 Required Skills:", result2.final_output)

    # Step 3: Possible jobs
    result3 = Runner.run_sync(job_agent, field, run_config=config)
    print("\n💼 Possible Jobs:", result3.final_output)

if __name__ == "__main__":
    main()
