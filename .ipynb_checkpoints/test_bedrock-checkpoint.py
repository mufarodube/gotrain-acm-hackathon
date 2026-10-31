   from strands import Agent
   from strands.models import BedrockModel

   model = BedrockModel(model_id="eu.anthropic.claude-sonnet-5-5", region_name="eu-central-1")
   Agent(model=model)("Hello, what can you do?")