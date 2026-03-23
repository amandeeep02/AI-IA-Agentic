import json
from tools.llm_client import call_llm
from orchestrator.prompt_templates import codegen_system_prompt

def generate_code(user_requirement):
    prompt_with_req = codegen_system_prompt.replace("<USER_REQUIREMENT>", user_requirement)
    response_content = call_llm(prompt_with_req, "Generate code for this requirement.", json_mode=True)
    try:
        return json.loads(response_content)
    except Exception as e:
        print(f"Error parsing code generation: {e}")
        return {"files": []}