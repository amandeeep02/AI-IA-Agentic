import json
from tools.llm_client import call_llm
from orchestrator.prompt_templates import testgen_system_prompt

def generate_tests(spec, code_snapshot):
    prompt_with_req = f"{testgen_system_prompt}\n\nSpec:\n{spec}\n\nCode Snapshot:\n{code_snapshot}"
    response_content = call_llm(testgen_system_prompt, prompt_with_req, json_mode=True)
    try:
        return json.loads(response_content)
    except Exception as e:
        print(f"Error parsing test generation: {e}")
        return {"tests": []}