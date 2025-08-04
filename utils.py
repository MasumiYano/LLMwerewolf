import os


def load_prompts(filename: str, **kwargs):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_dir = os.path.join(script_dir, "prompts")

    prompt_path = os.path.join(prompts_dir, filename)

    if not os.path.exists(prompt_path):
        raise FileNotFoundError("Prompt file not found")

    with open(prompt_path, "r", encoding="utf-8") as file:
        prompt_template = file.read()

    for key, value in kwargs.items():
        placeholder = "{" + key + "}"

        if isinstance(value, list):
            replacement = ", ".join(str(item) for item in value)
        else:
            replacement = str(value)

        prompt_template = prompt_template.replace(placeholder, replacement)

    return prompt_template
