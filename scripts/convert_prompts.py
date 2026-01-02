import json
import os

# Paths
source_file = "/home/kossiso/CascadeProjects/afrimed-chw/data/reference_docs/reference_prompts.jsonl"
target_file = "/home/kossiso/CascadeProjects/afrimed-chw/data/llama_finetune/reference_prompts_refined.json"

system_prompt = (
    "You are a Community Health Worker (CHW) assistant. Your goal is to provide accurate, "
    "accessible, and practical health advice relevant to resource-limited settings. "
    "Always prioritize patient safety, recognize danger signs that require urgent referral "
    "to a facility, and explain concepts simply. Do not provide advice outside your scope of practice."
)

def format_data_item(data):
    """Recursively formats data into a clean string, bolding keys in dicts."""
    if isinstance(data, list):
        if not data: return ""
        # If it's a list of primitive strings, just comma-join
        if all(isinstance(x, (str, int, float)) for x in data):
            return ", ".join(map(str, data))
        # If it contains lists or dicts
        return "; ".join([format_data_item(i) for i in data])
    elif isinstance(data, dict):
        parts = []
        for k, v in data.items():
            val = format_data_item(v)
            if val:
                parts.append(f"**{k}:** {val}")
            else:
                parts.append(f"**{k}**")
        return " ".join(parts)
    return str(data)

def format_section(title, content):
    """Formats a main section (Assessment, Action, Advice) matching the target style."""
    if not content: 
        return ""
    
    if isinstance(content, (list, dict)):
        items = []
        if isinstance(content, list):
            for item in content:
                items.append(format_data_item(item))
        else: # dict
            for k, v in content.items():
                val = format_data_item(v)
                if val:
                    items.append(f"**{k}:** {val}")
                else:
                    items.append(f"**{k}**")
        
        if title == "Action":
            # Action is typically a numbered list
            formatted_list = "\n".join([f"{i+1}. {item}" for i, item in enumerate(items)])
            return f"**{title}:** \n{formatted_list}"
        elif title == "Advice":
            # Advice is typically a bulleted list
            formatted_list = "\n".join([f"- {item}" for item in items])
            return f"**{title}:** \n{formatted_list}"
        else:
            # Assessment is usually a paragraph
            return f"**{title}:** {'; '.join(items)}"
    else:
        # String content
        return f"**{title}:** {content}"

def main():
    refined_data = []
    
    if not os.path.exists(source_file):
        print(f"Error: Source file {source_file} not found.")
        return

    with open(source_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                
                # Combine instruction and input
                # In the target format, 'input' is the actual question/prompt
                source_instr = item.get("instruction", "")
                source_input = item.get("input", "")
                combined_input = f"{source_instr} {source_input}".strip()
                
                # Format output
                source_output = item.get("output", {})
                
                sections = []
                for sec_title in ["Assessment", "Action", "Advice"]:
                    sec_content = source_output.get(sec_title)
                    if sec_content:
                        formatted_sec = format_section(sec_title, sec_content)
                        if formatted_sec:
                            sections.append(formatted_sec)
                
                final_output = "\n\n".join(sections)
                
                # Rigid format to match train_refined_patched.json
                refined_entry = {
                    "instruction": system_prompt,
                    "input": combined_input,
                    "output": final_output,
                    "is_refined": True
                }
                refined_data.append(refined_entry)
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")

    os.makedirs(os.path.dirname(target_file), exist_ok=True)
    with open(target_file, 'w', encoding='utf-8') as f:
        # indent=2 matched the head output of train_refined_patched.json
        json.dump(refined_data, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully converted {len(refined_data)} prompts to {target_file}")

if __name__ == "__main__":
    main()
