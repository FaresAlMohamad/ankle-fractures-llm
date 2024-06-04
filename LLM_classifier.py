from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random
import os
import argparse 

def get_classification(system_instruction, report, form, shots):
    ''' instructs the LLM to classify a radiology report with the given system_instruction and form, to specify the task
    and what shape the answer is expected to be. Examples can be provided with the shots argument for few-shot
    prompting'''
    
    text = system_instruction + form + " [/INST] " + " [INST] " + shots +report+" [/INST] "

    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    outputs = model.generate(**inputs, max_new_tokens=1,pad_token_id=tokenizer.eos_token_id)
    
    answer=tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    return answer

def fix_output(old_output, form):
    '''in case the get_classification() function returns an answer, that does not fit the expected shape, the 
    fix_output() function instructs the model to answer again using the old answer as well as an instruction
    to stick to the specified answer shape. This function is recursive if it fails to get the expected answer shape'''
    
    text = old_output + " [INST] " +  form + " [/INST] "
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    outputs = model.generate(**inputs, max_new_tokens=1,pad_token_id=tokenizer.eos_token_id)
    
    answer=tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    if answer[-1] in ["0", "1"]:
        return int(answer[-1])
    else:
        return fix_output(answer)

def classify_reports(reports):
    '''classify all reports using the get_classification() and fix_output() functions. The classifications are
    saved as a .pth file.'''
    
    classifications = {}
    for item in reports:
        image_id = item["image_id"]
        report = item["report"]
        classification = get_classification(args.system_instruction, report, args.form, args.shots)
        if classification[-1] in ["0", "1"]:
            classifications[image_id] = int(classification[-1])
        else:
            classifications[image_id] = fix_output(classification, args.form)
        torch.save(classifications, os.path.join(args.root_dir, "LLM_classifier", "classifications.pth"))

parser = argparse.ArgumentParser(description="Classify radiology reports using a LLM")
parser.add_argument('--system_instruction', type=str, default='<s> [INST] Beschreibt der folgende radiologische Befund das Vorhandensein einer Fraktur des oberen Sprunggelenks? Das OSG besteht aus der distalen Tibia, der distalen Fibula und dem Os Talus. Frakturen anderer Knochen sind irrelevant. Frakturen des Fußes können also ignoriert werden. Wenn ein V.a. eine OSG-Fraktur geäußert wird, zähl es als Fraktur. Wenn für eine Fraktur eine operative Versorgung durchgeführt werden musste, gilt sie weiterhin als Fraktur, es sei denn es steht explizit da, dass keine, nicht mal flaue, Frakturlinie oder Frakturspalt mehr erkennbar ist. ', help='system instruction for the LLM')
parser.add_argument('--form', type=str, default="Du darfst nur mit 0 oder 1 antworten", help='specify to the model which shape the answer should be')
parser.add_argument('--shots', type=str, default="", help='for fewshot learning, add examples')
parser.add_argument('--random_seed', type=float, default=0, help='set a random seed')
parser.add_argument('--root_dir', type=str, default='/sc-projects/sc-proj-cc06-ag-ki-radiologie/OSG', help='directory with the LLM folder and the LLM_classifier folder')
args = parser.parse_args()

random.seed(args.random_seed)
model_path = os.path.join(args.root_dir, "mixtral_8x_7b")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
reports = torch.load(os.path.join(args.root_dir, "LLM_classifier", "reports.pth"))
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    classify_reports(reports)

