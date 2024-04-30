import gradio as gr
import os
import tiktoken
from openai import OpenAI
from pypdf import PdfReader
from dotenv import load_dotenv
import time
import json
print(gr.__version__)

load_dotenv()  # take environment variables from .env.

client = OpenAI(
  api_key=os.environ.get("OPENAI_API_KEY"),
)

encoding = tiktoken.get_encoding("cl100k_base")

prompt_template = []

instruction_prompt = """Your goal is to summarize the patient health records with the instructions
                  provided below to help the physician. Keep your summary succinct and to the point, do not make up information and only fetch
                  information that is present in the document.
                  \n
                  Instructions:
                  Classify the input document as one of the three types: Prescription, Diagnosis, Lab Report. If the document does not fit into any of these three category, only output the type of the document with the description: Summary not available.
                  \n
                  If the document is of the type:

                  Prescription: list out all the medications prescribed to the patient with their correct dosages

                  Diagnosis: list the all diagnosis given by the doctor

                  Lab Report: list out reported values of all parameters for the lab investigation. If there is any 
                  abnormal values, report it. Use the bio reference interval provided in the document if given. Include details only about the investigation.
                  Do not give any notes or interpretations of your own.
                  \n
                  """

json_instruct_prompt = """Every item in the json array has the following values: 
                      'filename' to indicate which file the data has been extracted from.\n
                      'heading' can take one of the following values: Lab Report, Prescription, Diagnosis.\n
                      'subheading' indicates a larger data group that the data belongs to.\n
                      'date' the date of the document if available, in dd/mm/yyyy format.\n
                      'pagenumber' the page number of the document the information is extracted from.\n    .                  
                      'body' contains the main summary. Do not repeat any information that has been already been mentioned.
                      """

outputjson_template = {
    "summary" : [
        {
          "filename" : "value1",
          "heading" : "value2",
          "subheading" : "value3",
          "date" : "dd/mm/yyyy",
          "pagenumber" : 1,
          "body" : "value4",
        }
    ]
}



def get_gpt_response(model, content) :    
    response = client.chat.completions.create(
      model=model,
      response_format={ "type": "json_object" },
      messages=[
        {
          "role": "system",
          "content": "You are a medical assistant to a physician. Make sure to use technical terms specific to the medical field only."
        },
        {
          "role": "user",
          "content": instruction_prompt + 
          
                    "Give the summary as a json output with the following format:" + json.dumps(outputjson_template) + 
                    
                    "\nHere are the rules for crafting the json response: " + json_instruct_prompt +
                    
                    "\nThe documents are provided in a json format with filename, pagenumber and content below separated by ###.\n###\n"
                    
                    + content
        }
      ],
      temperature=0.01
    )
    return response.choices[0].message.content, response.usage

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def summarizer(model: str, files: list[str]) :
    
    input_list = []
    for file in files:
        pdf = PdfReader(file)
        page_number = 1
        input_json = {}
        input_json["filename"] = file.rpartition("\\")[-1]
        input_json["body"] = []
        for page in pdf.pages :
            text = page.extract_text()
            input_json["body"].append({"page_number": page_number, "content": text})
            page_number += 1
        input_list.append(input_json)
    input = json.dumps(input_list)
    #print(input)
    start_time = time.time()
    summary, usage = get_gpt_response(model, input)
    end_time = time.time()    

    analytics = "Total tokens: "+ str(usage.total_tokens) + "\nPrompt token: "+ str(usage.prompt_tokens) + "\nCompletion token: "+ str(usage.completion_tokens)
    result = "Time Taken: " + str(end_time - start_time) + "\n"+summary + "\n" + analytics
    return result

demo = gr.Interface(
    fn = summarizer,
    inputs=
    [
        gr.Dropdown(["gpt-4-turbo-2024-04-09", "gpt-3.5-turbo-0125"], label="Choose a Model"),
        gr.File(label="input medical records", file_types=['.pdf', '.jpeg', '.png' ], file_count="multiple")
    ],
    outputs=
    [
        gr.Textbox("Summary")
    ]
)

if __name__ == "__main__":
    demo.launch(share=False)

