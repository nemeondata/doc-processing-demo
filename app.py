import pandas as pd
import streamlit as st

from litellm import completion
from pydantic import BaseModel, Field
import instructor

import base64
import os
import pandas as pd

os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]

class TableRow(BaseModel):
    extraction_name: str = Field(description='Name of item extracted from document')
    extraction_value: str|int|float = Field(description='Value extracted from document')

class DocumentParsed(BaseModel):
    type_of_doc: str = Field(description='Type of Document uploaded, e.g. invoice, picking list, or MSDS')
    table_setup: list[TableRow] = Field(description='Values and descriptions for extration to summarise the document')

c  = instructor.from_litellm(
    completion
)

full_prompt = """You are an experienced logistics agent, trained in identifying and summarising logistics documents.
Extract the type of document you are asked to process, then in a table, summarise the document by extracting the most important information from the document."""

def send_data(materials, prompt, response_model, client, fallback = 'gpt-4o-mini'):
    try: 
        send = materials.copy()
        send.append(
            {
                "type": "text",
                "text": prompt
            }
        )

        ai_response = client.chat.completions.create(
            model='claude-3-5-haiku-20241022',
            max_tokens=4096,
            messages=[{"role": 'user', 'content': send}],
            response_model=response_model            
        )
    
    except: 
        send = []
        for i, mess in enumerate(materials):
            if 'image_url' in mess.keys():
                part = {'type':'file',
                'file': {
                    'filename': str(i)+'.pdf',
                    'file_data': mess['image_url']
                    }
                }
                send.append(part)
            else:
                send.append(mess)

        send.append(
            {
                "type": "text",
                "text": prompt
            }
        )

        ai_response = client.chat.completions.create(
            model=fallback,
            max_tokens=4096,
            messages=[{"role": 'user', 'content': send}],
            response_model=response_model            
        )            


    return ai_response

def main():
    st.title('Nemeon Document Intelligence Demo')

    files = st.file_uploader('Please upload a PDF!', type='.pdf', accept_multiple_files=False)

    if files:
        pdf_data = base64.standard_b64encode(files.getvalue()).decode('utf-8')

        content = [
            {
                'type': 'image_url',
                'image_url': f'data:application/pdf;base64,{pdf_data}'
            }
        ]

        parse = send_data(content, full_prompt, DocumentParsed, c)

        info = parse.model_dump()

        st.write(f"## Your document is a type of {info['type_of_doc']}")

        table = pd.DataFrame(info['table_setup'])

        st.dataframe(table)

if __name__ == "__main__":
    main()