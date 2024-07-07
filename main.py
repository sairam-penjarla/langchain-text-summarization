import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain.docstore.document import Document

# Set up the OpenAI API key
open_api_key = "<your OPENAI_API_KEY>"
os.environ["OPENAI_API_KEY"] = open_api_key

speech = """
People across the country, involved in government, political, and social activities, are dedicating their time to make the ‘Viksit Bharat Sankalp Yatra’ (Developed India Resolution Journey) successful. Therefore, as a Member of Parliament, it was my responsibility to also contribute my time to this program. So, today, I have come here just as a Member of Parliament and your ‘sevak’, ready to participate in this program, much like you.

In our country, governments have come and gone, numerous schemes have been formulated, discussions have taken place, and big promises have been made. However, my experience and observations led me to believe that the most critical aspect that requires attention is ensuring that the government’s plans reach the intended beneficiaries without any hassles. If there is a ‘Pradhan Mantri Awas Yojana’ (Prime Minister’s housing scheme), then those who are living in jhuggis and slums should get their houses. And he should not need to make rounds of the government offices for this purpose. The government should reach him. Since you have assigned this responsibility to me, about four crore families have got their ‘pucca’ houses. However, I have encountered cases where someone is left out of the government benefits. Therefore, I have decided to tour the country again, to listen to people’s experiences with government schemes, to understand whether they received the intended benefits, and to ensure that the programs are reaching everyone as planned without paying any bribes. We will get the real picture if we visit them again. Therefore, this ‘Viksit Bharat Sankalp Yatra’ is, in a way, my own examination. I want to hear from you and the people across the country whether what I envisioned and the work I have been doing aligns with reality and whether it has reached those for whom it was meant.

It is crucial to check whether the work that was supposed to happen has indeed taken place. I recently met some individuals who utilized the Ayushman card to get treatment for serious illnesses. One person met with a severe accident, and after using the card, he could afford the necessary operation, and now he is recovering well. When I asked him, he said: “How could I afford this treatment? Now that there is the Ayushman card, I mustered courage and underwent an operation. Now I am perfectly fine.” Such stories are blessings to me.

The bureaucrats, who prepare good schemes, expedite the paperwork and even allocate funds, also feel satisfied that 50 or 100 people who were supposed to get the funds have got it. The funds meant for a thousand villages have been released. But their job satisfaction peaks when they hear that their work has directly impacted someone’s life positively. When they see the tangible results of their efforts, their enthusiasm multiplies. They feel satisfied. Therefore, ‘Viksit Bharat Sankalp Yatra’ has had a positive impact on government officers. It has made them more enthusiastic about their work, especially when they witness the tangible benefits reaching the people. Officers now feel satisfied with their work, saying, “I made a good plan, I created a file, and the intended beneficiaries received the benefits.” When they find that the money has reached a poor widow under the Jeevan Jyoti scheme and it was a great help to her during her crisis, they realise that they have done a good job. When a government officer listens to such stories, he feels very satisfied.

There are very few who understand the power and impact of the ‘Viksit Bharat Sankalp Yatra’. When I hear people connected to bureaucratic circles talking about it, expressing their satisfaction, it resonates with me. I’ve heard stories where someone suddenly received 2 lakh rupees after the death of her husband, and a sister mentioned how the arrival of gas in her home transformed her lives. The most significant aspect is when someone says that the line between rich and poor has vanished. While the slogan ‘Garibi Hatao’ (Remove Poverty) is one thing, but the real change happens when a person says, “As soon as the gas stove came to my house, the distinction between poverty and affluence disappeared.
"""

class SummarizationUsingLangchain:
    """Class for summarizing text using Langchain's ChatOpenAI."""

    def __init__(self, speech):
        self.speech = speech

    def summarize(self):
        chat_messages = [
            SystemMessage(content='You are an expert assistant with expertise in summarizing speeches'),
            HumanMessage(content=f'Please provide a short and concise summary of the following speech:\n TEXT: {self.speech}')
        ]
        llm = ChatOpenAI(model_name='gpt-3.5-turbo')
        print("Number of tokens in speech:", llm.get_num_tokens(self.speech))
        summary = llm(chat_messages).content
        print("Summary Using Langchain:\n", summary)


class PromptTemplatesTextSummarization:
    """Class for summarizing text using prompt templates in Langchain."""

    def __init__(self, speech, language):
        self.speech = speech
        self.language = language

    def summarize(self):
        generic_template = '''
        Write a summary of the following speech:
        Speech : `{speech}`
        Translate the precise summary to {language}.
        '''
        prompt = PromptTemplate(input_variables=['speech', 'language'], template=generic_template)
        llm = ChatOpenAI(model_name='gpt-3.5-turbo')
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        summary = llm_chain.run({'speech': self.speech, 'language': self.language})
        print("Summary Using Prompt Templates:\n", summary)


class StuffDocumentChainSummarization:
    """Class for summarizing text using Stuff Document Chain in Langchain."""

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def summarize(self):
        pdfreader = PdfReader(self.pdf_path)
        text = ''.join([page.extract_text() for page in pdfreader.pages if page.extract_text()])
        docs = [Document(page_content=text)]
        llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
        template = '''Write a concise and short summary of the following speech.
        Speech: `{text}`
        '''
        prompt = PromptTemplate(input_variables=['text'], template=template)
        chain = load_summarize_chain(llm, chain_type='stuff', prompt=prompt, verbose=False)
        output_summary = chain.run(docs)
        print("Summary Using Stuff Document Chain:\n", output_summary)


class MapReduceSummarization:
    """Class for summarizing large documents using Map Reduce."""

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def summarize(self):
        pdfreader = PdfReader(self.pdf_path)
        text = ''.join([page.extract_text() for page in pdfreader.pages if page.extract_text()])
        llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
        chunks = text_splitter.create_documents([text])
        chain = load_summarize_chain(llm, chain_type='map_reduce', verbose=False)
        summary = chain.run(chunks)
        print("Summary Using Map Reduce:\n", summary)


class MapReduceWithCustomPrompts:
    """Class for summarizing text using Map Reduce with custom prompts."""

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def summarize(self):
        pdfreader = PdfReader(self.pdf_path)
        text = ''.join([page.extract_text() for page in pdfreader.pages if page.extract_text()])
        llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
        chunks = text_splitter.create_documents([text])
        chunks_prompt = """
        Please summarize the below speech:
        Speech:`{text}`
        Summary:
        """
        map_prompt_template = PromptTemplate(input_variables=['text'], template=chunks_prompt)
        final_combine_prompt = '''
        Provide a final summary of the entire speech with these important points.
        Add a Generic Motivational Title,
        Start the precise summary with an introduction and provide the
        summary in number points for the speech.
        Speech: `{text}`
        '''
        final_combine_prompt_template = PromptTemplate(input_variables=['text'], template=final_combine_prompt)
        summary_chain = load_summarize_chain(
            llm=llm,
            chain_type='map_reduce',
            map_prompt=map_prompt_template,
            combine_prompt=final_combine_prompt_template,
            verbose=False
        )
        output = summary_chain.run(chunks)
        print("Summary Using Map Reduce with Custom Prompts:\n", output)


class RefineChainSummarization:
    """Class for summarizing text using Refine Chain."""

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def summarize(self):
        pdfreader = PdfReader(self.pdf_path)
        text = ''.join([page.extract_text() for page in pdfreader.pages if page.extract_text()])
        llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
        chunks = text_splitter.create_documents([text])
        chain = load_summarize_chain(llm, chain_type='refine', verbose=True)
        output_summary = chain.run(chunks)
        print("Summary Using Refine Chain:\n", output_summary)


# Execute all summarization methods
speech = """
People across the country, involved in government, political, and social activities, are dedicating their time to make the ‘Viksit Bharat Sankalp Yatra’ (Developed India Resolution Journey) successful. Therefore, as a Member of Parliament, it was my responsibility to also contribute my time to this program. So, today, I have come here just as a Member of Parliament and your ‘sevak’, ready to participate in this program, much like you.
...
"""

summarizers = [
    SummarizationUsingLangchain(speech),
    PromptTemplatesTextSummarization(speech, 'Hindi'),
    StuffDocumentChainSummarization('apjspeech.pdf'),
    MapReduceSummarization('apjspeech.pdf'),
    MapReduceWithCustomPrompts('apjspeech.pdf'),
    RefineChainSummarization('apjspeech.pdf')
]

for summarizer in summarizers:
    summarizer.summarize()
