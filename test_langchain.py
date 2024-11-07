
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os

def test_langchain():
    # Initialize the Hugging Face endpoint by remotely
    repo_id = 'mistralai/Mistral-7B-Instruct-v0.3'
    llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=128, temperature=0.7, token=os.environ['huggingfaceid'])

    # Create a prompt template
    question = 'What is the capital of New Zealand?'
    template = '''Question: {question}
    Answer: Let's think step by step.'''
    prompt = PromptTemplate(template=template, input_variables=['question'])
    print(f"prompt:{prompt}")

    # Create a LangChain LLMChain with the prompt and the LLM
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    response = llm_chain.invoke({'question': question})
    print(f"response:{response}")

    # Initialize a Hugging Face pipeline by local
    model_id = 'gpt2'
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, max_new_tokens=100)
    hf = HuggingFacePipeline(pipeline=pipe)
    # Invoke the pipeline without prompt with 'text-generation'
    response = hf.invoke('Once upon a time')
    print(f"response:{response}")

    # # Initialize the pipeline for GPU usage，device argument can't work.
    gpu_llm = HuggingFacePipeline.from_model_id(
        model_id='gpt2',
        task='text-generation',
        pipeline_kwargs={'max_new_tokens': 100}
    )
    # # Create a prompt template
    template = """Question: {question}
    Answer: Let's think step by step."""
    prompt = PromptTemplate.from_template(template)
    # # Create a chain with the prompt and the GPU LLM
    chain = prompt | gpu_llm
    # Provide a question and invoke the chain
    question = 'How does photosynthesis work?'
    response = chain.invoke({"question": question})
    print(f"response:{response}")

test_langchain()