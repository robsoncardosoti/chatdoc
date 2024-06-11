import time, tempfile, os, torch
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()


custom_template = """Dada a seguinte pergunta, responda no idioma Português.

Pergunta: {question}"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

prompt_template = """Use o seguinte contexto para responder somente o que foi perguntado. Se nao souber a resposta, diga que não sabe, não tente inventar uma resposta.

{context}

Pergunta: {question}
Resposta: """

prompt_template_llamma3 = """
<|begin_of_text|>
   Você é um assistente prestativo, respeitoso e honesto que responde perguntas relacionadas ao documento do usuário. 
   Se o usuário perguntar fora do contexto do documento responda que não sabe a resposta. 
   Se o contexto fornecido não for suficiente não responda à pergunta.
<|eot_id|>
  Responda à pergunta do usuário com base no contexto fornecido abaixo:
  Contexto :{context}
  Pergunta: {question}
  Resposta: 
<|eot_id|>
"""


QA_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

list_embedding = ('BAAI/bge-base-en',  # 768 dimensions - Arc. BertModel
                  'BAAI/bge-large-en-v1.5',  # 1024 dimensions - BertModel
                  'PORTULAN/albertina-1b5-portuguese-ptbr-encoder',  # 1536 dimensions - DebertaV2ForMarkedLM
                  'google/canine-c',  # 768 dimensions - Canine Model
                  'intfloat/multilingual-e5-large',  # 1024 dimensions - XLMRobertaModel
                  'BAAI/bge-m3',  # 1024 dimensions - XMLRobertaModel
                  'stjiris/bert-large-portuguese-cased-legal-mlm-nli-sts-v0',  # 1024 dimensions - BertModel
                  'sentence-transformers/LaBSE', #3072 dimensions - BertModel
                  'OpenAI()')  # ,

list_model_loader = ('LlamaCpp',
                     'HuggingFaceInference',
                      'HuggingFaceHub',
                     'GPT4All',  #https://gpt4all.io/index.html
                     'AutoModel',
                     'AutoGPTQ',
                     # 'CTransformers',
                     'OpenAI')

list_vector_type = ('Faiss', 'Chroma', 'Elastic')
list_type_retriever = ('similarity','mmr','similarity_score_threshold')

diretorio = './modelos/'


def download_model(llm_model):
    from huggingface_hub import hf_hub_download
    from huggingface_hub import snapshot_download
    if llm_model == 'Meta-Llama-3-8B.Q8_0.gguf':
        hf_hub_download(repo_id="QuantFactory/Meta-Llama-3-8B-GGUF-v2", filename="Meta-Llama-3-8B.Q8_0.gguf",
                                local_dir=diretorio)  # https://huggingface.co/QuantFactory/Meta-Llama-3-8B-GGUF-v2/tree/main
    elif llm_model == 'Meta-Llama-3-8B-Instruct-v2.Q2_K.gguf':
        hf_hub_download(repo_id="QuantFactory/Meta-Llama-3-8B-Instruct-GGUF-v2",
                        filename="Meta-Llama-3-8B-Instruct-v2.Q2_K.gguf",
                        local_dir=diretorio)  # https://huggingface.co/QuantFactory/Meta-Llama-3-8B-GGUF-v2/tree/main
    elif llm_model == 'Meta-Llama-3-70B-Instruct-v2.Q2_K.gguf':
        hf_hub_download(repo_id="QuantFactory/Meta-Llama-3-70B-Instruct-GGUF-v2",
                        filename="Meta-Llama-3-70B-Instruct-v2.Q2_K.gguf",
                        local_dir=diretorio)
    elif llm_model == 'meta-llama/Meta-Llama-3-8B':
        print(".....")
    else:
        model = snapshot_download(repo_id=llm_model)  # https://huggingface.co/models?search=gptq


def list_llms(model_loader):
    global list_llm
    if model_loader == 'LlamaCpp':
        list_llm = ('Meta-Llama-3-8B.Q8_0.gguf', 'Meta-Llama-3-8B-Instruct-v2.Q2_K.gguf', 'Meta-Llama-3-70B-Instruct-v2.Q2_K.gguf')
    elif model_loader == 'HuggingFaceInference' or 'HuggingFaceHub':
        list_llm = ('meta-llama/Meta-Llama-3-8B', 'Meta-Llama-3-8B-Instruct', '22h/cabrita_7b_pt_850000', 'BAAI/Aquila2-7b',
            'google/gemma-7b', 'maritaca-ai/sabia-7b', 'lmsys/vicuna-7b-v1.5', 'Intel/neural-chat-7b-v3-3',
            'JJhooww/Mistral-7B-v0.2-Base_ptbr', 'adalbertojunior/Llama-3-8B-Instruct-Portuguese-v0.2-fft')
    elif model_loader == 'GPT4All':
        list_llm = ('Meta-Llama-3-8B.Q8_0.gguf', 'Meta-Llama-3-8B-Instruct-v2.Q2_K.gguf')
    elif model_loader == 'AutoModel':
        list_llm = ('astronomer/Llama-3-8B-GPTQ-4-Bit', 'neuralmagic/Mistral-7B-Instruct-v0.3-GPTQ-4bit',
                    'TheBloke/Wizard-Vicuna-13B-Uncensored-GPTQ', 'TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ')
    elif model_loader == 'AutoGPTQ':
        list_llm = ('Meta-Llama-3-8B',)
    elif model_loader == 'CTransformers':
        list_llm = ('Meta-Llama-3-8B.Q8_0.gguf', 'Meta-Llama-3-8B-Instruct.Q4_0.gguf')
    elif model_loader == 'OpenAI':
        list_llm = ('gpt-3.5-turbo',)
    return list_llm


def extract_raw_text(uploaded_files, extract_type):
    docs = []
    tempo_inicio = time.time()
    ### Load documents ###
    #temp_dir = tempfile.TemporaryDirectory()
    text = ""
    for file in uploaded_files:
        temp_filepath = os.path.join('./files/', file.name)  #
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        if extract_type == 'Texto':
            ### Extract Text ###
            if file.name.endswith(".pdf"):
                from PyPDF2 import PdfReader
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
                print("Tempo de raw text: ", time.time() - tempo_inicio)
                return text
        elif extract_type == 'Documento':
            ### Extract Document ###
            if file.name.endswith(".pdf") or file.name.endswith(".bin"):
                from langchain_community.document_loaders import PyMuPDFLoader
                loader = PyMuPDFLoader(temp_filepath)
            elif file.name.endswith(".txt"):
                from langchain_community.document_loaders import TextLoader
                loader = TextLoader(temp_filepath)
            elif file.name.endswith(".csv"):
                from langchain_community.document_loaders import CSVLoader
                loader = CSVLoader(file_path=temp_filepath, csv_args={'delimiter': ',',
                                                                      'quotechar': '"'})  # ,'fieldnames':['col1','col2','col3']
            elif file.name.endswith(".png") or file.name.endswith(".jpg"):
                from langchain_community.document_loaders import UnstructuredImageLoader
                loader = UnstructuredImageLoader(
                    temp_filepath)  # Unstructured cria diferentes elementos para diferentes chunks de texto, se preferir manter separados insira mode="elements"
            elif file.name.endswith(".doc") or file.name.endswith(".docx"):
                from langchain_community.document_loaders import UnstructuredWordDocumentLoader
                loader = UnstructuredWordDocumentLoader(temp_filepath)
            elif file.name.endswith(".ppt") or file.name.endswith(".pptx"):
                from langchain_community.document_loaders import UnstructuredPowerPointLoader
                loader = UnstructuredPowerPointLoader(temp_filepath)
            elif file.name.endswith(".xls") or file.name.endswith(".xlsx"):
                from langchain_community.document_loaders import UnstructuredExcelLoader
                loader = UnstructuredExcelLoader(temp_filepath, mode="elements")
            elif file.name.endswith(".odt"):
                from langchain_community.document_loaders import UnstructuredODTLoader
                loader = UnstructuredODTLoader(temp_filepath, mode="elements")
            elif file.name.endswith(".p7s"):
                import fitz
                pdf = fitz.open()
                pdf.insert_page(0)
                with open(temp_filepath, 'rb') as p7s_file:
                    loader = p7s_file.read()
            else:
                loader = []
                print("Arquivo não suportado!")
            docs.extend(loader.load())
    print("Tempo de raw text: ", time.time() - tempo_inicio)
    return docs


def split_text_chunks(text, chunk_size, chunk_type):
    global chunks
    tempo_inicio = time.time()
    ####################  Recursive chunking: Divide o texto em chunks de forma iterativa, realizando novas divisões de acordo com a lista de caracteres separadores até alcançar o tamanho desejado.   ####################
    if chunk_type == 'Recursiva':
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0, separators=["\n\n", "\n", " ", ""])
        chunks = text_splitter.split_text(text)
    ####################  Fixed size chunking: o mais comum, direto e simples. Segmenta pelo número de tokens.  ####################
    elif chunk_type == 'Fixa':
        from langchain.text_splitter import TextSplitter
        text_splitter = TextSplitter(chunk_size=chunk_size,chunk_overlap=0)
        chunks = text_splitter.split_text(text)
    ###################  Token chunking: Divide o texto usando tokens.  ####################
    elif chunk_type == 'Tokens':
        from langchain.text_splitter import TokenTextSplitter
        text_splitter = TokenTextSplitter(chunk_size=chunk_size,chunk_overlap=0)
        chunks = text_splitter.split_text(text)
    ###################  Semantic chunking: Divide o texto em partes menores que possuem semântica, possui um custo computacional maior que outros métodos.  ####################
    elif chunk_type == 'Semântica':
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_openai.embeddings import OpenAIEmbeddings
        text_splitter = SemanticChunker(OpenAIEmbeddings(), breakpoint_threshold_type="percentile")  #(HuggingFaceEmbeddings(model_name='BAAI/bge-base-en'))
        chunks = text_splitter.create_documents(text)
    elif chunk_type == 'Sentença':
        from llama_index.core.node_parser import SentenceSplitter
        text_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=0, )
        # chunks = text_splitter.split_text(text)
        chunks = text_splitter.get_nodes_from_documents(documents=[text], show_progress=True)


    print("Tempo de chunks: ", time.time() - tempo_inicio)
    return chunks


def split_document_chunks(text, chunk_size):
    tempo_inicio = time.time()
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    document_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0,
                                                       separators=["\n\n", "\n", " ", ""])
    chunks = document_splitter.split_documents(text)
    print("Tempo de chunks: ", time.time() - tempo_inicio)
    return chunks


def index_vectorstore(text_chunks, embeddings, vector_type,extract_type):
    tempo_inicio = time.time()
    if vector_type == 'Faiss':
        from langchain_community.vectorstores import FAISS
        if extract_type == 'Texto':
            vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        else:
            vectorstore = FAISS.from_documents(documents=text_chunks, embedding=embeddings)
        # faiss.save_local("faiss")
    elif vector_type == 'Chroma':
        from langchain_community.vectorstores import Chroma
        if extract_type == 'Texto':
            vectorstore = Chroma.from_texts(texts=text_chunks, embedding=embeddings, persist_directory='chroma')
        else:
            vectorstore = Chroma.from_documents(texts=text_chunks, embedding=embeddings, persist_directory='chroma')
        # chroma.persist()
    elif vector_type == 'Elastic':
        from langchain_community.vectorstores import elastic_vector_search
        if extract_type == 'Texto':
            vectorstore = elastic_vector_search.ElasticVectorSearch.from_texts(texts=text_chunks, embedding=embeddings,elasticsearch_url="http://localhost:9200", index_name="test_index")
        else:
            vectorstore = elastic_vector_search.ElasticVectorSearch.from_documents(texts=text_chunks, embedding=embeddings,elasticsearch_url="http://localhost:9200",index_name="test_index")
    else:
        vector_type = 'Outros'
        from langchain.vectorstores import DocArrayInMemorySearch
        if extract_type == 'Texto':
            vectorstore = DocArrayInMemorySearch.from_texts(text_chunks, embeddings)
        else:
            vectorstore = DocArrayInMemorySearch.from_documents(text_chunks, embeddings)
    print("Tipo de VectorStore: ", vector_type)
    print("Tempo de index: ", time.time() - tempo_inicio)
    return vectorstore


def index_retriever(vectorstore, type_retriever):
    if type_retriever == 'similarity':
        retriever = vectorstore.as_retriever(search_type=type_retriever, search_kwargs={"k": 2})
    elif type_retriever == 'mmr':
        retriever = vectorstore.as_retriever(search_type=type_retriever, search_kwargs={"k": 2, "fetch_k": 10})
    else:  # type_retriever == 'similarity_score_threshold'
        retriever = vectorstore.as_retriever(search_type=type_retriever, search_kwargs={"k": 2, "score_threshold": 0.5})
    return retriever



def pre_process(uploaded_files, chunk_size,embeddings,vector_type,extract_type):
    text = extract_raw_text(uploaded_files)
    text_chunks = split_text_chunks(text, chunk_size)
    vectorstore = index_vectorstore(text_chunks, embeddings, vector_type,extract_type)
    retriever = index_retriever(vectorstore)
    return retriever


def llm_loader(llm_model, max_tokens, context_window, temperature, model_loader):
    download_model(llm_model)
    from langchain.callbacks.manager import CallbackManager
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    if model_loader == 'OpenAI':
        from langchain import OpenAI
        from langchain_community.chat_models import ChatOpenAI
        llm = ChatOpenAI(model_name=llm_model, temperature=temperature, streaming=True)
    elif model_loader == 'LlamaCpp':
        from langchain_community.llms import LlamaCpp
        llm = LlamaCpp(
            model_path=diretorio + llm_model,
            n_gpu_layers=-1,  # Change this value based on your model and your GPU VRAM pool.
            n_batch=max_tokens,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
            n_ctx=context_window,  # context window
            callback_manager=callback_manager,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            top_k=40,
            verbose=True,
            f16_kv=True,

        )
    #######  EM DESENVOLVIMENTO   ###############
    elif model_loader == 'GPT4All':
        from langchain_community.llms import GPT4All
        callbacks = [StreamingStdOutCallbackHandler()]
        llm = GPT4All(
            model=diretorio + llm_model,
            max_tokens=context_window,
            temp=temperature,
            n_threads=8,
            callbacks=callbacks,
            verbose=True,  # Verbose is required to pass to the callback manager
        )
    ############   DEPRECATED CLASS   ################
    # elif model_loader=='HuggingFaceHub':
    #    from langchain_community.llms import HuggingFaceHub
    #    llm = HuggingFaceHub(repo_id=llm_model,
    #                         model_kwargs={"temperature":temperature, "max_length":context_window},
    #                         huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN"))

    elif model_loader=='HuggingFaceInference':
        from huggingface_hub import notebook_login
        notebook_login(os.getenv("HUGGINGFACE_API_TOKEN"))

        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
        model_checkpoint = llm_model
        model_config = AutoConfig.from_pretrained(model_checkpoint,
                                                  trust_remote_code=True,
                                                  max_new_tokens=1024,
                                                  )
        model = AutoModelForCausalLM.from_pretrained(model_checkpoint,
                                                     trust_remote_code=True,
                                                     config=model_config,
                                                     device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        from transformers import pipeline
        from langchain_huggingface.llms import HuggingFacePipeline

        pipeline = pipeline("text-generation",
                            model=model,
                            tokenizer=tokenizer,
                            torch_dtype=torch.float16,
                            max_length=3000,
                            device_map="auto", )

        llm = HuggingFacePipeline(pipeline=pipeline)

        #from huggingface_hub import login
        #login(token=os.getenv("HUGGINGFACE_API_TOKEN"))
        #from langchain_community.llms import HuggingFacePipeline
        #llm = HuggingFacePipeline.from_model_id(model_id=llm_model,task="text-generation", pipeline_kwargs={"max_new_tokens":200, "pad_token_id":50256})
    ############ Apresentando erro:  object has no attribute 'get_loading_attributes'  #########################
    elif model_loader == 'AutoModel':
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        config = AutoConfig.from_pretrained(llm_model)
        config.quantization_config["disable_exllama"] = True
        llm = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=llm_model, device_map="auto",
                                                   quantization_config=config)  # , quantization_config=quantization_config
    ############ Apresentando erro:    ##############################
    elif model_loader == 'AutoGPTQ':  ### GPTQ parameters: https://cdn-uploads.huggingface.co/production/uploads/6426d3f3a7723d62b53c259b/m89k1rHMUlTpGa9lrZdrK.png
        # from transformers import AutoModelForCausalLM
        # model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7b-Chat-GPTQ", torch_dtype=torch.float16, device_map="auto")

        from auto_gptq import AutoGPTQForCausalLM
        from transformers import AutoTokenizer, pipeline
        tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-base-en', use_fast=True)
        # Set the padding token to be equal to the end of sequence token (eos_token)
        tokenizer.pad_token = tokenizer.eos_token
        llm = AutoGPTQForCausalLM.from_quantized(model_name_or_path=diretorio + llm_model, device_map="auto",
                                                 model_basename="model", use_safetensors=True, trust_remote_code=True,
                                                 quantize_config=None, low_cpu_mem_usage=True, device='cuda',
                                                 use_triton=False,
                                                 max_memory={i: '8000MB' for i in range(torch.cuda.device_count())}, )
    ####### Apresenta erro: Segmentation fault (core dumped) ############
    # elif model_loader == 'CTransformers':
    #    from langchain_community.llms import CTransformers
    #    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    #    llm = CTransformers(model='./E:/LLMs/llama/' + llm_model, callbacks=[StreamingStdOutCallbackHandler()])
    else:
        from langchain_community.llms import OpenLLM
        llm = OpenLLM(model_name='falcon', model_id='tiiuae/falcon-7b')  # flan-t5	#google/flan-t5-large
    return llm


def get_conversation_chain(vectorstore, llm_model, context_window, max_tokens, temperature, model_loader,type_retriever):
    tempo_inicio = time.time()
    from langchain.chains import LLMChain
    if model_loader == 'customizado':
        llm = llm_loader(llm_model, max_tokens, context_window, temperature, model_loader)
        conversation_chain = llm(question=QA_PROMPT, context=CUSTOM_QUESTION_PROMPT,
                                 text_inputs='qual o nome do municipio')
    else:
        from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
        from langchain.chains import ConversationalRetrievalChain
        llm = llm_loader(llm_model, max_tokens, context_window, temperature, model_loader)
        retriever = index_retriever(vectorstore, type_retriever)
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        #memory = ConversationBufferWindowMemory(k=2,memory_key='chat_history')
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            condense_question_prompt=CUSTOM_QUESTION_PROMPT,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        )
    print("Tempo de chain: ", time.time() - tempo_inicio)

    return conversation_chain
