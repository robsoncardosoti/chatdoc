import langchain
import streamlit as st
from dotenv import load_dotenv
#from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
#from langchain import LLMChain
from template import css, bot_template, user_template, logo_image  #, head_template
#from PIL import Image
from utils import *
langchain.debug = True


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    tempo_inicio = time.time()
    load_dotenv()

    st.set_page_config(page_title="ChatTCMGO", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    with st.sidebar:
        tab1, tab2 = st.tabs(["Principal", "Configurações"])
        with tab1:
            embedding_model = st.selectbox('Selecione o embedding:', list_embedding)
            model_loader = st.selectbox('Selecione a carregador do modelo:', list_model_loader)
            llm_model = st.selectbox('Selecione o modelo:', list_llms(model_loader))
            # st.subheader("Documentos")
            uploaded_files = st.file_uploader("Selecione os documentos e clique em Processar",accept_multiple_files=True, type=['pdf', 'doc', 'docx'])
            print("Tempo de Main: ", time.time() - tempo_inicio)
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                extract_type = st.selectbox('Extração:', ('Texto', 'Documento'))
                vector_type = st.selectbox('Vetor:', list_vector_type)
                context_window = st.select_slider('Contexto:', options=['512', '1024', '2048', '4096'],
                                                  value='4096')
                chunk_size = st.select_slider('Segmento:', options=[256, 512, 1024, 2048], value=1024)
            with col2:
                chunk_type = st.selectbox('Segmentação:', ('Recursiva', 'Fixa', 'Semantica'))
                type_retriver = st.selectbox('Recuperador:', list_type_retriever)
                max_tokens = st.select_slider('Tokens:', options=[64, 128, 256, 512], value=128)
                temperature = st.select_slider('Temperatura:', options=[0, 0, 7, 0.8, 0.9, 1], value=0.0)


        if st.button("Processar"):
            with st.spinner("Processando"):

                tempo_inicio = time.time()
                # get pdf text
                raw_text = extract_raw_text(uploaded_files, extract_type)

                # get the text chunks
                if extract_type == 'Texto':
                    text_chunks = split_text_chunks(raw_text, chunk_size, chunk_type)
                else:
                    text_chunks = split_document_chunks(raw_text, chunk_size)

                # create conversation chain
                if embedding_model == 'OpenAI()':
                    from langchain.embeddings import OpenAIEmbeddings
                    embeddings = OpenAIEmbeddings()
                else:
                    from langchain.embeddings import HuggingFaceEmbeddings
                    embeddings = HuggingFaceEmbeddings(model_name=embedding_model,
                                                       model_kwargs={'device': 'cuda'},
                                                       encode_kwargs={'normalize_embeddings': True})
                print("o embedding e: ", embedding_model)
                print("o modelo LLM e: ", llm_model)
                # create vector store
                vectorstore = index_vectorstore(text_chunks,embeddings,vector_type,extract_type)
                st.session_state.conversation = get_conversation_chain(vectorstore, llm_model, context_window,
                                                                       max_tokens, temperature, model_loader,type_retriver)
                print("Tempo de execucao: ", time.time() - tempo_inicio)

    #st.image(Image.open('template/logotcm.png'))
    #st.markdown(head_template, unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(logo_image, unsafe_allow_html=True)
    with col2:
        st.header("DocChat")
    with col3:
        st.markdown("build:20240527")

    tab1, tab2 = st.tabs(["Principal", "Visualizar"])
    with tab1:
        if "conversation" not in st.session_state:
            st.session_state.conversation = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = None

        user_question = st.text_input("Informe uma pergunta sobre os seus documentos:")

        if user_question:
            handle_userinput(user_question)
    with tab2:
        #------------      Visualizacao do Arquivo     --------------#
        for file in uploaded_files:
            import base64
            #temp_dir = tempfile.TemporaryDirectory()
            temp_filepath = os.path.join('./files/', file.name)  # temp_dir.name
            with open(temp_filepath, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)


#------------------      Botao Limpar/Resetar conversas - EM DESENVOLVIMENTO     -----------------------#
# Setup memory for contextual conversation
# msgs = StreamlitChatMessageHistory()
# if len(msgs.messages) == 0 or st.sidebar.button("Limpar historico"):
# st.session_state.clear()
# msgs.clear()
# msgs.add_ai_message("Como eu posso te ajudar?")
# memory.clear()
# StreamlitChatMessageHistory.clear()
# st.session_state.conversation.conversation.clear()
# st.session_state.conversation.chat_history.clear()
# memory.chat_memory.clear()
# st.session_state.conversation = ['Con']
# st.session_state.chat_history = ['chat']
# del st.session_state.conversation
# del st.session_state.chat_history


if __name__ == '__main__':
    main()
