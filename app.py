import streamlit as st;
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI 
from langchain.chains import ConversationalRetrievalChain 
from htmlforUi import css, bot_template, user_template


#extract text ferom pdf
def get_pdf_text(pdf_docs):
    text="";
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf);
        for page in pdf_reader.pages:
            text+=page.extract_text();
    return text;

def get_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunk = text_splitter.split_text(raw_text);
    return chunk;

# #Add embeddings in a vectorStore using  OpenAI model
def get_vectorStore(text_chuncks):
    embeddings =OpenAIEmbeddings();
    vectorStore = FAISS.from_texts(texts=text_chuncks,embedding=embeddings);
    return vectorStore;


#Add embeddings in a vectorStore using Your own computer using Instructor Embeddings
# def get_vectorStore(text_chunks):
#     embeddings =HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl");
#     vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     return vectorstore


def getConversationChain(vectorstore):
    llm = ChatOpenAI();
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True);
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
    )
    return conversation_chain


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
    #load env files
    load_dotenv();

    #frontend
    st.set_page_config(page_title="Chat-With-Pdf", page_icon=":books:")
    st.write(css,unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat-With-Pdf :books:");

    user_question=st.text_input("Ask a question about the docu:")
    if user_question:
        handle_userinput(user_question);
    
    # st.write(user_template,unsafe_allow_html=True)
    # st.write(bot_template.replace(),unsafe_allow_html=True)


    with st.sidebar:
        st.subheader("Your Docu")
        pdf_docs=st.file_uploader(
            "Upload PDF here",accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Processing, Please Wait"):
                #gewt pdf text 
                raw_text = get_pdf_text(pdf_docs)
                #st.write(raw_text);

                #get the text chyunks  (divide in the chunks)
                text_chuncks = get_chunks(raw_text);
                # st.write(text_chuncks)


                #Create Vector Store
                vectorstore=get_vectorStore(text_chuncks)
                # st.write(vectorstore)      

                #create conversation chain 
                st.session_state.conversation = getConversationChain(vectorstore)      



if __name__ ==  '__main__':
    main();
