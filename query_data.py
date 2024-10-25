__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai.llms import GoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings


class RunRag:
    CHROMA_PATH = "chroma/"

    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:

    {context}

    ---

    Answer the question based on the above context: {question}
    """

    @staticmethod
    def get_embedding_function(api_key):
        embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001',
                                                  google_api_key=api_key)
        return embeddings

    def run(self, query_text, api_key):
        # Create CLI.
        return self.query_rag(query_text, api_key)

    def query_rag(self, query_text: str, api_key: str):
        # Prepare the DB.
        embedding_function = self.get_embedding_function(api_key)
        db = Chroma(persist_directory=self.CHROMA_PATH, embedding_function=embedding_function)

        # Search the DB.
        results = db.similarity_search_with_score(query_text, k=5)

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(self.PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        # print(prompt)

        model = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
        response_text = model.invoke(prompt)

        sources = [doc.metadata.get("id", None) for doc, _score in results]
        formatted_response = f"Response: {response_text}\nSources: {sources}"
        print(formatted_response)
        return response_text
