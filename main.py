import os
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Flask app setup
app = Flask(__name__)
app.secret_key = "secret_key_for_flask"
UPLOAD_FOLDER = "./uploads"
ALLOWED_EXTENSIONS = {"csv"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load environment variables
DB_CONNECTION_URL_2 = os.getenv("DB_CONNECTION_URL_2")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Helper functions
def validate_env_variables():
    if not DB_CONNECTION_URL_2:
        raise ValueError("Database connection URL is missing!")
    if not GOOGLE_API_KEY:
        raise ValueError("Google API key is missing!")
    if not GROQ_API_KEY:
        raise ValueError("Groq API key is missing!")

validate_env_variables()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_csv_data(file_path):
    texts = []
    try:
        print(f"\nProcessing CSV file: {file_path}\n")
        df = pd.read_csv(file_path)

        # Convert the CSV content into a single text block
        for _, row in df.iterrows():
            row_content = " ".join(map(str, row.values))
            texts.append(row_content)

        print(f"Successfully processed CSV file: {file_path}")
    except Exception as e:
        print(f"Error processing CSV file {file_path}: {e}")
    return texts


def chunk_texts(documents, chunk_size=500, overlap=50):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []
    for doc in documents:
        chunks.extend(splitter.split_text(doc))
    print(f"Generated {len(chunks)} chunks.")
    return chunks


def store_embeddings_pgvector(texts, table_name):
    try:
        embeddings_generator = GoogleGenerativeAIEmbeddings(
            api_key=GOOGLE_API_KEY,
            model="models/embedding-001"  
        )

        vectorstore = PGVector(
            connection_string=DB_CONNECTION_URL_2,
            embedding_function=embeddings_generator,
        )

        vectorstore.add_texts(texts, table_name=table_name)
        print(f"Data successfully stored in the '{table_name}' table.")
        return "success"
    except Exception as e:
        print(f"Error storing embeddings in PGVector: {e}")
        return f"Error storing embeddings: {e}"


def get_pgvector_retriever():
    try:
        # Initialize the embeddings generator
        embeddings_generator = GoogleGenerativeAIEmbeddings(
            api_key=GOOGLE_API_KEY,
            model="models/embedding-001"
        )

        # Initialize the vectorstore
        vectorstore = PGVector(
            connection_string=DB_CONNECTION_URL_2,
            embedding_function=embeddings_generator,
        )

        # Use as_retriever to wrap the vectorstore
        retriever = vectorstore.as_retriever(
            search_kwargs={"table_name": "langchain_pg_embedding", "k": 5}  # Retrieve top 5 results
        )
        return retriever
    except Exception as e:
        print(f"Error initializing retriever: {e}")
        return None


def query_groq_with_response(user_query):
    try:
        chatgroq = ChatGroq(
            api_key=GROQ_API_KEY,
            model="llama3-8b-8192",
            temperature=0.0,
            max_retries=2
        )

        prompt_template = PromptTemplate(
            input_variables=["context", "query"],
            template="Context: {context}\n\nQuery: {query}\n\nPlease generate a detailed response based on the context."
        )

        # Retrieve relevant chunks
        retriever = get_pgvector_retriever()
        if not retriever:
            return "Error initializing retriever. Please check the setup."

        # Use the retriever to get relevant documents
        retrieved_docs = retriever.get_relevant_documents(user_query)
        context = "\n".join(doc.page_content for doc in retrieved_docs)  # Combine the retrieved texts

        if not context.strip():
            context = "No relevant context retrieved."

        # Construct the LLM chain using the prompt template
        llm_chain = prompt_template | chatgroq

        # Generate a response from the model
        print(f"Generating response for query: {user_query}")
        response = llm_chain.invoke({"context": context, "query": user_query})

        # Return the response text
        return response.content if hasattr(response, 'content') else str(response)

    except Exception as e:
        print(f"Error in query generation: {e}")
        return "An error occurred while processing your query."


# Flask routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process_file_and_query():
    if "file" not in request.files or "query" not in request.form:
        flash("File or query missing!")
        return redirect(request.url)

    file = request.files["file"]
    user_query = request.form["query"]

    if file.filename == "":
        flash("No file selected!")
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Process CSV file
        csv_texts = load_csv_data(file_path)
        chunks = chunk_texts(csv_texts)

        # Store embeddings into PGVector
        result = store_embeddings_pgvector(chunks, table_name="langchain_pg_embedding")

        if result == "success":
            flash("Database connected successfully!")
            flash("Embeddings stored successfully!")
        else:
            flash(f"Error storing embeddings: {result}")
            return redirect(url_for("index"))

        # Query the LLM for a response
        response = query_groq_with_response(user_query)

        # Pass the response to the template
        return render_template("index.html", response=response)

    else:
        flash("Allowed file types are .csv")
        return redirect(request.url)


if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
