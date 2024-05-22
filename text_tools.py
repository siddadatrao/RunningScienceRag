import pandas as pd
import numpy as np
from PyPDF2 import PdfReader
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
from numpy.linalg import norm
from llama_cpp import Llama
from sentence_transformers import CrossEncoder

CLUMPSIZE = 5

def get_text_from_pdf(path_list):
    """
    :param path_list: Path of pdf file to use for RAG
    :return: df of two columns ['page_number', 'text']
    """
    df = pd.DataFrame(columns=['page_number', 'text'])
    for path in path_list:
        reader = PdfReader(path)
        page = []
        text = []

        for i in range(0, len(reader.pages)):
            text.append(reader.pages[i].extract_text().replace("\n", " "))
            page.append(i)

        sub_df = pd.DataFrame({'page_number': page, 'text': text}, columns=['page_number', 'text'])
        df = pd.concat([df, sub_df], ignore_index=True)

    return df


def clump_text(text, num_groups):
    """
    :param text: some set of text
    :param num_groups: number of sentences in one clump
    :return: array of clumped num_groups sized texts
    """
    clump_list = []
    sentences = sent_tokenize(text)

    temp_list = []
    c = 0
    for s in sentences:
        temp_list.append(s)
        c += 1
        if (c == num_groups):
            clump_list.append(''.join(temp_list))
            temp_list = []
            c = 0
    if (len(temp_list) > 0):
        clump_list.append(''.join(temp_list))

    return clump_list


def create_grouped_df(text_per_page, num_groups):
    """
    Creating a df with all clumps in individual row with associated page number of clumps
    :param text_per_page: Base df of page number and text on that page
    :param num_groups: number of sentences per clump
    :return: df with row of associated page number and clumps
    """
    page_nums = []
    grouped_texts = []

    for index, row in text_per_page.iterrows():
        clumped_texts = clump_text(row['text'], num_groups)
        number_of_clumped_strings = len(clumped_texts)
        page_nums = page_nums + [row['page_number']] * number_of_clumped_strings
        grouped_texts.extend(clumped_texts)
    clumped_df = pd.DataFrame({'page_nums': page_nums, 'grouped_texts': grouped_texts},
                              columns=['page_nums', 'grouped_texts'])

    return clumped_df


def load_model(use_case):
    """
    Load the model based on use case
    :param use_case: model identifier for local use
    :return: model to use
    """
    if use_case == "small":
        model_path = snapshot_download(repo_id="TaylorAI/gte-tiny", allow_patterns=["*.json", "model.safetensors"])
        model = SentenceTransformer(model_path)
        return model
    return None


def create_embeddings(clumped_df, model):
    """
    Create embeddings for text
    :param clumped_df: Clumped df with row of page number with associated clump
    :param model: model to use
    :return:
    """
    embeddings = []
    for index, row in clumped_df.iterrows():
        embeddings.append(model.encode(row['grouped_texts']))
    clumped_df['embedded_text'] = embeddings

    return clumped_df


def embed_text(text, model):
    """
    Encode text
    :param text: Input text
    :param model: model to use
    :return: Embeddings for text based on model
    """
    return model.encode(text)


def similarity(simtype, a, b):
    """
    Generic similarity function
    :param simtype: Type of similarity formula
    :param a: a text for comparison
    :param b: b text for comparison
    :return: similarity score between a and b
    """
    similarity = 0
    if simtype == 'cosine':
        similarity = np.dot(a,b)/(norm(a)*norm(b))
    return similarity


def get_embedding_similarity_text(text, num_of_similar, embeddings_df, model_to_use):
    """
    Get top k similar texts
    :param text: input text
    :param num_of_similar: desired number of similar texts
    :param embeddings_df: df of embeddings associated with text
    :param model_to_use: model to use
    :return: array of k similar texts
    """
    # Embed Input Text
    embed_input_text = embed_text(text, model_to_use)

    # Find similarity between input embedding and all embeddings
    result_array = np.array(
        embeddings_df.apply(lambda row: similarity('cosine', row['embedded_text'], embed_input_text), axis=1))

    # Get top x indices to use as reference
    top_k_indices = np.argsort(result_array)[-num_of_similar:]

    # Get text equivilent array
    similar_text_array = []
    for i in range(len(top_k_indices)):
        similar_text_array.append(embeddings_df.loc[top_k_indices[i], 'grouped_texts'])
    return similar_text_array


def pre_rag(context, action, step_statement, llm):
    """
    :param context:
    :param action:
    :param step_statement:
    :return:
    """
    output = llm(
          "Q: You are a platform that suggests text content that is focused on learning. " + action + context + step_statement + " A: ", # Prompt
          max_tokens=None, # Generate up to 32 tokens, set to None to generate up to the end of the context window
          stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
          echo=False # Echo the prompt back in the output
    ) # Generate a completion, can also call create_completion
    print(output['choices'][0]['text'])
    print('')
    return output['choices'][0]['text']


def rerank_similarity(number_similar, model_name, response, top_k_similar_texts):
    """
    :param number_similar:
    :param model_name:
    :param response:
    :param top_k_similar_texts:
    :return:
    """
    model = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2', max_length=512)
    scores = model.predict([(response, top_k_similar_texts[0]), (response, top_k_similar_texts[1]) , (response, top_k_similar_texts[2])])
    return scores


def setup():
    llm = Llama(
        model_path="../llmtools/Meta-Llama-3-8b-q4_K_S.gguf",
        # n_gpu_layers=-1, # Uncomment to use GPU acceleration
        # seed=1337, # Uncomment to set a specific seed
        # n_ctx=2048, # Uncomment to increase the context window
    )
    text_per_page = get_text_from_pdf(["med_texts/kaushik_test.pdf", "med_texts/nihms-1800005.pdf"])
    clumped_df = create_grouped_df(text_per_page, CLUMPSIZE)
    model_to_use = load_model("small")
    embeddings_df = create_embeddings(clumped_df, model_to_use)
    return llm, embeddings_df, model_to_use


def runt(context, action, step_statement, llm, embeddings_df, model_to_use):
    response = pre_rag(context, action, step_statement, llm)
    top_k_similar_texts = get_embedding_similarity_text(response, 5, embeddings_df, model_to_use)
    scores = rerank_similarity(3, "a", response, top_k_similar_texts)
    best_snippet = top_k_similar_texts[np.argmax(scores)]
    return best_snippet


llm, embeddings_df, model_to_use = setup()

while(True):
    user_input = input("Input: ")
    print(runt(user_input, "", " What is an alternative viewpoint?", llm, embeddings_df, model_to_use))