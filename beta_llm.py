import json
import os
import random
import re
import shutil

import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.callbacks import StreamingStdOutCallbackHandler, CallbackManager
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

K = 5
MAX_RETRY = 5

path_outputs = 'llm_sockshop'
os.makedirs(path_outputs, exist_ok=True)


def calc_threshold_met(df, service, met, filtered=False):
    d = df[df['NUSER'] == 1][met + "_" + service]
    if filtered:
        d = d[df["{}_{}".format(met, service)] > 0]
    if len(d) == 0:
        return 0
    return d.mean() + 3 * d.std()


def get_num_true_example(df_example_true, df_example_false):
    n_issues = len(df_example_true.index)
    n_no_issues = len(df_example_false.index)

    if n_no_issues == 0:
        return K
    elif n_issues == 0:
        return 0
    else:
        perc_issue = n_issues / (n_issues + n_no_issues)
        if perc_issue > (1 / K):
            if perc_issue < ((K - 1) / K):
                return round(perc_issue * K)
            else:
                return K - 1
        else:
            return 1


def get_true_example(df_true_example, num):
    nusers_all = list(set(df_true_example['NUSER']))
    nusers = random.sample(nusers_all, k=num)
    examples = [(0, 0, 0)] * num
    for i, n in enumerate(nusers):
        load = random.choice(list(df_true_example[df_true_example['NUSER'] == n]['LOAD']))
        sr = random.choice(
            list(df_true_example[(df_true_example['NUSER'] == n) & (df_true_example['LOAD'] == load)]['SR']))
        examples[i] = (n, load, sr)
    return examples


def generate_template(df_train, met, ser, ths):
    df_example_true = df_train[(df_train['NUSER'] > 1) & (df_train[met + '_' + ser] > ths)].reset_index(drop=True)
    df_example_false = df_train[(df_train['NUSER'] > 1) & (df_train[met + '_' + ser] <= ths)].reset_index(drop=True)

    N_TRUE_EXAMPLE = get_num_true_example(df_example_true, df_example_false)

    true_examples = get_true_example(df_example_true, N_TRUE_EXAMPLE)
    template = """Answer the question based on the following context:
{context}
Use this following constraints: 
NUSER has to be less than 51
NUSER has to be greater than 4
LOAD has to be 'normal' or 'stress_cart' or 'stress_shop'
SR has to be '1' or '5' or '10'

Follows these examples
"""

    for i, example in enumerate(true_examples):
        template += '{}) CHECK: {}_{} > {}\nYES\nCONFIGURATION: NUSER={},LOAD={},SR={}\n'.format(i + 1, met, ser, ths,
                                                                                                 example[0],
                                                                                                 example[1],
                                                                                                 example[2])
    for i in range(K - len(true_examples)):
        template += '{}) CHECK: {}_{} > {}\nNO\n'.format(i + 1 + len(true_examples), met, ser, ths)

    template += "{question}"
    # template += '{}) ISSUE: {}_{} > {}\nCONFIGURATION: '.format(K + 1, met, ser, ths)
    return template


# fare upper
def parse_output(output):
    output = output.replace(' ', '')
    output_split = [o for o in output.split() if o]

    if output_split[0] == 'YES':
        regex = r'CONFIGURATION:NUSER=\'?\d{1,2}\'?,LOAD=\'?\w+\'?,SR=\'?\d{1,2}\'?'

        match = re.search(regex, output_split[1])
        if match:
            splitted_config = output_split[1].split(sep=',')
            nuser = int(splitted_config[0][splitted_config[0].find('=') + 1:].replace("'", ''))
            load = splitted_config[1][splitted_config[1].find('=') + 1:].replace("'", '')
            sr = int(splitted_config[2][splitted_config[2].find('=') + 1:].replace("'", ''))
            return True, (nuser, load, sr)
    else:
        if output_split[0] == 'NO':
            return True, None
    return False, None


path_chroma = "sockshop_train_chroma_db"
embedding_function = GPT4AllEmbeddings
df_all = pd.read_csv('sockshop_df.csv')
df_all.to_csv('df_train_sockshop.csv', index=False)

if os.path.exists(path_chroma):
    vectorstore = Chroma(persist_directory=path_chroma, embedding_function=embedding_function())
else:
    # df_train = df_all[df_all['NUSER'].isin([i for i in range(1, 30, 2)] + [30])]
    # df_train.to_csv('df_train.csv', index=False)

    loader = CSVLoader(file_path='df_train_sockshop.csv')
    data = loader.load()
    vectorstore = Chroma.from_documents(documents=data, embedding=embedding_function(), persist_directory=path_chroma)

df_train = pd.read_csv('df_train_sockshop.csv')
retriever = vectorstore.as_retriever()

for rep in range(20):
    path_outputs_rep = os.path.join(path_outputs, 'rep' + str(rep) + "/generated_configs")
    if os.path.exists(path_outputs_rep):
        shutil.rmtree(path_outputs_rep)
    os.makedirs(path_outputs_rep)

    services = ["login", "get_catalogue1", "get_catalogue2", "get_catalogue3", "get_item", "get_related", "get_tags",
                "get_cart", "add_item_to_cart", "get_orders", "get_customer", "get_card", "get_address"]

    for met in ['RES_TIME', 'CPU', 'MEM']:
        for ser in services:
            for ret in range(MAX_RETRY):
                ths = calc_threshold_met(df_train, ser, met, False)
                template = generate_template(df_train, met, ser, ths)
                prompt = ChatPromptTemplate.from_template(template)

                model = Ollama(model="phi3", temperature=0, num_predict=25,
                               callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

                # RAG chain
                rag_chain = (
                        RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
                        | prompt
                        | model
                        | StrOutputParser()
                )

                chain = rag_chain.with_types(input_type=str)

                print('\n-------ANSWER-------\n')

                with open(os.path.join(path_outputs_rep, "configs_{}_{}.txt".format(met, ser)), "w") as f:
                    f.write(template + '\n\n')
                    output_model = chain.invoke(
                        input='\ncomplete this example\n{}) CHECK: {}_{} > {}\n'.format(K + 1, met, ser, ths))

                    f.write(output_model)
                    valid, config = parse_output(output_model)
                    if not valid:
                        if ret == MAX_RETRY - 1:
                            print('IMPOSSIBILE GENERARE CONFIGURAZIONE')
                        continue
                    else:
                        if config is not None:
                            with open(os.path.join(path_outputs_rep, "configs_{}_{}.json".format(met, ser)),
                                      "w") as f_json:
                                json.dump({'nusers': [config[0]], 'laods': [config[1]], 'spawn_rates': [config[2]],
                                           "anomalous_metrics": [met]}, f_json)
                        break
