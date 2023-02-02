import io
import torch
from transformers import pipeline
from sqlalchemy import create_engine
import numpy as np
from flask import Flask, jsonify, send_file, request
from flask_caching import Cache
import os
import openai
import psycopg2
import sqlalchemy as db
import pandas as pd
import json
from openai.embeddings_utils import get_embedding, cosine_similarity

app: Flask = Flask(__name__)


@app.route('/openChatAI')
def openChatAI():
    prompt = request.args.get('question')
    engine = request.args.get('engine')
    openai.api_key = os.getenv("OPENAI_API_KEY")
    #prompt = "Tenant 10 Expense"
    response = openai.Completion.create(
        #engine="davinci:ft-personal-2023-02-02-17-21-01",
        engine=engine,
        prompt=prompt,
        temperature=0,
        max_tokens=100
    )
    return response['choices'][0]['text']


@app.route('/openAIImage')
def openAIImage():
    openai.api_key = os.getenv("OPENAI_API_KEY")
    # prompt = "Write a tagline for a Construction Software Product"
    prompt = "Image for an AI Product for Construction"
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="1024x1024"
    )
    return response['data'][0]['url']


@app.route('/fetchData')
def fetchData():
    openai.api_key = os.getenv("OPENAI_API_KEY")
    conn = psycopg2.connect(host="localhost", database="postgres", user="postgres", password="lol123321")
    cursor = conn.cursor()
    # cursor.execute('select * from "Florida_prediction_by_city_new" where "_date" = \'22-12\' and "city" = \'Miami Beach\';')
    cursor.execute(
        'select * from "Florida_prediction_by_city_new" where "_date" = \'22-12\' and ("city" = \'Miami Beach\' or "city" = \'Doral\');')

    data = cursor.fetchall()
    # data = [" ".join([row[0], str(row[1]), str(row[2]), str(row[3]), str(row[4])]) for row in data]
    #data = [" ".join([str(row[0]), str(row[1]), str(row[7])]) for row in data]
    data = [" ".join([str(row[i]) for i in range(len(row))]) for row in data]
    data = [d.lower() for d in data]

    file = open('./response.txt', 'w')
    for i in data:
        file.write(i + "\n")
    file.close()

    return "Data Fetched Successfully."
    #return data


@app.route('/promptAI')
def promptAI():
    question = request.args.get('question')
    print(question)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    conn = psycopg2.connect(host="localhost", database="postgres", user="postgres", password="lol123321")
    cursor = conn.cursor()
    # cursor.execute('select * from "Florida_prediction_by_city_new" where "_date" = \'22-12\' and "city" = \'Miami Beach\';')
    cursor.execute(
        'select * from "Florida_prediction_by_city_new" where "_date" = \'22-12\' and ("city" = \'Miami Beach\' or "city" = \'Doral\');')

    data = cursor.fetchall()
    # data = [" ".join([row[0], str(row[1]), str(row[2]), str(row[3]), str(row[4])]) for row in data]
    #data = [" ".join([str(row[0]), str(row[1]), str(row[7])]) for row in data]
    data = [" ".join([str(row[i]) for i in range(len(row))]) for row in data]
    data = [d.lower() for d in data]

    df = pd.read_csv(r'C:\Users\kaush\PycharmProjects\OpenChatAI\budget.csv', header=0)
    print(df)
    data1 = df.T.reset_index().values.T.tolist()
    data1 = [" ".join([str(row[i]) for i in range(len(row))]) for row in data1]
    data1 = [d.lower() for d in data1]
    print(data1)

    #prompt = "\n".join(data1) + "\nWhat is the Budget of Tenant 40"
    prompt = "\n".join(data1) + "\n" + question
    completion = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=1,
        max_tokens=100
    )

    #return completion
    return completion['choices'][0]['text']

'''
    model_engine = completion.model
    # openai.Completion.create(engine=model_engine, contents=completion['choices'][0]['text'])
    model = openai.Completion.create(engine=model_engine, prompt=prompt, temperature=0.5,
                                     max_tokens=2048, top_p=1, frequency_penalty=1, presence_penalty=1)
    model.save(model_engine + "-finetuned")=

    model = openai.Model.load(model_engine + "-finetuned")
    prompt1 = "What is the ppsf of Miami Beach"
    completion1 = openai.Completion.create(
        model=model,
        prompt=prompt1,
        temperature=1,
        max_tokens=100
    )
'''
    #return completion['choices'][0]['text']


@app.route('/trainAI')
def trainAI():
    openai.api_key = os.getenv("OPENAI_API_KEY")
    conn = psycopg2.connect(host="localhost", database="postgres", user="postgres", password="lol123321")
    cursor = conn.cursor()
    # cursor.execute('select * from "Florida_prediction_by_city_new" where "_date" = \'22-12\' and ("city" = \'Miami Beach\' or "city" = \'Doral\');')
    query = 'select city, "Population_Growth" as population_growth from "Florida_prediction_by_city_new"' \
            'where "_date" = \'22-12\' and "city" = \'Miami Beach\';'
    df = pd.read_sql(query, con=conn)

    conn.close()

    #df['context'] = df.city + "\n" + df.price_low.astype(str) + "\n\n" + df.price.astype(
    #    str) + "\n\n" + df.price_high.astype(str)
    df['context'] = df.city + "\n" + df.population_growth.astype(str)

    def get_questions(context):
        question = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Write questions based on the text below\n\nText: {df['context']}\n\nQuestions:\n1.",
            temperature=0.5,
            max_tokens=257,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n\n"]
        )

        return question['choices'][0]['text']

    def get_answers(row):
        answer = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Write questions based on the text below\n\nText: {row.context}\n\nQuestions:\n{row.questions}\n\nAnswers:\n1.",
            temperature=0.5,
            max_tokens=257,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n\n"]
        )

        return answer['choices'][0]['text'] + "\n\n" + answer['choices'][0]['text']

    df['questions'] = df.context.apply(get_questions)
    df['questions'] = "1." + df.questions
    print(df[['questions']].values[0][0])

    df['answers'] = df.apply(get_answers, axis=1)
    df['answers'] = "1. " + df.answers
    df = df.dropna().reset_index().drop('index', axis=1)
    print(df[['answers']].values[0][0])

    print(df['city'])
    print(df['questions'])
    print(df['answers'])
    return df[['questions']].values[0][0] + "\n\n" + df[['answers']].values[0][0]


@app.route('/csvGeneration')
def csvGeneration():
    conn = psycopg2.connect(host="localhost", database="postgres", user="postgres", password="lol123321")
    cur = conn.cursor()
    #postgres_str = 'postgresql://postgres:lol123321@localhost:5432/postgres'
    #cnx = create_engine(postgres_str)
    cur.execute('select city, dom_low from "Florida_prediction_by_city_new" where "_date" = \'22-12\';')
    data = cur.fetchall()
    #query = 'select city, "Population_Growth" as population_growth from "Florida_prediction_by_city_new" where "_date" = \'22-12\';'
    df = pd.DataFrame(data, columns=["city", "dom_low"])
    #df = pd.read_sql_query(query, con=cnx)

    conn.close()

    # df['context'] = df.city + "\n" + df.price_low.astype(str) + "\n\n" + df.price.astype(
    #    str) + "\n\n" + df.price_high.astype(str)
    df['prompt'] = df.city.astype(str)
    df['completion'] = df.dom_low.astype(str)

    df.to_csv('cities.csv', encoding='utf-8', index=False, header=True)

    return send_file(
        io.BytesIO(df.to_csv(index=False, encoding='utf-8').encode()),
        as_attachment=True,
        attachment_filename='cities.csv',
        mimetype='text/csv',
        cache_timeout=0)


@app.route('/tableQA')
def tableQA():
    tqa = pipeline(task="table-question-answering",
                   model="google/tapas-base-finetuned-wtq")
    df = pd.read_csv("sample.csv")
    df = df.astype(str)

    query = "Which City has the Highest Population Growth?"

    return tqa(table=df, query=query)["answer"]


app.run()
