# IBM Hack Challenge

## User Query on Stack Overflow

Stack Overflow is a question and answer site for professional and enthusiast programmers.
It's built and run by the community of developer, as part of the
Stack Exchange network of Q&A sites. A lot of content is present in form of stack overflow
questions and answers, various studies point that developers face problems while
development life cycles and they ask questions on stack overflow which gets answered by
fellow developers across the globe.
In order for a new developer to understand a concept or solve an issue, it is very difficult to
identify the problems. It involves domain experts in form of experienced software
developers. The information present is overwhelming and at times can be too much to
handle for a budding developer.

# Goals

- To identify most relevant questions to a query [text similarity]
- Identify the matching tags and pick top relevant questions from stack overflow.
- To identify top k solutions of the problem.
  - (sentiment analysis of review content)

# Features

- Huge set of questions from API
- Further filtering of Questions using Universal Sentence Encoder
- This app also provides **Links for top answers**. 
- For top-k answers app displays **sentiments score and upvotes** along with the ans link
- Colouring is also done based on sentiments on how reliable the answer is based on the feedback of each answer.

## Getting Started

- These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

## Prerequisites

Things you need to install the software and instructions to install them:

- You should have python3 and pip3 installed:

- First check if python3 is intstalled by using the command:

  - ``` python3 -V ```

- Output:

  - ```Python 3.5.2```

- If not installed then type the following:
  ```bash
  sudo apt-get update
  sudo apt-get -y update
  sudo apt-get install python3-pip
  ```

## Instructions for installing few more packages and development tools for the programming environment

``` sudo apt-get install build-essential libssl-dev libffi-dev python-dev```

### Installing

- Getting our web app up and running is simple. 
- To run it you have to do what you do with most flask apps.

### Setting up the Github repository

```
git clone https://github.com/yudhik11/StackOverflow_UserQuery
```

### Additional Files to be downloaded [Pre-trained models]


- Download the utils folder from this [link](https://drive.google.com/file/d/1ESW6s3n58zo6kizs7OS_aMwpEUB-Ekq8/view?usp=sharing)

- Download the tf folder from this [link](https://drive.google.com/file/d/1v8CnvqLt6kruzYBF7kl0pwc-qJ-2vqEk/view?usp=sharing)


- Unzip the files:
  - place the **utils** folder to app/
  - **tf** folder to app/home/   

### Setting up virtual environment:

```bash
virtualenv -p /usr/bin/python3 name_of_your_environment
source name_of_your_environment/bin/activate
pip3 install -r requirements.txt
```

### Setting API Keys
* You need API keys to access stackoverflow APIs and an IBM Watson Sentiment Analyser key.
* Set the environment variables for the corresponding keys  
```bash
export IBM_WATSON_KEY=<your_ibm_watson_api_key>
export STACK_API_KEY=<your_stackoverflow_api_key>
```

### Some NLTK packages from python3 interpreter

```python
>>> import nltk
>>> nltk.download('stopwords')
>>> nltk.download('punkt')
```

### Initialize the database
```bash
python3 manage.py migrate
```
### Running the web app :

```bash
 python3 manage.py runserver # The web app can be run in http://127.0.0.1:8000
```






