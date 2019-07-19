# IBM Hack Challenge

## User Query on Stack Overflow

# Goals

- 

# Features

- 

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

### Setting up virtual environment:

```bash
virtualenv -p /usr/bin/python3 name_of_your_environment
source name_of_your_environment/bin/activate
pip3 install -r requirements.txt
```

### Running the web app :
```bash
python3 manage.py runserver # The web app can be run in http://127.0.0.1:8000
```






