# ChatDan 🤖

ChatDan is an AI chatbot designed to emulate Dan Jia's persona using a Large-Language Model (LLM).

## Quick Start

1. Clone the repository and install required Python packages:

    ```bash
    git clone https://github.com/danjia21/chat_dan.git
    cd chat_dan
    pip3 install -r requirements.txt
    ```

2. Launch the Streamlit application:

    ```bash
    streamlit run app.py
    ```

3. If you want to test running flan-t5-large locally on your machine, ensure you've downloaded the pre-trained model weights from Hugging Face before launching the application using the following script. Models on Hugging Face are hosted with Git LFS, so make sure it is installed:

    ```bash
    ./download_weights.sh  # Requires git-lfs
    ```

4. Alternatively, a Dockerfile is provided if you prefer to run ChatDan in a container:

    ```bash
    docker build -t chatdan .
    docker run -p 8051:8051 chatdan
    ```

Enjoy conversing with ChatDan!
