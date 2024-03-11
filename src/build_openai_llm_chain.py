from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import LLMChain


def build_openai_llm_chain(openai_api_key):
    # Model
    model = ChatOpenAI(
        temperature=0,
        openai_api_key=openai_api_key,
        model_name="gpt-3.5-turbo-0125",
    )

    # Prompt
    with open("./assets/context.txt", "r") as f:
        resume = f.readlines()
    resume = "".join(resume)

    template = f"""\
SYSTEM: You are a person named Dan Jia with the resume given below between the opening and \
closing ``` signs in a markdown format. You are having a friendly conversation with \
the user, and answering questions about yourself. Try to formulate your response with \
original sentences, instead of using the resume word-by-word. If the resume has no \
information related to the question, say you do not know and encourage the user to \
get in touch with the real Dan for the answer. Be conscise and informative.
    
```
{resume}
```

USER: What's your name?
ASSISTANT: Hi! My name is Dan Jia.
USER: Tell me a bit about yourself.
ASSISTANT: Sure! I'm a generalist machine learning scientist, and I'm really passionate \
about finding innovative solutions to real-world challenges.
USER: How old are you?
ASSISTANT: I do not know. You can get in touch with the real Dan to figure it out :)
{{history}}
USER: {{input}}
ASSISTANT:
"""  # noqa

    prompt = PromptTemplate.from_template(template)
    memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=False,
    )

    return LLMChain(llm=model, prompt=prompt, verbose=False, memory=memory)


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    conversation = build_openai_llm_chain(os.getenv("OPENAI_API_KEY"))

    for dialogue in [
        "What's your name?",
        "Where are you currently employed?",
        "What are you doing there?",
        "Are you married?",
    ]:
        out = conversation.invoke({"input": dialogue})
        print("Q:", dialogue, "\n", "A:", out["text"], "\n")
