from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferMemory

from typing import List
from langchain_core.messages import BaseMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage


# Path the memory class to obtain output in desired format.
class PatchedMemory(ConversationBufferMemory):
    def _buffer_as_str(self, messages: List[BaseMessage]) -> str:
        string_messages = []
        for m in messages:
            if isinstance(m, HumanMessage):
                role = "user"
            elif isinstance(m, AIMessage):
                role = "model"
            else:
                raise ValueError(f"Got unsupported message type: {m}")
            message = f"<start_of_turn>{role}\n{m.content}<end_of_turn>"
            string_messages.append(message)
        return "\n".join(string_messages)


def build_hf_llm_chain(hf_token):
    # repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    # repo_id = "google/gemma-7b-it"
    endpoint_url = (
        "https://api-inference.huggingface.co/models/google/gemma-7b-it"
    )
    model = HuggingFaceEndpoint(
        endpoint_url=endpoint_url,
        task="text-generation",
        # task="text2text-generation",
        # repo_id=repo_id,
        # temperature=0.5,
        huggingfacehub_api_token=hf_token,
        model_kwargs={"max_new_tokens": 1000},
    )

    # Prompt
    with open("./context.txt", "r") as f:
        resume = f.readlines()
    resume = "".join(resume)

    template = f"""\
<start_of_turn>user
You are a person named Dan Jia with the resume given below between the opening and \
closing ``` signs in a markdown format. You are having a friendly conversation with \
the user, and answering questions about yourself. Try to formulate your response with \
original sentences, instead of using the resume word-by-word. If the resume has no \
information related to the question, say you do not know and encourage the user to \
get in touch with the real Dan for the answer. Be conscise and informative.
    
```
{resume}
```

What's your name? <end_of_turn>
<start_of_turn>model
Hi! My name is Dan Jia. <end_of_turn>
<start_of_turn>user
Tell me a bit about yourself. <end_of_turn>
<start_of_turn>model
Sure! I'm a generalist machine learning scientist, and I'm really passionate about \
finding innovative solutions to real-world challenges. <end_of_turn>
<start_of_turn>user
How old are you? <end_of_turn>
<start_of_turn>model
I do not know. You can get in touch with the real Dan to find out. <end_of_turn>
{{history}}
<start_of_turn>user
user: {{input}}
model: <end_of_turn>
<start_of_turn>model
"""  # noqa

    prompt = PromptTemplate.from_template(template)

    memory = PatchedMemory(
        memory_key="history",
        return_messages=False,
    )

    return LLMChain(llm=model, prompt=prompt, verbose=False, memory=memory)


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    conversation = build_hf_llm_chain(os.getenv("HUGGINGFACEHUB_API_TOKEN"))

    for dialogue in [
        "What's your name?",
        "Where are you currently employed?",
        "What are you doing there?",
        "Are you married?",
    ]:
        out = conversation.invoke({"input": dialogue})
        print("Q:", dialogue, "\n", "A:", out["text"], "\n")
