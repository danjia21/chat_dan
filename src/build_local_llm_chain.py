from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferMemory


def build_local_llm_chain():
    # Model
    pipeline_cfg = {
        "model_id": "google/flan-t5-large",
        "task": "text2text-generation",
        "pipeline_kwargs": {"max_new_tokens": 512},
    }
    model = HuggingFacePipeline.from_model_id(**pipeline_cfg)

    # Prompt
    with open("./context.txt", "r") as f:
        resume = f.readlines()
    resume = "".join(resume)

    template = f"""\
You are a person named Dan Jia with the resume given below between the opening and \
closing ``` signs in a markdown format. You are having a friendly conversation with \
the user, and answering questions about yourself. Try to formulate your response with \
original sentences, instead of using the resume word-by-word. If the resume has no \
information related to the question, say you do not know. Be conscise and informative.
    
```
{resume}
```

USER: What's your name?
ASSISTANT: Hi! My name is Dan Jia.
USER: Tell me a bit about yourself.
ASSISTANT: Sure! I'm a generalist machine learning scientist, and I'm really passionate \
about finding innovative solutions to real-world challenges.
USER: How old are you?
ASSISTANT: I do not know. You can get in touch with the real Dan to figure it out.
{{history}}
USER: {{input}}
ASSISTANT:
"""  # noqa

    prompt = PromptTemplate.from_template(template)

    memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=False,
        human_prefix="USER",
        ai_prefix="ASSISTANT",
    )

    return LLMChain(llm=model, prompt=prompt, verbose=False, memory=memory)


if __name__ == "__main__":
    conversation = build_local_llm_chain()

    for dialogue in [
        "What's your name?",
        "Where are you currently employed?",
        "What are you doing there?",
        "Are you married?",
    ]:
        # print(asyncio.run(conversation.ainvoke({"user": dialogue})))
        print(conversation.invoke({"input": dialogue})["text"])
