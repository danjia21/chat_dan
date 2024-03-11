import asyncio
import time
import streamlit as st


def select_llm_model():
    # Ask the user which LLM model should be used
    def reset():
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state.clear()

    options = (
        "OpenAI (gpt-3.5-turbo-0125)",
        "Hugging Face (gemma-7b-it)",
        "Local host (flan-t5-large)",
    )

    option = st.sidebar.selectbox(
        "Which language model would you like to use?",
        options,
        on_change=reset,
        index=None,
        placeholder="Choose a model",
    )

    if option is None:
        st.markdown(
            "Hello and welcome to ChatDan 🤖, an AI chatbot designed to emulate "
            "Dan Jia (that's me) using a Large-Language Model (LLM). Feel free to "
            'engage in conversation with "him". Just remember, while ChatDan '
            "strives to provide accurate information, there's no guarantee of its "
            "accuracy, so take responses with a grain of salt. Also, don't hesitate "
            "to reach out to me, the real Dan 😃 \n\n\n\n\n"
            "Try the following:"
            "\n - Hi there! How are you?"
            "\n - What's your name?"
            "\n - Tell me a bit about yourself.\n\n\n\n\n"
            "Behind the scenes, ChatDan operates by utilizing prompt engineering "
            "and in-context learning with a pre-trained LLM. The prompt includes a "
            "sanitized version of my resume, along with instructions and example "
            "dialogues. If you're curious, you can explore the source code at this "
            "URL: \n\n\n\n\n"
            ":arrow_upper_left: **There are several LLM options available.**\n\n"
            "**Simply select one from the drop-down menu on the left to get started.**"
        )
        return False

    # Use OpenAI
    if option == options[0]:
        openai_api_key = st.sidebar.text_input(
            label="OpenAI API Key",
            type="password",
            value=(
                st.session_state["OPENAI_API_KEY"]
                if "OPENAI_API_KEY" in st.session_state
                else ""
            ),
            placeholder="sk-...",
        )

        if openai_api_key:
            st.session_state["OPENAI_API_KEY"] = openai_api_key
        else:
            st.error("Please add your OpenAI API key to continue.")
            st.info(
                "Obtain your key from this link: https://platform.openai.com/account/api-keys"
            )
            st.error(
                "Please be aware that using the OpenAI API "
                "comes with a cost. You can find detailed pricing information "
                "on the OpenAI website at this link: https://openai.com/pricing. "
                "The prompt for ChatDan comprises roughly 1500 tokens, equating "
                "to a cost of around $0.01 for 10 queries (as of March 9th, 2024)."
            )
            return False

    # Use Hugging Face
    elif option == options[1]:
        hf_api_key = st.sidebar.text_input(
            label="Hugging Face access token",
            type="password",
            value=(
                st.session_state["HF_API_KEY"]
                if "HF_API_KEY" in st.session_state
                else ""
            ),
            placeholder="hf_...",
        )

        if hf_api_key:
            st.session_state["HF_API_KEY"] = hf_api_key
        else:
            st.error("Please add your Hugging Face access token key to continue.")
            st.info(
                "Obtain your access token from this link: https://huggingface.co/settings/tokens."
            )
            st.info(
                "ChatDan leverages the Hugging Face inference API, which is provided free of charge."
            )
            return False

    # Run locally
    else:
        st.sidebar.markdown("This option runs the LLM locally on the app server.")

    return True


def display_linkedin():
    with st.sidebar:
        st.components.v1.html(
            """\
<script src="https://platform.linkedin.com/badges/js/profile.js" \
async defer type="text/javascript"></script>

<div class="badge-base LI-profile-badge" data-locale="en_US" \
data-size="medium" data-theme="light" data-type="VERTICAL" \
data-vanity="dan-jia-ml" data-version="v1">\
<a class="badge-base__link LI-simple-link" \
href="https://de.linkedin.com/in/dan-jia-ml?trk=profile-badge">\
</a>\
</div>
            """,
            height=500,
        )


@st.cache_resource
def build_llm():
    if "OPENAI_API_KEY" in st.session_state:
        from src.build_openai_llm_chain import build_openai_llm_chain

        return build_openai_llm_chain(st.session_state["OPENAI_API_KEY"])
    elif "HF_API_KEY" in st.session_state:
        from src.build_hf_llm_chain import build_hf_llm_chain

        return build_hf_llm_chain(st.session_state["HF_API_KEY"])
    else:
        from src.build_local_llm_chain import build_local_llm_chain

        return build_local_llm_chain()


async def ainvoke_llm_and_display_response(llm, prompt):
    msg_holder = st.empty()

    # Async invoke LLM, display a blinking typing bar while waiting for the result
    t = asyncio.create_task(llm.ainvoke({"input": prompt}))
    while not t.done():
        msg_holder.markdown("|")
        await asyncio.sleep(1.0)
        msg_holder.markdown("")
        await asyncio.sleep(1.0)
    response = t.result()
    response = response["text"]

    # Print result
    for i in range(1, len(response)):
        msg_holder.markdown(response[:i] + "|")
        time.sleep(0.01)
    msg_holder.markdown(response)

    return response


def main():
    # Page setup
    st.title("ChatDan")
    model_selected = select_llm_model()
    display_linkedin()
    if not model_selected:
        st.stop()

    llm = build_llm()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = asyncio.run(ainvoke_llm_and_display_response(llm, prompt))
            st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
