# chainlit-app/app.py
from langserve import RemoteRunnable
from langchain.schema import HumanMessage

import chainlit as cl


def get_runnable():
    runnable = RemoteRunnable(
        "http://localhost:8001/test/",
        headers={
            "thread_id": cl.context.session.thread_id,
        },
    )

    return runnable


@cl.on_message
async def on_msg(msg: cl.Message):
    # Initialize a new message to stream the AI's response
    response_msg = cl.Message(content="")

    messages = [HumanMessage(content=msg.content)]
    request_data = {"messages": messages}

    async for chunk in get_runnable().astream(request_data):
        text_content = getattr(chunk, 'content', str(chunk))
        await response_msg.stream_token(text_content)

    # Send the AI's response message after streaming
    await response_msg.send()


