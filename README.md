# Thoughtful AI Customer Support Agent

A CLI chatbot that answers questions about Thoughtful AI's agents using semantic
similarity over a predefined dataset, with an LLM fallback for everything else.

See the [Developer Notes](#developer-notes) for some information about how this
tool was made, opinions, etc.

## Setup

    cp .env.example .env        # add your OPENAI_API_KEY
    uv sync
    uv run chat

## Configuration

| Variable               | Default         | Description                        |
|------------------------|-----------------|------------------------------------|
| `OPENAI_API_KEY`       | —               | Enables live embeddings and LLM    |

Without `OPENAI_API_KEY`, the agent uses TF-IDF embeddings and a stub LLM.

## Developer Notes

This chatbot was made in response to a technical screening question on a job interview.
(I assume if you're reading this, I'm being screened?) The screen didn't say anything about
not using an AI agent, so I did, but I'd like to give a quick summary of that process
to give insight on how it went, and the choices I made.

I used Claude 4.6 running in OpenCode. Here's a rundown of things I told the agent:

```
Here's a prompt:

<Here, I pasted the full text from the screening tool>

I'd like you to make the most minimal version of this possible, while maintaining clear separation of code in file sructures. Be sure to separate your data layer cleanly.

The version of this that I would like initially is a LangChain based application in Python, which runs simply as a command-line chat. If possible, even though we're just using a CLI, maintain a basic MVC pattern so that other UIs could potentially be pluggable, but your abstraction layers should be incredibly miminalistic.
---
Use `uv` for dependency management and create a package structure.
---
Use uv for dependency and package management. Create a pyproject.toml and a standard package structure.
---
Go for it. Remember to be as minimal as is absolutely possible.
---
Let's use the langchain similarity API, rather than doing our own implementation.
---
Great. Now let's provide a wrapper prompt for the fallback path where search produces no results.
---
Great. Now, I'm wondering if there is a way to provide mocked or naive providers in the case that the OpenAI token isn't set.
---
OK, one more idea. Instead of _HashEmbeddings which is essentially meaningless, can we create some kind of BoW embeddings that would at least possibly be in the ballpark?
---
Use sklearn
---
Great. Let's allow users to set the threshold with an env var, but let's actually make some default values so that if no OpenAI key is set, we use a lower threshold, and if a key is present, we use a higher one. Give me a plan for how you'd do this?
---
Great. Please write a VERY CONCISE README with instructions for basic use of this tool.
---
Stub behavior can be minimally mentioned as you do. Get rid of project structure.
---
Great. Please print out for me all the messages that I've sent to you, each one separated by a blank line.
```

You may have noticed from the above workflow that the initial implementation vibe-rolled a cosine similarity score rather than using the LangChain API, and that there wasn't initially any sort of functional fallback when there was no OpenAI API key. Then, oncea fallback was created, it was hash-based, which would have met type requirements, but would have been meaningless. I suggested we make some minimal effort with Bag of Words (BoW), and the model did me one better and suggested TF-IDF. Since sklearn has some simple abstractions for these (and since I didn't mind adding that dependency), we got the current version.


