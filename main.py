from transformers import pipeline
from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

# tokens = tokenizer(
#     "heloo this is unbelieveable and comfortable"
# )

# print(tokens)

chatbot = pipeline(
    "text-generation",
    model="microsoft/phi-2"
)

response1 = chatbot(
    "You are a helpful assistant. Answer the following question clearly and briefly:\nwhat is an apple?",
    max_length=100,
    # same prompt different temperatures
    temperature = 0.2
)

response2 = chatbot(
    "You are a helpful assistant. Answer the following question clearly and briefly:\nwhat is an apple?",
    max_length=100,
    # more temp, more creative, less reliable
    temperature = 1.7
)

print(response1[0]["generated_text"])
print(response2[0]["generated_text"])


