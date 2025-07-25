from langchain.prompts import PromptTemplate

template = "What is a good name for a company that makes {product}?"
prompt = PromptTemplate.from_template(template)
print(prompt.format(product="chatbots"))
