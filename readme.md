Melhorias

1) Implementar botão para Limpar Historico ou Resetar a página.
2) Bloquear campo pesquisa até terminar processamento do arquivo.
3) Testar outros frameworks para inferencia:
https://python.langchain.com/v0.1/docs/guides/development/local_llms/
gpt4all:
ollama:
llamafile: https://huggingface.co/models?other=llamafile
4) Implementar agentes.




OBS: Para rodar o streamlit 1.34.0 e necessario criar a pasta .streamlit/config.toml dentro do projeto e adicinar o
conteudo:
[server]
enableXsrfProtection = false
enableCORS = false
