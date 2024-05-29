Melhorias

1) Implementar botão para Limpar Historico ou Resetar a página.
2) Implementar seleção de texto ou documento automaticamente.
   OBS: Quando extraído como documento é apresentado o erro: "expected string or bytes-like object, got 'list'".
3) Bloquear campo pesquisa até terminar processamento do arquivo.
4) Testar outros frameworks para inferencia:
https://python.langchain.com/v0.1/docs/guides/development/local_llms/
gpt4all:
ollama:
llamafile: https://huggingface.co/models?other=llamafile
5) Implementar agentes.




OBS: Para rodar o streamlit 1.34.0 e necessario criar a pasta .streamlit/config.toml dentro do projeto e adicinar o
conteudo:
[server]
enableXsrfProtection = false
enableCORS = false


Executar: streamlit run main.py build with-cuda torch_use_cuda_dsa 