# Histórico de Prompts — MST (Prim Denso), Profiling de Memória, Twice-around-the-tree e Auditoria

Registro curado dos *prompts* técnicos utilizados com assistente de IA (Google Antigravity / Gemini) na frente de desenvolvimento da Árvore Geradora Mínima, utilitários de instrumentação de memória, algoritmo aproximado *Twice-around-the-tree* e auditoria cruzada do projeto, acompanhados das respectivas decisões técnicas e verificações conduzidas pelo grupo.

---

## 1. Planejamento Arquitetural de `twice_around_the_tree` e MST
- **Prompt:** *"Leia o pdf em anexo e a estrutura do repositório já estabelecida. Vamos implementar o algoritmo aproximativo para o TSP twice_around_the_tree. Antes de tudo, apenas analise todo o contexto e planeje o que deve ser feito para esta tarefa, a fim de verificar se o planejamento está adequado."*
- **Auditoria / Decisão:** Validados previamente os contratos em `src/types.py` (`TourResult`) e `src/profiling.py` antes da escrita de código. Descartou-se o uso de `networkx` na construção da MST (evitando materializar $\Theta(N^2)$ arestas em objetos Python e custo $O(N^2 \log N)$) e de `scipy.sparse.csgraph.minimum_spanning_tree` (que converte para matriz esparsa em $O(N^2)$ e trata distância `0.0` de pontos coincidentes como ausência de aresta), definindo-se uma implementação própria, transparente e reutilizável em `christofides` e `src/features.py`.

## 2. Desacoplamento e Implementação da MST via Prim Denso (`src/mst.py`)
- **Prompt:** *"Começe por implementar o algoritmo de MST, em um arquivo separado, utilizando o algoritmo de Prim, usando busca em arrays simples ao invés de realizar o Prim original custoso que precisava de uma fila de prioridade."*
- **Auditoria / Decisão:** Isolada a sub-rotina em módulo dedicado (`src/mst.py`) com `MSTResult` (`parent`, `adjacency`, `total_weight`, `edges()` e `degrees()`), adotando Prim adaptado para grafos densos com busca linear em vetores 1D (`np.argmin`), atingindo tempo $O(N^2)$ e espaço auxiliar $O(N)$ sem copiar `dist_matrix`. Validada a otimalidade contra referência externa em `tests/test_mst.py`, incluindo pontos coincidentes (arestas de peso `0.0`) e invariância à raiz.

## 3. Auditoria de Baixo Nível de Alocações no NumPy (`improved` em `src/mst.py`)
- **Prompt:** *"O vetor extra \"improved\" utilizado na implementação do prim é apenas uma máscara para otimizar a atualização dos vizinhos nas iterações do algoritmo?"*
- **Auditoria / Decisão:** Inspecionado linha a linha o código gerado além dos três vetores teóricos clássicos do algoritmo de Prim (`min_edge_cost`, `parent`, `unvisited`). Confirmou-se que a pré-alocação da máscara booleana `improved` combinada a `np.less(..., out=improved)` e `np.copyto(..., where=improved)` evita a criação de três arrays temporários por iteração na indexação booleana ingênua do NumPy, eliminando alocações transitórias no *heap* e pressão no *garbage collector* dentro do laço — resultando no comportamento espacial estritamente linear $O(N)$ ($R^2 = 0{,}9996$) observado nos experimentos.

## 4. Isolamento de Profiling de Memória e Implementação de `twice_around_the_tree`
- **Prompt:** *"determine track_memory=False para que isso não contamine o restante dos testes. continue o percurso já traçado para a implementação do twice around the tree"*
- **Auditoria / Decisão:** Determinada explicitamente a diretriz `track_memory=False` por padrão em `track_peak_memory` (`src/profiling.py`) para impedir que o *overhead* do `tracemalloc` contaminasse `elapsed_time` e distorcesse as regressões de complexidade temporal, preservando também rastreamentos aninhados. Em `src/approximations.py`, implementou-se `_preorder_dfs` iterativo com pilha explícita em $O(N)$ (validado em árvore caminho degenerada de $N=1500$ sem `RecursionError`) e testes diferenciais em `tests/test_approximations.py` comprovando $\text{MST} \le \text{OPT} \le \text{TATT} \le 2 \cdot \text{MST} \le 2 \cdot \text{OPT}$ contra força bruta.