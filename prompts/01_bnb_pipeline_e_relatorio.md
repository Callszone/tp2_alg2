# Histórico de Prompts — Branch-and-Bound, Pipeline Experimental e Relatório

Registro curado dos *prompts* técnicos utilizados com assistente de IA (Google Antigravity / Gemini) ao longo do desenvolvimento do TP2 de DCC207, acompanhados das respectivas ações de auditoria e refatoração conduzidas pelo grupo.

---

## 1. Planejamento e Estruturação do Repositório
- **Prompt (resumo):** *"Com base no enunciado do TP2 (Branch-and-Bound, Twice-around-the-tree, Christofides, 5 estratégias de warm-start, modelagem estatística e análise de sensibilidade), sugira uma divisão entre 3 integrantes, um fluxo de trabalho até o prazo e prepare a estrutura inicial do repositório em Python 3 para o módulo de Branch-and-Bound."*
- **Auditoria / Decisão:** Definidos contratos comuns em `src/types.py` (`TourResult`, `BnBResult`) e isolamento de medição de memória (`track_memory=False` por padrão) em `src/profiling.py`.

## 2. Projeto e Implementação do Branch-and-Bound
- **Prompt (resumo):** *"Explique didaticamente as decisões de projeto do Branch-and-Bound para TSP e implemente `solve_branch_and_bound` junto com testes diferenciais contra força bruta e validação de timeout."*
- **Auditoria / Decisão:** Adotado DFS com ordenação *nearest-first*, `visited_mask` em *bitmask* (evitando cópias de matriz), limite inferior $O(1)$ incremental (`two_min`) e checagem periódica de *timeout* a cada 1024 nós. Reutilizado `track_peak_memory` de `src/profiling.py` após integração com o repositório remoto.

## 3. Benchmark Comparativo e Segundo Lower Bound (MST)
- **Prompt (resumo):** *"Execute um benchmark do Branch-and-Bound atual, versione o script e o CSV de resultados, e implemente um segundo lower bound baseado em Árvore Geradora Mínima (MST) para comparar poder de poda vs. custo por nó."*
- **Auditoria / Decisão:** Implementado `lower_bound_mode="mst"` em `src/branch_and_bound.py` e `scripts/benchmark_bnb.py`. O comparativo revelou o ponto de cruzamento em $N \ge 20$ (em $N=24$, redução de 59.403 para 434 nós expandidos).

## 4. Seleção de Instâncias TSPLIB (`EUC_2D`) e Parser
- **Prompt (resumo):** *"Implemente o parser TSPLIB `EUC_2D` e selecione de 12 a 16 instâncias conforme a faixa principal ($N \le 40$) e faixa limite ($N \in [45, 75]$)."*
- **Auditoria / Decisão:** Ao inspecionar o arquivo oficial `TSPLIB95`, verificou-se que apenas `eil51`, `berlin52` e `st70` são `EUC_2D` nativas até $N=75$. Optou-se por manter essas 3 na Faixa Limite (20%) e gerar 12 subinstâncias `EUC_2D` determinísticas na Faixa Principal (80%) com `floor(d + 0.5)` no parser.

## 5. Heurísticas de Warm-Start, Features, Runner e Análise Estatística
- **Prompt (resumo):** *"Implemente as heurísticas `cold_start`, `nearest_neighbor` e `convex_hull_insertion` em `src/heuristics.py`, o extrator de métricas geométricas/topológicas em `src/features.py`, o orquestrador paralelo em `src/runner.py` e o pipeline de regressão e sensibilidade em `src/analysis.py`."*
- **Auditoria / Decisão:** Usado *Andrew's Monotone Chain* no fecho convexo para suportar pontos colineares sem falhas numéricas, gravação incremental com `"NA"` para timeouts e testes automatizados em `tests/`.

## 6. Auditoria Crítica e Refatorações Pré-Experimento Oficial
- **Prompt (resumo):** *"Valide os apontamentos de revisão técnica sobre profiling de memória por sub-rotina, diversidade das subinstâncias, complexidade do `convex_hull_insertion`, otimização do Branch-and-Bound, confiabilidade do `runner` e rigor estatístico em `analysis.py`, implementando as correções cabíveis."*
- **Auditoria / Decisão:**
  - Instrumentado `subroutine_memory_bytes` em todas as sub-rotinas críticas.
  - Refatorado `convex_hull_insertion` de $O(N^3)$ para $O(N^2)$ com cache de melhor aresta.
  - Adicionada quebra de simetria de ciclo ($v_1 < v_{N-1}$) e buffers pré-alocados no `mst` do Branch-and-Bound.
  - Diversificadas as 12 subinstâncias com critérios geométricos distintos e calculados seus ótimos exatos via MILP/DFJ.
  - Adicionados *warm-up* + mediana de 15 repetições e *checkpoint/resume* no `runner.py`, além de correlação de Spearman, regressão Ridge e validação cruzada LOOCV em `analysis.py`.

## 7. Redação Técnica e Estruturação do Relatório LaTeX
- **Prompt (resumo):** *"Estruture o relatório em `report/report.tex` com as tabelas e figuras geradas, e revise a clareza das justificativas (como ordenação nearest-first e entropia espacial) antes da compilação final."*
- **Auditoria / Decisão:** Revisão manual de fluidez, remoção de termos e verificação numérica direta contra os CSVs de `results/`.
