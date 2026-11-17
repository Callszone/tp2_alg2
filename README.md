# tp2_alg2

Trabalho Prático 2 — DCC207 (Algoritmos 2, UFMG)  
**Análise Empírica e Otimização para o Caixeiro Viajante (TSP Geométrico)**

## Estrutura do Repositório

```text
tp2_alg2/
├── data/tsplib/          # Instâncias .tsp (EUC_2D) e soluções ótimas de referência
├── prompts/              # Histórico de prompts de LLM (anexo obrigatório)
├── report/               # Artigo científico em LaTeX (padrão SBC) e figuras
├── results/              # CSVs de métricas, profiling e modelagem
├── src/
│   ├── types.py          # Contratos de saída (TourResult, BnBResult)
│   ├── tsplib_parser.py  # Leitura de instâncias EUC_2D e matriz de distâncias
│   ├── heuristics.py     # Warm-start: Cold Start, Nearest Neighbor, Convex Hull
│   ├── approximations.py # Twice-around-the-tree e Christofides
│   ├── branch_and_bound.py # Algoritmo exato Branch-and-Bound
│   ├── features.py       # Extração de métricas geométricas/topológicas
│   ├── profiling.py      # Instrumentação de tempo e memória por subrotina
│   ├── runner.py         # Orquestração dos experimentos (timeout 30 min)
│   └── analysis.py       # Regressões assintóticas e análise de sensibilidade
├── tests/                # Testes unitários e validação de rotas
└── requirements.txt
```

## Setup (Python 3)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Contrato Comum de Dados

- **Entrada dos algoritmos:** `dist_matrix: np.ndarray` de formato `(N, N)` (e `coords: np.ndarray` `(N, 2)` quando necessário, ex: Convex Hull).
- **Rota (`tour`):** lista de índices `0..N-1` representando a ordem de visita (sem repetir o vértice inicial no final; o fechamento `tour[-1] -> tour[0]` é implícito no custo).
- **Timeout do B&B:** 1800 segundos (30 min). Em caso de estouro, `timed_out=True` e `elapsed_time=None` (gravado como `NA` no CSV), mantendo `nodes_expanded`, `nodes_pruned` e `max_depth`.
