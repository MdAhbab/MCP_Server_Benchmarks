# GPU run summary

- GPU: NVIDIA GeForce RTX 5070 Ti (15.92 GB), driver 616.64
- torch 2.11.0+cu128, CUDA 12.8, capability 12.0
- NVML energy counter: True
- Intel RAPL: False RAPL is exposed only on Linux

## E1 disclosure energy

- verbose: 3821 input tokens, 613.524 J GPU, 5.76 s
- progressive: 324 input tokens, 342.4832 J GPU, 3.12 s
- verbose_failure_loop: 11465 input tokens, 3304.846 J GPU, 39.06 s

- tokens cut 91.5%, GPU energy cut 44.2%

## E2 prefill against decode

- 324 input tokens: prefill 26.945 J, decode 302.4012 J, prefill share 0.0818
- 1306 input tokens: prefill 62.123 J, decode 339.5298 J, prefill share 0.1547
- 3821 input tokens: prefill 229.941 J, decode 346.9006 J, prefill share 0.3986

## E3 to E5

```
{
  "n_queries": 300,
  "k_budget": 94,
  "e3_query_construction": {
    "ragmcp_description_derived": 1.0,
    "ragmcp_natural": 0.9266666666666666,
    "mcpzero_description_derived": 0.37333333333333335,
    "mcpzero_natural": 0.27
  },
  "e4_mcpzero_server_stage": {
    "n_servers_regenerated": 308,
    "n_servers_requested": 308,
    "recall_published_descriptions": 0.27,
    "recall_model_written_descriptions": 0.2733333333333333
  },
  "e5_eats_natural": {
    "prefill_j_per_token": 0.06363633333333334,
    "rungs": {
      "R": {
        "recall": 0.9266666666666666,
        "tokens": 3767.383333333333,
        "routing_top1": 0.2,
        "joules_per_query": 239.742462
      },
      "C relative": {
        "recall": 0.9266666666666666,
        "tokens": 3767.383333333333,
        "routing_top1": 0.34,
        "joules_per_query": 239.742462
      },
      "+Pareto": {
        "recall": 0.9266666666666666,
        "tokens": 3767.383333333333,
        "routing_top1": 0.44,
        "joules_per_query": 239.742462
      },
      "+Budget": {
        "recall": 0.92,
        "tokens": 2862.2066666666665,
        "routing_top1": 0.44,
        "joules_per_query": 182.140338
      },
      "+Learned": {
        "recall": 0.92,
        "tokens": 2862.2066666666665,
        "routing_top1": 0.44,
        "joules_per_query": 182.140338
      }
    }
  }
}
```