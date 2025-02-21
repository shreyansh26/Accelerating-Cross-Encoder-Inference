---
pipeline_tag: text-classification
tags:
- transformers
- reranker
- cross-encoder
- transformers.js
language:
- multilingual
inference: false
license: cc-by-nc-4.0
library_name: transformers
---

<br><br>

<p align="center">
<img src="https://huggingface.co/datasets/jinaai/documentation-images/resolve/main/logo.webp" alt="Jina AI: Your Search Foundation, Supercharged!" width="150px">
</p>

<p align="center">
<b>Trained by <a href="https://jina.ai/"><b>Jina AI</b></a>.</b>
</p>

# jina-reranker-v2-base-multilingual

## Intended Usage & Model Info

The **Jina Reranker v2** (`jina-reranker-v2-base-multilingual`) is a transformer-based model that has been fine-tuned for text reranking task, which is a crucial component in many information retrieval systems. It is a cross-encoder model that takes a query and a document pair as input and outputs a score indicating the relevance of the document to the query. The model is trained on a large dataset of query-document pairs and is capable of reranking documents in multiple languages with high accuracy.

Compared with the state-of-the-art reranker models, including the previous released `jina-reranker-v1-base-en`, the **Jina Reranker v2** model has demonstrated competitiveness across a series of benchmarks targeting for text retrieval, multilingual capability, function-calling-aware and text-to-SQL-aware reranking, and code retrieval tasks.

The `jina-reranker-v2-base-multilingual` model is capable of handling long texts with a context length of up to `1024` tokens, enabling the processing of extensive inputs. To enable the model to handle long texts that exceed 1024 tokens, the model uses a sliding window approach to chunk the input text into smaller pieces and rerank each chunk separately.

The model is also equipped with a flash attention mechanism, which significantly improves the model's performance.


# Usage

_This model repository is licenced for research and evaluation purposes under CC-BY-NC-4.0. For commercial usage, please refer to Jina AI's APIs, AWS Sagemaker or Azure Marketplace offerings. Please [contact us](https://jina.ai/contact-sales) for any further clarifications._
1. The easiest way to use `jina-reranker-v2-base-multilingual` is to call Jina AI's [Reranker API](https://jina.ai/reranker/).

```bash
curl https://api.jina.ai/v1/rerank \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
  "model": "jina-reranker-v2-base-multilingual",
  "query": "Organic skincare products for sensitive skin",
  "documents": [
    "Organic skincare for sensitive skin with aloe vera and chamomile.",
    "New makeup trends focus on bold colors and innovative techniques",
    "Bio-Hautpflege für empfindliche Haut mit Aloe Vera und Kamille",
    "Neue Make-up-Trends setzen auf kräftige Farben und innovative Techniken",
    "Cuidado de la piel orgánico para piel sensible con aloe vera y manzanilla",
    "Las nuevas tendencias de maquillaje se centran en colores vivos y técnicas innovadoras",
    "针对敏感肌专门设计的天然有机护肤产品",
    "新的化妆趋势注重鲜艳的颜色和创新的技巧",
    "敏感肌のために特別に設計された天然有機スキンケア製品",
    "新しいメイクのトレンドは鮮やかな色と革新的な技術に焦点を当てています"
  ],
  "top_n": 3
}'
```

2. You can also use the `transformers` library to interact with the model programmatically.

Before you start, install the `transformers` and `einops` libraries:

```bash
pip install transformers einops
```

And then:
```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    'jinaai/jina-reranker-v2-base-multilingual',
    torch_dtype="auto",
    trust_remote_code=True,
)

model.to('cuda') # or 'cpu' if no GPU is available
model.eval()

# Example query and documents
query = "Organic skincare products for sensitive skin"
documents = [
    "Organic skincare for sensitive skin with aloe vera and chamomile.",
    "New makeup trends focus on bold colors and innovative techniques",
    "Bio-Hautpflege für empfindliche Haut mit Aloe Vera und Kamille",
    "Neue Make-up-Trends setzen auf kräftige Farben und innovative Techniken",
    "Cuidado de la piel orgánico para piel sensible con aloe vera y manzanilla",
    "Las nuevas tendencias de maquillaje se centran en colores vivos y técnicas innovadoras",
    "针对敏感肌专门设计的天然有机护肤产品",
    "新的化妆趋势注重鲜艳的颜色和创新的技巧",
    "敏感肌のために特別に設計された天然有機スキンケア製品",
    "新しいメイクのトレンドは鮮やかな色と革新的な技術に焦点を当てています",
]

# construct sentence pairs
sentence_pairs = [[query, doc] for doc in documents]

scores = model.compute_score(sentence_pairs, max_length=1024)
```

The scores will be a list of floats, where each float represents the relevance score of the corresponding document to the query. Higher scores indicate higher relevance.
For instance the returning scores in this case will be:
```bash
[0.8311430811882019, 0.09401018172502518,
 0.6334102749824524, 0.08269733935594559,
 0.7620701193809509, 0.09947021305561066,
 0.9263036847114563, 0.05834583938121796,
 0.8418256044387817, 0.11124119907617569]
```

The model gives high relevance scores to the documents that are most relevant to the query regardless of the language of the document.

Note that by default, the `jina-reranker-v2-base-multilingual` model uses [flash attention](https://github.com/Dao-AILab/flash-attention), which requires certain types of GPU hardware to run.
If you encounter any issues, you can try call `AutoModelForSequenceClassification.from_pretrained()` with `use_flash_attn=False`.
This will use the standard attention mechanism instead of flash attention.

If you want to use flash attention for fast inference, you need to install the following packages:
```bash
pip install ninja # required for flash attention
pip install flash-attn --no-build-isolation
```
Enjoy the 3x-6x speedup with flash attention! ⚡️⚡️⚡️


3. You can also use the `transformers.js` library to run the model directly in JavaScript (in-browser, Node.js, Deno, etc.)!

If you haven't already, you can install the [Transformers.js](https://huggingface.co/docs/transformers.js) JavaScript library (v3) using:
```bash
npm i xenova/transformers.js#v3
```

Then, you can use the following code to interact with the model:
```js
import { AutoTokenizer, XLMRobertaModel } from '@xenova/transformers';

const model_id = 'jinaai/jina-reranker-v2-base-multilingual';
const model = await XLMRobertaModel.from_pretrained(model_id, { dtype: 'fp32' });
const tokenizer = await AutoTokenizer.from_pretrained(model_id);

/**
 * Performs ranking with the CrossEncoder on the given query and documents. Returns a sorted list with the document indices and scores.
 * @param {string} query A single query
 * @param {string[]} documents A list of documents
 * @param {Object} options Options for ranking
 * @param {number} [options.top_k=undefined] Return the top-k documents. If undefined, all documents are returned.
 * @param {number} [options.return_documents=false] If true, also returns the documents. If false, only returns the indices and scores.
 */
async function rank(query, documents, {
    top_k = undefined,
    return_documents = false,
} = {}) {
    const inputs = tokenizer(
        new Array(documents.length).fill(query),
        { text_pair: documents, padding: true, truncation: true }
    )
    const { logits } = await model(inputs);
    return logits.sigmoid().tolist()
        .map(([score], i) => ({
            corpus_id: i,
            score,
            ...(return_documents ? { text: documents[i] } : {})
        })).sort((a, b) => b.score - a.score).slice(0, top_k);
}

// Example usage:
const query = "Organic skincare products for sensitive skin"
const documents = [
    "Organic skincare for sensitive skin with aloe vera and chamomile.",
    "New makeup trends focus on bold colors and innovative techniques",
    "Bio-Hautpflege für empfindliche Haut mit Aloe Vera und Kamille",
    "Neue Make-up-Trends setzen auf kräftige Farben und innovative Techniken",
    "Cuidado de la piel orgánico para piel sensible con aloe vera y manzanilla",
    "Las nuevas tendencias de maquillaje se centran en colores vivos y técnicas innovadoras",
    "针对敏感肌专门设计的天然有机护肤产品",
    "新的化妆趋势注重鲜艳的颜色和创新的技巧",
    "敏感肌のために特別に設計された天然有機スキンケア製品",
    "新しいメイクのトレンドは鮮やかな色と革新的な技術に焦点を当てています",
]

const results = await rank(query, documents, { return_documents: true, top_k: 3 });
console.log(results);
```


That's it! You can now use the `jina-reranker-v2-base-multilingual` model in your projects.


In addition to the `compute_score()` function, the `jina-reranker-v2-base-multilingual` model also provides a `model.rerank()` function that can be used to rerank documents based on a query. You can use it as follows:

```python
result = model.rerank(
    query,
    documents,
    max_query_length=512,
    max_length=1024,
    top_n=3
)
```

Inside the `result` object, you will find the reranked documents along with their scores. You can use this information to further process the documents as needed.

The `rerank()` function will automatically chunk the input documents into smaller pieces if they exceed the model's maximum input length. This allows you to rerank long documents without running into memory issues.
Specifically, the `rerank()` function will split the documents into chunks of size `max_length` and rerank each chunk separately. The scores from all the chunks are then combined to produce the final reranking results. You can control the query length and document length in each chunk by setting the `max_query_length` and `max_length` parameters. The `rerank()` function also supports the `overlap` parameter (default is `80`) which determines how much overlap there is between adjacent chunks. This can be useful when reranking long documents to ensure that the model has enough context to make accurate predictions.

3. Alternatively, `jina-reranker-v2-base-multilingual` has been integrated with `CrossEncoder` from the `sentence-transformers` library.

Before you start, install the `sentence-transformers` libraries:

```bash
pip install sentence-transformers
```

The [`CrossEncoder`](https://sbert.net/docs/package_reference/cross_encoder/cross_encoder.html) class supports a [`predict`](https://sbert.net/docs/package_reference/cross_encoder/cross_encoder.html#sentence_transformers.cross_encoder.CrossEncoder.predict) method to get query-document relevance scores, and a [`rank`](https://sbert.net/docs/package_reference/cross_encoder/cross_encoder.html#sentence_transformers.cross_encoder.CrossEncoder.rank) method to rank all documents given your query.

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder(
    "jinaai/jina-reranker-v2-base-multilingual",
    automodel_args={"torch_dtype": "auto"},
    trust_remote_code=True,
)

# Example query and documents
query = "Organic skincare products for sensitive skin"
documents = [
    "Organic skincare for sensitive skin with aloe vera and chamomile.",
    "New makeup trends focus on bold colors and innovative techniques",
    "Bio-Hautpflege für empfindliche Haut mit Aloe Vera und Kamille",
    "Neue Make-up-Trends setzen auf kräftige Farben und innovative Techniken",
    "Cuidado de la piel orgánico para piel sensible con aloe vera y manzanilla",
    "Las nuevas tendencias de maquillaje se centran en colores vivos y técnicas innovadoras",
    "针对敏感肌专门设计的天然有机护肤产品",
    "新的化妆趋势注重鲜艳的颜色和创新的技巧",
    "敏感肌のために特別に設計された天然有機スキンケア製品",
    "新しいメイクのトレンドは鮮やかな色と革新的な技術に焦点を当てています",
]

# construct sentence pairs
sentence_pairs = [[query, doc] for doc in documents]

scores = model.predict(sentence_pairs, convert_to_tensor=True).tolist()
"""
[0.828125, 0.0927734375, 0.6328125, 0.08251953125, 0.76171875, 0.099609375, 0.92578125, 0.058349609375, 0.84375, 0.111328125]
"""

rankings = model.rank(query, documents, return_documents=True, convert_to_tensor=True)
print(f"Query: {query}")
for ranking in rankings:
    print(f"ID: {ranking['corpus_id']}, Score: {ranking['score']:.4f}, Text: {ranking['text']}")
"""
Query: Organic skincare products for sensitive skin
ID: 6, Score: 0.9258, Text: 针对敏感肌专门设计的天然有机护肤产品
ID: 8, Score: 0.8438, Text: 敏感肌のために特別に設計された天然有機スキンケア製品
ID: 0, Score: 0.8281, Text: Organic skincare for sensitive skin with aloe vera and chamomile.
ID: 4, Score: 0.7617, Text: Cuidado de la piel orgánico para piel sensible con aloe vera y manzanilla
ID: 2, Score: 0.6328, Text: Bio-Hautpflege für empfindliche Haut mit Aloe Vera und Kamille
ID: 9, Score: 0.1113, Text: 新しいメイクのトレンドは鮮やかな色と革新的な技術に焦点を当てています
ID: 5, Score: 0.0996, Text: Las nuevas tendencias de maquillaje se centran en colores vivos y técnicas innovadoras
ID: 1, Score: 0.0928, Text: New makeup trends focus on bold colors and innovative techniques
ID: 3, Score: 0.0825, Text: Neue Make-up-Trends setzen auf kräftige Farben und innovative Techniken
ID: 7, Score: 0.0583, Text: 新的化妆趋势注重鲜艳的颜色和创新的技巧
"""
```

# Evaluation

We evaluated Jina Reranker v2 on multiple benchmarks to ensure top-tier performance and search relevance.

|           Model Name            |   Model Size | MKQA(nDCG@10, 26 langs) 	| BEIR(nDCG@10, 17 datasets) 	| MLDR(recall@10, 13 langs) | CodeSearchNet (MRR@10, 3 tasks) 	| AirBench (nDCG@10, zh/en) 	| ToolBench (recall@3, 3 tasks) 	| TableSearch (recall@3) 	|
| :-----------------------------: | :----------: | ------------------------- | ---------------------------- | --------------------------- | --------------------------------- | --------------------------- | ------------------------------- | ------------------------ |
| jina-reranker-v2-multilingual 	|    278M    	|          54.83          	|            53.17           	|           68.95           	|              71.36              	|           61.33           	|             77.75             	|          93.31         	|
|       bge-reranker-v2-m3      	|    568M    	|          54.17          	|            53.65           	|           59.73           	|              62.86              	|           61.28           	|             78.46             	|          74.86         	|
|  mmarco-mMiniLMv2-L12-H384-v1 	|    118M    	|          53.37          	|            45.40           	|           28.91           	|              51.78              	|           56.46           	|             58.39             	|          53.60         	|
|    jina-reranker-v1-base-en   	|    137M    	|            -            	|            52.45           	|             -             	|                -                	|             -             	|             74.13             	|          72.89         	|

Note:
- NDCG@10 and MRR@10 measure ranking quality, with higher scores indicating better search results
- recall@3 measures the proportion of relevant documents retrieved, with higher scores indicating better search results