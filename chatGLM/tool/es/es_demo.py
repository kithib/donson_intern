"""
    pip install elasticsearch
"""

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk




from elasticsearch import Elasticsearch

USERNAME = "elastic"
PASSWORD = "dGrpGquemieevI=+H+lu"
ELATICSEARCH_ENDPOINT = "localhost:9200"

url = f'http://{USERNAME}:{PASSWORD}@{ELATICSEARCH_ENDPOINT}'
print("url: " + url)
es = Elasticsearch(url, verify_certs=False)
resp = es.info()
print(resp)

def insert(index,doc):
    es.index(index=index, document=doc)

def batch_insert(index,docs_list):
    bulk(es, docs_list, index=index)

def search(index,body):
    results = es.search(index=index, body=body)
    return results

if __name__ == "__main__":
    # 单条插入
    doc = {
      "title": "Apple iPhone Marketing Docu",
      "content": "The iPhone 13 is the most advanced iPhone ever. It has impressive camera system and great performance.",
      "brand": "Apple",
      "category": "Phone",
      # 其他字段
    }

    # es.index(index="products2", document=doc)

    # 批量插入
    # docs = [
    #   {
    #     "title": "Apple iPhone Marketing Docu",
    #     "content": "The iPhone 13 is the most advanced iPhone ever. It has impressive camera system and great performance.",
    #     "brand": "Apple",
    #     "category": "Phone",
    #     # 其他字段
    #   },
    #   # 其他文档
    # ]
    #
    # # helpers.bulk(es, docs, index="products2")
    # bulk(es, docs, index="products2")


    # 查询
    query = {
      "query": {
        "match_all": {}
      }
    }


    combine_query = {
  "query": {
    "bool": {
      "should": [
        {
          "match": {
            "brand": "Apple"
          }
        },
        {
          "script_score": {
            "query": {
              "match_all": {}
            },
            "script": {
              "source": "cosineSimilarity(params.queryVector, 'product_vector') + 1.0",
              "params": {
                "queryVector": [0.1, 0.2, 0.3, 0.4, 0.5]
              }
            }
          }
        }
      ]
    }
  },
  "sort": [
    {
      "time": {
        "order": "desc"
      }
    }
  ]
}


    script_query ={
        "script_score": {
            "query": {
                "match_all": {
                }
            },
            "script": {
                "params": {
                    "queryVector": [
                        0.1,
                        0.2,
                        0.3,
                        0.4,
                        0.5
                    ]
                },
                "source": "cosineSimilarity(params.queryVector, 'product_vector') + 1.0"
            }
        }
    }

    # results = es.search(index="products2", body=query)
    results =es.search(index="products",body={"size": 3,"query": script_query})


    print(results)

    # rs = search("products",{"size": 3, "query": query})
    rs = search("products",combine_query)
    print('rs',rs)


