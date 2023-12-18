from es_demo import *
from similarity_demo import *

#ES_INDEX_NAME = "test_products_ys"
ES_INDEX_NAME = "kit_test_newdata"


def load_data():
    data_path = r"/home/kit/kit/pangu/end_data.json"
    with open(data_path, 'r', encoding='utf8') as f:
        data = f.readlines()
    print(f'raw data loaded, len:{len(data)}')
    data = [eval(d) for d in data]
    print(len(data))
    for dictionary in data:  
        for key, value in dictionary.items():  
            if value is None:  
                dictionary[key] = []  
    return data


def prepare_es_data(data):
    cnt = 0
    for item in data:
        ctime = '20231125'
        title = item['title']
        content = item['content']
        brand = item['brand_name_list'][1:-1]  # 去掉[]
        category = item['category_name_list'][1:-1]  # 去掉[]
        # category = item['brand_3class']  # 去掉[]
        feature = item['baichuan2_keyword_clean']
        crowd = item['crowd_keyword_list'][1:-1]  # 去掉[]
        fmt_text = f"品牌：{brand}，品类：{category}，卖点：{feature}，受众人群：{crowd}"
        product_vector = get_embedding(fmt_text)

        doc = {"title": title, "content": content, "brand": brand, "category": category, "feature": feature,
               "crowd": crowd, "ctime": ctime, "product_vector": product_vector}
        insert(ES_INDEX_NAME, doc)

        cnt += 1
        print(f'insert cnt:{cnt}')
        # if cnt == 100:
        #     print('注意：当前只加载100条数据用于测试，生产时需要删除这段代码！！！')
        #     break


def init_es():
    data = load_data()
    prepare_es_data(data)

# 输入：品牌、品类、卖点、受众人群
def combine_search(_doc, topn=3,fields=None,must=['category']):
    mapping = {"brand": "品牌", "category": "品类", "feature": "卖点", "crowd": "受众人群"}
    fields = ["title", "content", "time"] if not fields else fields
    query_fmt = "，".join([f"{mapping[k]}：{_doc[k]}" for k in _doc])
    #print(f"query_fmt:{query_fmt}")
    query_vector = get_embedding(query_fmt)
    must_list = [{"match":{f"{k}":f"{_doc[k]}"}} for k in must]  # 交，且

    should_list = []  # 并，或
    match_list = [{"match":{f"{k}":f"{_doc[k]}"}} for k in _doc if k not in must]  # must加了，should不要重复加

    script_vector = {"script_score": {
        "query": {
            "match_all": {}
        },
        "script": {
            "source": "cosineSimilarity(params.queryVector, 'product_vector') + 1.0",
            "params": {
                "queryVector": query_vector
            }
        }
    }}
    # should_list.extend(match_list)  # 加入字段  # TODO 不要关键词并集，只留must和向量

    should_list.append(script_vector)  # 加入向量搜索

    body_fmt = {
        "size": topn,
        "_source": fields,

        "query": {
            "bool": {
                "must":must_list,
                "should": should_list
            }
        },
        "sort": [
            {
                "_score": {
                    "order": "desc"
                }
            },
            {
                "ctime": {
                    "order": "desc"
                }
            }
        ]
    }
    print(body_fmt)
    results = search(ES_INDEX_NAME, body_fmt)
    return results


# 输入：品牌、品类、卖点、受众人群
def vector_search(_doc, topn=3,fields=None):
    mapping = {"brand": "品牌", "category": "品类", "feature": "卖点", "crowd": "受众人群"}
    fields = ["title", "content", "time"] if not fields else fields
    query_fmt = "，".join([f"{mapping[k]}：{_doc[k]}" for k in _doc])
    print(f"query_fmt:{query_fmt}")
    query_vector = get_embedding(query_fmt)
    should_list = []  # 并，或
    match_list = [{"match":{f"{k}":f"{_doc[k]}"}} for k in _doc]

    script_vector = {"script_score": {
        "query": {
            "match_all": {}
        },
        "script": {
            "source": "cosineSimilarity(params.queryVector, 'product_vector') + 1.0",
            "params": {
                "queryVector": query_vector
            }
        }
    }}
    # should_list.extend(match_list)  # 加入字段
    should_list.append(script_vector)  # 加入向量搜索

    body_fmt = {
        "size": topn,
        "_source": fields,

        "query": {
            "bool": {
                "should": should_list
            }
        },
        "sort": [
            {
                "_score": {
                    "order": "desc"
                }
            },
            {
                "ctime": {
                    "order": "desc"
                }
            }
        ]
    }

    results = search(ES_INDEX_NAME, body_fmt)
    return results


# 输入：品牌、品类、卖点、受众人群
def keyword_search(_doc, topn=3,fields=None):
    mapping = {"brand": "品牌", "category": "品类", "feature": "卖点", "crowd": "受众人群"}
    fields = ["title", "content", "time"] if not fields else fields
    query_fmt = "，".join([f"{mapping[k]}：{_doc[k]}" for k in _doc])
    print(f"query_fmt:{query_fmt}")
    query_vector = get_embedding(query_fmt)
    should_list = []  # 并，或
    match_list = [{"match":{f"{k}":f"{_doc[k]}"}} for k in _doc]

    script_vector = {"script_score": {
        "query": {
            "match_all": {}
        },
        "script": {
            "source": "cosineSimilarity(params.queryVector, 'product_vector') + 1.0",
            "params": {
                "queryVector": query_vector
            }
        }
    }}
    should_list.extend(match_list)  # 加入字段
    # should_list.append(script_vector)  # 加入向量搜索

    body_fmt = {
        "size": topn,
        "_source": fields,

        "query": {
            "bool": {
                "should": should_list
            }
        },
        "sort": [
            {
                "_score": {
                    "order": "desc"
                }
            },
            {
                "ctime": {
                    "order": "desc"
                }
            }
        ]
    }

    results = search(ES_INDEX_NAME, body_fmt)
    return results

def check_all_data():
    data = load_data()

    import pandas as pd
    total_results = []
    for line in data:  # TODO
        try:
            ctime = '2023-11-08'
            title = line['title']
            content = line['content']
            brand = line['brand_name_list'][1:-1]  # 去掉[]
            category = line['category_name_list'][1:-1]  # 去掉[]
            # category = line['brand_3class']  # 去掉[]
            feature = line['baichuan2_keyword_clean']
            crowd = line['crowd_keyword_list'][1:-1]  # 去掉[]
            query = {"brand":brand,"category":category,"feature":feature,"crowd":crowd}
            query_item = {'score':0,'title':title,'content':content,'brand':brand,'category':category,'feature':feature,'crowd':crowd,'ctime':ctime,'type':'query'}
            print(query_item)
            res = [query_item]
            results = combine_search(query, topn=5,
                                     fields=['brand', 'category', 'feature', 'crowd', 'ctime', 'title', 'content'])
            for item in results['hits']['hits']:
                print(item)
                _res = {'score':item['_score'],'title':item['_source']['title'],'content':item['_source']['content'],'brand':item['_source']['brand'],'category':item['_source']['category'],'feature':item['_source']['feature'],'crowd':item['_source']['crowd'],'ctime':item['_source']['ctime'],'type':'result'}
                print(_res)
                res.append(_res)
            total_results.extend(res)
        except Exception as e:
            print(f"error, line:{line}, {e}")
    df = pd.DataFrame(total_results)
    df.to_excel("/home/kit/kit/search/pangu_result.xlsx",index=False)


if __name__ == "__main__":
    # 初始化
    init_es()

    # 遍历
    #check_all_data()

    # # 查询
    #doc = {"brand":"雅萌YA-MAN","category":"美容仪","feature":"大美女张嘉倪、雅萌ACE、种草、美容仪器、租、平价护肤、亲测好用","crowd":"美女"}
    # # doc = {"brand":"雅萌YA-MAN","category":"美容仪","feature":"种草、平价","crowd":"女生"}
    # # doc = {"brand":"雅萌","category":"美容仪","feature":"种草、平价","crowd":"女生"}
    #
    # # doc = {"brand":"汤臣倍健BYHEALTH","category":"乳母营养, 维生素, 褪黑素, 胶原蛋白, 膳食纤维, 孕妇叶酸, 鱼油, 儿童维生素 D, 孕妇多维复合维生素, 叶酸, 钙片, 儿童钙片, 益生菌, 儿童维生素 AD, 蛋白粉, 螺旋藻, 叶黄素, 儿童维生素, 蛋白粉测试, 葡萄籽精华, 辅酶Q10","feature":"75折优惠、减肥人群、上班族、营养不均衡、长期熬夜","crowd":""}
    # # doc = {"brand":"汤臣倍健","category":"维生素","feature":"75折优惠、减肥人群、上班族、营养不均衡、长期熬夜","crowd":""}
    # # doc = {"brand":"汤臣背健","category":"维生素","feature":"75折优惠、减肥人群、上班族、营养不均衡、长期熬夜","crowd":""}
    #
    # # doc = {"brand":"卡尔文·克莱恩Calvin Klein","category":"内衣","feature":"舒服、运动内衣、内搭、撞色设计、简约、好搭配、软乎乎、面料、上身、无束缚、无钢圈、承托、聚拢性、显胸大、小胸内衣、双十二清单、无钢圈舒适内衣、平价好穿内衣、又纯又欲、内衣测评、双十一囤货、无痕内衣、双十一定宝好物、舒适内衣","crowd":""}
    # # doc = {"brand":"卡尔文·克莱恩","category":"内衣","feature":"舒服、运动内衣、内搭、撞色设计、简约、好搭配、软乎乎、面料、上身、无束缚、无钢圈、承托、聚拢性、显胸大、小胸内衣、双十二清单、无钢圈舒适内衣、平价好穿内衣、又纯又欲、内衣测评、双十一囤货、无痕内衣、双十一定宝好物、舒适内衣","crowd":""}
    # # doc = {"brand":"Calvin Klein","category":"内衣","feature":"运动内衣","crowd":""}
    #
    #results = combine_search(doc,topn=3,fields=['brand','category','feature','crowd','ctime','title','content'])
    # # results = vector_search(doc,topn=3,fields=['brand','category','feature','crowd','ctime','title','content'])
    # # results = keyword_search(doc,topn=3,fields=['brand','category','feature','crowd','ctime','title','content'])
    # print(f"query:{doc}")
    #print(results)
    #
    # import pandas as pd
    # query_item = {'score':0,'title':'','content':'','brand':doc['brand'],'category':doc['category'],'feature':doc['feature'],'crowd':doc['crowd'],'ctime':'','type':'query'}
    # print(query_item)
    # res = [query_item]
    # for item in results['hits']['hits']:
    #     print(item)
    #     _res = {'score':item['_score'],'title':item['_source']['title'],'content':item['_source']['content'],'brand':item['_source']['brand'],'category':item['_source']['category'],'feature':item['_source']['feature'],'crowd':item['_source']['crowd'],'ctime':item['_source']['ctime'],'type':'result'}
    #     print(_res)
    #     res.append(_res)
    # df = pd.DataFrame(res)
    # df.to_excel("data/pangu_result.xlsx",index=False)
