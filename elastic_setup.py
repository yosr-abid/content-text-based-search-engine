from elasticsearch import Elasticsearch
mappings = { "mappings": {
       "properties": {
          "ImageURL": {"type": "text","index" : True},
          "Tags": {"type": "text","index" : True},
          "Vector": {"type": "dense_vector","dims": 1000,"index" : True, "similarity": "cosine"}
        }
    }
}


client = Elasticsearch(
    "http://localhost:9200",  # Elasticsearch endpoint 
)

print(client.ping())

client.indices.create(index="test4",body=mappings)



# from elasticsearch import Elasticsearch
# mappings = { "mappings": {
#        "properties": {
#           "imageId": {"type": "text","index" : True},
#           "OriginalMDS":{"type": "text","index" : True},
#           "ImageURL": {"type": "text","index" : True},
#           "Size": {"type": "integer"},
#           "Tags": {"type": "text","index" : True},
#           "Vector": {"type": "dense_vector","dims": 1000,"index" : True, "similarity": "cosine"}
#         }
#     }
# }


# client = Elasticsearch(
#     "http://localhost:9200",  # Elasticsearch endpoint 
# )

# print(client.ping())

# client.indices.create(index="test4",body=mappings)
