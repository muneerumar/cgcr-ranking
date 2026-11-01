"""optional adaptation: official Shield scores gate Qwen inputs, no TFSG.

Preserve ten fixed slots, withholding the lower five documents. This is an
explicit adaptation, not the official two-stage answer-generation pipeline.
"""


def filter_top_k(docs,scores,k=5):
    if len(docs)!=len(scores) or not 0<k<=len(docs):raise ValueError('Invalid Shield filter')
    selected=set(sorted(range(len(scores)),key=lambda i:(-scores[i],i))[:k])
    return [doc if i in selected else '[Withheld by Shield Reranker top-5 filter.]' for i,doc in enumerate(docs)]
