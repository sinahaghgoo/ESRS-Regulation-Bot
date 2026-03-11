#!/usr/bin/env python3
"""Host-side embedding script. Run from the repo root on the host machine."""
import sys, json, time, numpy as np
sys.path.insert(0, '.')

from rag.config import ARTIFACTS_DIR
from rag.embedder import embed_texts
from rag.indexer import VectorIndex
from rag.chunker import Chunk

def build_embeddings(strategy):
    chunks_file = ARTIFACTS_DIR / f'chunks_{strategy}.json'
    with open(chunks_file) as f:
        data = json.load(f)
    print(f'[{strategy}] Loaded {len(data)} chunks')
    
    texts = [d['text'] for d in data]
    
    # Embed in batches
    batch_size = 100
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        t0 = time.time()
        emb = embed_texts(batch, batch_size=batch_size)
        elapsed = time.time() - t0
        all_embs.append(emb)
        done = min(i+batch_size, len(texts))
        print(f'  Batch {i//batch_size+1}: {done}/{len(texts)} in {elapsed:.1f}s')
    
    embeddings = np.vstack(all_embs)
    print(f'  Embeddings shape: {embeddings.shape}')
    
    # Save raw embeddings for reuse
    np.save(str(ARTIFACTS_DIR / f'embeddings_{strategy}.npy'), embeddings)
    
    # Build vector index
    collection = f'esrs_reports_{strategy}'
    chunks = []
    for d in data:
        chunks.append(Chunk(
            chunk_id=d['chunk_id'], text=d['text'], company=d['company'],
            reporting_year=d['reporting_year'], source_file=d['source_file'],
            header_hierarchy=d['header_hierarchy'], page_number=d['page_number'],
            page_span=d['page_span'], chunk_index=d['chunk_index'],
            chunking_strategy=d['chunking_strategy'], token_count=d['token_count'],
            contains_table=d.get('contains_table', False),
            contains_figure=d.get('contains_figure', False),
        ))
    
    vec_index = VectorIndex(collection, use_qdrant=True)
    vec_index.upsert_chunks(chunks, embeddings)
    print(f'  Indexed {vec_index.count()} vectors')
    return vec_index

if __name__ == '__main__':
    t0 = time.time()
    for s in ['structure_aware', 'fixed_size']:
        build_embeddings(s)
    print(f'\nTotal time: {time.time()-t0:.0f}s')
    print('EMBEDDING_COMPLETE')
