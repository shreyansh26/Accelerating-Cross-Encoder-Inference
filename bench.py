import time
import torch
from torch.utils.data import DataLoader
import random
import nltk
nltk.download('punkt')

def load_and_sample_sentences(num_pairs=1024, min_sentences=2, max_sentences=15, base_seed=100):
    with open('data/sample_data.txt', 'r') as f:
        text = f.read()
    
    sentences = nltk.sent_tokenize(text)
    
    # Generate random pairs
    pairs = []
    for i in range(num_pairs):
        # Set unique seed for each iteration if base_seed provided
        random.seed(base_seed + i)
            
        n1 = random.randint(min_sentences, max_sentences)
        text1 = ' '.join(random.sample(sentences, n1))
        
        n2 = random.randint(min_sentences, max_sentences)
        text2 = ' '.join(random.sample(sentences, n2))
        
        pairs.append([text1, text2])
        
    return pairs

def inference(model, sentence_pairs):
    scores = model.predict(sentence_pairs, convert_to_tensor=True, batch_size=64)
    
    return scores

def test_model(model):
    sentence_pairs = load_and_sample_sentences(num_pairs=64, base_seed=42)
    # print(sentence_pairs[:10])
    with torch.inference_mode():
        scores = inference(model, sentence_pairs)
        print(scores.tolist()[:10])

def benchmark(model, print_scores=False, num_runs=10, trace=None, seed=100, on_sorted_inputs=False):  
    sentence_pairs_warmup = load_and_sample_sentences(num_pairs=512, base_seed=seed)
    sentence_pairs = load_and_sample_sentences(num_pairs=1024, base_seed=2*seed)

    print(max([len(model.tokenizer.encode(x)) for x in sentence_pairs]))
    print(max([len(model.tokenizer.encode(x)) for x in sentence_pairs_warmup]))

    # print([len(model.tokenizer.encode(x)) for x in sentence_pairs])
    print(f"Total pairs: {len(sentence_pairs)}")

    with torch.inference_mode():
        # Warmup
        print("Warming up...")
        for i in range(10):
            sentence_pairs_warmup = load_and_sample_sentences(num_pairs=2048, base_seed=seed + i)
            _ = inference(model, sentence_pairs_warmup)

        # Multiple benchmark runs
        print("Benchmarking...")
        times = []

        for i in range(num_runs):
            sentence_pairs = load_and_sample_sentences(num_pairs=1024, base_seed=2*seed + i)
            
            if on_sorted_inputs:
                # Sort by max length of each pair
                lengths = [(max(len(model.tokenizer.encode(p[0])), len(model.tokenizer.encode(p[1]))), i) 
                          for i, p in enumerate(sentence_pairs)]
                sorted_indices = [i for _, i in sorted(lengths, reverse=True)]
                sentence_pairs_sorted = [sentence_pairs[i] for i in sorted_indices]
            else:
                sentence_pairs_sorted = sentence_pairs
                sorted_indices = None

            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            scores = inference(model, sentence_pairs_sorted)
            
            if on_sorted_inputs:
                # Restore original order
                original_scores = torch.empty_like(scores)
                for new_idx, orig_idx in enumerate(sorted_indices):
                    original_scores[orig_idx] = scores[new_idx]
                scores = original_scores
                
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        if trace is not None:
            with torch.profiler.profile() as prof:
                for i in range(1, 2):
                    scores = inference(model, sentence_pairs)
                    prof.step()

            prof.export_chrome_trace(f"trace/{trace}.json")

    if print_scores:
        print(scores.tolist()[:10])
        
    mean_time = sum(times[-5:]) / len(times[-5:])
    std_time = (sum((t - mean_time) ** 2 for t in times[-5:]) / len(times[-5:])) ** 0.5
    
    print(f"Inference times: {[f'{t:.4f}' for t in times]}")
    print(f"Mean time: {mean_time:.4f} ± {std_time:.4f} seconds")