import time
import torch
from torch.utils.data import DataLoader
import random
import nltk
nltk.download('punkt')

def load_and_sample_sentences(num_pairs=1024, min_sentences=2, max_sentences=10):
    # Read the file
    with open('data/sample_data.txt', 'r') as f:
        text = f.read()
    
    # Split into sentences
    sentences = nltk.sent_tokenize(text)
    
    # Generate random pairs
    pairs = []
    for _ in range(num_pairs):
        # Sample random sentences for first text
        n1 = random.randint(min_sentences, max_sentences)
        text1 = ' '.join(random.sample(sentences, n1))
        
        # Sample random sentences for second text
        n2 = random.randint(min_sentences, max_sentences)
        text2 = ' '.join(random.sample(sentences, n2))
        
        pairs.append([text1, text2])
    
    return pairs

def inference(model, sentence_pairs):
    scores = model.predict(sentence_pairs, convert_to_tensor=True, batch_size=32)
    
    return scores

def test_model(model):
    sentence_pairs = load_and_sample_sentences(num_pairs=64)
    scores = model.predict(sentence_pairs, convert_to_tensor=True)
    print(scores.tolist())

def benchmark(model, print_scores=False, num_runs=5, cuda_graph=False, trace=None, on_sorted_inputs=False):
    sentence_pairs_warmup = load_and_sample_sentences(num_pairs=512)
    sentence_pairs = load_and_sample_sentences(num_pairs=1024)

    indices = None
    if on_sorted_inputs:
        # Sort by length before timing
        indices = list(range(len(sentence_pairs)))
        indices.sort(key=lambda i: len(model.tokenizer.encode(sentence_pairs[i][0] + sentence_pairs[i][1])), reverse=True)
        sentence_pairs = [sentence_pairs[i] for i in indices]

    print(max([len(model.tokenizer.encode(x)) for x in sentence_pairs]))
    print(max([len(model.tokenizer.encode(x)) for x in sentence_pairs_warmup]))

    print([len(model.tokenizer.encode(x)) for x in sentence_pairs])
    print(f"Total pairs: {len(sentence_pairs)}")

    # Warmup
    print("Warming up...")
    for _ in range(10):
        sentence_pairs_warmup = load_and_sample_sentences(num_pairs=512)
        _ = inference(model, sentence_pairs_warmup)

    # Multiple benchmark runs
    print("Benchmarking...")
    times = []

    for i in range(num_runs):
        sentence_pairs = load_and_sample_sentences(num_pairs=1024)
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        scores = inference(model, sentence_pairs)
            
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    # Restore original order if sorted
    if on_sorted_inputs and scores is not None:
        original_scores = torch.zeros_like(scores)
        original_scores[indices] = scores
        scores = original_scores

    if trace is not None:
        with torch.profiler.profile() as prof:
            for i in range(1, 2):
                scores = inference(model, sentence_pairs)
                prof.step()

        prof.export_chrome_trace(f"trace/{trace}.json")

    if print_scores:
        print(scores.tolist())
        
    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
    
    print(f"Inference times: {[f'{t:.4f}' for t in times]}")
    print(f"Mean time: {mean_time:.4f} ± {std_time:.4f} seconds")