import time
import torch
from torch.utils.data import DataLoader

def inference(model, inp_dataloader):
    scores = model.predict(inp_dataloader)
    return scores

def test_model(model):
    query = "Organic skincare products for sensitive skin"
    documents = [
        "Organic skincare for sensitive skin with aloe vera and chamomile.",
        "New makeup trends focus on bold colors and innovative techniques",
    ] * 40
    sentence_pairs = [[query, doc] for doc in documents]
    scores = model.predict(sentence_pairs, convert_to_tensor=True)
    print(scores.tolist())

def benchmark(model, print_scores=False, num_runs=5, cuda_graph=False, trace=None):
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
        "The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore while contemplating the meaning of life and watching the sunset paint vibrant colors across the horizon.",
        "Coffee brewing.",
        "The ancient ruins stood silently, their weathered stones telling stories of civilizations long forgotten and empires that had crumbled to dust.",
        "Birds sing melodies.",
        "Quantum physics challenges our fundamental understanding of reality and consciousness in ways we are only beginning to comprehend.",
        "Fresh bread baking in the morning.",
        "Technology evolves rapidly.",
        "The mountain climber persevered through harsh conditions, determined to reach the snow-capped peak before nightfall.",
        "Waves crash on empty beaches.",
        "Children's laughter echoed through the playground as clouds drifted lazily overhead in the summer sky.",
        "Books transport minds.",
        "The old grandfather clock in the hallway has been keeping time faithfully for over two hundred years.",
        "Music soothes souls.",
        "Artificial intelligence researchers work tirelessly to push the boundaries of machine learning and neural networks.",
        "Gardens bloom in spring sunshine.",
        "The chef expertly crafted each dish with precision and artistic flair, transforming simple ingredients into culinary masterpieces.",
        "Rain falls softly.",
        "Astronauts floating in the International Space Station observe our beautiful blue planet from above.",
        "Time flows endlessly forward.",
        "The mysterious alchemist's laboratory contained bubbling potions and ancient scrolls filled with cryptic formulas.",
        "Desert winds sculpt intricate patterns in the golden sand dunes stretching endlessly to the horizon.",
    ] * 10

    sentence_pairs_rev = [[doc + " " + doc, query] for doc in documents]
    sentence_pairs = [[query, doc + " " + doc] for doc in documents]

    print(max([len(model.tokenizer.encode(x)) for x in sentence_pairs]))
    print(max([len(model.tokenizer.encode(x)) for x in sentence_pairs_rev]))

    print([len(model.tokenizer.encode(x)) for x in sentence_pairs])
    print(f"Total pairs: {len(sentence_pairs)}")

    inp_dataloader_rev = DataLoader(
        sentence_pairs_rev,
        batch_size=32,
        collate_fn=model.smart_batching_collate_text_only,
        num_workers=0,
        shuffle=False,
    )

    inp_dataloader = DataLoader(
        sentence_pairs,
        batch_size=32,
        collate_fn=model.smart_batching_collate_text_only,
        num_workers=0,
        shuffle=False,
    )

    if cuda_graph:
        # Create static input tensors for graph capture
        static_inputs = next(iter(inp_dataloader_rev))
        for name in static_inputs:
            static_inputs[name] = static_inputs[name].to(model.device)
        
        # Warmup
        _ = inference(model, static_inputs)
        
        # Capture graph
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            static_scores = inference(model, static_inputs)
        torch.cuda.current_stream().wait_stream(stream)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_scores = inference(model, static_inputs)

    # Warmup
    print("Warming up...")
    for _ in range(10):
        if cuda_graph:
            # Copy new data into static tensors
            batch = next(iter(inp_dataloader_rev))
            for name in batch:
                static_inputs[name].copy_(batch[name].to(model.device))
            g.replay()
        else:
            _ = inference(model, inp_dataloader_rev)

    # Multiple benchmark runs
    print("Benchmarking...")
    times = []

    for i in range(num_runs):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        if cuda_graph:
            # Use the same graph with new data
            batch = next(iter(inp_dataloader))
            for name in batch:
                static_inputs[name].copy_(batch[name].to(model.device))
            g.replay()
            scores = static_scores
        else:
            scores = inference(model, inp_dataloader)
            
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        times.append(end_time - start_time)
        
    if trace is not None:
        with torch.profiler.profile() as prof:
            for i in range(1, 2):
                scores = inference(model, inp_dataloader)
                prof.step()

        prof.export_chrome_trace(f"trace/{trace}.json")

    if print_scores:
        print(scores.tolist())
        
    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
    
    print(f"Inference times: {[f'{t:.4f}' for t in times]}")
    print(f"Mean time: {mean_time:.4f} ± {std_time:.4f} seconds")