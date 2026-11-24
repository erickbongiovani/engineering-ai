import math
import random


def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]


def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)

    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples


def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = max(random.random(), 1e-12)
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples


def simulate_ab_test(n_visitors=2000, seed=42):
    random.seed(seed)

    # 50% para variante A (0) e 50% para variante B (1)
    variants = sample_categorical([0.5, 0.5], n=n_visitors)

    # Tempo médio de carregamento: 1.8s na A vs 1.4s na B
    load_times_a = sample_normal_box_muller(mu=1.8, sigma=0.4, n=n_visitors)
    load_times_b = sample_normal_box_muller(mu=1.4, sigma=0.35, n=n_visitors)

    conversions_a, conversions_b = [], []
    times_a, times_b = [], []

    for var, t_a, t_b in zip(variants, load_times_a, load_times_b):
        if var == 0:
            # Páginas mais lentas reduzem levemente a chance de conversão
            p_conv = max(0.01, min(0.95, 0.14 - 0.02 * (t_a - 1.5)))
            conv = sample_bernoulli(p_conv, n=1)[0]
            conversions_a.append(conv)
            times_a.append(t_a)
        else:
            p_conv = max(0.01, min(0.95, 0.18 - 0.02 * (t_b - 1.5)))
            conv = sample_bernoulli(p_conv, n=1)[0]
            conversions_b.append(conv)
            times_b.append(t_b)

    rate_a = sum(conversions_a) / len(conversions_a)
    rate_b = sum(conversions_b) / len(conversions_b)
    avg_time_a = sum(times_a) / len(times_a)
    avg_time_b = sum(times_b) / len(times_b)

    return {
        "visitors_a": len(conversions_a),
        "visitors_b": len(conversions_b),
        "conv_rate_a": rate_a,
        "conv_rate_b": rate_b,
        "lift": rate_b - rate_a,
        "avg_load_a": avg_time_a,
        "avg_load_b": avg_time_b,
    }


results = simulate_ab_test()
print(f"Visitantes -> A: {results['visitors_a']} | B: {results['visitors_b']}")
print(f"Conversão  -> A: {results['conv_rate_a']:.2%} | B: {results['conv_rate_b']:.2%}")
print(f"Diferença (B - A): {results['lift']:+.2%}")
print(f"Tempo médio -> A: {results['avg_load_a']:.2f}s | B: {results['avg_load_b']:.2f}s")