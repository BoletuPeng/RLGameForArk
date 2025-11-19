"""
测试不同网络架构的参数量
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

def estimate_params(arch, input_dim=29, actor_output=6, critic_output=1):
    """估算网络参数量"""
    # Actor网络参数
    actor_params = 0
    prev_dim = input_dim
    for hidden_dim in arch:
        actor_params += prev_dim * hidden_dim + hidden_dim
        prev_dim = hidden_dim
    actor_params += prev_dim * actor_output + actor_output

    # Critic网络参数
    critic_params = 0
    prev_dim = input_dim
    for hidden_dim in arch:
        critic_params += prev_dim * hidden_dim + hidden_dim
        prev_dim = hidden_dim
    critic_params += prev_dim * critic_output + critic_output

    return actor_params + critic_params

# 测试所有网络架构
network_configs = {
    "small": [64, 64],
    "medium": [128, 128],
    "large": [256, 256],
    "xlarge": [256, 256, 128]
}

print("\n" + "=" * 70)
print("PPO 网络架构参数量对比")
print("=" * 70)
print(f"{'架构':<10} {'网络结构':<20} {'参数量':<15} {'相比small':<15}")
print("-" * 70)

small_params = estimate_params(network_configs["small"])

for name, arch in network_configs.items():
    params = estimate_params(arch)
    ratio = params / small_params
    print(f"{name:<10} {str(arch):<20} {params:>10,} 个  {ratio:>10.1f}x")

print("=" * 70)

# 详细展示medium架构的参数分配
print("\n详细参数分配 (medium 网络: [128, 128]):")
print("-" * 70)

arch = [128, 128]
input_dim = 29

print("\nActor 网络 (策略网络):")
print(f"  输入层 → 隐藏层1: {input_dim} × 128 + 128(bias) = {input_dim * 128 + 128:,}")
print(f"  隐藏层1 → 隐藏层2: 128 × 128 + 128(bias) = {128 * 128 + 128:,}")
print(f"  隐藏层2 → 动作输出: 128 × 6 + 6(bias) = {128 * 6 + 6:,}")

actor_params = (input_dim * 128 + 128) + (128 * 128 + 128) + (128 * 6 + 6)
print(f"  Actor 总参数: {actor_params:,}")

print("\nCritic 网络 (价值网络):")
print(f"  输入层 → 隐藏层1: {input_dim} × 128 + 128(bias) = {input_dim * 128 + 128:,}")
print(f"  隐藏层1 → 隐藏层2: 128 × 128 + 128(bias) = {128 * 128 + 128:,}")
print(f"  隐藏层2 → 价值输出: 128 × 1 + 1(bias) = {128 * 1 + 1:,}")

critic_params = (input_dim * 128 + 128) + (128 * 128 + 128) + (128 * 1 + 1)
print(f"  Critic 总参数: {critic_params:,}")

print(f"\n总参数量: {actor_params + critic_params:,}")
print("=" * 70)
