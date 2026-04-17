import json
import random
import codecs

# 可调参数
SUPPLY_TOTAL = 3000         # 40GP 供应总量
DEMAND_TOTAL = 2500         # 40GP 需求总量
NUM_SUPPLY = 50              # 供应站数量（可调整）
NUM_DEMAND = 50              # 需求站数量（可调整）
RANDOM_SEED = 42             # 随机种子，保证可复现（可删除或修改）

def main():
    random.seed(RANDOM_SEED)  # 固定随机结果，方便测试

    # 1. 读取车站列表，获取总站数
    with codecs.open("stations.json", "r", encoding="utf-8") as f:
        stations = json.load(f)["stations"]
    n = len(stations)

    # 2. 从所有车站索引中随机选择供应站和需求站（保证无交集）
    all_indices = list(range(n))
    supply_indices = set(random.sample(all_indices, NUM_SUPPLY))
    # 剩余索引中选需求站
    remaining = [i for i in all_indices if i not in supply_indices]
    demand_indices = set(random.sample(remaining, NUM_DEMAND))

    # 3. 随机分配供应量（总和 = SUPPLY_TOTAL，每个站至少 1）
    supply_values = [1] * NUM_SUPPLY
    remaining_supply = SUPPLY_TOTAL - NUM_SUPPLY
    # 随机分配到 NUM_SUPPLY 个站
    for _ in range(remaining_supply):
        idx = random.randrange(NUM_SUPPLY)
        supply_values[idx] += 1

    # 4. 随机分配需求量（总和 = DEMAND_TOTAL，每个站至少 1）
    demand_values = [1] * NUM_DEMAND
    remaining_demand = DEMAND_TOTAL - NUM_DEMAND
    for _ in range(remaining_demand):
        idx = random.randrange(NUM_DEMAND)
        demand_values[idx] += 1

    # 5. 构建新的 40GP 数组（初始全 0）
    new_supply_40 = [0] * n
    new_demand_40 = [0] * n

    # 填入供应站
    for idx, val in zip(supply_indices, supply_values):
        new_supply_40[idx] = val

    # 填入需求站
    for idx, val in zip(demand_indices, demand_values):
        new_demand_40[idx] = val

    # 6. 直接构建仅含 40GP 的新数据字典（不再依赖原文件）
    supply_data = {"40GP": new_supply_40}
    demand_data = {"40GP": new_demand_40}

    # 7. 写回文件
    with codecs.open("supply_data_TOPK.json", "w", encoding="utf-8") as f:
        json.dump(supply_data, f, ensure_ascii=False, indent=2)
    with codecs.open("demand_data_TOPK.json", "w", encoding="utf-8") as f:
        json.dump(demand_data, f, ensure_ascii=False, indent=2)

    # 打印简要统计
    print("✅ 40GP 数据已更新！")
    print(f"供应站数量：{NUM_SUPPLY}，总供应量：{sum(supply_values)}")
    print(f"需求站数量：{NUM_DEMAND}，总需求量：{sum(demand_values)}")
    print("供应站索引（示例）：", sorted(list(supply_indices))[:5])
    print("需求站索引（示例）：", sorted(list(demand_indices))[:5])

if __name__ == "__main__":
    main()