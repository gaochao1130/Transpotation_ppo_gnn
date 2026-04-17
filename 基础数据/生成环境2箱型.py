import json
import random

# ==================== 可调参数区（请在此修改配置） ====================
RANDOM_SEED = 42  # 随机种子（保证可复现）
# 核心约束参数（严格按照您的要求设置）
TOTAL_ACTIVE_STATIONS = 100  # 总活跃车站数（源/目的站都从这里选）
SUPPLY_STATIONS_PER_BOX = 50  # 每个箱型的源站数量
DEMAND_STATIONS_PER_BOX = 50  # 每个箱型的目的站数量
DUAL_SUPPLY_STATIONS = 20  # 同时供应两种箱型的车站数
DUAL_DEMAND_STATIONS = 20  # 同时需求两种箱型的车站数
# 集装箱类型及总量配置
BOX_TYPES = {
    "20GPBZ": {"supply_total": 400, "demand_total": 380},  # 20英尺标准箱
    "40GPBZ": {"supply_total": 300, "demand_total": 280}  # 40英尺标准箱
}
# 数值生成波动控制（不同箱型之间的波动，暂未使用，保留原参数）
FLUCTUATION_RANGE = 0.1


# ====================================================================


def generate_supply_values(total_target, num_sites, min_val=1):
    """
    供应数值生成：每个供应站至少有 min_val，剩余量随机增量分配。
    """
    values = [min_val] * num_sites
    remaining = total_target - sum(values)
    for _ in range(remaining):
        idx = random.randrange(num_sites)
        values[idx] += 1
    return values


def generate_demand_values(total_target, num_sites, min_val=1):
    """
    需求数值生成：与供应数值生成逻辑一致
    """
    values = [min_val] * num_sites
    remaining = total_target - sum(values)
    for _ in range(remaining):
        idx = random.randrange(num_sites)
        values[idx] += 1
    return values


def main():
    random.seed(RANDOM_SEED)

    # ---------- 1. 读取车站列表 ----------
    try:
        with open("stations.json", "r", encoding="utf-8") as f:
            stations = json.load(f)["stations"]
        total_stations = len(stations)
        if total_stations == 0:
            raise ValueError("stations.json 中没有车站数据！")
        print(f"📌 总车站数量：{total_stations}")
    except FileNotFoundError:
        print("❌ 错误：未找到 stations.json 文件，请检查文件路径！")
        return
    except Exception as e:
        print(f"❌ 读取车站文件失败：{str(e)}")
        return

    # ---------- 2. 校验参数合法性 ----------
    if TOTAL_ACTIVE_STATIONS > total_stations:
        raise ValueError(f"活跃车站数 {TOTAL_ACTIVE_STATIONS} 超过总车站数 {total_stations}")

    # 校验供应参数：20单供 + 40单供 + 双供 = 总活跃站
    single_20_supply = SUPPLY_STATIONS_PER_BOX - DUAL_SUPPLY_STATIONS
    single_40_supply = SUPPLY_STATIONS_PER_BOX - DUAL_SUPPLY_STATIONS
    no_supply = TOTAL_ACTIVE_STATIONS - single_20_supply - single_40_supply - DUAL_SUPPLY_STATIONS
    if any(x < 0 for x in [single_20_supply, single_40_supply, no_supply]):
        raise ValueError("供应参数不合法：双供站数不能超过单箱型供应站数")

    # 校验需求参数：20单需 + 40单需 + 双需 = 总活跃站
    single_20_demand = DEMAND_STATIONS_PER_BOX - DUAL_DEMAND_STATIONS
    single_40_demand = DEMAND_STATIONS_PER_BOX - DUAL_DEMAND_STATIONS
    no_demand = TOTAL_ACTIVE_STATIONS - single_20_demand - single_40_demand - DUAL_DEMAND_STATIONS
    if any(x < 0 for x in [single_20_demand, single_40_demand, no_demand]):
        raise ValueError("需求参数不合法：双需站数不能超过单箱型需求站数")

    print(f"\n✅ 参数校验通过：")
    print(
        f"  供应端：单供20GP={single_20_supply} | 单供40GP={single_40_supply} | 双供={DUAL_SUPPLY_STATIONS} | 不供应={no_supply}")
    print(
        f"  需求端：单需20GP={single_20_demand} | 单需40GP={single_40_demand} | 双需={DUAL_DEMAND_STATIONS} | 不需求={no_demand}")

    # ---------- 3. 随机选择100个活跃车站 ----------
    all_indices = list(range(total_stations))
    active_stations = random.sample(all_indices, TOTAL_ACTIVE_STATIONS)
    inactive_stations = [i for i in all_indices if i not in active_stations]
    print(f"\n📌 已选择 {len(active_stations)} 个活跃车站，{len(inactive_stations)} 个车站供需全为0")

    # ---------- 4. 分配供应站（严格按照比例） ----------
    # 打乱活跃站顺序，随机分配
    shuffled_active = random.sample(active_stations, len(active_stations))

    # 分配供应站
    dual_supply_sites = shuffled_active[:DUAL_SUPPLY_STATIONS]
    single_20_supply_sites = shuffled_active[DUAL_SUPPLY_STATIONS:DUAL_SUPPLY_STATIONS + single_20_supply]
    single_40_supply_sites = shuffled_active[
        DUAL_SUPPLY_STATIONS + single_20_supply:DUAL_SUPPLY_STATIONS + single_20_supply + single_40_supply]

    # 构建每个箱型的供应站列表
    supply_20 = dual_supply_sites + single_20_supply_sites
    supply_40 = dual_supply_sites + single_40_supply_sites
    random.shuffle(supply_20)
    random.shuffle(supply_40)

    print(f"\n📦 供应站分配完成：")
    print(f"  20GPBZ 供应站总数：{len(supply_20)} (目标：{SUPPLY_STATIONS_PER_BOX})")
    print(f"  40GPBZ 供应站总数：{len(supply_40)} (目标：{SUPPLY_STATIONS_PER_BOX})")
    print(f"  同时供应两种箱型的车站数：{len(dual_supply_sites)} (目标：{DUAL_SUPPLY_STATIONS})")

    # ---------- 5. 生成供应数据 ----------
    supply_data = {"stations": stations}
    # 初始化所有车站供应量为0
    supply_20_array = [0] * total_stations
    supply_40_array = [0] * total_stations

    # 为20GPBZ生成供应量
    supply_20_vals = generate_supply_values(BOX_TYPES["20GPBZ"]["supply_total"], len(supply_20))
    for idx, val in zip(supply_20, supply_20_vals):
        supply_20_array[idx] = val

    # 为40GPBZ生成供应量
    supply_40_vals = generate_supply_values(BOX_TYPES["40GPBZ"]["supply_total"], len(supply_40))
    for idx, val in zip(supply_40, supply_40_vals):
        supply_40_array[idx] = val

    supply_data["20GPBZ"] = supply_20_array
    supply_data["40GPBZ"] = supply_40_array

    print(f"\n📊 供应总量校验：")
    print(f"  20GPBZ 实际供应：{sum(supply_20_array)} / 目标：{BOX_TYPES['20GPBZ']['supply_total']}")
    print(f"  40GPBZ 实际供应：{sum(supply_40_array)} / 目标：{BOX_TYPES['40GPBZ']['supply_total']}")

    # ---------- 6. 分配需求站（严格按照比例，且满足互斥约束：需求站不能是同箱型的供应站） ----------
    # 重新打乱活跃站顺序，与供应站分配独立
    shuffled_active_demand = random.sample(active_stations, len(active_stations))

    # 先找出每个箱型不能作为需求站的站点（同箱型的供应站）
    forbidden_20_demand = set(supply_20)
    forbidden_40_demand = set(supply_40)

    # 筛选可用的需求站候选
    available_20_demand = [s for s in shuffled_active_demand if s not in forbidden_20_demand]
    available_40_demand = [s for s in shuffled_active_demand if s not in forbidden_40_demand]

    # 先分配双需站（必须同时不在两个箱型的禁止列表中）
    available_dual_demand = [s for s in shuffled_active_demand if
                             s not in forbidden_20_demand and s not in forbidden_40_demand]
    if len(available_dual_demand) < DUAL_DEMAND_STATIONS:
        raise ValueError(
            f"可用双需站不足！需要 {DUAL_DEMAND_STATIONS} 个，实际只有 {len(available_dual_demand)} 个。请减少双需站数或增加活跃站数。")

    dual_demand_sites = random.sample(available_dual_demand, DUAL_DEMAND_STATIONS)

    # 从剩余可用站中分配单需站
    remaining_20 = [s for s in available_20_demand if s not in dual_demand_sites]
    remaining_40 = [s for s in available_40_demand if s not in dual_demand_sites]

    if len(remaining_20) < single_20_demand:
        raise ValueError(f"20GPBZ 可用单需站不足！需要 {single_20_demand} 个，实际只有 {len(remaining_20)} 个")
    if len(remaining_40) < single_40_demand:
        raise ValueError(f"40GPBZ 可用单需站不足！需要 {single_40_demand} 个，实际只有 {len(remaining_40)} 个")

    single_20_demand_sites = random.sample(remaining_20, single_20_demand)
    single_40_demand_sites = random.sample(remaining_40, single_40_demand)

    # 构建每个箱型的需求站列表
    demand_20 = dual_demand_sites + single_20_demand_sites
    demand_40 = dual_demand_sites + single_40_demand_sites
    random.shuffle(demand_20)
    random.shuffle(demand_40)

    print(f"\n📦 需求站分配完成：")
    print(f"  20GPBZ 需求站总数：{len(demand_20)} (目标：{DEMAND_STATIONS_PER_BOX})")
    print(f"  40GPBZ 需求站总数：{len(demand_40)} (目标：{DEMAND_STATIONS_PER_BOX})")
    print(f"  同时需求两种箱型的车站数：{len(dual_demand_sites)} (目标：{DUAL_DEMAND_STATIONS})")
    print(f"  ✅ 满足互斥约束：需求站都不是同箱型的供应站")

    # ---------- 7. 生成需求数据 ----------
    demand_data = {"stations": stations}
    # 初始化所有车站需求量为0
    demand_20_array = [0] * total_stations
    demand_40_array = [0] * total_stations

    # 为20GPBZ生成需求量
    demand_20_vals = generate_demand_values(BOX_TYPES["20GPBZ"]["demand_total"], len(demand_20))
    for idx, val in zip(demand_20, demand_20_vals):
        demand_20_array[idx] = val

    # 为40GPBZ生成需求量
    demand_40_vals = generate_demand_values(BOX_TYPES["40GPBZ"]["demand_total"], len(demand_40))
    for idx, val in zip(demand_40, demand_40_vals):
        demand_40_array[idx] = val

    demand_data["20GPBZ"] = demand_20_array
    demand_data["40GPBZ"] = demand_40_array

    print(f"\n📊 需求总量校验：")
    print(f"  20GPBZ 实际需求：{sum(demand_20_array)} / 目标：{BOX_TYPES['20GPBZ']['demand_total']}")
    print(f"  40GPBZ 实际需求：{sum(demand_40_array)} / 目标：{BOX_TYPES['40GPBZ']['demand_total']}")

    # ---------- 8. 写入文件 ----------
    try:
        with open("supply_data_one.json", "w", encoding="utf-8") as f:
            json.dump(supply_data, f, ensure_ascii=False, indent=2)
        with open("demand_data_one.json", "w", encoding="utf-8") as f:
            json.dump(demand_data, f, ensure_ascii=False, indent=2)
        print("\n✅ 数据已成功写入文件！")
    except PermissionError:
        desktop_path = "C:\\Users\\86188\\Desktop\\"
        try:
            with open(desktop_path + "supply_data_one.json", "w", encoding="utf-8") as f:
                json.dump(supply_data, f, ensure_ascii=False, indent=2)
            with open(desktop_path + "demand_data_one.json", "w", encoding="utf-8") as f:
                json.dump(demand_data, f, ensure_ascii=False, indent=2)
            print("⚠️ 原路径写入失败，已将文件保存到桌面！")
        except Exception as e:
            print(f"❌ 桌面写入也失败：{str(e)}")
            return
    except Exception as e:
        print(f"❌ 写入文件失败：{str(e)}")
        return

    # ---------- 9. 示例站点信息 ----------
    print(f"\n📌 示例站点信息：")
    if dual_supply_sites:
        idx = dual_supply_sites[0]
        print(f"  双供应站（索引 {idx}）：20GPBZ={supply_20_array[idx]}, 40GPBZ={supply_40_array[idx]}")

    if single_20_supply_sites:
        idx = single_20_supply_sites[0]
        print(f"  单供应20GPBZ站（索引 {idx}）：20GPBZ={supply_20_array[idx]}, 40GPBZ={supply_40_array[idx]}")

    if dual_demand_sites:
        idx = dual_demand_sites[0]
        print(f"  双需求站（索引 {idx}）：20GPBZ={demand_20_array[idx]}, 40GPBZ={demand_40_array[idx]}")
        print(f"    验证互斥：该站20GPBZ供应量={supply_20_array[idx]}, 40GPBZ供应量={supply_40_array[idx]} (均应为0)")


if __name__ == "__main__":
    main()