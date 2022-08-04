import copy
import math
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

##############################################
# 對 4.1 最小式利用 Lagrange 對 w_ij(u_ij^m) 做微分 = m*u_ij^(m-1) * (ζ(δ′_ij)^2 + η*||xj − ci||^2) - λ_j = 0
# => u_ij^(m-1) = (λ_j / m) * ( 1 / mu_ij^(m-1) * (ζ(δ′_ij)^2 + η||xj − ci||^2))
# => u_ij = (λ_j / m)^(1/m-1) * ( 1 / mu_ij^(m-1) * (ζ(δ′_ij)^2 + η||xj − ci||^2))^(1/m-1)
# 1 = Σ(i=1,K)u_ij = Σ(i=1,K)(λ_j / m)^(1/m-1) * ( 1 / mu_ij^(m-1) * (ζ(δ′_ij)^2 + η||xj − ci||^2))^(1/m-1)
# (λ_j / m)^(1/m-1) = 1 / (Σ(l=1,K) 1/(ζ(δ′_ij)^2 + η||xj − ci||^2)^(2/m) )
# u_ij = 1 / (Σ(l=1,K) ((ζ(δ′_ij)^2 + η||xj − ci||^2) / (ζ(δ′_ij)^2 + η||xj − cl||^2))^(2/m-1) )

# 對 4.1 最小式利用 Lagrange 對 ci 做微分 = Σ(j=1,N) u_ij^m * 2 * (xj − ci)^1 * (0 - 1)
# => Σ(j=1,N) -2u_ij^m * (xj − ci) = 0
# => Σ(j=1,N) (u_ij^m * xj) - (Σ(j=1,N) (u_ij^m)) * ci = 0
# => ci = (Σ(j=1,N) (u_ij^m * xj)) / (Σ(j=1,N) (u_ij^m))  這其實是加權後平均
##############################################

# 以下為論文文中設定的測試參數
global Epsilon  # ε 收斂常數、收斂閥值 0 <= ε <= 1  愈小所需執行時間越久 所需時間程度為指數
Epsilon = 0.00000001

global Sigma  # σ 論文預設 1
Sigma = 1

global Delta  # δ
Delta = 0

global Zeta  # ζ  ratio coefficients z+n=1  設1將忽略群心導致只有一群
Zeta = 0.01

global Eta  # η   ratio coefficients z+n=1
Eta = 1 - Zeta

global DeltaBar  # δ′ 範圍為 [0,1]  social distance
DeltaBar = math.e ** ((-1 * Delta**2) / (2 * Sigma**2))


def print_matrix(list: list) -> None:
    """輸出串列 list (Debug用)

    Args:
        list (list): 要輸出的串列
    """
    for i in range(0, len(list)):
        print(list[i])


def initialize_U(data: list[list[float]], cluster_n: int) -> list[list[float]]:
    """
    隨機產生每個資料對應的隸屬值串列.
    在此相當於隨機分配點給群.

    Args:
        data (list[list[float]]): 要分群之資料
        cluster_n (int): 群數

    Returns:
        list[list[float]]: 產生出的隸屬值
    """
    U = []
    # 遍歷全部座標點
    for i in range(0, len(data)):
        current = []
        rand_sum = 0.0
        for j in range(0, cluster_n):
            dummy = random.randint(1, int(100000))  # 100000 > cluster_n
            current.append(dummy)
            rand_sum += dummy

        # 轉成百分比且點與各群之隸屬值之總和必定為1
        for j in range(0, cluster_n):
            current[j] = current[j] / rand_sum

        U.append(current)
    return U


def calDist(point1: list[float], point2: list[float]) -> float:
    """使用歐基里德距離公式計算兩點距離

    Args:
        point1 (list[float]): 第一個座標點
        point2 (list[float]): 第二個座標點

    Returns:
        float: 兩點距離
    """

    # 確保輸入正確
    if len(point1) != len(point2):
        return -1
    dummy = 0.0
    # 做成k-way list也可以處裡的方式
    for i in range(0, len(point1)):
        dummy += abs(point1[i] - point2[i]) ** 2
    return math.sqrt(dummy)


def end_conditon(U: list[list[float]], U_last: list[list[float]]) -> bool:
    """
    收斂判斷

    根據論文 Table2,S4   結束判斷: |cfm - cf'm| <= e

    Args:
        U (list[list[float]]): 上次收斂值串列
        U_last (list[list[float]]): 當前收斂值串列

    Returns:
        bool: 結果是否收斂了(表示結果已經不會改變了可以停了)
    """
    global Epsilon
    for i in range(0, len(U)):
        for j in range(0, len(U[0])):  # len(U[0]) = cluster_n
            if abs(U[i][j] - U_last[i][j]) > Epsilon:
                return False
    return True


def normalise_U(U: list[list[float]]) -> list[list[float]]:
    """
    分群結束時標準化隸屬值, 使該點分配之群之隸屬值為1其餘0

    Args:
        U (list[list[float]]): 全部座標點之個別隸屬值

    Returns:
        list[list[float]]: 標準化後的座標隸屬值
    """
    for i in range(0, len(U)):
        maximum = max(U[i])
        for j in range(0, len(U[0])):  # len(U[0]) = 群數
            U[i][j] = 0 if U[i][j] != maximum else 1
    return U


def algorithm(data: list[list[float]], cluster_n: int) -> list[float]:
    """主邏輯函數

    Args:
        data (list[list[float]]): 要分群的資料
        cluster_n (int): 群數

    Returns:
        list[float]: 返回各點與群的最終隸屬值
    """

    # 初始化隸屬值串列
    U = initialize_U(data, cluster_n)

    # Table2 S2 S3 循環執行直到收斂
    while True:
        U_old = copy.deepcopy(U)  # 保存上次執行結果
        C = []  # 保存各群中心點

        for j in range(0, cluster_n):
            cluster_center = []  # Ci 公式為拉葛朗日乘子對Ci微分為0
            for i in range(0, len(data[0])):  # k-way 做法
                denominator = 0.0  # Cj 中的各點隸屬值
                numerator = 0.0
                for k in range(0, len(data)):
                    denominator += U[k][j]  * data[k][i]  # 分子
                    numerator += U[k][j]  # 分母
                cluster_center.append(denominator / numerator)  # 新增群中心
            C.append(cluster_center)

        distance = []  # 保存距離向量後續用來計算隸屬值
        for i in range(0, len(data)):
            current = []
            for j in range(0, cluster_n):
                current.append(calDist(data[i], C[j]))  # 計算資料與群中心的距離
            distance.append(current)

        # 更新隸屬值
        for j in range(0, cluster_n):
            for i in range(0, len(data)):
                sum = 0.0
                for k in range(0, cluster_n):
                    # Σ(l=1,K) ((ζ(δ′_ij)^2 + η||xj − ci||^2) / (ζ(δ′_ij)^2 + η||xj − cl||^2))^(2/m-1)
                    a = Eta * (distance[i][j] ** 2) + (Zeta * DeltaBar**2)
                    b = Eta * (distance[i][k] ** 2) + (Zeta * DeltaBar**2)
                    sum += (a / b)**1.1  # 分母
                U[i][j] = 1 / sum # 轉成隸屬值

        # 檢查是否收斂了
        if end_conditon(U, U_old):
            break
    print_matrix(U) # 印出分群結果
    U = normalise_U(U)
    print("Done!")
    return U


def randCriclePoint(radius: float, x_center: float, y_center: float) -> list[float]:
    """
    產生以座標x_center, y_center為中心,
    半徑radius內的隨機座標點

    Args:
        radius (float): 圓半徑
        x_center (float): 圓心x座標
        y_center (float): 圓心y座標

    Returns:
        list[float]: 隨機座標點
    """
    r = (radius) * math.sqrt(random.uniform(0, 1))
    theta = (2 * math.pi) * random.uniform(0, 1)  # random angle
    return [r * math.cos(theta) + x_center, r * math.sin(theta) + y_center]


def randNCriclePoint(
    radius: float, x_center: float, y_center: float, n: int
) -> list[list[float]]:
    """
    產生n個以座標x_center, y_center為中心,
    半徑radius內的隨機座標點

    Args:
        radius (float): 圓半徑
        x_center (float): 圓心x座標
        y_center (float): 圓心y座標
        n (int): 要產生的座標點數量

    Returns:
        list[list[float]]: 1x(nx2) 的座標點數量串列
    """
    nums = []
    for i in range(0, n):
        nums.append(randCriclePoint(radius, x_center, y_center))  # uniform
    return nums


if __name__ == "__main__":
    cluster_n = 5  # 分群數 群數越高耗時越久
    data_count = 300 # 資料數(UE數)
    data_cx = 500
    data_cy = 500
    data_r = 500
    colors = list(mcolors.TABLEAU_COLORS if cluster_n < 11 else mcolors.CSS4_COLORS) # 顏色表

    # 產生測試數據
    data = randNCriclePoint(data_r, data_cx, data_cy, data_count)
    res_U = algorithm(data, cluster_n)

    # 製作圖表方便觀看
    fig, ax = plt.subplots()
    for i, val in enumerate(data):
        ax.scatter(val[0], val[1], c=colors[res_U[i].index(1)])

    # 繪製圓圈確保數據落點正確
    cir = plt.Circle((data_cx, data_cy), data_r, color="r", fill=False)
    ax.set_aspect("equal", adjustable="datalim")
    ax.add_patch(cir)
    # 以上繪製圓圈

    plt.show()
