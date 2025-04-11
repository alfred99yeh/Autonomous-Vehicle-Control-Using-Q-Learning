import math as m
import random as r
from simple_geometry import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# 設定隨機種子，確保結果可重複
r.seed(42)
np.random.seed(42)

# 定義車體類別
class Car():
    def __init__(self) -> None:
        self.radius = 6               # 車體半徑為 6 單位
        self.angle_min = -90          # 車體允許的最小角度 -90 度
        self.angle_max = 270          # 車體允許的最大角度 270 度
        self.wheel_min = -40          # 方向盤允許的最小轉角 -40 度
        self.wheel_max = 40           # 方向盤允許的最大轉角 40 度
        self.xini_max = 4.5           # 車體初始 x 座標上限
        self.xini_min = -4.5          # 車體初始 x 座標下限

        self.reset()                  # 呼叫重置方法初始化車體狀態

    @property
    def diameter(self):
        return self.radius/2          # 返回車體直徑的一半（可能代表車體的寬度）

    def reset(self):
        self.angle = 90               # 重置車體角度為 90 度
        self.wheel_angle = 0          # 重置方向盤角度為 0 度

        # 計算車體初始 x 座標範圍
        xini_range = (self.xini_max - self.xini_min - self.radius)
        left_xpos = self.xini_min + self.radius//2
        # 隨機產生車體初始 x 座標（大約在 [-3, 3] 範圍內）
        self.xpos = r.random()*xini_range + left_xpos
        self.ypos = 0                 # 初始 y 座標設定為 0


    def setWheelAngle(self, angle):
        # 根據輸入角度設定方向盤角度，同時限制在允許的範圍內
        self.wheel_angle = angle if self.wheel_min <= angle <= self.wheel_max else (
            self.wheel_min if angle <= self.wheel_min else self.wheel_max)

    def setPosition(self, newPosition: Point2D):
        # 設定車體位置為傳入的 Point2D 物件中的 x 與 y 座標
        self.xpos = newPosition.x
        self.ypos = newPosition.y

    def getPosition(self, point='center') -> Point2D:
        # 根據參數返回車體的不同部位位置，預設返回中心位置
        if point == 'right':
            right_angle = self.angle - 45                      # 右側位置：車體角度減 45 度
            right_point = Point2D(self.radius/2, 0).rorate(right_angle)  # 取得右側偏移向量（旋轉後）
            return Point2D(self.xpos, self.ypos) + right_point # 返回車體中心加上右側偏移向量
        elif point == 'left':
            left_angle = self.angle + 45                       # 左側位置：車體角度加 45 度
            left_point = Point2D(self.radius/2, 0).rorate(left_angle)  # 取得左側偏移向量
            return Point2D(self.xpos, self.ypos) + left_point  # 返回車體中心加上左側偏移向量
        elif point == 'front':
            # 計算車體前方的位置（以車體半徑的一半作為偏移距離）
            fx = m.cos(self.angle/180*m.pi)*self.radius/2 + self.xpos
            fy = m.sin(self.angle/180*m.pi)*self.radius/2 + self.ypos
            return Point2D(fx, fy)                             # 返回前方位置
        else:
            return Point2D(self.xpos, self.ypos)               # 返回車體中心位置

    def getWheelPosPoint(self):
        # 計算方向盤所在位置，根據方向盤角度與車體角度共同計算
        wx = m.cos((-self.wheel_angle+self.angle)/180*m.pi) * self.radius/2 + self.xpos
        wy = m.sin((-self.wheel_angle+self.angle)/180*m.pi) * self.radius/2 + self.ypos
        return Point2D(wx, wy)

    def setAngle(self, new_angle):
        # 設定車體角度，先取模 360 保證在 0-360 之間
        new_angle %= 360
        if new_angle > self.angle_max:
            new_angle -= self.angle_max - self.angle_min   # 超過最大角度則調整
        self.angle = new_angle                              # 更新車體角度

    def tick(self):
        '''
        更新車體狀態：根據當前角度與方向盤角度，計算下一個時刻的車體位置和角度
        '''
        car_angle = self.angle/180*m.pi        # 將車體角度轉換成弧度
        wheel_angle = self.wheel_angle/180*m.pi  # 將方向盤角度轉換成弧度
        # 根據運動模型計算新的 x 座標
        new_x = self.xpos + m.cos(car_angle+wheel_angle) + m.sin(wheel_angle)*m.sin(car_angle)
        # 根據運動模型計算新的 y 座標
        new_y = self.ypos + m.sin(car_angle+wheel_angle) - m.sin(wheel_angle)*m.cos(car_angle)
        # 根據運動模型計算新的車體角度（轉換回角度制）
        new_angle = (car_angle - m.asin(2*m.sin(wheel_angle) / (self.radius*1.5))) / m.pi * 180

        new_angle %= 360                        # 保證角度在 0-360 之間
        if new_angle > self.angle_max:
            new_angle -= self.angle_max - self.angle_min  # 調整超出範圍的角度

        self.xpos = new_x                      # 更新 x 座標
        self.ypos = new_y                      # 更新 y 座標
        self.setAngle(new_angle)               # 更新車體角度

# 定義 Playground 環境類別，負責處理軌道、感測器及碰撞偵測等
class Playground():
    def __init__(self):
        self.path_line_filename = "軌道座標點.txt"  # 軌道座標資料檔案名稱
        self._setDefaultLine()                    # 使用預設的軌道線
        self.decorate_lines = [
            Line2D(-6, 0, 6, 0),                 # 裝飾線：起點線
            Line2D(0, 0, 0, -3),                 # 裝飾線：中間參考線
        ]
        self.car = Car()                          # 建立車體物件
        self.reset()                              # 重置環境狀態

        # 新增 Q-Learning 相關屬性
        self.Q_table = {}     # Q表，儲存每個離散狀態的 Q 值向量
        self.epsilon = 0.3    # 探索率

    def _setDefaultLine(self):
        print('use default lines')                 # 印出提示訊息：使用預設軌道線
        self.destination_line = Line2D(18, 40, 30, 37)  # 設定終點區域邊界（以一條線表示）
        # 設定跑道邊界線，依序連接成一個封閉的路徑
        self.lines = [
            Line2D(-6, -3, 6, -3),
            Line2D(6, -3, 6, 10),
            Line2D(6, 10, 30, 10),
            Line2D(30, 10, 30, 50),
            Line2D(18, 50, 30, 50),
            Line2D(18, 22, 18, 50),
            Line2D(-6, 22, 18, 22),
            Line2D(-6, -3, -6, 22),
        ]
        self.car_init_pos = None                    # 初始車體位置（若有檔案讀取則更新）
        self.car_init_angle = None                  # 初始車體角度

    def _readPathLines(self):
        try:
            # 嘗試從指定檔案讀取軌道資料
            with open(self.path_line_filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # 讀取第一行：車體起始位置和角度
                pos_angle = [float(v) for v in lines[0].split(',')]
                self.car_init_pos = Point2D(*pos_angle[:2])
                self.car_init_angle = pos_angle[-1]
                # 讀取第二、三行：終點區域的兩個座標點
                dp1 = Point2D(*[float(v) for v in lines[1].split(',')])
                dp2 = Point2D(*[float(v) for v in lines[2].split(',')])
                self.destination_line = Line2D(dp1, dp2)
                # 從第四行開始讀取跑道邊界節點
                self.lines = []
                inip = Point2D(*[float(v) for v in lines[3].split(',')])
                for strp in lines[4:]:
                    p = Point2D(*[float(v) for v in strp.split(',')])
                    line = Line2D(inip, p)
                    inip = p
                    self.lines.append(line)
        except Exception:
            # 若檔案讀取失敗，則使用預設軌道
            self._setDefaultLine()

    def predictAction(self, state):
        '''
        此 function 為模擬時，給予車子隨機動作（隨機選擇方向盤角度）。
        未來可依據 Q-Learning 模型來替換此隨機動作。
        '''
        '''
        根據當前狀態使用 Q-Learning 模型預測動作（方向盤角度）。
        此處採用 ε-greedy 策略：
        - 以 epsilon 機率隨機選擇動作（探索）
        - 否則選擇 Q 值最大的動作（利用）
        若狀態尚未存在於 Q 表中，則先初始化該狀態的 Q 值向量為全 0。
        '''
        # 將連續狀態離散化，假設 discretize_state() 為已實作的函式
        state_disc = discretize_state(state)
        
        # 如果該離散狀態尚未在 Q 表中，則初始化
        if state_disc not in self.Q_table:
            self.Q_table[state_disc] = np.zeros(self.n_actions)
        
        # 以 ε-greedy 策略選擇動作
        if r.random() < self.epsilon:
            # 探索：隨機選擇一個動作
            return r.randint(0, self.n_actions - 1)
        else:
            # 利用：選擇 Q 值最大的動作
            q_values = self.Q_table[state_disc]
            return int(np.argmax(q_values))

    @property
    def n_actions(self):
        # 動作數量：根據方向盤允許轉角範圍計算離散動作數
        return (self.car.wheel_max - self.car.wheel_min + 1)

    @property
    def observation_shape(self):
        return (len(self.state),)

    @property
    def state(self):
        # 根據感測器交點計算前、右、左三個方向的距離
        front_dist = -1 if len(self.front_intersects) == 0 else self.car.getPosition().distToPoint2D(self.front_intersects[0])
        right_dist = -1 if len(self.right_intersects) == 0 else self.car.getPosition().distToPoint2D(self.right_intersects[0])
        left_dist = -1 if len(self.left_intersects) == 0 else self.car.getPosition().distToPoint2D(self.left_intersects[0])
        return [front_dist, right_dist, left_dist]
    
        # ------------------- 新增 getReward 方法 -------------------
    def getReward(self):
        """
        根據車體中心是否進入終點區域與碰撞情況回傳獎勵：
          - 到達終點區域: +100
          - 碰撞 (但未到達終點): -100
          - 否則: -1 (每一步都有步數懲罰)
        """
        # 取得車體中心位置
        cpos = self.car.getPosition('center')
        # 計算車體中心到終點線的距離
        distance_to_destination = cpos.distToLine2D(self.destination_line)
        # print(f"Distance to destination: {distance_to_destination:.2f}")
        
        if cpos.isInRect(self.destination_line.p1, self.destination_line.p2):
            print("到達終點區域")
            return 100   # 到達終點區域獎勵
        elif self.done:
            return -100  # 碰撞懲罰
        else:
            return 0.02*(50 - distance_to_destination) # 每一步的獎勵（距離越近獎勵越高）

    def _checkDoneIntersects(self):
        if self.done:
            return self.done

        # 取得車體各部位的位置：中心、前、右、左
        cpos = self.car.getPosition('center')     # 車體中心
        cfront_pos = self.car.getPosition('front')   # 前方
        cright_pos = self.car.getPosition('right')   # 右側
        cleft_pos = self.car.getPosition('left')     # 左側
        diameter = self.car.diameter                 # 車體直徑的一半

        # 判斷車體中心是否位於終點區域內
        isAtDestination = cpos.isInRect(self.destination_line.p1, self.destination_line.p2)
        done = False if not isAtDestination else True

        # 初始化各方向感測器交點的儲存與狀態
        front_intersections, find_front_inter = [], True
        right_intersections, find_right_inter = [], True
        left_intersections, find_left_inter = [], True

        # 針對每一條跑道邊界線進行檢查
        for wall in self.lines:
            dToLine = cpos.distToLine2D(wall)   # 計算車體中心到牆線的距離
            p1, p2 = wall.p1, wall.p2
            dp1, dp2 = (cpos-p1).length, (cpos-p2).length  # 計算車體中心到牆線端點的距離
            wall_len = wall.length              # 牆線長度

            # 判斷是否觸碰牆線
            p1_touch = (dp1 < diameter)
            p2_touch = (dp2 < diameter)
            body_touch = (dToLine < diameter and (dp1 < wall_len and dp2 < wall_len))
            # 檢查車體前方、右側、左側射線是否與牆線交叉
            front_touch, front_t, front_u = Line2D(cpos, cfront_pos).lineOverlap(wall)
            right_touch, right_t, right_u = Line2D(cpos, cright_pos).lineOverlap(wall)
            left_touch, left_t, left_u = Line2D(cpos, cleft_pos).lineOverlap(wall)

            if p1_touch or p2_touch or body_touch or front_touch:
                if not done:
                    done = True

            # 若有交點，計算各感測器射線與牆線的交點位置
            if find_front_inter and front_u and 0 <= front_u <= 1:
                front_inter_point = (p2 - p1) * front_u + p1
                if front_t:
                    if front_t > 1:  # 僅選取位於車前方的交點
                        front_intersections.append(front_inter_point)
                    elif front_touch:  # 若交點重疊則不使用
                        front_intersections = []
                        find_front_inter = False

            if find_right_inter and right_u and 0 <= right_u <= 1:
                right_inter_point = (p2 - p1) * right_u + p1
                if right_t:
                    if right_t > 1:  # 僅選取車前方的交點
                        right_intersections.append(right_inter_point)
                    elif right_touch:
                        right_intersections = []
                        find_right_inter = False

            if find_left_inter and left_u and 0 <= left_u <= 1:
                left_inter_point = (p2 - p1) * left_u + p1
                if left_t:
                    if left_t > 1:  # 僅選取車前方的交點
                        left_intersections.append(left_inter_point)
                    elif left_touch:
                        left_intersections = []
                        find_left_inter = False

        # 設定各感測器的交點
        self._setIntersections(front_intersections, left_intersections, right_intersections)
        self.done = done  # 更新模擬是否結束的狀態
        return done

    def _setIntersections(self, front_inters, left_inters, right_inters):
        # 依據與車體前方的距離排序交點，便於取得最接近的交點
        self.front_intersects = sorted(front_inters, key=lambda p: p.distToPoint2D(self.car.getPosition('front')))
        self.right_intersects = sorted(right_inters, key=lambda p: p.distToPoint2D(self.car.getPosition('right')))
        self.left_intersects = sorted(left_inters, key=lambda p: p.distToPoint2D(self.car.getPosition('left')))

    def reset(self):
        self.done = False                # 重置模擬結束狀態為 False
        self.car.reset()                 # 重置車體狀態

        if self.car_init_angle and self.car_init_pos:
            self.setCarPosAndAngle(self.car_init_pos, self.car_init_angle)
        self._checkDoneIntersects()      # 檢查是否碰撞或到達終點
        return self.state              # 返回當前狀態

    def setCarPosAndAngle(self, position: Point2D = None, angle=None):
        if position:
            self.car.setPosition(position)  # 設定車體位置
        if angle:
            self.car.setAngle(angle)         # 設定車體角度
        self._checkDoneIntersects()          # 更新模擬狀態

    def calWheelAngleFromAction(self, action):
        # 根據動作索引計算對應的方向盤角度
        angle = self.car.wheel_min + action * (self.car.wheel_max - self.car.wheel_min) / (self.n_actions - 1)
        return angle

    def step(self, action=None):
        '''
        執行一步模擬：根據指定的動作更新車體狀態，並回傳新的狀態
        '''
        if action:
            angle = self.calWheelAngleFromAction(action=action)  # 根據動作索引計算方向盤角度
            self.car.setWheelAngle(angle)                        # 設定方向盤角度

        if not self.done:
            self.car.tick()                # 更新車體狀態
            self._checkDoneIntersects()    # 檢查是否碰撞或到達終點
            return self.state              # 回傳新狀態
        else:
            return self.state

def discretize_state(state, num_bins=5, sensor_min=0, sensor_max=30):
    bins = np.linspace(sensor_min, sensor_max, num_bins+1)
    discrete_state = tuple(np.digitize([s], bins)[0] for s in state)
    return discrete_state

# ------------------- 訓練函式 -------------------
def train(env, num_episodes, alpha=0.1, gamma=0.9):
    """
    使用 Q-Learning 對環境進行訓練，更新 Q_table，
    並在每個回合結束時印出總獎勵。
    """
    max_reward = -float('inf')  # 初始化最大獎勵為負無窮
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        step_count = 0
        total_reward = 0  # 累計每回合的獎勵
        while not done:
            state_disc = discretize_state(state)
            # ε-greedy 選擇動作
            if r.random() < env.epsilon:
                action = r.randint(0, env.n_actions - 1)
            else:
                if state_disc not in env.Q_table:
                    env.Q_table[state_disc] = np.zeros(env.n_actions)
                action = int(np.argmax(env.Q_table[state_disc]))
            # 在訓練過程中每個回合後降低 epsilon（例如乘以0.99）
            env.epsilon *= 0.99

            # 執行動作
            next_state = env.step(action)
            reward = env.getReward()
            total_reward += reward  # 累計獎勵
            next_state_disc = discretize_state(next_state)
            if state_disc not in env.Q_table:
                env.Q_table[state_disc] = np.zeros(env.n_actions)
            if next_state_disc not in env.Q_table:
                env.Q_table[next_state_disc] = np.zeros(env.n_actions)
            old_value = env.Q_table[state_disc][action]
            next_max = np.max(env.Q_table[next_state_disc])
            # Q-Learning 更新公式
            new_value = old_value + alpha * (reward + gamma * next_max - old_value)
            env.Q_table[state_disc][action] = new_value

            state = next_state
            done = env.done
            step_count += 1
        print(f"Episode {episode+1} finished in {step_count} steps, total reward: {total_reward}")
        if total_reward > max_reward:
            max_reward = total_reward
            max_reward_episode = episode + 1
    print("Training complete.")
    print(f"Highest reward was {max_reward} in episode {max_reward_episode}")
        
def run_gui(env):
    # 使用訓練後的環境，並關閉探索(設定 epsilon = 0)
    env.epsilon = 0  
    env.reset()

    fig, ax = plt.subplots(figsize=(4, 5))
    ax.set_aspect('equal')
    ax.set_xlim(-20, 40)
    ax.set_ylim(-20, 60)
    ax.set_title("自走車模擬 (使用Q-Learning策略)")

    # ==================== 繪製軌道邊界並收集頂點 ====================
    points = set()
    for wall in env.lines:
        x_vals = [wall.p1.x, wall.p2.x]
        y_vals = [wall.p1.y, wall.p2.y]
        ax.plot(x_vals, y_vals, 'k-', lw=2)
        points.add((wall.p1.x, wall.p1.y))
        points.add((wall.p2.x, wall.p2.y))
    for (px, py) in points:
        ax.text(px, py, f"({px},{py})", color='blue', fontsize=8,
                ha='center', va='bottom')

    # ==================== 繪製終點區域 ====================
    dest = env.destination_line
    ax.plot([dest.p1.x, dest.p2.x], [dest.p1.y, dest.p2.y], 'r-', lw=2, label='終點')
    
    # ==================== 繪製裝飾線 (如起點線) ====================
    for dline in env.decorate_lines:
        ax.plot([dline.p1.x, dline.p2.x], [dline.p1.y, dline.p2.y], 'b--', lw=2)

    # ==================== 車體 (圓形) ====================
    car_body = Circle((env.car.xpos, env.car.ypos), env.car.radius/2, fc='r', ec='k')
    ax.add_patch(car_body)
    
    # ==================== 車體方向箭頭 ====================
    arrow_length = env.car.radius
    car_arrow = ax.arrow(env.car.xpos, env.car.ypos,
                         arrow_length*m.cos(env.car.angle/180*m.pi),
                         arrow_length*m.sin(env.car.angle/180*m.pi),
                         head_width=1.0, head_length=2.0, fc='k', ec='k')
    
    # ==================== 感測器射線 ====================
    sensor_lines = {}
    sensor_colors = {'front': 'm', 'right': 'c', 'left': 'y'}
    for sensor in ['front', 'right', 'left']:
        line_obj, = ax.plot([], [], sensor_colors[sensor]+'-', lw=1.5)
        sensor_lines[sensor] = line_obj
    
    # ==================== 車輛行走軌跡 ====================
    path_positions = [env.car.getPosition('center')]
    path_line, = ax.plot([env.car.getPosition('center').x],
                         [env.car.getPosition('center').y], 'r--', lw=1)
    
    # ==================== 顯示感測器距離文字 ====================
    sensor_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)
    
    def update(frame):
        nonlocal car_arrow
        if env.done:
            ani.event_source.stop()
            return
        
        state = env.state
        # 這裡 predictAction 會依據訓練後的 Q_table 選擇最佳動作
        action = env.predictAction(state)
        env.step(action)
        
        # 更新車體位置與方向箭頭
        center = env.car.getPosition('center')
        car_body.center = (center.x, center.y)
        car_arrow.remove()
        arrow_dx = arrow_length*m.cos(env.car.angle/180*m.pi)
        arrow_dy = arrow_length*m.sin(env.car.angle/180*m.pi)
        car_arrow = ax.arrow(center.x, center.y, arrow_dx, arrow_dy,
                             head_width=1.0, head_length=2.0, fc='k', ec='k')
        
        # 更新感測器射線
        for sensor in ['front', 'right', 'left']:
            sensor_pos = env.car.getPosition(sensor)
            if sensor == 'front' and len(env.front_intersects) > 0:
                inter_pt = env.front_intersects[0]
            elif sensor == 'right' and len(env.right_intersects) > 0:
                inter_pt = env.right_intersects[0]
            elif sensor == 'left' and len(env.left_intersects) > 0:
                inter_pt = env.left_intersects[0]
            else:
                if sensor == 'front':
                    angle = env.car.angle
                elif sensor == 'right':
                    angle = env.car.angle - 45
                elif sensor == 'left':
                    angle = env.car.angle + 45
                inter_pt = Point2D(sensor_pos.x + 20*m.cos(angle/180*m.pi),
                                   sensor_pos.y + 20*m.sin(angle/180*m.pi))
            sensor_lines[sensor].set_data([sensor_pos.x, inter_pt.x],
                                          [sensor_pos.y, inter_pt.y])
        
        # 更新軌跡
        path_positions.append(center)
        xs = [pt.x for pt in path_positions]
        ys = [pt.y for pt in path_positions]
        path_line.set_data(xs, ys)
        
        print(env.state, center.x, center.y)
        
        sensor_info = f"前: {state[0]:.2f} | 右: {state[1]:.2f} | 左: {state[2]:.2f}"
        sensor_text.set_text(sensor_info)
    
    ani = animation.FuncAnimation(fig, update, interval=100)
    plt.show()
        
if __name__ == "__main__":
    # run_example()
    env = Playground()
    train(env, num_episodes=5000)  # 訓練 20000 回合
    print("訓練完成，現在展示模擬結果：")
    run_gui(env)