#!/home/pi/ultrasonic_gesture/venv/bin/python3
"""
接物下落游戏 (Pygame版本) - 漏接只统计好球
底部接盘由超声波距离控制
"""
import os
import sys

# 修复VNC显示问题 - 强制使用X11
os.environ['SDL_VIDEODRIVER'] = 'x11'
os.environ['DISPLAY'] = ':0'

import RPi.GPIO as GPIO
import time
import pygame
import random
import json
from datetime import datetime

# 引脚定义
TRIG = 23
ECHO = 24

# 游戏配置
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
GAME_DURATION = 120  # 秒

# 颜色定义
COLORS = {
    'WHITE': (255, 255, 255),
    'BLACK': (0, 0, 0),
    'RED': (255, 0, 0),
    'GREEN': (0, 255, 0),
    'BLUE': (0, 0, 255),
    'YELLOW': (255, 255, 0),
    'GRAY': (128, 128, 128),
    'DARK_GRAY': (64, 64, 64),
    'CYAN': (0, 255, 255),
    'PURPLE': (255, 0, 255),
    'ORANGE': (255, 165, 0)
}


class UltrasonicSensor:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(TRIG, GPIO.OUT)
        GPIO.setup(ECHO, GPIO.IN)
        
        # 距离映射参数
        self.min_distance = 10  # cm
        self.max_distance = 40  # cm
        
    def get_distance(self):
        """获取单次距离测量"""
        GPIO.output(TRIG, False)
        time.sleep(0.005)
        
        GPIO.output(TRIG, True)
        time.sleep(0.00001)
        GPIO.output(TRIG, False)
        
        pulse_start = time.time()
        timeout = pulse_start + 0.04
        
        while GPIO.input(ECHO) == 0 and pulse_start < timeout:
            pulse_start = time.time()
        
        if pulse_start >= timeout:
            return None
        
        pulse_end = time.time()
        while GPIO.input(ECHO) == 1 and pulse_end < timeout:
            pulse_end = time.time()
        
        if pulse_end >= timeout:
            return None
        
        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17150
        
        if 2 <= distance <= 400:
            return distance
        return None
    
    def get_position(self):
        """获取距离映射到屏幕位置 (0 ~ SCREEN_WIDTH)"""
        dist = self.get_distance()
        if dist is None:
            return None
        
        if dist < self.min_distance:
            dist = self.min_distance
        elif dist > self.max_distance:
            dist = self.max_distance
        
        position = SCREEN_WIDTH - ((dist - self.min_distance) / (self.max_distance - self.min_distance) * SCREEN_WIDTH)
        return int(position)
    
    def cleanup(self):
        GPIO.cleanup()


class FallingBall:
    def __init__(self, is_good=True):
        self.radius = 25
        self.x = random.randint(self.radius, SCREEN_WIDTH - self.radius)
        self.y = -self.radius
        self.speed = 3 + random.random() * 2.5
        self.is_good = is_good
        self.color = COLORS['GREEN'] if is_good else COLORS['RED']
        self.caught = False
        self.missed = False
        self.scored = False
    
    def update(self):
        self.y += self.speed
    
    def is_off_screen(self):
        return self.y > SCREEN_HEIGHT + self.radius
    
    def draw(self, screen):
        # 发光效果
        for i in range(3, 0, -1):
            alpha_color = tuple(c - i*30 for c in self.color)
            alpha_color = tuple(max(0, c) for c in alpha_color)
            pygame.draw.circle(screen, alpha_color, (self.x, int(self.y)), self.radius + i*3, 1)
        
        pygame.draw.circle(screen, self.color, (self.x, int(self.y)), self.radius)
        pygame.draw.circle(screen, COLORS['WHITE'], (self.x, int(self.y)), self.radius, 2)
        
        # 标签
        label = "OK" if self.is_good else "NO"
        font = pygame.font.Font(None, 28)
        text = font.render(label, True, COLORS['WHITE'])
        text_rect = text.get_rect(center=(self.x, int(self.y)))
        screen.blit(text, text_rect)
    
    def check_catch(self, paddle_x, paddle_width):
        """检查是否被接住"""
        if self.caught or self.missed:
            return None
        
        if self.y + self.radius >= SCREEN_HEIGHT - 35:
            if paddle_x - paddle_width//2 <= self.x <= paddle_x + paddle_width//2:
                self.caught = True
                return 'caught'
            else:
                self.missed = True
                return 'missed'
        return None


class Game:
    def __init__(self):
        # 初始化Pygame
        pygame.init()
        
        # 设置显示模式
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Catching Game - Ultrasonic Control")
        
        # 创建中文字体
        self.chinese_font = None
        font_names = [
            'WenQuanYiMicroHei',
            'WenQuanYi Zen Hei',
            'DroidSansFallback',
            'NotoSansCJK',
            'NotoSansCJK-Regular',
            'SimHei',
            'MicrosoftYaHei',
            'Arial Unicode MS'
        ]
        
        for font_name in font_names:
            try:
                self.chinese_font = pygame.font.Font(font_name, 24)
                break
            except:
                continue
        
        if self.chinese_font is None:
            try:
                self.chinese_font = pygame.font.SysFont('Arial', 24)
            except:
                self.chinese_font = pygame.font.Font(None, 24)
        
        # 字体
        self.font_large = pygame.font.Font(None, 52)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        self.font_tiny = pygame.font.Font(None, 18)
        
        # 游戏状态
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False
        
        # 游戏变量
        self.score = 0
        self.good_caught = 0
        self.bad_caught = 0
        self.good_missed = 0      # 只统计好球漏接
        self.bad_missed = 0       # 坏球漏接（不显示，只用于统计）
        self.start_time = None
        self.elapsed_time = 0
        self.game_over = False
        
        # 接盘
        self.paddle_width = 80
        self.paddle_height = 15
        self.paddle_x = SCREEN_WIDTH // 2
        self.paddle_y = SCREEN_HEIGHT - 35
        
        # 球
        self.balls = []
        self.spawn_timer = 0
        self.spawn_interval = 1.0
        
        # 传感器
        self.sensor = UltrasonicSensor()
        
        # 日志
        self.log_data = []
        
        # 帧率
        self.frame_count = 0
        self.fps = 0
        self.fps_timer = 0
        
        # 距离显示
        self.current_distance = None
        
        # 浮动消息
        self.game_messages = []
    
    def spawn_ball(self):
        """生成新的球"""
        is_good = random.random() < 0.7
        ball = FallingBall(is_good)
        self.balls.append(ball)
    
    def update_paddle(self):
        """更新接盘位置"""
        pos = self.sensor.get_position()
        if pos is not None:
            self.paddle_x += (pos - self.paddle_x) * 0.25
            self.current_distance = self.sensor.get_distance()
    
    def update_balls(self):
        """更新所有球 - 漏接只统计好球"""
        for ball in self.balls[:]:
            ball.update()
            
            if not ball.caught and not ball.missed:
                if ball.y + ball.radius >= SCREEN_HEIGHT - 35:
                    if self.paddle_x - self.paddle_width//2 <= ball.x <= self.paddle_x + self.paddle_width//2:
                        # 接住了
                        ball.caught = True
                        if ball.is_good:
                            self.score += 1
                            self.good_caught += 1
                            self.game_messages.append(('+1', time.time(), COLORS['GREEN']))
                        else:
                            self.score -= 1
                            self.bad_caught += 1
                            self.game_messages.append(('-1', time.time(), COLORS['RED']))
                    else:
                        # 漏接了
                        ball.missed = True
                        if ball.is_good:
                            # 只有好球漏接才计数
                            self.good_missed += 1
                            self.game_messages.append(('MISS!', time.time(), COLORS['YELLOW']))
                        else:
                            # 坏球漏接不计数，只显示提示
                            self.bad_missed += 1
                            self.game_messages.append(('Bad Miss', time.time(), COLORS['ORANGE']))
            
            # 球离开屏幕
            if ball.is_off_screen():
                if not ball.caught and not ball.missed:
                    # 安全保护：如果球离开屏幕还没被处理
                    ball.missed = True
                    if ball.is_good:
                        self.good_missed += 1
                        self.game_messages.append(('MISS!', time.time(), COLORS['YELLOW']))
                    else:
                        self.bad_missed += 1
                self.balls.remove(ball)
        
        # 清理旧消息
        current_time = time.time()
        self.game_messages = [msg for msg in self.game_messages if current_time - msg[1] < 1.5]
    
    def draw_background(self):
        """绘制背景"""
        for i in range(SCREEN_HEIGHT):
            color_value = int(10 + (i / SCREEN_HEIGHT) * 40)
            color = (color_value, color_value, color_value + 30)
            pygame.draw.line(self.screen, color, (0, i), (SCREEN_WIDTH, i))
        
        pygame.draw.line(self.screen, COLORS['GRAY'], (0, SCREEN_HEIGHT - 20), (SCREEN_WIDTH, SCREEN_HEIGHT - 20), 2)
        pygame.draw.line(self.screen, COLORS['DARK_GRAY'], 
                        (0, SCREEN_HEIGHT - 40), (SCREEN_WIDTH, SCREEN_HEIGHT - 40), 1)
    
    def draw_text(self, text, x, y, color=COLORS['WHITE'], font=None, center=False):
        """绘制文本"""
        if font is None:
            font = self.chinese_font
        
        try:
            text_surface = font.render(text, True, color)
        except:
            try:
                default_font = pygame.font.Font(None, 24)
                text_surface = default_font.render(text, True, color)
            except:
                pygame.draw.rect(self.screen, color, (x, y, len(text)*10, 20), 1)
                return
        
        if center:
            text_rect = text_surface.get_rect(center=(x, y))
            self.screen.blit(text_surface, text_rect)
        else:
            self.screen.blit(text_surface, (x, y))
    
    def draw(self):
        """绘制所有内容"""
        self.draw_background()
        
        # 绘制接盘
        shadow_rect = (self.paddle_x - self.paddle_width//2 + 3, self.paddle_y + 3, self.paddle_width, self.paddle_height)
        pygame.draw.rect(self.screen, COLORS['DARK_GRAY'], shadow_rect)
        pygame.draw.rect(self.screen, COLORS['BLUE'], 
                        (self.paddle_x - self.paddle_width//2, self.paddle_y, 
                         self.paddle_width, self.paddle_height))
        pygame.draw.rect(self.screen, COLORS['WHITE'], 
                        (self.paddle_x - self.paddle_width//2, self.paddle_y, 
                         self.paddle_width, self.paddle_height), 2)
        
        # 绘制所有球
        for ball in self.balls:
            ball.draw(self.screen)
        
        # 浮动消息
        for msg, timestamp, color in self.game_messages:
            elapsed = time.time() - timestamp
            y_offset = int(elapsed * 60)
            self.draw_text(msg, SCREEN_WIDTH//2, 150 - y_offset, color, self.font_large, center=True)
        
        self.draw_ui()
        pygame.display.flip()
    
    def draw_ui(self):
        """绘制UI"""
        # Score
        score_text = self.font_large.render(f"Score: {self.score}", True, COLORS['WHITE'])
        self.screen.blit(score_text, (20, 10))
        
        # Timer
        if self.start_time:
            remaining = max(0, GAME_DURATION - (time.time() - self.start_time))
            minutes = int(remaining // 60)
            seconds = int(remaining % 60)
            time_text = self.font_medium.render(f"Time: {minutes:02d}:{seconds:02d}", True, COLORS['YELLOW'])
            time_rect = time_text.get_rect(center=(SCREEN_WIDTH//2, 25))
            self.screen.blit(time_text, time_rect)
        
        # Stats - 显示好球漏接
        stats = f"Good:{self.good_caught} Bad:{self.bad_caught} Miss:{self.good_missed}"
        stats_text = self.font_small.render(stats, True, COLORS['WHITE'])
        self.screen.blit(stats_text, (SCREEN_WIDTH - 260, 15))
        
        # 如果有好球漏接，高亮显示
        if self.good_missed > 0:
            miss_text = self.font_small.render(f"Missed: {self.good_missed}", True, COLORS['RED'])
            self.screen.blit(miss_text, (SCREEN_WIDTH - 260, 45))
        
        # Distance
        if self.current_distance:
            dist_text = self.font_small.render(f"Dist: {self.current_distance:.1f}cm", True, COLORS['WHITE'])
            self.screen.blit(dist_text, (20, SCREEN_HEIGHT - 30))
        
        # FPS
        fps_text = self.font_small.render(f"FPS: {self.fps:.0f}", True, COLORS['GRAY'])
        self.screen.blit(fps_text, (SCREEN_WIDTH - 80, SCREEN_HEIGHT - 30))
        
        # Pause
        if self.paused:
            pause_text = self.font_large.render("PAUSED", True, COLORS['YELLOW'])
            pause_rect = pause_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
            self.screen.blit(pause_text, pause_rect)
        
        # 操作提示
        tip_text = self.font_tiny.render("Hand near -> Right | Hand far -> Left | SPACE=Pause | ESC=Exit", True, COLORS['GRAY'])
        tip_rect = tip_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT - 5))
        self.screen.blit(tip_text, tip_rect)
    
    def log_event(self, event_type):
        """记录事件"""
        self.log_data.append({
            'timestamp': time.time(),
            'elapsed': time.time() - self.start_time if self.start_time else 0,
            'event': event_type,
            'score': self.score,
            'paddle_x': self.paddle_x,
            'distance': self.current_distance,
            'good_missed': self.good_missed,
            'good_caught': self.good_caught,
            'bad_caught': self.bad_caught
        })
    
    def save_log(self):
        """保存日志"""
        if not self.log_data:
            return
        
        log_dir = 'game_logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"game_log_{timestamp}.json"
        filepath = os.path.join(log_dir, filename)
        
        log_summary = {
            'game_duration': GAME_DURATION,
            'final_score': self.score,
            'good_caught': self.good_caught,
            'bad_caught': self.bad_caught,
            'good_missed': self.good_missed,
            'bad_missed': self.bad_missed,
            'accuracy': self.calculate_accuracy(),
            'logs': self.log_data
        }
        
        with open(filepath, 'w') as f:
            json.dump(log_summary, f, indent=2)
        
        print(f"\n✅ Game log saved to: {filepath}")
        return filepath
    
    def calculate_accuracy(self):
        """计算准确率 = 好球接住 / (好球接住 + 好球漏接)"""
        total_good = self.good_caught + self.good_missed
        if total_good == 0:
            return 0.0
        return self.good_caught / total_good * 100
    
    def show_game_over(self):
        """游戏结束画面"""
        self.screen.fill(COLORS['BLACK'])
        
        accuracy = self.calculate_accuracy()
        
        title = self.font_large.render("GAME OVER!", True, COLORS['YELLOW'])
        title_rect = title.get_rect(center=(SCREEN_WIDTH//2, 70))
        self.screen.blit(title, title_rect)
        
        score_text = self.font_large.render(f"Final Score: {self.score}", True, COLORS['WHITE'])
        score_rect = score_text.get_rect(center=(SCREEN_WIDTH//2, 140))
        self.screen.blit(score_text, score_rect)
        
        stats = [
            f"Good caught: {self.good_caught}",
            f"Bad caught: {self.bad_caught}",
            f"Good missed: {self.good_missed}",
            f"Total good balls: {self.good_caught + self.good_missed}",
            f"Accuracy: {accuracy:.1f}%"
        ]
        
        for i, stat in enumerate(stats):
            # 准确率用黄色高亮
            if "Accuracy" in stat:
                color = COLORS['YELLOW']
            elif "missed" in stat and self.good_missed > 5:
                color = COLORS['RED']
            else:
                color = COLORS['WHITE']
            text = self.font_medium.render(stat, True, color)
            text_rect = text.get_rect(center=(SCREEN_WIDTH//2, 220 + i * 45))
            self.screen.blit(text, text_rect)
        
        tip = self.font_small.render("Press R to Restart | ESC to Exit", True, COLORS['GRAY'])
        tip_rect = tip.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT - 40))
        self.screen.blit(tip, tip_rect)
        
        pygame.display.flip()
        
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        waiting = False
                        self.restart()
                    elif event.key == pygame.K_ESCAPE:
                        waiting = False
                        self.running = False
    
    def restart(self):
        """重新开始"""
        self.score = 0
        self.good_caught = 0
        self.bad_caught = 0
        self.good_missed = 0
        self.bad_missed = 0
        self.balls = []
        self.log_data = []
        self.game_messages = []
        self.start_time = time.time()
        self.game_over = False
        self.spawn_timer = 0
    
    def run(self):
        """主循环"""
        print("="*60)
        print("🎮 Catching Game - Ultrasonic Control")
        print("="*60)
        print("\nGame Instructions:")
        print("  🤚 Hand near sensor -> Paddle moves RIGHT")
        print("  🤚 Hand far sensor  -> Paddle moves LEFT")
        print("  🟢 Catch GREEN ball -> +1 point")
        print("  🔴 Catch RED ball   -> -1 point")
        print("  ❌ Miss GREEN ball  -> Missed (only good balls count)")
        print("  ⏱ Game duration: 120 seconds")
        print("\nControls:")
        print("  SPACE: Pause/Resume")
        print("  ESC: Exit game")
        print("-"*60)
        
        input("\nPress ENTER to start...")
        
        self.start_time = time.time()
        self.running = True
        
        while self.running:
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.fps_timer >= 1.0:
                self.fps = self.frame_count
                self.frame_count = 0
                self.fps_timer = current_time
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_ESCAPE:
                        self.running = False
            
            if self.paused:
                self.draw()
                self.clock.tick(30)
                continue
            
            self.elapsed_time = time.time() - self.start_time
            if self.elapsed_time >= GAME_DURATION:
                self.game_over = True
                self.show_game_over()
                if self.running:
                    continue
                break
            
            self.spawn_timer += 1 / 30
            if self.spawn_timer >= self.spawn_interval:
                self.spawn_ball()
                self.spawn_timer = 0
                self.spawn_interval = 0.8 + random.random() * 0.6
            
            self.update_paddle()
            self.update_balls()
            
            if self.frame_count % 30 == 0:
                self.log_event('frame_update')
            
            self.draw()
            self.clock.tick(30)
        
        if self.log_data:
            self.save_log()
        
        self.sensor.cleanup()
        pygame.quit()
        print("\n👋 Game exited")


if __name__ == "__main__":
    game = Game()
    game.run()
